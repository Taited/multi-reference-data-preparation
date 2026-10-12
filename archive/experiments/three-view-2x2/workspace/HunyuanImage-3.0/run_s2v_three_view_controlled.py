#!/usr/bin/env python3
"""Controlled multi-keyframe three-view runner for HunyuanImage 3 Distil."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from s2v_three_view_common import (  # noqa: E402
    PROMPT, SYSTEM_PROMPT, append_jsonl, base_manifest_entry, border_white_ratio,
    clean_object_name, contact_sheet, load_jobs, manifest_path, read_manifests,
    reference_paths, save_image_atomic, stable_seed, utc_now,
)

DEFAULT_JOBS = ROOT / "dataset/s2v_three_view_controlled_inputs/jobs.prepared.jsonl"
DEFAULT_MODEL = ROOT / "weights/HunyuanImage-3.0-Instruct-Distil"


def parse_args():
    parser = argparse.ArgumentParser("Hunyuan controlled S2V three-view generation")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-mode", choices=["full_frames", "bbox_crops"], required=True)
    parser.add_argument("--model-id", default=str(DEFAULT_MODEL))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", default="768x1536")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--gpu-reserve-gib", type=float, default=8.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-context-length", type=int, default=27000,
        help="Raised above the 22,800 checkpoint default so four 1024-grid references fit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_model(model_id: Path):
    required = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json"]
    missing = [name for name in required if not (model_id / name).is_file()]
    shards = sorted(model_id.glob("model-*-of-0032.safetensors"))
    if missing or len(shards) != 32:
        raise FileNotFoundError(
            f"Incomplete Hunyuan checkpoint {model_id}: missing={missing}, shards={len(shards)}/32"
        )
    config = json.loads((model_id / "config.json").read_text(encoding="utf-8"))
    if not (config.get("cfg_distilled") and config.get("use_meanflow")):
        raise ValueError("The controlled runner requires HunyuanImage-3.0-Instruct-Distil")


def load_model(model_id: Path, args):
    import torch
    from hunyuan_image_3 import HunyuanImage3ForCausalMM

    max_memory = {}
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_gib = free_bytes / 1024**3
        budget = free_gib - args.gpu_reserve_gib
        if budget <= 0:
            raise RuntimeError(f"visible GPU {index} has only {free_gib:.1f} GiB free")
        max_memory[index] = f"{budget:.0f}GiB"
        print(f"visible GPU {index}: {free_gib:.1f} GiB free, model budget={budget:.0f} GiB", flush=True)
    model = HunyuanImage3ForCausalMM.from_pretrained(
        str(model_id), attn_implementation="sdpa", dtype="auto", device_map="auto",
        moe_impl="eager", moe_drop_tokens=True, local_files_only=True, max_memory=max_memory,
    )
    model.load_tokenizer(str(model_id))
    model.generation_config.diff_guidance_scale = args.guidance_scale
    # Four conditioning images occupy up to ~25.5k tokens. The tokenizer explicitly supports a
    # larger max_length, and the custom rotary embedding is computed from the runtime positions.
    # Two H200s provide sufficient KV/attention headroom for this modest extension.
    model.generation_config.max_length = args.max_context_length
    model.config.max_position_embeddings = args.max_context_length
    return model


def main():
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.steps != 8:
        raise ValueError("Hunyuan Distil controlled experiment is fixed to 8 steps")
    jobs_path = Path(args.jobs).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_id = Path(args.model_id).expanduser().resolve()
    validate_model(model_id)
    selected = load_jobs(jobs_path, args.limit)
    jobs = [job for index, job in enumerate(selected) if index % args.num_shards == args.shard_index]
    if not jobs:
        raise ValueError("No jobs assigned to this shard")
    output_dir, compare_dir = output_root / "edited", output_root / "compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    completed_rows = read_manifests(output_root)
    completed = {
        name for name, row in completed_rows.items()
        if row.get("output_path") and Path(row["output_path"]).is_file()
    }
    manifest = manifest_path(output_root, args.shard_index, args.num_shards, args.dry_run)
    error_path = output_root / f"errors.shard{args.shard_index}-of{args.num_shards}.jsonl"
    print(
        f"model=HunyuanImage-3.0-Instruct-Distil mode={args.input_mode} "
        f"shard={args.shard_index}/{args.num_shards} jobs={len(jobs)}/{len(selected)} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} output={output_root}", flush=True,
    )
    model, failures = None, 0
    with manifest.open("a", encoding="utf-8") as manifest_handle, \
            error_path.open("a", encoding="utf-8") as error_handle:
        for job in tqdm(jobs, unit="product"):
            name = job["job_name"]
            output_path = output_dir / f"{name}.png"
            if not args.overwrite and name in completed and output_path.is_file():
                continue
            refs = reference_paths(job, args.input_mode)
            seed = stable_seed(name, args.seed)
            entry = base_manifest_entry(
                job, args.input_mode, "HunyuanImage-3.0-Instruct-Distil", seed, refs, output_path
            )
            entry.update({
                "checkpoint": str(model_id), "steps": args.steps,
                "guidance_scale": args.guidance_scale, "image_size_argument": args.image_size,
                "max_context_length": args.max_context_length,
                "shard_index": args.shard_index, "num_shards": args.num_shards,
            })
            if args.dry_run:
                entry["dry_run"] = True
                append_jsonl(manifest_handle, entry)
                continue
            try:
                if model is None:
                    model = load_model(model_id, args)
                started = time.monotonic()
                _, samples = model.generate_image(
                    prompt=PROMPT.format(object_name=clean_object_name(job)),
                    image=[str(path) for path in refs], seed=seed, image_size=args.image_size,
                    use_system_prompt="custom", system_prompt=SYSTEM_PROMPT, bot_task="image",
                    infer_align_image_size=False, diff_infer_steps=args.steps,
                    max_new_tokens=args.max_new_tokens, verbose=0,
                )
                result = samples[0].convert("RGB")
                if result.size != (1536, 768):
                    result = result.resize((1536, 768), Image.Resampling.LANCZOS)
                save_image_atomic(result, output_path)
                reference_sheet = contact_sheet([Image.open(path).convert("RGB") for path in refs])
                left = reference_sheet.resize((1536, 768), Image.Resampling.LANCZOS)
                comparison = Image.new("RGB", (3072, 768), "white")
                comparison.paste(left, (0, 0))
                comparison.paste(result, (1536, 0))
                save_image_atomic(comparison, compare_dir / f"{name}.jpg", fmt="JPEG", quality=90)
                entry.update({
                    "actual_output_size": list(result.size),
                    "border_white_ratio": border_white_ratio(result),
                    "seconds": round(time.monotonic() - started, 3), "completed_at": utc_now(),
                })
                append_jsonl(manifest_handle, entry)
                print(
                    f"COMPLETED {name} seconds={entry['seconds']:.1f} "
                    f"white={entry['border_white_ratio']:.4f}", flush=True,
                )
            except Exception as error:
                failures += 1
                append_jsonl(error_handle, {
                    "job_name": name, "input_mode": args.input_mode,
                    "model": "HunyuanImage-3.0-Instruct-Distil",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(), "failed_at": utc_now(),
                })
                print(f"FAILED {name}: {type(error).__name__}: {error}", flush=True)
    print(f"FINISHED shard={args.shard_index} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
