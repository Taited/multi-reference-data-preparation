#!/usr/bin/env python3
"""Controlled multi-keyframe three-view runner for Qwen-Image-Edit-2511."""

import argparse
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

import run_s2v_recontext_qwen_image_edit_2511_jobs as qwen_base  # noqa: E402
from s2v_three_view_common import (  # noqa: E402
    NEGATIVE_PROMPT, PROMPT, append_jsonl, base_manifest_entry, border_white_ratio,
    clean_object_name, contact_sheet, load_jobs, manifest_path, read_manifests,
    reference_paths, save_image_atomic, stable_seed, utc_now,
)

DEFAULT_JOBS = ROOT / "dataset/s2v_three_view_controlled_inputs/jobs.prepared.jsonl"


def parse_args():
    parser = argparse.ArgumentParser("Qwen controlled S2V three-view generation")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-mode", choices=["full_frames", "bbox_crops"], required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--lightning-steps", type=int, default=8, choices=[8])
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if (args.width, args.height) != (1536, 768):
        raise ValueError("controlled experiment output is fixed at 1536x768")
    if args.no_lora or args.num_inference_steps != 8:
        raise ValueError("controlled Qwen experiment is fixed to the 8-step Lightning LoRA")
    jobs_path = Path(args.jobs).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
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
        f"model=Qwen-Image-Edit-2511 mode={args.input_mode} "
        f"shard={args.shard_index}/{args.num_shards} jobs={len(jobs)}/{len(selected)} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} output={output_root}", flush=True,
    )
    pipeline, failures = None, 0
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
                job, args.input_mode, "Qwen-Image-Edit-2511-Lightning-8step", seed, refs, output_path
            )
            entry.update({
                "checkpoint": str(qwen_base.LOCAL_MODEL_DIR),
                "lightning_lora": str(qwen_base.LORA_BY_STEPS[8]), "steps": 8,
                "cfg_scale": args.cfg_scale, "lora_alpha": args.lora_alpha,
                "negative_prompt": NEGATIVE_PROMPT,
                "shard_index": args.shard_index, "num_shards": args.num_shards,
            })
            if args.dry_run:
                entry["dry_run"] = True
                append_jsonl(manifest_handle, entry)
                continue
            try:
                if pipeline is None:
                    pipeline = qwen_base.load_pipeline(args)
                images = [Image.open(path).convert("RGB") for path in refs]
                started = time.monotonic()
                result = pipeline(
                    PROMPT.format(object_name=clean_object_name(job)),
                    negative_prompt=NEGATIVE_PROMPT, edit_image=images, seed=seed,
                    num_inference_steps=8, height=768, width=1536,
                    edit_image_auto_resize=True, zero_cond_t=True, cfg_scale=args.cfg_scale,
                    progress_bar_cmd=lambda values, **kwargs: values,
                ).convert("RGB")
                if result.size != (1536, 768):
                    result = result.resize((1536, 768), Image.Resampling.LANCZOS)
                save_image_atomic(result, output_path)
                reference_sheet = contact_sheet(images).resize((1536, 768), Image.Resampling.LANCZOS)
                comparison = Image.new("RGB", (3072, 768), "white")
                comparison.paste(reference_sheet, (0, 0))
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
                    "model": "Qwen-Image-Edit-2511-Lightning-8step",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(), "failed_at": utc_now(),
                })
                print(f"FAILED {name}: {type(error).__name__}: {error}", flush=True)
    print(f"FINISHED shard={args.shard_index} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
