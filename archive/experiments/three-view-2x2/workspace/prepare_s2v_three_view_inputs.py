#!/usr/bin/env python3
"""Prepare one shared set of full-frame and bbox-crop inputs for both models."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from s2v_three_view_common import bbox_crop, contact_sheet, load_jobs, save_image_atomic


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser("Prepare controlled S2V three-view references")
    parser.add_argument("--jobs", default=str(ROOT / "dataset/s2v_multiview_product/jobs.jsonl"))
    parser.add_argument(
        "--output-root", default=str(ROOT / "dataset/s2v_three_view_controlled_inputs")
    )
    parser.add_argument("--bbox-padding", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bbox_padding < 0:
        raise ValueError("--bbox-padding must be non-negative")
    jobs_path = Path(args.jobs).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    crop_root = output_root / "bbox_crops"
    audit_root = output_root / "audit"
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    for job in load_jobs(jobs_path):
        full_paths, crop_paths, crop_rows = [], [], []
        full_images, crop_images = [], []
        for index, view in enumerate(job.get("views") or []):
            source_path = Path(view["path"]).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"{job['job_name']}: missing {source_path}")
            source = Image.open(source_path).convert("RGB")
            crop, crop_box = bbox_crop(source, view.get("bbox_norm"), args.bbox_padding)
            crop_path = crop_root / job["job_name"] / f"view-{index:02d}.png"
            if args.overwrite or not crop_path.is_file():
                save_image_atomic(crop, crop_path)
            full_paths.append(str(source_path))
            crop_paths.append(str(crop_path.resolve()))
            full_images.append(source)
            crop_images.append(crop)
            crop_rows.append({
                "source_path": str(source_path),
                "source_size": list(source.size),
                "bbox_norm": view.get("bbox_norm"),
                "crop_box_px": list(crop_box),
                "crop_size": list(crop.size),
                "crop_path": str(crop_path.resolve()),
            })
        if len(full_paths) < 2:
            raise ValueError(f"{job['job_name']}: fewer than two keyframes")
        audit = Image.new("RGB", (384 * len(full_images), 768), "white")
        audit.paste(contact_sheet(full_images), (0, 0))
        audit.paste(contact_sheet(crop_images), (0, 384))
        audit_path = audit_root / f"{job['job_name']}.jpg"
        save_image_atomic(audit, audit_path, fmt="JPEG", quality=92)
        row = dict(job)
        row["controlled_inputs"] = {
            "full_frames": full_paths,
            "bbox_crops": crop_paths,
            "bbox_padding": args.bbox_padding,
            "crop_metadata": crop_rows,
            "audit_sheet": str(audit_path.resolve()),
        }
        prepared.append(row)
    prepared_path = output_root / "jobs.prepared.jsonl"
    temporary = prepared_path.with_name(f".{prepared_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, prepared_path)
    config = {
        "source_jobs": str(jobs_path),
        "prepared_jobs": str(prepared_path.resolve()),
        "jobs": len(prepared),
        "bbox_padding": args.bbox_padding,
        "bbox_definition": "target-object bbox plus proportional context on all four sides",
        "full_and_crop_order_identical": True,
    }
    config_path = output_root / "preparation.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(prepared)} jobs -> {prepared_path}")
    print(f"Shared bbox crops -> {crop_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
