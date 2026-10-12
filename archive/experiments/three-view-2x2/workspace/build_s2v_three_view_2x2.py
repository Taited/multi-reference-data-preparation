#!/usr/bin/env python3
"""Incrementally build model-by-input 2x2 review sheets, optionally waiting for completion."""

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from s2v_three_view_common import load_jobs, read_manifests, save_image_atomic, utc_now


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
PREPARED = DATASET / "s2v_three_view_controlled_inputs/jobs.prepared.jsonl"
OUTPUT_ROOT = DATASET / "s2v_three_view_2x2_comparison"
CONDITIONS = [
    ("HunyuanImage-3.0 | FULL FRAMES", DATASET / "s2v_three_view_hunyuan_image_3_full_frames"),
    ("HunyuanImage-3.0 | BBOX CROPS", DATASET / "s2v_three_view_hunyuan_image_3_bbox_crops"),
    ("Qwen-Image-Edit-2511 | FULL FRAMES", DATASET / "s2v_three_view_qwen_image_edit_2511_full_frames"),
    ("Qwen-Image-Edit-2511 | BBOX CROPS", DATASET / "s2v_three_view_qwen_image_edit_2511_bbox_crops"),
]


def font(size=24):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def image_path(root, name):
    return root / "edited" / f"{name}.png"


def make_sheet(job, destination):
    name = job["job_name"]
    paths = [image_path(root, name) for _, root in CONDITIONS]
    if not all(path.is_file() for path in paths):
        return False
    label_height = 44
    canvas = Image.new("RGB", (3072, (768 + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    text_font = font(23)
    for index, ((label, _), path) in enumerate(zip(CONDITIONS, paths)):
        column, row = index % 2, index // 2
        x, y = column * 1536, row * (768 + label_height)
        draw.rectangle((x, y, x + 1536, y + label_height), fill=(30, 30, 30))
        draw.text((x + 18, y + 8), label, fill="white", font=text_font)
        image = Image.open(path).convert("RGB")
        if image.size != (1536, 768):
            image = image.resize((1536, 768), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y + label_height))
    save_image_atomic(canvas, destination, fmt="JPEG", quality=92)
    return True


def make_overviews(jobs, page_size=10):
    """Compact pages with one row per job and four aligned condition thumbnails."""
    overview_dir = OUTPUT_ROOT / "overviews"
    overview_dir.mkdir(parents=True, exist_ok=True)
    text_font, small_font = font(18), font(15)
    for page_start in range(0, len(jobs), page_size):
        page_jobs = jobs[page_start:page_start + page_size]
        header_height, row_label_height, image_height = 40, 26, 192
        canvas = Image.new(
            "RGB", (1536, header_height + len(page_jobs) * (row_label_height + image_height)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for column, (label, _) in enumerate(CONDITIONS):
            x = column * 384
            draw.rectangle((x, 0, x + 384, header_height), fill=(30, 30, 30))
            draw.text((x + 8, 9), label.replace(" | ", " / "), fill="white", font=small_font)
        for row_index, job in enumerate(page_jobs):
            y = header_height + row_index * (row_label_height + image_height)
            label = f"{page_start + row_index + 1:02d}  {job.get('object_name') or job['job_name']}"
            draw.rectangle((0, y, 1536, y + row_label_height), fill=(225, 225, 225))
            draw.text((8, y + 3), label, fill=(20, 20, 20), font=text_font)
            for column, (_, root) in enumerate(CONDITIONS):
                image = Image.open(image_path(root, job["job_name"])).convert("RGB")
                image = image.resize((384, image_height), Image.Resampling.LANCZOS)
                canvas.paste(image, (column * 384, y + row_label_height))
        destination = overview_dir / f"overview_page_{page_start // page_size + 1:02d}.jpg"
        save_image_atomic(canvas, destination, fmt="JPEG", quality=88)


def condition_summary(label, root, total):
    entries = read_manifests(root)
    valid = [row for row in entries.values() if row.get("output_path") and Path(row["output_path"]).is_file()]
    times = [float(row["seconds"]) for row in valid if row.get("seconds") is not None]
    whites = [float(row["border_white_ratio"]) for row in valid if row.get("border_white_ratio") is not None]
    return {
        "condition": label,
        "output_root": str(root),
        "completed": len(valid),
        "total": total,
        "mean_generation_seconds": round(sum(times) / len(times), 3) if times else None,
        "mean_border_white_ratio": round(sum(whites) / len(whites), 6) if whites else None,
    }


def write_summary(jobs, sheets):
    summary = {
        "updated_at": utc_now(),
        "jobs": len(jobs),
        "comparison_sheets_completed": sheets,
        "conditions": [condition_summary(label, root, len(jobs)) for label, root in CONDITIONS],
        "metric_note": (
            "Border-white ratio checks only background compliance; it cannot judge object identity, "
            "hand removal, exact view compliance, reconstruction quality, or overall winner. Use the "
            "paired 2x2 sheets for those criteria."
        ),
        "layout": {
            "top_left": CONDITIONS[0][0], "top_right": CONDITIONS[1][0],
            "bottom_left": CONDITIONS[2][0], "bottom_right": CONDITIONS[3][0],
        },
    }
    path = OUTPUT_ROOT / "summary.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_once(jobs):
    sheets_dir = OUTPUT_ROOT / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for job in jobs:
        destination = sheets_dir / f"{job['job_name']}.jpg"
        if destination.is_file() or make_sheet(job, destination):
            completed += 1
    write_summary(jobs, completed)
    if completed == len(jobs):
        make_overviews(jobs)
    print(f"2x2 comparison sheets: {completed}/{len(jobs)}", flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    jobs = load_jobs(PREPARED)
    while True:
        completed = build_once(jobs)
        if completed == len(jobs) or not args.wait:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
