#!/usr/bin/env python3
"""Export self-contained, document-ready inputs into all four experiment directories."""

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from s2v_three_view_common import load_jobs, reference_paths, save_image_atomic


ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "dataset/s2v_three_view_controlled_inputs/jobs.prepared.jsonl"
SETTINGS = [
    ("full_frames", ROOT / "dataset/s2v_three_view_hunyuan_image_3_full_frames"),
    ("bbox_crops", ROOT / "dataset/s2v_three_view_hunyuan_image_3_bbox_crops"),
    ("full_frames", ROOT / "dataset/s2v_three_view_qwen_image_edit_2511_full_frames"),
    ("bbox_crops", ROOT / "dataset/s2v_three_view_qwen_image_edit_2511_bbox_crops"),
]


def get_font(size):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def export_image(source: Path, destination: Path):
    """Decode and save consistently, so exports do not depend on external source paths."""
    if destination.is_file():
        return Image.open(destination).convert("RGB")
    image = Image.open(source).convert("RGB")
    save_image_atomic(image, destination, fmt="PNG", compress_level=3)
    return image


def make_document_sheet(images, object_name, input_mode):
    cell_width, cell_height = 512, 512
    label_height = 76
    canvas = Image.new("RGB", (cell_width * len(images), cell_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, view_font = get_font(25), get_font(19)
    title = f"Input: {'full keyframes' if input_mode == 'full_frames' else 'bbox crops'} | {object_name}"
    draw.rectangle((0, 0, canvas.width, label_height), fill=(32, 32, 32))
    draw.text((18, 9), title, fill="white", font=title_font)
    for index, image in enumerate(images):
        cell = ImageOps.contain(image, (cell_width, cell_height), Image.Resampling.LANCZOS)
        x = index * cell_width + (cell_width - cell.width) // 2
        y = label_height + (cell_height - cell.height) // 2
        canvas.paste(cell, (x, y))
        draw.rectangle((index * cell_width + 8, label_height + 8,
                        index * cell_width + 112, label_height + 38), fill=(32, 32, 32))
        draw.text((index * cell_width + 17, label_height + 11), f"view {index + 1:02d}",
                  fill="white", font=view_font)
    return canvas


def main():
    jobs = load_jobs(JOBS)
    for input_mode, setting_root in SETTINGS:
        individual_root = setting_root / "input_images"
        sheet_root = setting_root / "input_contact_sheets"
        individual_root.mkdir(parents=True, exist_ok=True)
        sheet_root.mkdir(parents=True, exist_ok=True)
        index_rows = []
        for job in jobs:
            name = job["job_name"]
            sources = reference_paths(job, input_mode)
            exported, images = [], []
            for index, source in enumerate(sources):
                destination = individual_root / name / f"view-{index:02d}.png"
                images.append(export_image(source, destination))
                exported.append(str(destination.resolve()))
            object_name = " ".join(str(job.get("object_name") or "object").split())
            sheet = make_document_sheet(images, object_name, input_mode)
            sheet_path = sheet_root / f"{name}.jpg"
            if not sheet_path.is_file():
                save_image_atomic(sheet, sheet_path, fmt="JPEG", quality=95)
            index_rows.append({
                "job_name": name,
                "object_name": object_name,
                "input_mode": input_mode,
                "input_images": exported,
                "input_contact_sheet": str(sheet_path.resolve()),
            })
        index_path = setting_root / "input_images.jsonl"
        temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in index_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, index_path)
        print(
            f"{setting_root.name}: {sum(len(row['input_images']) for row in index_rows)} images, "
            f"{len(index_rows)} contact sheets"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
