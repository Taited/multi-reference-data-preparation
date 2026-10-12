"""Shared definitions for the controlled S2V three-view experiment."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps


EXPERIMENT_VERSION = "s2v-three-view-2x2-v1"

SYSTEM_PROMPT = """You are a precise multi-reference product rendering engine. Every input image
shows the same physical target object in a video. Combine evidence from all inputs to preserve the
object's identity, geometry, proportions, materials, colours, texture, construction, and authentic
visible markings. Return exactly one edited image and no explanatory text."""

PROMPT = """The input images are video keyframes of the exact same {object_name}. Study all of them
together and use only their consistent visual evidence to reconstruct that same physical object.
Remove every person, hand, finger, arm, body part, other object, and the complete original scene.
Reconstruct portions hidden by the interaction conservatively, continuing the object's visible
shape, proportions, material, colour, texture, construction, and genuine markings.

Create one clean orthographic three-view product sheet containing exactly three depictions of this
same {object_name}, arranged left to right in one horizontal row: (1) a straight-on FRONT view at
object-centre height, (2) a straight-on RIGHT-SIDE view at the same height, and (3) a true TOP view
photographed vertically downward at exactly 90 degrees. Do not use three-quarter, oblique, elevated,
or isometric angles. Keep the three depictions consistent in identity, appearance, lighting, scale,
and orientation. Make each depiction whole, separated, and fully visible with generous even spacing.

The entire canvas and all space between views must be uniform pure white RGB(255,255,255), with soft
even studio lighting and at most a faint contact shadow. Show no people, hands, original background,
props, dividers, labels, captions, arrows, borders, watermarks, invented writing, invented logos,
extra objects, or extra views. Photorealistic, sharp, catalogue-grade product photography, not a
drawing, diagram, or collage of the source images."""

NEGATIVE_PROMPT = (
    "person, people, hand, hands, fingers, arm, body, clothing, face, original background, room, "
    "table, prop, other object, different products, inconsistent identity, duplicate view, fourth "
    "object, three-quarter view, oblique view, isometric view, elevated front view, incorrect top "
    "view, grey background, gradient background, panel divider, border, caption, view label, arrow, "
    "watermark, invented text, invented logo, distorted shape, cropped object, blurry, illustration"
)


def clean_object_name(job: dict) -> str:
    return " ".join(str(job.get("object_name") or "object").split())


def stable_seed(job_name: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256(
        f"{EXPERIMENT_VERSION}|{job_name}|{base_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def load_jobs(path: Path, limit: int = 0) -> list[dict]:
    jobs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                jobs.append(json.loads(line))
                if limit and len(jobs) >= limit:
                    break
    return jobs


def reference_paths(job: dict, input_mode: str) -> list[Path]:
    try:
        values = job["controlled_inputs"][input_mode]
    except KeyError as error:
        raise KeyError(
            f"{job.get('job_name')}: prepared jobs lack controlled_inputs.{input_mode}"
        ) from error
    paths = [Path(value).expanduser().resolve() for value in values]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{job.get('job_name')}: missing reference {missing[0]}")
    if len(paths) < 2:
        raise ValueError(f"{job.get('job_name')}: only {len(paths)} usable references")
    return paths


def bbox_crop(image: Image.Image, bbox_norm: list[float], padding: float):
    if not bbox_norm or len(bbox_norm) != 4:
        raise ValueError("bbox crop requires four normalized bbox values")
    x0, y0, x1, y1 = [float(value) for value in bbox_norm]
    x0, x1 = sorted((x0 * image.width, x1 * image.width))
    y0, y1 = sorted((y0 * image.height, y1 * image.height))
    width, height = x1 - x0, y1 - y0
    if width < 2 or height < 2:
        raise ValueError(f"invalid bbox after scaling: {bbox_norm}")
    crop_box = (
        max(0, round(x0 - width * padding)),
        max(0, round(y0 - height * padding)),
        min(image.width, round(x1 + width * padding)),
        min(image.height, round(y1 + height * padding)),
    )
    return image.crop(crop_box), crop_box


def contact_sheet(images: list[Image.Image], thumb=(384, 384)) -> Image.Image:
    cells = []
    for image in images:
        cell = ImageOps.contain(image.convert("RGB"), thumb, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", thumb, "white")
        canvas.paste(cell, ((thumb[0] - cell.width) // 2, (thumb[1] - cell.height) // 2))
        cells.append(canvas)
    sheet = Image.new("RGB", (thumb[0] * len(cells), thumb[1]), "white")
    for index, cell in enumerate(cells):
        sheet.paste(cell, (index * thumb[0], 0))
    return sheet


def border_white_ratio(image: Image.Image, band=0.05, threshold=245) -> float:
    gray = image.convert("L")
    width, height = gray.size
    bw, bh = max(1, round(width * band)), max(1, round(height * band))
    regions = (
        gray.crop((0, 0, width, bh)),
        gray.crop((0, height - bh, width, height)),
        gray.crop((0, 0, bw, height)),
        gray.crop((width - bw, 0, width, height)),
    )
    white = total = 0
    for region in regions:
        histogram = region.histogram()
        white += sum(histogram[threshold:])
        total += sum(histogram)
    return round(white / max(1, total), 6)


def manifest_path(output_root: Path, shard_index: int, num_shards: int, dry_run=False) -> Path:
    suffix = ".dryrun" if dry_run else ""
    return output_root / f"manifest.shard{shard_index}-of{num_shards}{suffix}.jsonl"


def read_manifests(output_root: Path) -> dict[str, dict]:
    entries = {}
    for path in sorted(output_root.glob("manifest.shard*-of*.jsonl")):
        if ".dryrun." in path.name:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("job_name"):
                    entries[row["job_name"]] = row
    return entries


def append_jsonl(handle, entry: dict) -> None:
    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def save_image_atomic(image: Image.Image, path: Path, fmt="PNG", **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format=fmt, **kwargs)
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def base_manifest_entry(job: dict, input_mode: str, model: str, seed: int,
                        refs: list[Path], output_path: Path) -> dict:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "job_name": job["job_name"],
        "sample_id": job.get("sample_id"),
        "entity_id": job.get("entity_id"),
        "object_name": clean_object_name(job),
        "model": model,
        "input_mode": input_mode,
        "bbox_padding": job.get("controlled_inputs", {}).get("bbox_padding"),
        "reference_paths": [str(path) for path in refs],
        "output_path": str(output_path),
        "seed": seed,
        "prompt": PROMPT.format(object_name=clean_object_name(job)),
        "target_views": ["front", "right_side", "top"],
        "output_size": [1536, 768],
    }
