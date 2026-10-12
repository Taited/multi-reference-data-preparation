#!/usr/bin/env python3
"""Show process health, completed counts, timing-based ETA, and GPU usage."""

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
RUNTIME = DATASET / "s2v_three_view_runtime"
CONDITIONS = [
    ("Hunyuan / full", DATASET / "s2v_three_view_hunyuan_image_3_full_frames", 1),
    ("Hunyuan / bbox", DATASET / "s2v_three_view_hunyuan_image_3_bbox_crops", 1),
    ("Qwen / full", DATASET / "s2v_three_view_qwen_image_edit_2511_full_frames", 2),
    ("Qwen / bbox", DATASET / "s2v_three_view_qwen_image_edit_2511_bbox_crops", 2),
]


def rows_in_manifests(root):
    entries = {}
    for path in sorted(root.glob("manifest.shard*-of*.jsonl")):
        if ".dryrun." in path.name:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = row.get("output_path")
            if row.get("job_name") and output and Path(output).is_file():
                entries[row["job_name"]] = row
    return list(entries.values())


def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "等待首个样本计时"
    seconds = max(0, round(seconds))
    return str(timedelta(seconds=seconds))


def process_lines():
    lines = []
    for path in sorted((RUNTIME / "pids").glob("*.pid")):
        try:
            pid = int(path.read_text().strip())
            os.kill(pid, 0)
            process_state = Path(f"/proc/{pid}/stat").read_text().split()[2]
            if process_state != "Z":
                state = "RUNNING"
            else:
                log_path = RUNTIME / "logs" / f"{path.stem}.log"
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-8192:] if log_path.is_file() else ""
                final_summary = DATASET / "s2v_three_view_2x2_comparison/summary.json"
                final_done = path.stem == "finalizer" and final_summary.is_file() and '"comparison_sheets_completed": 20' in final_summary.read_text()
                state = "DONE" if "WORKER_DONE" in log_tail or final_done else "ZOMBIE"
        except (ValueError, OSError):
            pid, state = path.read_text().strip(), "STOPPED"
        lines.append(f"  {path.stem:<28} {state:<7} pid={pid}")
    return lines


def show_once():
    prepared = DATASET / "s2v_three_view_controlled_inputs/jobs.prepared.jsonl"
    total = sum(1 for line in prepared.open(encoding="utf-8") if line.strip()) if prepared.is_file() else 20
    now = datetime.now(timezone.utc)
    print(f"S2V three-view 2x2 status — {now.isoformat(timespec='seconds')}")
    etas, grand_done = [], 0
    print("\nConditions:")
    for label, root, shards in CONDITIONS:
        rows = rows_in_manifests(root)
        done = len(rows)
        grand_done += done
        average = sum(float(row["seconds"]) for row in rows if row.get("seconds")) / done if done else None
        shard_etas = []
        for shard in range(shards):
            shard_rows = [row for row in rows if int(row.get("shard_index", 0)) == shard]
            shard_total = len(range(shard, total, shards))
            shard_average = (
                sum(float(row["seconds"]) for row in shard_rows if row.get("seconds")) / len(shard_rows)
                if shard_rows else average
            )
            shard_etas.append((shard_total - len(shard_rows)) * shard_average if shard_average else None)
        eta = max((value for value in shard_etas if value is not None), default=None)
        if eta is not None:
            etas.append(eta)
        white_values = [float(row["border_white_ratio"]) for row in rows if row.get("border_white_ratio") is not None]
        white = f" white={sum(white_values) / len(white_values):.4f}" if white_values else ""
        print(f"  {label:<18} {done:>2}/{total}  avg={format_duration(average):>12}  ETA={format_duration(eta):>12}{white}")
    overall_eta = max(etas) if len(etas) == len(CONDITIONS) else None
    overall_text = format_duration(overall_eta)
    if etas and len(etas) < len(CONDITIONS):
        overall_text = "等待所有条件首样本"
    print(f"  {'TOTAL':<18} {grand_done:>2}/{total * len(CONDITIONS)}  overall ETA={overall_text}")
    print("\nProcesses:")
    lines = process_lines()
    print("\n".join(lines) if lines else "  no PID files yet")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        print("\nGPUs (index / MiB / util%):")
        print("  " + " | ".join(" / ".join(part.strip() for part in line.split(",")) for line in gpu))
    except (OSError, subprocess.CalledProcessError):
        pass
    if grand_done == total * len(CONDITIONS):
        print(f"\nCOMPLETE: comparison sheets are in {DATASET / 's2v_three_view_2x2_comparison'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    args = parser.parse_args()
    while True:
        show_once()
        if not args.watch:
            return 0
        print("\n" + "=" * 100 + "\n", flush=True)
        time.sleep(max(1, args.watch))


if __name__ == "__main__":
    raise SystemExit(main())
