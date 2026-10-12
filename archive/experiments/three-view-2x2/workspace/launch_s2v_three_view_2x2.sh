#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$ROOT/dataset/s2v_three_view_runtime"
LOGS="$RUNTIME/logs"
PIDS="$RUNTIME/pids"
mkdir -p "$LOGS" "$PIDS"

pid_is_live() {
  local candidate="$1"
  [[ -n "$candidate" ]] || return 1
  kill -0 "$candidate" 2>/dev/null || return 1
  [[ -r "/proc/$candidate/stat" ]] || return 1
  [[ "$(awk '{print $3}' "/proc/$candidate/stat")" != "Z" ]]
}

"$ROOT/DiffSynth-Studio/.venv/bin/python" "$ROOT/prepare_s2v_three_view_inputs.py" \
  --jobs "$ROOT/dataset/s2v_multiview_product/jobs.jsonl" \
  --output-root "$ROOT/dataset/s2v_three_view_controlled_inputs" \
  --bbox-padding 0.20

launch_worker() {
  local label="$1"
  local gpus="$2"
  local model="$3"
  local mode="$4"
  local shard="$5"
  local shards="$6"
  local pid_file="$PIDS/$label.pid"
  local log_file="$LOGS/$label.log"
  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(tr -dc '0-9' < "$pid_file")"
    if pid_is_live "$old_pid"; then
      echo "already running: $label pid=$old_pid gpus=$gpus"
      return
    fi
  fi
  nohup setsid env CUDA_VISIBLE_DEVICES="$gpus" PYTHONUNBUFFERED=1 \
    bash "$ROOT/run_s2v_three_view_worker.sh" "$model" "$mode" "$shard" "$shards" \
    </dev/null >>"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "started: $label pid=$pid gpus=$gpus log=$log_file"
}

launch_worker "hunyuan_full_shard0" "0,1" hunyuan full_frames 0 1
launch_worker "hunyuan_bbox_shard0" "2,3" hunyuan bbox_crops 0 1
launch_worker "qwen_full_shard0" "4" qwen full_frames 0 2
launch_worker "qwen_full_shard1" "5" qwen full_frames 1 2
launch_worker "qwen_bbox_shard0" "6" qwen bbox_crops 0 2
launch_worker "qwen_bbox_shard1" "7" qwen bbox_crops 1 2

finalizer_pid="$PIDS/finalizer.pid"
if [[ -f "$finalizer_pid" ]]; then
  old_finalizer="$(tr -dc '0-9' < "$finalizer_pid")"
else
  old_finalizer=""
fi
if ! pid_is_live "$old_finalizer"; then
  nohup setsid env PYTHONUNBUFFERED=1 \
    bash "$ROOT/run_s2v_three_view_finalizer.sh" \
    </dev/null >>"$LOGS/finalizer.log" 2>&1 &
  echo "$!" > "$finalizer_pid"
  echo "started: finalizer pid=$! log=$LOGS/finalizer.log"
fi

echo
echo "All GPU workers are detached from this SSH session."
echo "Progress: $ROOT/s2v_three_view_status.py"
