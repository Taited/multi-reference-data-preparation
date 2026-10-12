#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?model is required}"
INPUT_MODE="${2:?input mode is required}"
SHARD_INDEX="${3:?shard index is required}"
NUM_SHARDS="${4:?number of shards is required}"

JOBS="$ROOT/dataset/s2v_three_view_controlled_inputs/jobs.prepared.jsonl"
case "$MODEL:$INPUT_MODE" in
  hunyuan:full_frames)
    PYTHON="$ROOT/HunyuanImage-3.0/.venv/bin/python"
    RUNNER="$ROOT/HunyuanImage-3.0/run_s2v_three_view_controlled.py"
    OUTPUT="$ROOT/dataset/s2v_three_view_hunyuan_image_3_full_frames"
    ;;
  hunyuan:bbox_crops)
    PYTHON="$ROOT/HunyuanImage-3.0/.venv/bin/python"
    RUNNER="$ROOT/HunyuanImage-3.0/run_s2v_three_view_controlled.py"
    OUTPUT="$ROOT/dataset/s2v_three_view_hunyuan_image_3_bbox_crops"
    ;;
  qwen:full_frames)
    PYTHON="$ROOT/DiffSynth-Studio/.venv/bin/python"
    RUNNER="$ROOT/DiffSynth-Studio/run_s2v_three_view_controlled.py"
    OUTPUT="$ROOT/dataset/s2v_three_view_qwen_image_edit_2511_full_frames"
    ;;
  qwen:bbox_crops)
    PYTHON="$ROOT/DiffSynth-Studio/.venv/bin/python"
    RUNNER="$ROOT/DiffSynth-Studio/run_s2v_three_view_controlled.py"
    OUTPUT="$ROOT/dataset/s2v_three_view_qwen_image_edit_2511_bbox_crops"
    ;;
  *)
    echo "Unknown worker configuration: $MODEL $INPUT_MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT"
attempt=1
max_attempts=3
while (( attempt <= max_attempts )); do
  echo "WORKER_START model=$MODEL mode=$INPUT_MODE shard=$SHARD_INDEX/$NUM_SHARDS attempt=$attempt at=$(date --iso-8601=seconds)"
  "$PYTHON" "$RUNNER" \
    --jobs "$JOBS" \
    --output-root "$OUTPUT" \
    --input-mode "$INPUT_MODE" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS"
  code=$?
  if (( code == 0 )); then
    echo "WORKER_DONE model=$MODEL mode=$INPUT_MODE shard=$SHARD_INDEX/$NUM_SHARDS at=$(date --iso-8601=seconds)"
    exit 0
  fi
  echo "WORKER_RETRY model=$MODEL mode=$INPUT_MODE shard=$SHARD_INDEX/$NUM_SHARDS exit=$code attempt=$attempt"
  ((attempt++))
  if (( attempt <= max_attempts )); then
    sleep 30
  fi
done
echo "WORKER_GAVE_UP model=$MODEL mode=$INPUT_MODE shard=$SHARD_INDEX/$NUM_SHARDS at=$(date --iso-8601=seconds)" >&2
exit 1
