#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/DiffSynth-Studio/.venv/bin/python" \
  "$ROOT/build_s2v_three_view_2x2.py" --wait --interval 30
