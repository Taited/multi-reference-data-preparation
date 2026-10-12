# Historical experiment archive

Archived on 2026-10-12. These scripts are historical snapshots, not current pipeline entry points.
No datasets, images, checkpoints, credentials, virtual environments or logs are included.

## Restore before running

See manifest.json for every original workspace-relative path and SHA-256 checksum.
Back up any existing destination, then restore each file from its archived path to its original
path under the workspace root. Do not run scripts directly here: their relative paths and imports
expect the original layout. Restoring a group does not guarantee compatibility with current dependencies.

## Three-view 2x2 experiment

experiments/three-view-2x2/workspace/ preserves the original workspace layout:
shared scripts and experiment notes at its root, plus HunyuanImage-3.0/ and DiffSynth-Studio/ runners.
Restore these files together into the parent workspace. The Qwen runner additionally requires
DiffSynth-Studio/run_s2v_recontext_qwen_image_edit_2511_jobs.py and both original project environments.
Existing inputs and generated outputs remain under the shared workspace dataset/ directory.
