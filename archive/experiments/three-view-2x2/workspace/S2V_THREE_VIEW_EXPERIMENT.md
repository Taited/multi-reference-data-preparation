# S2V three-view 2x2 controlled experiment

This experiment compares two variables on the same 20 video/object jobs:

| | Full video keyframes | Shared bbox crops |
|---|---|---|
| HunyuanImage-3.0 Distil | `s2v_three_view_hunyuan_image_3_full_frames` | `s2v_three_view_hunyuan_image_3_bbox_crops` |
| Qwen-Image-Edit-2511 Lightning | `s2v_three_view_qwen_image_edit_2511_full_frames` | `s2v_three_view_qwen_image_edit_2511_bbox_crops` |

All output directories are under `dataset/`. Both models consume the exact same source files for a
given input setting. The bbox inputs are prepared once from the supplied normalized target-object
bbox with 20% proportional context on each side. The positive prompt, target views
(front/right-side/top), output resolution (1536x768), job order, and per-job seed are identical.
Each model uses its corresponding local `.venv` and its recommended distilled 8-step setup.

GPU allocation is fixed: Hunyuan/full uses 0-1, Hunyuan/bbox uses 2-3, Qwen/full uses 4-5, and
Qwen/bbox uses 6-7. Hunyuan needs two visible GPUs for automatic checkpoint placement; each Qwen
condition is split into two independent single-GPU shards.

Commands:

```bash
./launch_s2v_three_view_2x2.sh
./s2v_three_view_status.py
./s2v_three_view_status.py --watch 30
```

Workers use `nohup` plus a new session, so they survive SSH disconnects. Successful jobs are resumed
from per-shard manifests. Each worker retries unresolved failures up to three passes. As soon as all
four outputs for a job exist, the background finalizer creates a labeled 2x2 sheet in
`dataset/s2v_three_view_2x2_comparison/sheets/`.

Each of the four setting directories is also self-contained for documentation:

- `input_images/<job_name>/view-XX.png` contains every individual model input.
- `input_contact_sheets/<job_name>.jpg` is a labeled, document-ready input strip.
- `input_images.jsonl` maps every job to those exported files.

The status report includes mean border-white compliance, but that metric is not an overall quality
score. Determine the winner from the paired sheets using object identity, complete hand/person and
scene removal, correctness of the three requested views, reconstruction plausibility, and background
purity. The layout keeps model fixed across columns and input setting fixed across rows, so both main
effects can be reviewed without changing any other experiment variable.
