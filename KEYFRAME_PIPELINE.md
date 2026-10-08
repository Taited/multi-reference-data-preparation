# 关键帧筛选、图像生成与一致性检测流程

本流程从原始视频出发，先筛出「物体清晰、视角互不重复」的关键帧及其前景 mask，
再用 HunyuanImage-3.0 将同一物体的多视角关键帧生成为干净的物体图，最后由 Gemma 检查物体身份、
生成指令、几何质量和跨视角一致性。

代码分布在四个项目里：`mast3r/`（静止帧过滤 + 视角判定）、`gemma/`（物体识别、bbox 与生成结果审核）、
`sam2/`（抠图）、`HunyuanImage-3.0/`（多视角图像生成）。

## 路径约定与运行起点

本仓库只存放流程文档，四个代码仓库与它位于同一工作区。工作区可以放在任意位置，
下文使用 `WORKSPACE_ROOT` 表示其绝对路径，不依赖原部署机器的挂载点或用户名。
仓库链接、分支和依赖关系见 [README.md](README.md)。

```text
<workspace>/
├── multi-reference-data-preparation/  # 本文档所在仓库
├── mast3r/                           # Taited/MASt3R，克隆时指定小写目录名
├── gemma/
├── sam2/
├── HunyuanImage-3.0/
├── dataset/                          # 共享输入与输出
└── weights/                          # 自行准备的模型权重
```

在本仓库根目录执行一次，后续命令在同一个 shell 中运行：

```bash
export WORKSPACE_ROOT="$(cd .. && pwd)"
export GEMMA_MODEL_DIR="$WORKSPACE_ROOT/weights/gemma-4-31B-it"
cd "$WORKSPACE_ROOT"
```

`GEMMA_MODEL_DIR` 应指向实际的 Hugging Face 权重目录，也可以设为工作区外已有的
模型快照目录；这里不会下载或移动权重。Gemma 脚本内仍有原机器的默认路径，因此下文
显式传入 `--model "$GEMMA_MODEL_DIR"` 覆盖它。

各命令块通过 `cd "$WORKSPACE_ROOT/<project>"` 定位项目，避免连续执行时相对目录错位。
说明文字中的 `dataset/`、`weights/` 默认相对于工作区；命令中的 `dataset/` 则通过各项目
的软链接访问共享目录。新环境需自行建立 `<project>/dataset -> ../dataset`，克隆代码不会
自动提供数据、权重或虚拟环境。已有 jobs/manifests 如含旧机器的绝对图像路径，应重新生成
或迁移路径，设置 `WORKSPACE_ROOT` 不会自动改写 JSON 内容。

## 0. 总览

```
原始 mp4
  │  ① mast3r/extract_keyframes.py           灰度 MSE 排序，丢弃 90% 低差异帧
  ▼
dataset/datatang_session2_videos_fps24_012_keyframes/<video_stem>/frame_%06d.jpg
  │  ② gemma/scripts/gemma31b_keyframe_bbox.py   Gemma-4-31B：物体是否出现 / 是否模糊 / bbox
  ▼
dataset/gemma_keyframe_bbox.json
  │  ③ sam2/tools/qwen3vl_bbox_to_mask.py    以 bbox 为 prompt 做前景抠图
  ▼
dataset/gemma_keyframe_masks/<video_stem>/frame_%06d.png   （调色板索引 PNG）
  │  ④ mast3r/filter_frames_gemma.py         丢模糊/小框 + MASt3R 共视性去重
  ▼
dataset/gemma_keyframe_filtered/filtered.json + <video_stem>/frame_%06d.jpg
  │  ⑤ 准备 S2V multi-view jobs，HunyuanImage-3.0-Instruct-Distil 逐视角生成物体图
  ▼
dataset/s2v_object_only-hunyuan-distil/{full_bbox_direct_v2,extra_views_latest_all}/
  │  ⑥ gemma/scripts/gemma31b_edited_frame_consistency.py
  │     对照源图和同组其他视角，检查身份 / prompt / 几何 / 跨视角一致性
  ▼
dataset/gemma_edited_frame_consistency.jsonl
```

各步只通过 `dataset/` 下的文件传递，`gemma/dataset`、`sam2/dataset`、`mast3r/dataset`、
`HunyuanImage-3.0/dataset` 都是指向 `wentai/dataset` 的软链接，所以四个项目看到的是同一份数据，
路径可以直接对齐。

> **注意第④→⑤步的数据接口**：前四步的示例产物是 `gemma_keyframe_filtered/filtered.json`，
> 而现有 Hunyuan 批处理脚本读的是 S2V JSONL jobs（需包含 `job_name`、`sample_id`、`entity_id`、
> `frame_path`、`frame_index`、`object_name` 和归一化 `bbox_norm`）。仓库目前没有把该
> `filtered.json` 直接转成 S2V jobs 的 adapter；⑤⑥节记录的是现有 production S2V jobs 的生成与
> 审核流程，不要把 `filtered.json` 直接传给 `--jobs`。

各项目的 Python 解释器（互相独立，不要混用）：

| 项目 | 解释器 |
|---|---|
| mast3r | `mast3r/.venv/bin/python` |
| gemma | `gemma/.venv-hf/bin/python`（transformers 环境） |
| sam2 | `sam2/.venv/sam2/bin/python`，或 `cd sam2 && source env.sh` |
| HunyuanImage-3.0 | `HunyuanImage-3.0/.venv/bin/python` |

以上是现有部署的解释器位置。新机器需按各仓库说明独立安装环境；Git 仓库不包含这些虚拟环境。

---

## 前置条件

### 1) 视频目录与命名

```
dataset/datatang_session2_videos_fps24_012/
    010743_human-object-interaction_frames_64-242___yoga_block.mp4
    011358_human-object-interaction_5_frames_62-251___chocolate_bar.mp4
```

命名规则：`<video_base>___<object_slug>.mp4`（物体名里的空格用 `_` 代替）。
后续所有目录名都直接用 mp4 的文件名主干（`video_stem`），第②步是按
`<video_base>___*` 去匹配关键帧目录的，**没有 `___<slug>` 后缀会匹配不上**。

如果拿到的是没重命名的视频，可以用 `dataset/gemma_result.json` 批量改名：

```bash
python3 - <<'PY'
import json, os
root = "dataset/datatang_session2_videos_fps24_012"
g = json.load(open("dataset/gemma_result.json"))
for key, v in g.items():
    objs = v.get("objects") or []
    if not objs:
        continue
    base = os.path.basename(key).replace(".mp4", "")
    slug = objs[0]["name"].replace(" ", "_")
    src, dst = f"{root}/{base}.mp4", f"{root}/{base}___{slug}.mp4"
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
PY
```

### 2) `dataset/gemma_result.json`（视频级物体识别结果，本流程的输入，不由这四步生成）

```json
{
  "datatang_session2_videos_fps24_012/011939_human-object-interaction_1_frames_0-189.mp4": {
    "has_object_in_hand": true,
    "objects": [{"name": "brush", "confidence": 1.0}]
  }
}
```

要求：key 是「相对路径形式的 mp4 名」，只用到 basename；`objects` 为空的条目会被整段跳过。

### 3) 权重

| 用途 | 路径 |
|---|---|
| Gemma-4-31B-it | `$GEMMA_MODEL_DIR`（示例：`weights/gemma-4-31B-it`） |
| SAM2.1 hiera-large | `sam2/checkpoints/sam2.1_hiera_large.pt`（缺失时 `bash sam2/checkpoints/download_ckpts.sh`） |
| MASt3R metric | `mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` |
| HunyuanImage-3.0-Instruct-Distil | `weights/HunyuanImage-3.0-Instruct-Distil` |

---

## ① 静止帧过滤（mast3r/extract_keyframes.py）

逐帧转灰度 → INTER_AREA 缩到 128×128 → 与前一帧算 MSE；按 MSE 降序取 top-k（默认 10%）。
用排序而不是阈值，是因为近乎静止的视频里大量 diff 相同，阈值比较会「要么全留要么全丢」。

```bash
cd "$WORKSPACE_ROOT/mast3r"
.venv/bin/python extract_keyframes.py \
    --src dataset/datatang_session2_videos_fps24_012 \
    --out dataset/datatang_session2_videos_fps24_012_keyframes \
    --keep-ratio 0.10 --jobs 16
```

**输入**：`--src` 下的 `*.mp4`（只扫一层，不递归）。
**输出**：

```
dataset/datatang_session2_videos_fps24_012_keyframes/
    <video_stem>/frame_000123.jpg      # 123 = 该 mp4 里 0-based 的解码序号
    manifest.json                      # 每个视频的 fps/分辨率/实际阈值/保留帧号列表
```

注意帧号是**解码序号**，与文件名里的 `frames_64-242` 无关，不要拿它去对齐原视频区间。

| 超参 | 默认 | 说明 |
|---|---|---|
| `--keep-ratio` | 0.10 | 保留比例，0.10 即丢掉 MSE 最低的 90%。取值 (0,1] |
| `--metric` | mse | 可选 `mse` / `mad`（平均绝对差） |
| `--jpeg-quality` | 95 | 输出 JPEG 质量 |
| `--jobs` | 16 | 进程数，纯 CPU，按核数调 |
| `--no-keep-first` | 关 | 默认强制保留第 0 帧（它没有前驱帧），加上此开关则不保留 |

---

## ② Gemma 识别物体 / 判断模糊 / 输出 bbox（gemma/scripts/gemma31b_keyframe_bbox.py）

对第①步留下的每一帧，按 `gemma_result.json` 给出的物体名提问：物体是否出现、是否运动模糊、
bbox 是多少（0–1000 归一化网格，输出时同时换算成绝对像素）。
8 卡数据并行，必须用 `torchrun` 启动；worklist 按帧切分（`jobs[rank::world]`）。

```bash
cd "$WORKSPACE_ROOT/gemma"
source .venv-hf/bin/activate      # transformers 环境，不是 .venv
torchrun --nproc_per_node 8 scripts/gemma31b_keyframe_bbox.py \
    --model "$GEMMA_MODEL_DIR" \
    --output-json dataset/gemma_keyframe_bbox.json \
    --max-new-tokens 256 --save-every 50
```

**输入**：`dataset/gemma_result.json` + `dataset/datatang_session2_videos_fps24_012_keyframes/`
（这两个路径写死在脚本头部的 `GEMMA_JSON` / `KEYFRAMES_DIR`，要换数据集就改这两行）。

**输出**：`dataset/gemma_keyframe_bbox.json`

```json
{
  "meta": {"model": "gemma-4-31B-it ...", "num_videos": 220, "total_detections": 3123},
  "results": {
    "010743_human-object-interaction_frames_64-242": {
      "gemma_key": "....mp4",
      "has_object_in_hand": true,
      "objects": ["yoga block"],
      "keyframe_dir": "010743_human-object-interaction_frames_64-242___yoga_block",
      "frames": {
        "frame_000008.jpg": {
          "path": "dataset/.../frame_000008.jpg",
          "width": 1920, "height": 1080,
          "detections": [{
            "label": "yoga block",
            "box_2d_norm1000": [ymin, xmin, ymax, xmax],
            "bbox_2d": [x1, y1, x2, y2],
            "motion_blur": false
          }]
        }
      }
    }
  }
}
```

- 没检测到物体 → `detections: []`，并保留 `raw_output`（前 500 字符）便于排查。
- `motion_blur` 为 `true/false`，模型没给就是 `null`（第④步把 `null` 当作「不模糊」处理）。
- **断点续跑**：每个 rank 写 `<output>.rank<K>.tmp`，重跑时自动跳过已完成帧；rank0 在最后合并。
  中途挂了直接用同样命令重启即可。合并只在 8 个 rank 都跑完时发生，所以别删 tmp 文件。

| 超参 | 默认 | 说明 |
|---|---|---|
| `--model` | 脚本内置原部署路径 | 本文命令显式用 `$GEMMA_MODEL_DIR` 覆盖；需为 HF 格式权重目录 |
| `--output-json` | `dataset/gemma_keyframe_bbox.json` | 输出（同时决定 `.rankN.tmp` 的位置） |
| `--max-new-tokens` | 256 | 一帧多实例时别调太小，否则 JSON 被截断 |
| `--max-items` | -1 | 只跑前 N 帧，冒烟测试用 |
| `--save-every` | 50 | 每处理 N 帧落一次盘 |

单卡跑不动 31B 时可用 E4B 版本 `gemma/scripts/gemma_bbox_infer.py`（`--ckpt`/`--out`/`--limit`/
`--max-new-tokens`/`--probe`）。注意它的 prompt **不输出 `motion_blur`**，第④步的模糊过滤会失效，
只剩面积过滤和视角去重。

---

## ③ SAM2 抠图（sam2/tools/qwen3vl_bbox_to_mask.py）

脚本名带 qwen3vl 只是历史原因，它读的就是上一步那套 JSON 结构，直接把 `--bbox-json` 指到 gemma 的输出即可。
每帧把该帧所有 `bbox_2d` 一起喂给 SAM2 image predictor，输出一张调色板索引 PNG：
0=背景，i=该帧第 i 个检测（重叠时后面的覆盖前面的）。调色板固定写死在脚本里，跨帧跨视频颜色一致。

```bash
cd "$WORKSPACE_ROOT/sam2"
source env.sh          # 或直接用 .venv/sam2/bin/python
python tools/qwen3vl_bbox_to_mask.py \
    --bbox-json dataset/gemma_keyframe_bbox.json \
    --out-dir dataset/gemma_keyframe_masks \
    --checkpoint checkpoints/sam2.1_hiera_large.pt \
    --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml
```

**输入**：`gemma_keyframe_bbox.json`；帧图片路径取自 JSON 里的 `path` 字段（仓库相对路径），
所以**必须在 `sam2/` 目录下运行**，靠 `sam2/dataset -> ../dataset` 软链接解析。

**输出**：`dataset/gemma_keyframe_masks/<keyframe_dir>/frame_%06d.png`
（目录名取 JSON 里的 `keyframe_dir`，与关键帧目录一一对应；`detections` 为空的帧也会写一张全 0 的空 mask，
保证第④步能按文件名一一对上）。

| 超参 | 默认 | 说明 |
|---|---|---|
| `--bbox-json` | `dataset/qwen3vl_keyframe_bbox.json` | **必须改成 gemma 的输出** |
| `--out-dir` | `dataset/qwen3vl_keyframe_masks` | 建议改成 `dataset/gemma_keyframe_masks` |
| `--checkpoint` / `--model-cfg` | sam2.1 hiera-large | model-cfg 是 hydra 名字，不是文件路径 |
| `--device` | cuda | 单卡跑；没有 GPU 会自动退回 CPU |
| `--multimask` | 关 | 每个框出 3 个候选 mask 取最高 IoU；框有歧义时效果更好，但更慢 |
| `--overwrite` | 关 | 默认已存在的 PNG 直接跳过（天然可续跑） |

---

## ④ MASt3R 判断新视角 + 去重（mast3r/filter_frames_gemma.py）

每个物体目录内部按三段处理：

1. **模糊/无框过滤**：`detections` 为空 → `no_bbox`；主检测（面积最大那个）`motion_blur == true` → `blurred`。
2. **面积过滤**：绝对下限 `bbox 面积 / 画幅 < --min-area-frac`（默认 0.002 = 0.2%）直接丢；
   再在 log 面积空间做一次 MAD 稳健下界 `exp(median - k·1.4826·MAD)`，丢掉明显偏小的离群帧
   （帧数 ≥4 才启用 MAD，否则只用绝对下限）。
3. **视角去重**：按 bbox 面积**从大到小**遍历，把已保留帧 A 的物体像素经 MASt3R 匹配 warp 到候选帧 B，
   栅格化落点得到 Wm：

   ```
   coverage_B = |Wm ∩ mask_B| / |mask_B|    # B 的可见表面有多少已在 A 中出现过
   precision  = |Wm ∩ mask_B| / |Wm|        # 落点是否真落在 B 的物体上（信任门）
   ```

   `precision >= --min-prec` 且 `coverage_B >= --tau-dup` → 判为重复视角丢弃；
   precision 低说明这次 MASt3R 匹配不可靠，此时的 coverage 不采信，该帧保留。
   从大到小遍历保证每个视角簇留下的是最清晰/最大的那一帧。

   匹配前会把两帧各自按 mask 质心裁一个正方形窗口（边长 = `--pad` × mask 最长边），
   并把窗口内 mask 之外的像素涂成灰色 128。全图直接送 MASt3R 是不行的——它的刚性场景先验会把
   独立运动的物体当成静止背景，压根跟不上；抑制背景后才变成 object↔object 匹配。

```bash
cd "$WORKSPACE_ROOT/mast3r"
# 冒烟测试：只跑一个目录
.venv/bin/python filter_frames_gemma.py --limit 1 --copy-frames

# 全量：8 卡分片并行
mkdir -p scratchpad_filterlogs
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python filter_frames_gemma.py \
      --nshards 8 --shard $i --copy-frames --resume \
      > scratchpad_filterlogs/shard$i.log 2>&1 &
done; wait
```

**输入**：`dataset/gemma_keyframe_bbox.json`（拿 `keyframe_dir` 和逐帧 `detections`）、
关键帧 JPG 目录、第③步的 mask PNG 目录。三者的子目录名必须一致（都等于 `keyframe_dir`），
mask 文件名是同名 JPG 换 `.png`。

**输出**：

```
dataset/gemma_keyframe_filtered/
    filtered.json                       # 单进程时
    filtered.part<K>of<N>.json          # 分片时，每个分片一份
    <keyframe_dir>/frame_%06d.jpg       # --copy-frames 时拷贝保留帧
```

JSON 结构：

```json
{
  "params": {"min_area_frac": 0.002, "area_mad_k": 2.5, "tau_dup": 0.5, "min_prec": 0.5,
             "pad": 1.6, "dilate": 7, "size": 512},
  "num_dirs": 220, "total_kept": 648,
  "dirs": [{
    "keyframe_dir": "010743_..._yoga_block",
    "n_frames": 16, "n_candidates": 12, "n_after_area": 11, "n_kept": 6,
    "kept_frame_ids": [8, 55, 60, 67, 104, 136],
    "frames": [
      {"frame_id": 0,  "area_frac": 0.00353, "decision": "area_outlier"},
      {"frame_id": 7,  "decision": "no_bbox"},
      {"frame_id": 8,  "area_frac": 0.00433, "max_coverage_vs_kept": 0.4607,
       "precision": 0.8848, "decision": "kept"}
    ]
  }]
}
```

`decision` 取值：`no_bbox` / `blurred` / `area_outlier` / `no_mask`（mask 缺失或不足 50 像素）/
`duplicate`（附 `dup_of` = 被判重复于哪一帧）/ `kept`。

分片跑完后合并成一份 `filtered.json`：

```bash
cd "$WORKSPACE_ROOT/mast3r"
.venv/bin/python - <<'PY'
import json, glob
from pathlib import Path
d = Path("dataset/gemma_keyframe_filtered")
parts = sorted(d.glob("filtered.part*of*.json"))
merged, dirs = json.loads(parts[0].read_text()), []
for p in parts:
    dirs += json.loads(p.read_text())["dirs"]
dirs.sort(key=lambda x: x["keyframe_dir"])
merged["dirs"] = dirs
merged["num_dirs"] = len(dirs)
merged["total_kept"] = sum(x["n_kept"] for x in dirs)
(d / "filtered.json").write_text(json.dumps(merged, indent=2))
print(merged["num_dirs"], merged["total_kept"])
PY
```

| 超参 | 默认 | 说明 |
|---|---|---|
| `--bbox-json` | `dataset/gemma_keyframe_bbox.json` | 第②步输出 |
| `--keyframes-dir` / `--mask-dir` / `--out` | 见上文默认路径 | mask-dir 要与第③步 `--out-dir` 一致 |
| `--ckpt` | MASt3R metric 权重 | |
| `--min-area-frac` | 0.002 | bbox 面积占画幅比例的绝对下限（0.2%） |
| `--area-mad-k` | 2.5 | log 面积 MAD 下界系数，调小 = 丢得更狠 |
| `--tau-dup` | 0.5 | `coverage_B ≥ 此值` 判为同视角。调大 = 保留更多帧 |
| `--min-prec` | 0.5 | 信任门；`precision` 低于此值不采信这次匹配，候选帧保留 |
| `--pad` | 1.6 | 裁剪窗口边长 = pad × mask 最长边 |
| `--dilate` | 7 | warp 落点栅格化后的膨胀核（像素），补稀疏采样的空洞 |
| `--size` | 512 | 送进 MASt3R 的分辨率 |
| `--copy-frames` | 关 | 是否把保留帧拷进 `--out/<dir>/` |
| `--limit` / `--resume` / `--nshards` / `--shard` | 0 / 关 / 1 / 0 | 调试、续跑、分片 |

> `--resume` 是按「目录」粒度续跑的（读已有 JSON 里的 `keyframe_dir` 跳过），
> 中断的那个目录会整个重算，不会残留半份结果。

### 可视化检查

```bash
cd "$WORKSPACE_ROOT/mast3r"
.venv/bin/python viz_filtered.py --show-dropped     # 每个目录一张拼图 -> <out>/viz/<dir>.png
```
参数：`--json`（默认 `dataset/gemma_keyframe_filtered/filtered.json`）、`--keyframes-dir`、
`--mask-dir`、`--out`、`--show-dropped`（多显示一行被判重复的帧及其 `dup_of`）、`--limit`。

---

## ⑤ HunyuanImage-3.0 多视角物体图生成

本阶段使用 `HunyuanImage-3.0-Instruct-Distil` 做图生图：将每个关键视角的 bbox crop 作为参考图，
移除人物、手、其他物体和原场景，补全被遮挡部分，并生成白底、完整、居中的同一件物体。
同一 `sample_id + entity_id` 下的 anchor 和 extra views 必须全部保留分组字段，供第⑥步做跨视角比较。

当前 production 队列分为两份：

| 类型 | jobs | 输出根目录 |
|---|---|---|
| anchor | `dataset/s2v_keyframe_recontext/jobs.latest_all.jsonl` | `dataset/s2v_object_only-hunyuan-distil/full_bbox_direct_v2` |
| extra views | `dataset/s2v_keyframe_recontext/jobs.latest_all.extra_views.jsonl` | `dataset/s2v_object_only-hunyuan-distil/extra_views_latest_all` |

如果从新的 easy-HOI `index.jsonl` 开始，也可以先运行
`HunyuanImage-3.0/build_s2v_multiview_jobs.py` 生成统一的 `jobs.multiview.jsonl`；完整构建参数见
`HunyuanImage-3.0/README_S2V_MULTIVIEW.md`。该 builder 读取的是 easy-HOI index，不是第④步的
`filtered.json`。

先分别对两个队列做小规模 dry-run，检查图片、bbox crop、输出命名与 checkpoint，不会加载模型：

```bash
cd "$WORKSPACE_ROOT/HunyuanImage-3.0"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python run_s2v_multiview_hunyuan_distil.py \
  --jobs dataset/s2v_keyframe_recontext/jobs.latest_all.jsonl \
  --output-root dataset/s2v_object_only-hunyuan-distil/full_bbox_direct_v2 \
  --limit 8 --dry-run

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python run_s2v_multiview_hunyuan_distil.py \
  --jobs dataset/s2v_keyframe_recontext/jobs.latest_all.extra_views.jsonl \
  --output-root dataset/s2v_object_only-hunyuan-distil/extra_views_latest_all \
  --limit 8 --dry-run
```

确认 dry-run manifest 后去掉 `--dry-run` 和 `--limit` 正式生成：

```bash
cd "$WORKSPACE_ROOT/HunyuanImage-3.0"

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python run_s2v_multiview_hunyuan_distil.py \
  --jobs dataset/s2v_keyframe_recontext/jobs.latest_all.jsonl \
  --output-root dataset/s2v_object_only-hunyuan-distil/full_bbox_direct_v2 \
  --prompt-profile strict_v2 --input-mode bbox

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python run_s2v_multiview_hunyuan_distil.py \
  --jobs dataset/s2v_keyframe_recontext/jobs.latest_all.extra_views.jsonl \
  --output-root dataset/s2v_object_only-hunyuan-distil/extra_views_latest_all \
  --prompt-profile strict_v2 --input-mode bbox
```

Distil MeanFlow checkpoint 固定使用 **8 steps**，脚本会拒绝其他 `--diff-infer-steps`。
模型通过 `device_map=auto` 放置，建议每个 worker 暴露两张 GPU；批量运行时用
`--num-shards/--shard-index` 启动互不重叠的 worker。

每个输出根目录包含：

```text
<output-root>/
    videos/<sample_id>/
        edited/*.jpg             # Hunyuan 生成的物体图，第⑥步的待审核图
        reference_inputs/*.jpg   # 实际送入模型的 bbox crop，第⑥步的源参考图
        compare/*.jpg            # 原帧与生成图的并排预览
    manifest.shard_XX.jsonl      # source job、输入/输出路径、prompt、seed、尺寸和质量状态
    invalid_input.shard_XX.jsonl
    review_required.shard_XX.jsonl
```

默认再次运行相同命令会根据 manifest 和已存在图片跳过已完成 job，可直接断点续跑；
只有明确要重做时才加 `--overwrite`。manifest 的 `quality_status` 只检查白底边缘，
不能替代下一步的语义和跨视角一致性检测。

---

## ⑥ Gemma 生成结果一致性检测（gemma/scripts/gemma31b_edited_frame_consistency.py）

Gemma 对每张 Hunyuan 生成图同时查看：

1. 该视角的源参考 crop；
2. 当前待审核生成图；
3. 最多 `--max-peers` 张同一 `sample_id + entity_id` 的其他视角生成图。

它检查四项：是否仍是源图中的同一件物体、是否符合 Hunyuan 生成指令、部件与几何是否合理、
是否与其他生成视角保持同一物体身份。四项全部为 `true` 时 `verdict.pass` 才为 `true`。

先单进程关联 source jobs 与 Hunyuan manifests，生成稳定的审核 worklist：

```bash
cd "$WORKSPACE_ROOT/gemma"
source .venv-hf/bin/activate

python scripts/gemma31b_edited_frame_consistency.py \
  --prepare-only \
  --main-jobs dataset/s2v_keyframe_recontext/jobs.latest_all.jsonl \
  --extra-jobs dataset/s2v_keyframe_recontext/jobs.latest_all.extra_views.jsonl \
  --main-root dataset/s2v_object_only-hunyuan-distil/full_bbox_direct_v2 \
  --extra-root dataset/s2v_object_only-hunyuan-distil/extra_views_latest_all \
  --worklist dataset/gemma_edited_frame_consistency.jobs.jsonl
```

终端会打印 `source_jobs`、`matched_outputs`、`groups` 和 worklist job 数；
正式审核前应确认 `matched_outputs` 与实际完成的 Hunyuan 输出数量相符。然后复用 worklist 做 8 卡数据并行：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/gemma31b_edited_frame_consistency.py \
  --model "$GEMMA_MODEL_DIR" \
  --reuse-worklist \
  --worklist dataset/gemma_edited_frame_consistency.jobs.jsonl \
  --output-jsonl dataset/gemma_edited_frame_consistency.jsonl \
  --batch-size 1 --max-peers 3
```

成功记录示例：

```json
{
  "review_id": "source-job-name",
  "group_id": "sample-id||entity-id",
  "status": "ok",
  "verdict": {
    "pass": true,
    "object_identity_match": true,
    "prompt_match": true,
    "physically_plausible": true,
    "consistent_with_other_views": true,
    "defects": ["none"],
    "reason": "The edited object matches the reference and remains consistent across views.",
    "confidence": 0.96
  }
}
```

各 rank 会逐条追加到 `<output>.rank<N>.jsonl` 并立即落盘；同样的命令重启时只跳过
`status == "ok"` 的记录，失败项会自动重试。所有 rank 完成后，rank 0 合并为
`dataset/gemma_edited_frame_consistency.jsonl`。

### 生成—检测闭环

- `verdict.pass == true`：该生成图通过，可进入下游数据集。
- `status != "ok"`：属于推理或解析失败，直接续跑 Gemma。
- `status == "ok"` 但 `verdict.pass == false`：属于图像质量或一致性失败；根据 `defects` 和
  `reason` 回查源 crop、Hunyuan prompt/seed，重建失败 job 后重新生成，再重新准备 worklist 并审核。
- 常见缺陷包括 `identity_mismatch`、`missing_part`、`extra_part`、`deformed`、
  `wrong_material_or_color`、`person_or_hand_remains`、`bad_background` 和
  `cross_view_inconsistent`。

当前脚本负责检测和记录 verdict，**不会自动调用 Hunyuan 重生成**；失败 job 的提取、换 seed/prompt
和重新入队仍需在调度层完成。

---

## 另一条支线：select_new_views.py

`mast3r/select_new_views.py` 是同一套共视性计算的**另一种选帧策略**：不做「按面积从大到小两两去重」，
而是维持一个 anchor，逐帧与 anchor 比，`coverage_B < --tau-new`（默认 0.4）就判为新视角并接管 anchor，
适合「按时间顺序抓视角变化」的场景。它的默认路径指向 qwen3vl 那套产物，用 gemma 产物时要显式覆盖：

```bash
cd "$WORKSPACE_ROOT/mast3r"
.venv/bin/python select_new_views.py \
    --bbox-json dataset/gemma_keyframe_bbox.json \
    --mask-dir dataset/gemma_keyframe_masks \
    --out dataset/gemma_keyframe_newviews
```
输出 `newviews.json`，`decision` ∈ `anchor` / `new_view` / `same_view` / `low_conf` / `dropped_small`。
它**不读 `motion_blur`**（面积过滤用的是 mask 面积，默认 `--min-area-frac 0.004` 且叠加
`--rel-area 0.3 × 目录内中位面积`），所以正式流程用的是 `filter_frames_gemma.py`。

---

## 常见问题

- **第②步跑完 `results` 是空的**：多半是视频没按 `<base>___<slug>.mp4` 命名，
  或关键帧目录名与 mp4 主干对不上（脚本用 `base + "___"` 前缀匹配）；也可能 `gemma_result.json` 里
  `objects` 为空——这类条目会被整段跳过。
- **第③步大量 `[WARN] missing image`**：没有在 `sam2/` 目录下运行，JSON 里的 `path` 是仓库相对路径。
- **第④步某个目录 `n_kept` 为 0**：先看 `frames` 里的 `decision` 分布，`blurred` 太多就是 Gemma 判模糊过严，
  `area_outlier` 太多就放宽 `--min-area-frac` / 调大 `--area-mad-k`。
- **保留帧偏多/偏少**：只调 `--tau-dup`。调大（如 0.7）更严格地认定「同视角」→ 保留更多；
  调小（如 0.3）→ 保留更少。`--min-prec` 是匹配可信度门槛，不要拿它来控制数量。
