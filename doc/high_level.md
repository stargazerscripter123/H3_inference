Q:

我现在想用minimax h3 替换条wan2.2 来做首帧+尾帧视频生成。research 技术报告，huggingface / github repo，网上成功案例，给我出一个详细的 可行的 plan。包括所有的步骤，从inference 开始，然后深入到lora 来解决off the shelf model 解决不了的问题

A:

# MiniMax H3 替换 Wan2.2 首帧＋尾帧视频生成：研究结论与执行方案

## 一、结论

**可以替换，但目前不应该直接下线 Wan2.2。**

最合理的架构是：

1. **先把 MiniMax H3 接成一条并行的高质量 FL2VA 路线**；
2. 用 H3 API 建立能力上限；
3. 再用本地 ComfyUI/Hugging Face 权重验证质量、显存和延迟；
4. 只有在确认失败模式确实属于模型能力缺口，而不是输入、量化、scheduler 或 workflow 问题后，才进入 LoRA；
5. 在 H3 达到质量、稳定性和吞吐门槛前，保留你现在的 Wan2.2 作为低延迟和故障回退路线。

H3 的优势非常明确：它官方原生支持首帧、尾帧或首尾帧同时输入，并联合生成 24 FPS 视频和原生立体声音频；API 支持 4–15 秒、768P/2K。H3 于 **2026 年 7 月 31 日**发布，但官方完整 Technical Report 仍标记为“即将发布”，因此目前训练部分必须根据开放权重和开源实现反向还原，而不是照搬官方训练 recipe。([MiniMax][1])

本地生态在过去几天刚刚形成：

* ComfyUI 原生 H3 支持已经在 **2026 年 8 月 3 日**合并；
* Hugging Face Diffusers 的 H3 实现已经提交，但目前仍是 draft PR；
* DiffSynth-Studio 已有 BF16、NF4、CPU/disk offload 和首尾帧示例，但 PR 仍未合并；
* 我没有找到已经公开、可复现的 MiniMax H3 LoRA 训练成功案例，也没有在 musubi-tuner、ai-toolkit、OneTrainer、sd-scripts 中找到现成 H3 trainer。

因此现在的准确定位是：

> **H3 inference 已经可行；H3 LoRA 在架构上可行，但不是 turnkey，需要自己补训练器。**

---

## 二、H3 与你当前 Wan2.2 路线的关键差异

| 维度         | MiniMax H3                             | 你当前 Wan2.2 pipeline              |
| ---------- | -------------------------------------- | -------------------------------- |
| 首尾帧能力      | 官方原生 FL2VA                             | 通过现有 Wan workflow / wrapper 实现   |
| 核心模型       | 单个约 33B joint audio-video packed DiT   | HIGH/LOW MoE 路径                  |
| 音频         | 同一次 denoising 原生生成 32 kHz stereo       | 通常独立生成或无音频                       |
| CFG        | guidance 已蒸馏进权重，没有 CFG、negative prompt | 你当前有 ScheduledCFG                |
| 推理步数       | 30 sigma 点＝29 次模型 forward；50＝49 次      | 你当前约 2 HIGH＋2 LOW＝4 次主模型 forward |
| 本地 BF16 体积 | 仅 Transformer 约 61.7 GB                | 明显更小且工程成熟                        |
| 帧率         | 原生 24 FPS                              | 你当前先 16 FPS，再 RIFE 到 32 FPS      |
| LoRA       | 技术上支持 PEFT，但没有公开 trainer recipe        | musubi、DiffSynth 等生态成熟           |
| 量化         | INT8、pruned INT8、NF4 等刚出现              | FP8、INT8、cache、distillation 已成熟  |
| License    | MiniMax H3 自定义社区协议                     | Wan2.2 Apache 2.0                |
| 推荐生产角色     | 高质量、强 endpoint control、原生音频            | 低延迟、高吞吐、稳定 fallback              |

Diffusers PR 明确写明 H3 是单一 packed sequence 上运行的 joint video/audio Transformer，没有 CFG，每个 sigma 只做一次模型 forward；30 个 sigma grid points 实际对应 29 次 forward。其 BF16 Transformer 本身约 61.7 GB，单张 80 GB 卡运行 768P 仍需 component offload。

这意味着它**不是**你当前 Wan2.2 2 HIGH＋2 LOW worker 的性能等价替换。即使 H3 单步效率很高，29–49 次 33B 模型 forward 与你现在 4 次主模型 forward 的延迟级别也完全不同。

另外，H3 权重使用自定义 `minimax-h3-community-license-agreement`，而 Wan2.2 I2V 是 Apache 2.0。正式商用部署前，应单独完成 H3 license 审查。([Hugging Face][2])

---

# 三、目前最有价值的代码和实现

## 1. 官方 API：先用它建立 H3 的质量上限

官方 `/v2/video_generation` 支持：

* `role=first_frame`
* `role=last_frame`
* 0、1 或 2 张 endpoint image
* 4–15 秒整数时长
* 768P 或 2K
* 首尾帧模式下，输出比例自动跟随输入
* 异步提交、轮询和下载
* H3-Context-IR prompt 增强
* 768P 视频 in-context regeneration 到 2K

官方还支持 reference image/video/audio，但 API 文档将 FL2VA 与 reference generation 分成不同模式；不要未经验证就假设本地开放权重能同时接受首尾帧和额外 identity/motion reference。([MiniMax API Docs][3])

最小可运行 API：

```python
from __future__ import annotations

import os
import time
from pathlib import Path

import requests


BASE_URL = "https://api.minimax.io"
MODEL = "MiniMax-H3"


def create_fl2va_task(
    *,
    prompt: str,
    first_frame_url: str,
    last_frame_url: str,
    duration: int = 5,
    resolution: str = "768P",
) -> str:
    if duration < 4 or duration > 15:
        raise ValueError("MiniMax H3 duration must be an integer from 4 to 15 seconds.")

    api_key = os.environ["MINIMAX_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": first_frame_url},
                "role": "first_frame",
            },
            {
                "type": "image_url",
                "image_url": {"url": last_frame_url},
                "role": "last_frame",
            },
        ],
        "duration": duration,
        "resolution": resolution,
    }

    response = requests.post(
        f"{BASE_URL}/v2/video_generation",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["task_id"]


def wait_for_task(task_id: str, poll_seconds: int = 10) -> str:
    api_key = os.environ["MINIMAX_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = requests.get(
            f"{BASE_URL}/v2/query/video_generation/{task_id}",
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

        task = response.json()["task"]
        status = task["status"]

        if status == "succeeded":
            return task["content"]["url"]

        if status in {"failed", "cancelled"}:
            raise RuntimeError(
                f"H3 task ended with status={status}: {task.get('error')}"
            )

        time.sleep(poll_seconds)


def download_video(url: str, output_path: str) -> None:
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    Path(output_path).write_bytes(response.content)
```

你的 first/last frame 已经可以放到 R2，因此可以直接使用短时有效的 presigned URLs。

---

## 2. ComfyUI：当前最适合本地 inference 验证

ComfyUI H3 core support 已合并到 master。实现包含：

* `MiniMaxH3ImageToVideo`
* `MiniMaxH3ReferenceToVideo`
* `EmptyMiniMaxH3LatentAV`
* `MiniMaxH3SigmaShift`
* joint video/audio latent
* Qwen3-VL-32B text/multimodal encoder
* FL2VA 和 Ref2VA 两套 Transformer
* BF16、INT8 和 pruned INT8 模型识别

该 PR 于 2026 年 8 月 3 日合并，因此应固定到合并 commit 或其后的已知 commit，不能直接无版本地跟踪 master。

Comfy-Org 的 Hugging Face repack 提供：

```text
diffusion_models/
  minimax_h3_fl2va_bf16.safetensors
  minimax_h3_fl2va_int8_convrot.safetensors
  minimax_h3_fl2va_pruned_int8_convrot.safetensors

text_encoders/
  qwen3vl_32b_minimax_h3_bf16.safetensors
  qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

vae/
  minimax_h3_video_vae_fp16.safetensors
  minimax_h3_audio_vae_fp32.safetensors
```

你只做首尾帧生成时，第一轮无需下载 `ref2va` Transformer。([Hugging Face][4])

### 建议的本地权重组合

**RTX 5090：**

```text
FL2VA DiT:  minimax_h3_fl2va_pruned_int8_convrot
Text model: qwen3vl_32b_minimax_h3_nvfp4_awq
Fallback:   qwen3vl_32b_minimax_h3_int8_convrot
Video VAE:  minimax_h3_video_vae_fp16
Audio VAE:  minimax_h3_audio_vae_fp32
```

**RTX 6000 Ada：**

```text
FL2VA DiT:  minimax_h3_fl2va_pruned_int8_convrot
Text model: qwen3vl_32b_minimax_h3_int8_convrot
Video VAE:  minimax_h3_video_vae_fp16
Audio VAE:  minimax_h3_audio_vae_fp32
```

不要把 audio VAE cast 到 BF16。Diffusers 集成测试发现，BF16 audio VAE 解码音量可能比 FP32 低约 20 dB，因此 audio VAE 应保持 FP32。

### ComfyUI 参数起点

第一轮只做正确性验证：

```yaml
task: FL2VA
width: 832
height: 480
frames: 124
fps: 24
sigma_points:
  - 30
  - 50
video_shift: 12.0
audio_shift: 3.0
cfg: none
negative_prompt: none
batch_size: 1
seed: fixed
rife: disabled
```

124 帧在 24 FPS 下实际约为 **5.17 秒**。H3 本地 temporal grid 要求帧数满足：

```text
frames = 17k + 5
```

当前 ComfyUI 节点将 124–362 帧视为约 5–15 秒的训练范围；节点默认画布是 768 short edge、最大面积约 `768 × 1344`。

### 一个很容易被忽略的问题

当前 ComfyUI 实现对 endpoint image 的处理不对称：

* first frame：直接 stretch 到目标 canvas；
* last frame：保持比例后 center cover-crop。

因此在进入 H3 前，应由你自己的 preprocess service 先把两张图统一成相同画布，避免：

* 首帧人物变宽或变窄；
* 尾帧人物被 center crop 截掉；
* 两张图人物 scale 不一致；
* endpoint 相似度评估使用了错误的 reference。

这一行为可以直接从已合并的 H3 节点代码中确认。

---

## 3. Diffusers：最适合用来搭 LoRA trainer

Diffusers PR #14355 当前仍为 draft、未合并，但它是现阶段最适合训练开发的实现，因为：

* `MiniMaxH3Transformer3DModel` 继承了 `PeftAdapterMixin`；
* 支持 gradient checkpointing；
* attention 是标准 `to_q / to_k / to_v / to_out`；
* FFN 是标准 SwiGLU；
* 实现了完整 packed layout；
* 实现了 H3 特有的 scheduler；
* 作者报告对 15 种官方 use case 的 30-step denoising trajectory 达成 bit-for-bit parity。

但它目前没有专用 `MiniMaxH3LoraLoaderMixin`，也没有 LoRA training example。

H3 Transformer 结构包括：

```text
50 Transformer blocks
hidden size: 5376
56 attention heads
head dim: 128
FFN dim: 14336
video latent channels: 24
audio latent channels: 32
video patch: 1 × 2 × 2
text embedding dim: 5120
```

文本、首尾帧 condition、目标视频和目标音频全部被打包进同一个一维 sequence 做 full self-attention，不存在传统 cross-attention。

这也是为什么 H3 在低分辨率和高分辨率之间的训练成本差异极大。

---

## 4. DiffSynth-Studio：适合低显存/NF4 inference 研究

DiffSynth-Studio PR #1548 提供了：

* BF16 inference；
* CPU offload；
* NF4；
* disk offload；
* VRAM limit；
* T2VA；
* 首尾帧 FL2VA；
* joint video/audio MP4 输出。

该 PR 于 2026 年 8 月 3 日创建，当前仍未合并。

它的公开示例已经直接包含：

```python
video, audio = pipe(
    prompt=prompt,
    height=832,
    width=480,
    num_frames=124,
    num_inference_steps=50,
    seed=0,
    keyframes=[first_frame, last_frame],
    keyframe_indices=[0, -1],
)
```

也就是说，这不是只停留在模型定义层面的支持，而是已经有明确的 FL2VA 调用示例。

其低显存示例会把 NF4 模型 offload 到 disk/CPU，再按 VRAM limit 分层加载。它适合证明“能否在 32 GB 卡上执行”，但不应直接当作生产吞吐方案，因为 disk layer swapping 很容易被 NVMe I/O 限制。

---

# 四、网上成功案例应该如何解读

目前 H3 才发布几天，证据分为三个等级。

## Level A：官方结果

官方展示了：

* film opening titles；
* animated posters；
* product pages；
* advertising/e-commerce；
* multimodal reference；
* native stereo audio；
* 2K generation；
* video motion transfer。

这些能证明官方服务的能力上限，但不能证明开放权重、INT8 或 NF4 本地路径能达到同样质量。([MiniMax][5])

## Level B：开源工程验证

目前最强的开源证据是：

* ComfyUI core integration 已合并；
* Diffusers contributor 报告 15 个官方配置 numerical parity；
* DiffSynth 提供 BF16 和 NF4 首尾帧示例；
* ComfyUI 的 INT8 H3 至少在一台 RTX 5090、96 GB host RAM 机器上做过实际加载测试。

这些可以证明本地 inference 路线是现实可行的，但还没有足够的独立 latency、quality 和长期稳定性数据。

## Level C：LoRA 成功案例

截至 **2026 年 8 月 3 日**，没有找到可复现的：

* H3 LoRA trainer；
* H3 LoRA checkpoint；
* H3 LoRA Hugging Face training card；
* H3 LoRA Reddit 成功/失败报告；
* H3 LoRA musubi-tuner recipe；
* H3 QLoRA recipe。

所以 LoRA 部分必须按“新模型 trainer bring-up”管理，而不是普通参数调节任务。

---

# 五、建议的迁移架构

```text
                           ┌──────────────────────────┐
first frame ──────────────▶│ Endpoint preprocessing   │
last frame ───────────────▶│ - aspect normalization  │
prompt ───────────────────▶│ - crop / scale alignment│
                           │ - input validation       │
                           └─────────────┬────────────┘
                                         │
                            ┌────────────▼─────────────┐
                            │ Generation Router        │
                            └───────┬───────────┬──────┘
                                    │           │
                       high-quality │           │ low-latency/fallback
                                    │           │
                  ┌─────────────────▼───┐   ┌───▼─────────────────┐
                  │ MiniMax H3 FL2VA    │   │ Existing Wan2.2     │
                  │ API or local worker │   │ 2 HIGH + 2 LOW      │
                  └─────────┬───────────┘   └─────────┬───────────┘
                            │                         │
                  ┌─────────▼───────────┐             │
                  │ Video + stereo audio│             │
                  │ decode / mux        │             │
                  └─────────┬───────────┘             │
                            └────────────┬─────────────┘
                                         │
                           ┌─────────────▼─────────────┐
                           │ QC / candidate reranking │
                           │ - endpoint fidelity      │
                           │ - identity               │
                           │ - anatomy                │
                           │ - temporal stability     │
                           └─────────────┬─────────────┘
                                         │
                           ┌─────────────▼─────────────┐
                           │ Optional interpolation   │
                           │ upload to R2             │
                           └───────────────────────────┘
```

第一轮 benchmark 必须禁用 RIFE。否则你会把：

* H3 原生 24 FPS；
* Wan 原生 16 FPS；
* Wan 经 RIFE 变成 32 FPS；

混在一起比较，无法判断 blur、face quality 和 motion smoothness 到底来自 generator 还是 interpolation。

---

# 六、逐阶段执行计划

## Phase 0：冻结 Wan baseline 和测试集

建立 `benchmark_manifest.jsonl`，每条记录至少包含：

```json
{
  "id": "case_0001",
  "first_frame": "inputs/case_0001_first.png",
  "last_frame": "inputs/case_0001_last.png",
  "prompt": "...",
  "duration_sec": 5,
  "category": "multi_person",
  "expected_camera": "static",
  "expected_motion": "...",
  "wan_seed": 1234,
  "wan_output": "wan_outputs/case_0001.mp4"
}
```

建议先做 150–300 组，并有意识地覆盖：

| 类别                  |  建议数量 |
| ------------------- | ----: |
| 单人近景、面部身份           | 30–50 |
| 全身、手部、复杂人体动作        | 30–50 |
| 双人/多人交互             | 20–40 |
| 相机移动、zoom、pan、orbit | 20–30 |
| 强光照或背景变化            | 20–30 |
| 产品、logo、文字          | 20–30 |
| 首尾帧差异非常大的困难样本       | 20–30 |

### 必须记录的指标

**Endpoint fidelity**

* output frame 0 对齐处理后的 first frame；
* output final frame 对齐处理后的 last frame；
* LPIPS；
* DINO feature cosine；
* SSIM 只作为辅助，不作为主要感知指标。

**Identity**

* ArcFace/AdaFace；
* 对视频中每隔 8–12 帧抽样；
* 多人场景用 face tracking＋Hungarian matching；
* 记录最差帧和 p10，而不只是均值。

**Anatomy**

* 人数是否变化；
* 手臂、腿、手指异常；
* person detector count；
* pose track continuity；
* 人工 blind judge。

**Temporal**

* optical-flow warp error；
* flicker；
* sudden scene cut；
* background swimming；
* final-frame rushing：最后 10% 视频突然快速变形来追尾帧。

**工程指标**

* prompt encoding 时间；
* denoise 时间；
* video VAE decode；
* audio decode；
* mux；
* peak VRAM；
* peak host RAM；
* disk read；
* 总 wall time；
* OOM 和失败率。

---

## Phase 1：用 API 做 H3 capability oracle

先只跑：

```text
duration: 5 seconds
resolution: 768P
candidate count: 1
prompt: 原始 prompt
```

第二轮：

```text
duration: 5 seconds
resolution: 768P
candidate count: 2–4
prompt: H3-Context-IR 增强版
```

第三轮只对优胜样本：

```text
768P output
  → H3 in-context regeneration
  → 2K
```

### Prompt 模板

H3 没有 negative prompt，因此不要继续沿用 Wan 的“正向 prompt＋negative prompt＋ScheduledCFG”思路。使用完整的正向约束：

```text
Subject invariants:
The same woman remains present throughout the entire video.
Her face, hairstyle, clothes and body proportions remain unchanged.

Opening state:
At the beginning, she is standing in the exact pose shown in the first image.

Motion trajectory:
During the first third, ...
During the middle, ...
During the final third, she gradually transitions into the exact pose and
composition shown in the final image.

Camera:
The camera remains static.
No cut, no change of lens, no sudden reframing.

Scene continuity:
The room, furniture, lighting direction and background layout remain stable.

Ending state:
The final frame naturally matches the supplied last frame without a sudden
speed-up or abrupt morph.

Audio:
Only subtle room ambience; no speech and no music.
```

对于相机运动，可以测试官方支持的 `[pan]`、`[zoom]`、`[static]` 等 camera instructions。([MiniMax API Docs][3])

### API 阶段的 go/no-go

只有满足以下条件才继续大规模本地部署：

* H3 相对当前 Wan baseline 的人工 blind preference 至少达到约 60%；
* endpoint identity 明显改善；
* severe anatomy failure 不高于 Wan；
* 没有系统性的 final-frame rushing；
* API H3 的失败模式是可重复、可分类的。

如果 API 版 H3 在目标领域都无法超过 Wan，直接训练开放权重 LoRA 的风险会非常高。先不要训练。

---

## Phase 2：本地 H3 inference bring-up

### Step 1：固定版本

建议固定：

```text
ComfyUI:
  commit >= 57500fc5bc92566a63f2046824f522cd55c335ca

Diffusers research branch:
  PR #14355 head abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc

DiffSynth research branch:
  PR #1548 head 0f2086f733c90ef94d4ebc4154deb255daede2e6
```

不要让三个实现共用同一个 Python environment。分别用三个 Docker image：

```text
h3-comfy-inference
h3-diffusers-reference
h3-diffsynth-lowvram
```

### Step 2：先跑最小规模

```text
832 × 480
124 frames
30 sigma points
seed 0
first + last frame
audio decode enabled
no RIFE
```

然后依次测试：

```text
30 steps → 50 steps
832×480 → 1344×768
pruned INT8 → full INT8 → BF16 oracle
Qwen NVFP4 → Qwen INT8 → BF16 oracle
```

不要一次同时改变模型量化、分辨率、steps 和 prompt。

### Step 3：本地和 API 分开判断

你至少要有四个质量层级：

```text
A. H3 API 768P
B. local H3 BF16/offload
C. local H3 pruned INT8
D. local H3 NF4
```

如果：

```text
API good
BF16 good
INT8 bad
```

这不是 LoRA 问题，而是量化问题。

如果：

```text
API good
BF16 bad
```

优先怀疑：

* open checkpoint 和 API 模型差异；
* packing；
* first/last resize；
* text encoding；
* sampler；
* precision；
* audio/video timestep pairing。

如果：

```text
API bad
BF16 bad
```

才可能是 off-the-shelf model capability gap。

---

## Phase 3：production inference benchmark matrix

建议固定如下实验：

| 实验   | Model          | Resolution |           Steps | Quant        |
| ---- | -------------- | ---------: | --------------: | ------------ |
| H3-A | API            |       768P | service default | service      |
| H3-B | API＋Context-IR |       768P | service default | service      |
| H3-C | Comfy          |    832×480 |              30 | pruned INT8  |
| H3-D | Comfy          |    832×480 |              50 | pruned INT8  |
| H3-E | Comfy          |   1344×768 |              30 | pruned INT8  |
| H3-F | Diffusers      |    832×480 |              30 | BF16/offload |
| H3-G | DiffSynth      |    832×480 |              50 | NF4          |
| WAN  | 当前 production  |        当前值 |             2＋2 | 当前值          |

### 推荐初始部署策略

```text
H3 route:
  endpoint-critical
  high-value content
  native audio needed
  human/object identity important
  latency budget relaxed

Wan route:
  low latency
  high throughput
  bulk generation
  H3 timeout/OOM/failure fallback
```

---

# 七、LoRA 前必须完成的工作

LoRA 不应该用来修复以下问题：

| 失败                    | 正确处理                  | 是否适合 LoRA |
| --------------------- | --------------------- | --------- |
| 首尾帧 aspect ratio 不一致  | 统一 canvas             | 否         |
| first frame 被 stretch | 自己预处理                 | 否         |
| last frame 被 crop     | 自己预处理                 | 否         |
| INT8 造成脸部模糊           | BF16/更好量化             | 否         |
| scheduler sign 错误     | 修代码                   | 否         |
| audio VAE BF16 音量异常   | 保持 FP32               | 否         |
| 两张图在三维几何上不可连接         | storyboard/中间关键帧      | 通常否       |
| 多个 seed 只有部分成功        | best-of-N/rerank      | 未必        |
| 固定领域动作始终错误            | 领域视频 LoRA             | 是         |
| 固定相机语言始终错误            | motion/camera LoRA    | 是         |
| 特定人物/产品持续漂移           | identity/product LoRA | 是         |
| 特定人体交互反复崩坏            | motion/anatomy LoRA   | 是         |

### 在 LoRA 前先加 best-of-N

对每个请求生成 2–4 个 seed，自动打分：

```text
score =
    w_first    × first_frame_similarity
  + w_last     × last_frame_similarity
  + w_identity × minimum_track_identity
  + w_motion   × temporal_motion_score
  - w_anatomy  × anatomy_failure_penalty
  - w_cut      × scene_cut_penalty
```

如果 best-of-4 已能把失败率降到生产要求，就不一定需要立即训练。

---

# 八、H3 LoRA trainer 的技术方案

## 1. 选择 Diffusers PR 作为训练底座

不要从 ComfyUI workflow 直接开始训练。

原因：

* Diffusers Transformer 已继承 `PeftAdapterMixin`；
* q/k/v 是独立 Linear，适合 PEFT；
* gradient checkpointing 已实现；
* scheduler 和 packed layout 清晰；
* 可以直接调用 Transformer forward；
* ComfyUI 的主要目标是 inference，不是 autograd/FSDP trainer。

H3 Transformer 的 PEFT 和 checkpointing 接口已经在当前代码中存在。

建议建立独立 repo：

```text
h3-fl2va-lora/
├── configs/
│   ├── train_480p_r8.yaml
│   ├── train_480p_r16.yaml
│   └── eval.yaml
├── data/
│   ├── manifests/
│   └── splits/
├── cache/
│   ├── text/
│   ├── video_latents/
│   ├── audio_latents/
│   └── keyframe_latents/
├── h3_training/
│   ├── dataset.py
│   ├── cache_text.py
│   ├── cache_vae.py
│   ├── packing.py
│   ├── noise.py
│   ├── losses.py
│   ├── lora.py
│   ├── fsdp.py
│   └── checkpoint.py
├── scripts/
│   ├── train_h3_fl2va_lora.py
│   ├── eval_h3_lora.py
│   ├── export_peft_adapter.py
│   └── convert_adapter_to_comfy.py
└── tests/
    ├── test_scheduler_target.py
    ├── test_packing_parity.py
    ├── test_base_inference_parity.py
    └── test_overfit_8_clips.py
```

---

## 2. 训练数据格式

第一轮统一为：

```text
resolution: 832×480 or 480×832
fps: 24
frames: 124
duration: approximately 5.17 seconds
no scene cut
first frame: target video frame 0
last frame: target video frame 123
audio: original synchronized audio where available
```

Manifest：

```json
{
  "video": "clips/train/000001.mp4",
  "prompt": "A woman slowly turns toward the camera...",
  "fps": 24,
  "num_frames": 124,
  "width": 832,
  "height": 480,
  "has_audio": true,
  "split": "train",
  "identity_id": "person_013",
  "scene_id": "room_027",
  "motion_class": "turn_and_sit"
}
```

首尾帧必须在数据 pipeline 内从最终解码后的 124 帧视频中提取，而不是从另一个压缩版本或原始 source 中单独拿图。否则：

* compression；
* crop；
* color conversion；
* frame indexing；

都会制造伪 endpoint mismatch。

### 数据阶段

**Stage 0：8–16 clips**

目标不是泛化，而是验证 trainer 是否能 overfit。

**Stage 1：100–300 clips**

验证一个非常窄的动作或领域是否能改善。

**Stage 2：500–3,000 clips**

建立 production adapter。

**Stage 3：更大数据**

只有 Stage 2 明显 underfit 才扩展。H3 是 33B base，不应默认需要数万条数据。

### Split 原则

不能随机按相邻 clip split。应按：

* identity；
* scene；
* source video；
* location；
* recording session；

做 group split，防止同一视频的相邻切片进入 train 和 validation。

---

## 3. 缓存策略

训练过程中不加载 Qwen3-VL-32B 和两个 VAE。

离线缓存：

### Text cache

缓存：

```text
Qwen3-VL hidden states used by H3
shape roughly [L, 5120]
token modality tags
attention/layout metadata
```

必须使用 H3 自己的 text encoder path。不能直接使用普通 Qwen3-VL 最后一个 hidden state，因为 H3 使用的是特定层和特定 multimodal token/tag 处理。

### Video cache

缓存：

```text
clean video latent:
  [24, 37, H/16, W/16]

first-frame condition latent
last-frame condition latent
```

124 帧通过 H3 video VAE 对应约 37 个 latent time positions。

### Audio cache

缓存：

```text
[32, 2, T_audio]
T_audio ≈ duration × 40 Hz
```

约 5.17 秒对应约 207 个 audio latent positions，每个位置有 stereo 两行。

---

## 4. H3 的正确 flow-matching objective

H3 不能直接套用普通 Diffusers FlowMatch scheduler，因为 velocity sign 与 timestep 定义不同。

H3 使用：

```text
t = 1 - sigma
t = 1 means clean
```

forward process：

[
x_t = t x_0 + (1-t)\epsilon
]

模型预测 data-ward velocity：

[
v = x_0 - \epsilon
]

因此：

[
\hat{x}_0 = x_t + \sigma \hat{v}
]

而不是很多 flow model 使用的减号。Diffusers 的 H3 scheduler 专门指出了这个差异；直接复用普通 FlowMatch scheduler 会把训练 target 和 inference update 方向做反。

### Video 和 audio 使用不同 shift

先采样同一个基础噪声位置：

[
u \sim U(0,1)
]

初始 smoke test 可以用 uniform；后续再比较 logit-normal。

Video：

[
\sigma_v = \frac{12u}{1+11u}
]

[
t_v = 1-\sigma_v
]

Audio：

[
\sigma_a = \frac{3u}{1+2u}
]

[
t_a = 1-\sigma_a
]

然后：

[
x_{v,t} = t_v x_{v,0} + (1-t_v)\epsilon_v
]

[
y_{v} = x_{v,0} - \epsilon_v
]

[
x_{a,t} = t_a x_{a,0} + (1-t_a)\epsilon_a
]

[
y_{a} = x_{a,0} - \epsilon_a
]

### 首尾帧 condition rows

首尾帧 condition 不参与 denoise target loss。

它们应：

1. 使用 H3 video VAE 编码；
2. 使用接近 clean 的 condition noise augmentation；
3. 插入 packed sequence；
4. 每一步保持固定；
5. 不被 scheduler update；
6. 不计入 target-row MSE。

当前 inference 实现使用接近 `0.999` 的 keyframe noise augmentation。训练器必须复用相同的 condition packing，否则训练和生产 conditioning path 不一致。

---

## 5. Packed sequence

训练 forward 的 sequence 顺序必须与 inference 一致：

```text
[text rows
 | first-frame condition video rows
 | last-frame condition video rows
 | target audio rows
 | target video rows]
```

同时构造：

```text
position_ids: [sequence, 3]     # temporal, height, width
token_tags:                     # video=0, text=1, audio=2
timestep_indices
video_indices
audio_indices
text_indices
```

不能把 H3 改成传统：

```text
video latent + cross attention text
```

因为它根本没有传统 cross-attention。所有模态通过同一个 full self-attention stack 交互。

---

## 6. Loss 设计

第一版只需要：

[
L_v = \operatorname{MSE}(\hat{v}_v, v_v)
]

[
L_a = \operatorname{MSE}(\hat{v}_a, v_a)
]

[
L = L_v + \lambda_a L_a
]

不要把 video 和 audio token 直接 concat 后统一 mean。Video token 数远大于 audio token，audio objective 会几乎消失。应先分别求 mean，再加权。

### 如果产品不需要音频

仍然应保留 audio rows，因为它们是 joint packed sequence 的一部分，并会影响 video attention。

建议：

```text
lambda_audio = 0.05–0.2
```

用真实同步 audio 做一个小权重 preservation objective。最终产品可以不保存 audio。

如果完全把 audio loss 设为零：

* shared Transformer 的 LoRA 可能破坏 audio；
* audio rows 的分布可能漂移；
* 漂移后的 audio rows 又会通过 self-attention 影响 video。

### 如果产品需要原生音频

建议：

```text
lambda_audio = 0.2–1.0
```

并建立额外的：

* lip-sync；
* audio-video event alignment；
* clipping；
* loudness；
* speech intelligibility；

验证集。

### Endpoint temporal weighting

如果模型经常：

* 开头几帧偏离 first frame；
* 末尾追不上 last frame；
* 最后 10% 突然快速 morph；

可以对 target video latent 的两端加权：

```text
first 2 latent frames: weight 1.5–2.0
middle latent frames:  weight 1.0
last 2 latent frames:  weight 1.5–2.0
```

这是一个实验性增强，不应在 trainer parity 尚未通过前加入。

---

## 7. 第一版 LoRA target

建议第一版：

```python
target_modules = [
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
]
```

配置：

```yaml
rank: 8
alpha: 8
dropout: 0.0
dtype:
  base: bf16
  lora_trainable: fp32
```

50 层全部 attention、rank 8 大约是 **20M trainable parameters**；rank 16 约 40M，仍远小于 33B base。

### Ablation 顺序

**A1：推荐起点**

```text
all 50 blocks
q/k/v/out
rank 8
```

**A2：更集中在后半层**

```text
blocks 25–49
q/k/v/out
rank 16
```

适合 appearance、identity、局部 motion correction。

**A3：更强 motion capacity**

```text
all attention
+ last 12–16 blocks FFN linears
```

只有 A1 明显 underfit 才进入。

### 第一轮不要训练

* Qwen3-VL；
* token refiner；
* context embedder；
* time embedder；
* AdaLN projection；
* norms；
* video VAE；
* audio VAE；
* full output heads。

H3 部分输入、时间和输出模块使用 FP32 参数，第一轮不碰这些混合精度敏感模块。

### 可选的 video-specific adapter

如果你的目标只是视觉 style/texture，而不是动作理解，可以额外实验：

```text
proj_in
proj_out
```

它们是 video-specific projection，理论上比 shared attention 更少破坏 audio。但它们不能充分解决复杂 motion grammar，因此不应替代 attention LoRA。

---

## 8. 训练分辨率为什么必须从 480×832 开始

H3 video VAE：

```text
spatial compression: 16×16
Transformer patch:   2×2
```

### 832×480

```text
latent spatial: 52×30
patched grid:   26×15 = 390 rows per latent frame
latent frames:  37
target video rows: approximately 14,430
two endpoint rows: approximately 780
audio rows: approximately 414
total: roughly 15.6K + text rows
```

### 1344×768

```text
latent spatial: 84×48
patched grid:   42×24 = 1008 rows per latent frame
latent frames:  37
target video rows: approximately 37,296
two endpoint rows: approximately 2,016
audio rows: approximately 414
total: roughly 40K + text rows
```

因为是 full self-attention，粗略 attention compute 比例接近：

[
\left(\frac{40K}{15.6K}\right)^2 \approx 6.6
]

所以从 480×832 直接跳到 768×1344，不只是像素增加约 2.6 倍，attention 部分的理论计算量接近增加 6–7 倍。

---

## 9. 多卡训练配置

### 推荐硬件路线

对你现有本地多卡资源，第一版建议：

```text
4 × 48 GB GPU
BF16 frozen Transformer
FSDP FULL_SHARD
activation checkpointing
Qwen embeddings precomputed
VAE latents precomputed
batch size 1 per GPU
gradient accumulation
```

BF16 Transformer 约 61.7 GB，理论上 4 卡权重 shard 后每卡约 15.4 GB raw weights；主要压力会转移到 packed-sequence activation，而不是 optimizer state。LoRA optimizer state 很小。Transformer BF16 大小和单 80 GB offload 情况可由当前 Diffusers PR 确认。

### 不推荐第一版用两张 5090 做 BF16 FSDP

两卡时每卡 raw shard 接近 31 GB，几乎没有 activation 空间。要使用双 5090，必须叠加：

* block swap；
* CPU offload；
* quantized frozen base；
* 或 sequence/context parallel。

这些都会同时增加 trainer bring-up 难度。第一轮先在 4×48 GB 上证明 objective 和 LoRA 正确。

### FSDP 关键配置

```python
use_orig_params = True
sharding_strategy = FULL_SHARD
limit_all_gathers = True
forward_prefetch = True
backward_prefetch = BACKWARD_PRE
mixed_precision = BF16
activation_checkpointing = every_transformer_block
```

`use_orig_params=True` 对 frozen base＋trainable LoRA 的 mixed `requires_grad` 模型尤其重要。

### 优化器起点

```yaml
optimizer: AdamW
learning_rate_sweep:
  - 1.0e-5
  - 2.0e-5
  - 5.0e-5
weight_decay:
  - 0.0
  - 0.01
max_grad_norm: 1.0
effective_batch_size: 8–16
warmup_ratio: 0.03–0.05
scheduler: cosine_or_constant
```

第一轮 LoRA 参数很少，没必要为了 optimizer memory 强制使用 8-bit Adam。标准 fused AdamW 更容易排除实现问题。

---

## 10. 训练阶段与门槛

### Stage T0：base parity

在 adapter disabled 时：

* trainer forward 输出应与 reference inference Transformer 对齐；
* scheduler target unit test 通过；
* packing indices 与 Diffusers implementation 一致；
* 同 seed 重复结果一致。

如果 base parity 不通过，不得开始 LoRA。

### Stage T1：8–16 clips overfit

目标：

* training loss 明显下降；
* 固定 seed 下 endpoint 和动作逼近训练 clip；
* adapter off 后恢复 base output；
* adapter scale 0 时与 base 完全一致；
* checkpoint save/load 无误。

如果 8–16 clips 都不能 overfit，优先排查：

* velocity target sign；
* timestep；
* condition row；
* packed index；
* latent scaling；
* target row slicing；
* FSDP/PEFT module placement。

不要用更大数据掩盖 trainer bug。

### Stage T2：100–300 clips narrow pilot

目标：

* held-out identity/scene 上仍有改善；
* base-domain retention 不明显下降；
* final-frame rushing 减少；
* audio 没有明显崩溃；
* adapter weight 0.6–1.0 范围可控。

### Stage T3：500–3,000 clips production LoRA

加入：

```text
80% first+last
10% first-only
5% last-only
5% no-image/T2V retention
```

这样可以避免模型完全过拟合“双 endpoint 总是存在”的模式。

建议额外混入 20–30% generic retention clips，尤其覆盖：

* 普通 walking；
* speaking；
* static camera；
* pan；
* zoom；
* single person；
* multi-person；
* indoor/outdoor。

---

# 九、LoRA inference 与 ComfyUI 兼容问题

## Diffusers 路线

Diffusers Transformer 已经是 `PeftAdapterMixin`，因此第一版可以直接：

```python
transformer.load_adapter(adapter_path)
transformer.set_adapters(["production_h3_lora"], weights=[0.8])
```

但目前不能假设：

```python
pipe.load_lora_weights(...)
```

已经完整支持 H3，因为 PR 自己明确说明尚无专用 H3 LoRA loader。

## ComfyUI 路线

ComfyUI 通用 LoRA loader 会为所有 `diffusion_model.*.weight` 建立 generic key，因此原则上可以加载 H3 adapter。

但存在一个转换问题：

* Diffusers H3：`to_q`、`to_k`、`to_v` 分开；
* Comfy/Open checkpoint：可能使用 fused `qkv_proj`。

所以不能只做字符串 rename。

如果每个 q/k/v adapter rank 都是 `r`，要合并成一个 fused qkv LoRA，可以构造 rank `3r`：

[
A_{fused} =
\begin{bmatrix}
A_q \
A_k \
A_v
\end{bmatrix}
]

[
B_{fused} =
\begin{bmatrix}
B_q & 0 & 0 \
0 & B_k & 0 \
0 & 0 & B_v
\end{bmatrix}
]

这样：

[
B_{fused} A_{fused}
===================

\begin{bmatrix}
B_q A_q \
B_k A_k \
B_v A_v
\end{bmatrix}
]

第一版最稳的方案是：

1. 在 Diffusers runtime 验证 LoRA；
2. LoRA 质量确认后再写 qkv converter；
3. converter 完成后逐层比较 merged weight delta；
4. 用固定 latents 验证 Comfy 和 Diffusers 输出。

不要一开始就同时调 trainer 和 adapter converter。

---

# 十、生产部署方案

## Worker 设计

当前应按：

```text
one process
one GPU
batch size 1
one loaded H3 model
```

设计。

多张 GPU 应优先做并行 worker：

```text
GPU0 → H3 worker 0
GPU1 → H3 worker 1
GPU2 → H3 worker 2
GPU3 → H3 worker 3
```

而不是立即把一个 H3 request tensor-parallel 到四张卡。当前开放实现还没有经过充分验证的 H3 TP/sequence-parallel production path。

## Host memory 与 storage

对于 5090 offload 路线：

* 96 GB RAM 已出现过实际测试环境；
* production 更建议 128–192 GB；
* 模型 cache 使用本地快速 NVMe；
* 不把 Hugging Face cache 放在网络文件系统；
* worker 启动时预热一条 480P/124-frame 请求；
* 对 text encoder、DiT、VAE 分别记录 load/offload 时间。

这里的 128–192 GB 是工程余量建议，不是官方最低要求。

## 输出 contract

H3 输出是：

```text
video frames
stereo audio waveform
audio sample rate
```

service 应明确选择：

```text
mode A: mux native audio into MP4
mode B: discard H3 audio
mode C: replace H3 audio with existing TTS/music pipeline
```

即使最终丢弃音频，也不要假设 inference 可以完全删除 audio latent rows；它们参与 joint Transformer forward。

## RIFE

H3 原生 24 FPS。

只在确有 48 FPS 要求时使用：

```text
24 FPS H3
  → RIFE 2×
  → 48 FPS
```

不要把 24 FPS 插到 32 FPS；这会造成非整数 cadence。若业务必须 32 FPS，更适合最终统一转码，而不是直接 RIFE 2×。

---

# 十一、推荐的 go/no-go gates

## Gate 1：API

继续条件：

```text
H3 API blind win rate >= approximately 60%
endpoint fidelity better than current Wan
severe anatomy failure not worse
```

## Gate 2：本地 BF16

继续条件：

```text
local BF16 quality close to API
no systematic resize or packing bug
same seed reproducible
```

## Gate 3：量化

生产量化版本：

```text
identity degradation <= acceptable threshold
endpoint score drop <= 5% relative
severe failure increase is negligible
```

如果 INT8/NF4 达不到，H3 只能作为 API 或高显存 route。

## Gate 4：LoRA overfit

继续条件：

```text
8–16 clips can overfit
adapter disable restores base
checkpoint reload exactly reproduces result
```

## Gate 5：LoRA generalization

继续条件：

```text
held-out identity and scene improve
base retention acceptable
audio does not regress beyond SLA
adapter scale is stable
```

## Gate 6：production replacement

只有在以下全部满足后，才考虑减少 Wan traffic：

```text
quality gate
latency gate
OOM/failure gate
license approval
adapter packaging
automatic fallback
observability
```

---

# 十二、我建议你实际采用的最终路线

## 路线 A：立即执行的 inference 主线

```text
1. 建立 150–300 对首尾帧 benchmark
2. 用 H3 API 768P/5s 跑能力上限
3. 固定 ComfyUI H3 merge commit
4. 5090 使用 pruned INT8 DiT + NVFP4/INT8 Qwen
5. 480×832、124 帧、30/50 sigma points
6. 与 API、BF16、NF4、当前 Wan 做同输入比较
7. 建立 H3/Wan production router
```

## 路线 B：LoRA 主线

```text
1. Pin Diffusers PR #14355
2. 复制并固定 packing/scheduler implementation
3. 预缓存 Qwen、video VAE、audio VAE 输出
4. 完成 base forward parity tests
5. 用 8–16 clips overfit
6. attention-only rank-8 LoRA
7. 4×48GB FSDP，480×832，124 帧
8. 扩到 100–300 clips
9. 再扩到 500–3,000 clips
10. 最后处理 Diffusers → Comfy qkv LoRA 转换
```

## 路线 C：生产策略

```text
MiniMax H3:
  high-quality
  endpoint-critical
  native audio
  premium queue

Wan2.2:
  low latency
  high throughput
  fallback
  bulk queue
```

**最终判断：H3 很有可能成为比 Wan2.2 更适合首帧＋尾帧约束的质量路线，但以当前模型大小、29–49 次 forward 和刚刚形成的本地生态，它暂时更像 premium generator，而不是你现有 2＋2 step Wan worker 的低延迟直接替代品。LoRA 值得做，但正确顺序必须是 API oracle → local BF16 parity → quantization diagnosis → small-set overfit → production LoRA，而不是直接用大量数据开训。**

[1]: https://www.minimax.io/blog/minimax-h3 "https://www.minimax.io/blog/minimax-h3"
[2]: https://huggingface.co/Comfy-Org/MiniMax-H3 "https://huggingface.co/Comfy-Org/MiniMax-H3"
[3]: https://platform.minimax.io/docs/guides/video-generation?utm_source=chatgpt.com "Video Generation - MiniMax API Docs"
[4]: https://huggingface.co/Comfy-Org/MiniMax-H3?utm_source=chatgpt.com "Comfy-Org/MiniMax-H3 · Hugging Face"
[5]: https://www.minimax.io/blog/minimax-h3?utm_source=chatgpt.com "MiniMax H3: An Open Model Breaking the Boundaries Between Tasks and Modalities - MiniMax Research | MiniMax"
