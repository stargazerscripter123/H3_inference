Q:

你现在的部署都是基于 comfy，如果换成vllm omni 会更快吗？

A:

# 结论

**会。对你的单机 4×RTX 6000 Ada，vLLM-Omni 作为 MiniMax H3 推理后端，大概率会比当前 ComfyUI 原生 H3 workflow 更快。**

但要区分两件事：

* **ComfyUI 继续作为 workflow/UI**；
* **H3 的 Qwen、DiT、VAE 和 MP4 输出改由 vLLM-Omni 服务执行**。

这是最合适的架构，而不是把整条业务逻辑从 ComfyUI 删除。vLLM-Omni 已提供官方 ComfyUI 节点，可以把 ComfyUI 放在 CPU 或另一台机器上，然后向本地或远程 vLLM-Omni 服务发送视频生成请求。

不过，**vLLM-Omni 本身不会把 50-step 变成 10-step，也不会自动把 3 分钟变成 30 秒**。它能解决的是：

1. 让一个请求真正使用四张 GPU；
2. 减少 CPU/GPU 权重换入换出；
3. 并行 Qwen3-VL、DiT attention 和 VAE；
4. 使用更适合 serving 的 attention、FP8、warmup 和长期驻留机制。

---

# 一、为什么 vLLM-Omni 在你的机器上更有优势

## ComfyUI 当前路径

ComfyUI 的官方 H3 实现已经做得不差，包括：

* pruned checkpoint，将时间/AdaLN 相关参数减少约 40%；
* INT8 ConvRot；
* fused activation＋quantize；
* DynamicVRAM 和异步 offload；
* video VAE tiling 和 temporal chunking。

但是其原生 H3 workflow 本质上仍是：

```text
一个 ComfyUI 进程
    ↓
一张主要计算 GPU
    ↓
Qwen → DiT → VAE 依次换入/换出
```

四张 GPU 通常只能开四个 ComfyUI worker，提高吞吐量；它不会自动把一个 H3 视频拆到四张卡共同完成。

## vLLM-Omni 路径

vLLM-Omni 的 H3 原生支持已经合并，首帧、尾帧和首尾帧 `[0, -1]` 输入矩阵也已于 **2026 年 8 月 5 日**合并。

它可以把同一个请求拆成：

```text
Qwen3-VL:
    text-encoder tensor parallel

H3 DiT:
    tensor parallel
    或 tensor parallel × Ulysses sequence parallel

Video/audio VAE:
    tile patch parallel

Output:
    persistent server-side decode + mux
```

因此你的四张 6000 Ada 可以同时服务一个视频，而不是只有一张卡做主要 denoise。

---

# 二、预期能快多少

目前没有公开的：

```text
4×RTX 6000 Ada
同一 prompt
同一首尾帧
同一分辨率
同一步数
ComfyUI vs vLLM-Omni
```

严格 A/B benchmark，所以下面是基于模型结构、显存驻留和多卡通信的工程估算，不是已测结果。

| 当前 Comfy 状态                     |           换 vLLM-Omni 后的合理预期 |
| ------------------------------- | ---------------------------: |
| DiT 已完全驻留单张 6000 Ada，未使用 cache  |                   约 1.5–2.5× |
| 每一步存在明显 CPU→GPU block streaming |                       约 2–4× |
| Comfy 已使用激进 TeaCache/Spectrum   | vLLM lossless 可能只快一点，甚至不一定更快 |
| 多个连续请求                          |      vLLM 吞吐量优势通常比单请求延迟优势更明显 |

四卡不会达到 4×线性加速，因为 RTX 6000 Ada 是 48GB、PCIe Gen4 x16 卡，多卡 TP/sequence parallel 会引入 PCIe collective 通信。([NVIDIA][1])

---

# 三、现有 benchmark 能说明什么

## 1. vLLM-Omni 不是自动的十倍加速器

vLLM-Omni 的四卡 H3 优化分支在 4×B300 上，将 8.7 秒、209 帧、50-step FL2VA 的 E2E 从约 84.5 秒降到约 82.3 秒；其中 DiT denoise 仍占约 78.4 秒。也就是说，当模型已经完全驻留后，缓存 RoPE、metadata、VAE input 等 runtime 优化只能再拿到几个百分点，**真正的大头始终是 DiT forward 数量和每次 forward 的计算量**。

## 2. FP8 是有效加速点

vLLM-Omni 已经合并 H3 online FP8，支持 TP sharding，并保留 patch、timestep 和 final projections 为 BF16/FP32。其 H100 质量测试中，FP8 将 peak memory 从 68.52 GiB/GPU 降到 53.51 GiB/GPU，并通过视频 LPIPS 和音频相似度门槛。

早期同一 FP8 路线在 4×B300、1344×768、124 帧、50 steps 上测得：

```text
BF16: 79.08 s
FP8:  58.42 s
```

约为 1.35× 加速。这个具体 benchmark 来自被后续实现取代的开发 PR，因此应视为方向性证据，而不是对 6000 Ada 的承诺。

FP8 runtime 本身支持 Ada/Hopper，并明确将 MiniMax H3 列为 online FP8 支持模型；但 H3 的 Ada 质量和速度仍需要你自己的 A/B。FP8 当前也不能与 H3 layerwise offload 组合。

## 3. Offload 仍然会非常慢

vLLM-Omni 的 2×5090 distributed layerwise offload 实验中，1344×768、124 帧、50-step 请求仍需要约 **8 分 38 秒**。这说明当模型必须持续通过 PCIe streaming weights 时，换 framework 也无法创造奇迹。

你的 4×48GB 比 2×32GB 更有优势，因为可以采用 TP4，把更多甚至全部重复 DiT blocks 留在 GPU 上。

---

# 四、4×RTX 6000 Ada 的正确 vLLM-Omni 拓扑

## 首选：TP4＋USP1

不要从官方 B300 的纯 `USP4` 配置直接开始。

H3 的 BF16 DiT 约 66.3GB，Qwen3-VL retained encoder 约 51.5GB。纯 Ulysses 通常切 sequence、复制模型权重，66.3GB DiT 本身就超过单张 6000 Ada 的 48GB；而 TP4 会将主要 Linear 权重分到四张卡。

推荐第一条服务：

```bash
export MODEL=/models/MiniMax-H3/FL2VA

CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

这条路线的逻辑是：

```text
DiT weights       → TP4
Qwen3-VL weights  → TP4
VAE tiles         → 4 GPU
sequence          → 不先拆分
```

对于 PCIe-only workstation，这通常是最稳妥的第一个 topology。

## 第二个测试：TP2＋USP2

```bash
--tensor-parallel-size 2
--usp 2
```

它会：

* 两路切模型权重；
* 两路切长 sequence；
* 仍使用四张 GPU。

H3 的序列很长，USP2 可能降低每张卡的 attention 工作量；但 Ulysses 需要 all-to-all，在 6000 Ada 的 PCIe topology 上也可能吃掉收益。因此必须和 TP4 实测，不能直接假设哪一个更快。

## 第三个测试：TP4＋FP8

在第一条命令上增加：

```bash
--quantization fp8
```

不要同时添加 layerwise offload。

如果 TP4 BF16 已经能完全驻留，FP8 的主要价值是：

```text
更小权重
更小 memory bandwidth
更快 GEMM
更多 activation headroom
```

---

# 五、用于首帧＋尾帧的请求

当前 vLLM-Omni 已支持 `[0, -1]` 首尾帧。

```bash
export FIRST_FRAME=/data/first.png
export LAST_FRAME=/data/last.png

curl -sS -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F 'prompt=The subject moves naturally from the exact first-frame state to the exact final-frame state, maintaining identity, clothing, scene geometry and lighting throughout.' \
  -F 'width=864' \
  -F 'height=480' \
  -F 'fps=24' \
  -F 'num_inference_steps=12' \
  -F 'flow_shift=12' \
  -F 'seed=42' \
  -F 'extra_params={"task":"fl2va","duration":5.0,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
  -F "input_references=@${FIRST_FRAME};type=image/png" \
  -F "input_references=@${LAST_FRAME};type=image/png" \
  -o output.mp4
```

---

# 六、你应该怎样做严格 A/B

保持以下内容完全一致：

```text
prompt
first frame
last frame
seed
864×480
124 frames
12 steps
flow shift 12
audio shift 3
无额外 cache
```

测试矩阵：

| ID | Backend        | Topology   | Precision   |
| -- | -------------- | ---------- | ----------- |
| C0 | ComfyUI native | 1×6000 Ada | 当前 INT8/FP8 |
| V1 | vLLM-Omni      | TP4        | BF16        |
| V2 | vLLM-Omni      | TP4        | FP8         |
| V3 | vLLM-Omni      | TP2＋USP2   | BF16        |
| V4 | vLLM-Omni      | TP2＋USP2   | FP8         |

每个配置：

```text
第一次：相同 shape warmup，不记录
第二至第四次：记录
最终使用 median
```

记录：

```text
text/keyframe encode
DiT denoise
seconds per denoise step
video/audio VAE decode
MP4 mux
total E2E
peak VRAM per GPU
GPU utilization
PCIe/NCCL utilization
```

vLLM-Omni 可以加入：

```bash
--enable-diffusion-pipeline-profiler
```

---

# 七、能不能达到 30 秒

## 1344×768、50 steps

**不能指望靠换 vLLM-Omni 达到 30 秒。**

即使 4×B300 的开发 benchmark，50-step、124 帧的 FP8 路径也仍在约一分钟级别。

## 864×480、10–12 steps

这是你的可行区间。

30 秒预算：

```text
Qwen + endpoint VAE encode:  2–4 s
DiT denoise:                20–23 s
VAE decode + mux:            3–5 s
----------------------------------
Total:                      25–32 s
```

因此：

```text
12 steps → 每步 DiT 最好低于约 1.8 s
10 steps → 每步 DiT 最好低于约 2.2 s
```

**4×6000 Ada＋vLLM-Omni TP4/FP8 有机会进入这个区域，但没有现成 benchmark 能保证。**

如果 V2/V4 的 12-step denoise 仍超过 30 秒、E2E 超过 40 秒，继续调 framework 小参数意义不大；下一层必须是：

```text
Cache-DiT / H3-specific block cache
或
8-step acceleration LoRA / consistency distillation
```

---

# 八、最终推荐架构

```text
ComfyUI
  - 只负责 workflow
  - 可以用 CPU 启动
  - 管理首帧、尾帧、prompt、业务逻辑
        │
        │ HTTP
        ▼
vLLM-Omni MiniMax H3 server
  - 4×RTX 6000 Ada
  - TP4 或 TP2×USP2
  - text encoder TP4
  - VAE patch parallel 4
  - FP8 A/B
        │
        ▼
H.264 + stereo AAC MP4
```

官方 ComfyUI-vLLM-Omni 插件正是为这种分离模式设计的，也明确支持 MiniMax H3 FL2VA 和 Ref2VA。

**我的判断是：值得换，而且应优先在 4×6000 Ada 上测试 vLLM-Omni。**但迁移目标应该定义为“使用四卡并消除单卡瓶颈”，而不是认为 vLLM 这个名字本身就会自动将 H3 压到 30 秒。第一条应跑的配置是 **TP4＋864×480＋12 steps＋BF16**，随后是同拓扑 FP8，再和 **TP2＋USP2** 做严格 warm A/B。

[1]: https://www.nvidia.com/en-au/products/workstations/rtx-6000/?utm_source=chatgpt.com "NVIDIA RTX 6000 Ada Generation Graphics Card"
