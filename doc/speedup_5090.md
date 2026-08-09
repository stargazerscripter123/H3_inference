# 结论

按你当前这类 **2×RTX 5090 32 GB、128 GB RAM** 的机器，MiniMax H3 做 5 秒首帧＋尾帧视频，目标应拆成两档：

| 目标                                         | 当前可行性                                                       |
| ------------------------------------------ | ----------------------------------------------------------- |
| **864×480、124 帧、warm latency 约 30 秒**      | 有机会，需要 8–12 次真实 DiT forward、DiT 常驻显存、低步数＋缓存                 |
| **1344×768、124 帧、完整 20–50 次 forward、30 秒** | stock ComfyUI＋5090 基本不现实，需要 few-step distillation 或更强的多卡运行时 |

这里的 **30 秒必须是 warm latency**：模型已经加载、同一分辨率已经 warmup，从任务进入 sampler 到 MP4 完成。不应把首次加载几十 GB 权重、kernel 初始化和文件 mmap 算进去。

作为参照，SGLang 已合并的 H3 benchmark 中，即使在 **4×H200、1344×768、124 帧、50 steps** 上，lossless 路径仍约 75 秒；经过专门校准的 Cache-DiT high-quality 路径约 53.7 秒，速度提升 1.4 倍，代价是输出不再 bit-exact。

所以你的主线不能只是“换一个 attention kernel”，而必须同时完成：

[
\text{低 token 数}
+
\text{低 NFE}
+
\text{模型常驻}
+
\text{避免重复 forward}
]

---

# 一、先检查有没有在无意中做双倍计算

MiniMax H3 本身没有传统 CFG，也没有 negative prompt；官方运行参数固定为 `guidance_scale=1.0`、`negative_prompt=None`。

Comfy workflow 应满足：

```text
Guider: BasicGuider
CFG: 1.0
Negative conditioning: none
每个 sigma: 1 次模型 forward
```

不要使用：

```text
CFG > 1
positive + negative 双分支
Heun
DPM2
需要两次 model evaluation 的二阶 sampler
```

需要关注的不是 UI 上显示的 `steps`，而是：

```text
NFE = Number of Function Evaluations
```

例如：

```text
12 steps × 1 forward/step = 12 NFE       正常
12 steps × CFG positive/negative = 24 NFE
12 steps × Heun = 24 NFE
12 steps × CFG × Heun = 48 NFE
```

如果你当前 workflow 错用了双分支 CFG 或两次求值 sampler，改为 `BasicGuider + res_multistep/Euler` 后，延迟可能直接接近减半。

---

# 二、分辨率是第一大加速杠杆

官方 Comfy H3 模板默认采用：

```text
1344 × 768
124 frames
pruned INT8 FL2VA
Qwen3-VL NVFP4
```

这本身是一个偏质量而不是偏速度的配置。官方模板还列出了 864×480、736×416、608×352 等低分辨率档位。

## 为什么降到 864×480 会快很多

H3 的视频 VAE 空间压缩是 16 倍，DiT 再使用 2×2 patch。

### 1344×768

```text
latent spatial = 84 × 48
DiT grid       = 42 × 24
tokens/frame   = 1008
latent frames  ≈ 37

target video tokens ≈ 37 × 1008 = 37,296
```

### 864×480

```text
latent spatial = 54 × 30
DiT grid       = 27 × 15
tokens/frame   = 405
latent frames  ≈ 37

target video tokens ≈ 37 × 405 = 14,985
```

H3 把 text、endpoint conditions、audio 和 video 放在一个 packed sequence 中做 full self-attention。当前实现仍有 50 个 DiT blocks。

仅视频 token 的 attention pair 数量变化大约是：

[
\left(\frac{37,296}{14,985}\right)^2 \approx 6.2
]

完整模型不会严格快 6.2 倍，因为 MLP 是线性复杂度，权重 GEMM、VAE 和 offload 也占时间；但实际通常仍会比 1344×768 快很多。

## 我建议的分辨率档位

| 档位           |    H3 生成分辨率 | 使用场景                |
| ------------ | ----------: | ------------------- |
| Quality      |     960×544 | 接受更高延迟              |
| **Balanced** | **864×480** | 第一轮冲击 30 秒          |
| Fast         |     736×416 | 864×480 仍超过 35–40 秒 |
| Emergency    |     608×352 | 只适合大景别、后续强 upscale  |

生产输出需要 720P/768P 时，先生成 864×480，再用第二张 5090 做视频超分。对于 H3 这种 full-attention video DiT，这通常比直接在 1344×768 denoise 便宜得多。

---

# 三、模型选择：先比较官方 pruned FP8 和 pruned INT8

第一轮只测试两个官方模型：

```text
minimax_h3_fl2va_pruned_int8_convrot.safetensors
minimax_h3_fl2va_pruned_fp8_scaled.safetensors
```

Text encoder 保持：

```text
qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

官方 H3 Comfy integration 明确支持 original 和 pruned checkpoint；pruned 版本把非常庞大的时间/AdaLN 参数替换成预计算曲线，主要减少模型体积和权重搬运。

## 不要直接假设 FP8 一定比 INT8 快

在 5090 上，FP8 Tensor Core 理论峰值更高，但最终速度还取决于：

* Comfy quant kernel 是否命中最优路径；
* 是否需要运行时反量化；
* 模型有没有完全驻留显存；
* GEMM 和 attention 哪一个是主要瓶颈；
* FP8 模型是否触发额外 cast；
* 当前 PyTorch/CUDA/comfy-kitchen 版本。

因此必须用相同：

```text
prompt
first frame
last frame
seed
resolution
frames
steps
sampler
```

跑 FP8/INT8 A/B。

**如果 INT8 和 FP8 每步速度接近，选质量更稳定的那个。**

**如果较小的模型能够让 DiT 从 partial/offloaded 变成 fully resident，即使单个 GEMM 没更快，端到端也可能快很多。**

---

# 四、显存策略：目标是 denoise 阶段 DiT 完全常驻

你的显存不可能同时舒适地容纳：

```text
Qwen3-VL text encoder
H3 DiT
video VAE
audio VAE
大规模 124-frame activations
```

正确的阶段式布局是：

```text
阶段 1:
Qwen + VAE encoder 在 GPU
→ 生成 text states 和首尾帧 latents

阶段 2:
Qwen/VAE encoder offload
H3 DiT 全部载入 GPU
→ 完成整个 denoise loop，不再逐层搬权重

阶段 3:
DiT offload 或释放足够空间
video/audio VAE decode
```

## 推荐启动命令

先使用默认 DynamicVRAM，不要一开始就加 `--gpu-only`：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --preview-method none \
  --async-offload 2 \
  --reserve-vram 1.5
```

ComfyUI 当前在 NVIDIA 上默认启用 async offload，也提供 `--highvram`、`--reserve-vram`、`--vram-headroom` 和多种 attention/fast flags。

### 然后测试第二个启动 profile

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --preview-method none \
  --async-offload 2 \
  --reserve-vram 1.5 \
  --highvram
```

`--highvram` 不是必然更快：

* 如果它让 DiT 在 sampler 阶段保持完整常驻：保留；
* 如果它让 Qwen 占着显存，导致 DiT partial load：删除；
* 如果发生 OOM：删除或换更小 quant；
* 不要使用 `--lowvram` 或 `--novram` 来追求低 latency；
* 不要设置很大的 `--vram-headroom`，它会故意空出显存，可能迫使权重 offload。

### 从 Comfy 日志判断

理想情况应接近：

```text
diffusion model loaded completely
```

危险信号：

```text
loaded partially
lowvram patches
block loaded/offloaded repeatedly
```

如果每个 denoise step 都在 CPU→GPU 传 DiT block，那么无论减少多少 kernel 时间，都很难到 30 秒。

---

# 五、先把 sampler 改成 12 steps

第一轮推荐：

```yaml
width: 864
height: 480
length: 124
fps: 24

steps: 12
sampler: res_multistep
scheduler: simple

video_sigma_shift: 12.0
audio_sigma_shift: 3.0

guidance: BasicGuider
cfg: 1.0
negative: none
```

124 帧在 24 FPS 下实际约为：

[
124 / 24 = 5.167 \text{ seconds}
]

这是 H3 接近“5 秒”的合法 temporal grid。

## 测试顺序

不要直接从 50 跳到 4：

```text
20 steps
16 steps
12 steps
10 steps
8 steps
```

每一步都固定 seed，手工比较：

* 第一帧是否仍准确；
* 最后一帧是否能自然抵达；
* 最后 10% 是否突然加速追尾帧；
* 面部和手部是否恶化；
* 背景是否游动；
* 双人 identity 是否交换；
* 音频是否出现跳变。

未经蒸馏的 base H3 在 4–6 steps 下通常不应被期待维持原始质量。**8–12 steps 才是目前合理的 off-the-shelf 探索区间。**

---

# 六、再加入 H3 TeaCache，但把它视为有损加速

目前有 H3 专用的 Comfy TeaCache wrapper。一个公开 benchmark 在：

```text
1024×576
124 frames
20 steps
FL2VA INT8
GA100-class 64GB GPU
```

上将总时间从约 306 秒降至 102 秒，约 3 倍；但该项目自己仍将多 prompt sweep、polynomial calibration 和质量 gate 列为未完成事项。

安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Icyoung/ComfyUI-MiniMaxH3-TeaCache.git
cd ComfyUI-MiniMaxH3-TeaCache
git rev-parse HEAD
```

记录 commit，重启 Comfy。

连接位置：

```text
UNETLoader
   ↓
MiniMax H3 TeaCache
   ↓
BasicGuider / Sampler
```

## 12-step 推荐参数

先从保守值开始：

```yaml
rel_l1_thresh: 0.08
start_step: 2
end_step: -2
total_steps: 12
```

然后依次测试：

```text
0.08
0.10
0.12
0.15
```

我建议 production 起点：

```text
0.10
```

不要直接把 `0.15` 当作生产默认。

### 为什么要谨慎

这个 implementation 会在判断相邻 timestep 输入变化足够小时，直接复用之前的 model output，从而跳过整个 DiT forward。

这不是数学上的无损优化。SGLang 的 H3 implementation 甚至明确拒绝把通用 TeaCache 描述为 lossless contract，因为 H3 的 video/audio 是耦合 denoise 的。

所以必须同时检查：

```text
video quality
endpoint fidelity
native audio quality
A/V synchronization
```

不要只看中间某一帧。

---

# 七、当前不要盲目启用 SageAttention

当前 Comfy H3 的 SageAttention 路径有一个值得注意的问题：H3 的 partial split-half RoPE 会产生 channel-wise magnitude outliers，SageAttention 的低精度 INT8 QK 路径可能直接产生 noise。修复 PR 提议为 H3 调用设置：

```python
low_precision_attention=False
```

但该 PR 目前是 closed、未 merge。

因此：

```text
默认 optimized attention：先使用
FlashAttention：可以单独 A/B
SageAttention：当前不要作为 production 默认
```

至少先检查：

```bash
grep -n "optimized_attention" \
  ComfyUI/comfy/ldm/minimax/model.py
```

如果 H3 调用没有显式禁用 low-precision QK，不应直接启用 `--use-sage-attention`。

---

# 八、`torch.compile` 和 `--fast` 不是第一优先级

目前 H3 的 packed multimodal sequence、quantized linear、模型 prefetch 和 Python-side layout 都让 `torch.compile` 的收益不稳定。更重要的是：

* compile 首次运行很慢；
* shape 改变可能重编译；
* quant custom ops 可能 graph break；
* 当前公开 H3 runtime 测试中，compile 的 steady-state 收益并不突出；
* 数值输出可能发生变化。

先完成：

```text
分辨率
NFE
模型常驻
cache
```

再测试 compile。

Comfy 的 `--fast` 标志本身被标为实验性、可能影响质量。

后续可以做一个独立 A/B：

```bash
--fast fp8_matrix_mult cublas_ops
```

但不要直接使用：

```bash
--fast
```

开启所有实验优化。

---

# 九、精确 benchmark matrix

固定一组比较困难的首尾帧、prompt 和 seed，按下面顺序跑。

每个配置：

```text
第 1 次：warmup，不计
第 2–4 次：计时
取 median
```

| Run    |  Resolution |  Frames |  Steps | Model        | Cache     |
| ------ | ----------: | ------: | -----: | ------------ | --------- |
| B0     |        当前设置 |      当前 |     当前 | 当前           | Off       |
| B1     |     864×480 |     124 |     20 | pruned INT8  | Off       |
| B2     |     864×480 |     124 |     12 | pruned INT8  | Off       |
| B3     |     864×480 |     124 |     12 | pruned FP8   | Off       |
| B4     |     864×480 |     124 |     12 | B2/B3 winner | 0.08      |
| **B5** | **864×480** | **124** | **12** | **winner**   | **0.10**  |
| B6     |     864×480 |     124 |     10 | winner       | 0.10      |
| B7     |     736×416 |     124 |     10 | winner       | 0.10      |
| B8     |     736×416 |     124 |      8 | winner       | 0.10–0.12 |

每次记录：

```text
total wall time
text encoding time
first/last VAE encoding time
sampler time
seconds per real DiT forward
实际执行 forward 数
video VAE decode time
audio VAE decode time
mux/save time
peak VRAM
peak host RAM
```

---

# 十、使用 GPU 监控快速定位瓶颈

运行 Comfy 时开另一个终端：

```bash
nvidia-smi dmon -s pucvmet -d 1 -o DT
```

也可以：

```bash
nvtop
```

## 情况 A：计算受限

表现：

```text
GPU utilization: 95–100%
PCIe RX/TX: 不高
每个 step 持续稳定
```

结论：

```text
权重已常驻
继续减少 resolution / NFE / cache
换 offload 参数没有明显帮助
```

## 情况 B：权重搬运受限

表现：

```text
GPU utilization 周期性掉到很低
PCIe RX 很高
CPU RAM 带宽很高
每层或每步都有明显停顿
```

结论：

```text
DiT 没有完整常驻
优先换更小 checkpoint
调整 DynamicVRAM/highvram
释放 Qwen/VAE 占用
```

## 情况 C：后处理受限

表现：

```text
sampler 已结束
GPU utilization 下降
但任务仍需要很多秒才完成
```

结论：

```text
video VAE decode
audio VAE decode
CPU ffmpeg encode
PNG/frame preview
```

如果不需要 H3 native audio，仍然保留 joint audio latent 参与 denoise，但可以跳过：

```text
audio VAE decode
audio mux
```

这可能省下最后几秒，但不会显著减少 DiT denoise，因为 audio/video 在 H3 中是联合建模的。

---

# 十一、30 秒的阶段预算

要做到约 30 秒，建议把预算控制成：

| 阶段                     |       目标 |
| ---------------------- | -------: |
| Qwen＋endpoint encoding |   ≤3–4 s |
| DiT denoise            | ≤20–22 s |
| VAE decode＋encode/mux  |   ≤4–6 s |
| Total                  | ≤30–32 s |

因此如果 864×480 下：

```text
单次真实 DiT forward = 2.5 s
```

那么最多只能执行大约：

```text
20 / 2.5 = 8 次真实 forward
```

这意味着：

```text
12 nominal steps
TeaCache skip 3–4 steps
real NFE ≈ 8–9
```

才有机会达到目标。

如果单次真实 forward 已经是 4 秒，那么再怎么调 async offload，也不可能通过 12 次真实 forward 达到 30 秒。必须进一步：

```text
降到 736×416
减少真实 NFE
或训练 few-step model
```

---

# 十二、两张 5090 应该怎样使用

## 对单个请求

Stock ComfyUI 不会因为机器有两张 5090，就自动把一个 H3 DiT request 拆到两张卡上。

启动两个 Comfy worker：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --port 8188 ...
CUDA_VISIBLE_DEVICES=1 python main.py --port 8189 ...
```

主要作用是：

```text
吞吐量约翻倍
两个视频并行生成
```

而不是把单个视频从 60 秒变成 30 秒。

而且你只有 128 GB RAM，两个 H3 process 同时保留 Qwen、DiT 和 VAE 的 CPU-side weights，可能引发 host RAM pressure。先把单 worker latency 做到稳定，再测试双 worker。

## 对单请求 latency 更有效的双卡设计

后续可以改成：

```text
GPU 1:
  Qwen text/image encode
  first/last VAE encode
                ↓
       只传 text states 和 latents
                ↓
GPU 0:
  H3 DiT 全程常驻并 denoise
                ↓
       只传最终 video/audio latents
                ↓
GPU 1:
  VAE decode
  spatial upscale
  NVENC
```

这不会让 DiT 本身快一倍，但能：

* 避免 Qwen、DiT、VAE 在同一张卡反复换入换出；
* 让 GPU 0 永久作为 denoise worker；
* 让 GPU 1 永久作为 pre/post-processing worker；
* 提高连续请求 latency 稳定性；
* 在 GPU 1 decode 当前视频时，让 GPU 0 开始下一个视频。

这需要自定义 Python service 或设备感知 custom nodes，不是 stock Comfy workflow 一键配置。

---

# 十三、如果 864×480＋12 steps＋cache 仍超过 40 秒

这时不要继续花大量时间微调：

```text
async streams
torch.compile
微小 kernel flags
```

它们通常只能给出个位数到十几个百分点的改善，解决不了 40 秒到 30 秒甚至 60 秒到 30 秒的问题。

下一步应是 **H3 few-step distillation LoRA**。

普通 motion/style LoRA 不会让模型更快；甚至会增加额外低秩 GEMM。要提速，LoRA 必须让 H3 在更少 timestep 下仍保持质量。

## 正确目标

```text
Teacher:
  H3 20–30 steps

Student:
  same H3 base + acceleration LoRA
  8 steps
  后续尝试 6 steps
```

## 训练路线

```text
Stage 1:
20-step teacher → 12-step student

Stage 2:
12-step student → 8-step student

Stage 3:
8-step student → 6-step experimental
```

训练 objective 至少包括：

```text
teacher velocity matching
multi-step trajectory consistency
first-frame reconstruction
last-frame reconstruction
temporal latent consistency
joint audio preservation
generic retention
```

LoRA target 起点：

```text
all 50 blocks:
  qkv projection
  attention output projection

必要时再加入:
  后 12–16 层 MLP
```

训练完成后把 acceleration LoRA merge 到 quantization 前的 base checkpoint，再做：

```text
merged BF16
→ calibrated FP8 / INT8
→ Comfy checkpoint conversion
```

用你那台 **4×RTX 6000 Ada 48 GB** 做 trainer bring-up，比直接在 2×32 GB 5090 上处理 H3 teacher/student 更合理。

---

# 我建议你现在直接采用的配置

## 第一目标：质量与速度平衡

```yaml
resolution: 864x480
frames: 124
steps: 12
sampler: res_multistep
scheduler: simple

model:
  A/B:
    - minimax_h3_fl2va_pruned_int8_convrot
    - minimax_h3_fl2va_pruned_fp8_scaled

text_encoder:
  qwen3vl_32b_minimax_h3_nvfp4_awq

guidance:
  BasicGuider
  CFG 1.0
  no negative prompt

TeaCache:
  rel_l1_thresh: 0.10
  start_step: 2
  end_step: -2
  total_steps: 12

runtime:
  one persistent Comfy worker
  exact-shape warmup
  no sampler previews
  DiT fully resident during denoise
  async offload enabled
```

## 如果仍超过 35 秒

```yaml
resolution: 736x416
frames: 124
steps: 10
TeaCache: 0.10
postprocess:
  upscale on second 5090
```

## 如果必须硬性压到 30 秒以内

```yaml
resolution: 736x416
steps: 8
TeaCache: 0.10–0.12
native audio decode: disable if unused
postprocess GPU: second 5090
```

这会进入明显的质量交换区间。若 B8 仍超过约 35–40 秒，正确路线就不再是继续调 Comfy 参数，而是开始 **8-step H3 acceleration LoRA / consistency distillation**。

先执行 B1→B5。最有决定意义的三个数字是：**warm total time、sampler seconds per real forward、denoise 时 peak VRAM/PCIe activity**。
