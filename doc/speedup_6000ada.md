# 核心结论

针对你现有的 **单机 4×RTX 6000 Ada 48GB**，最优加速策略和 2×5090 不一样：

> **不要继续让 ComfyUI 单进程只用一张 6000 Ada 跑 H3。**
> ComfyUI 保留为 workflow/UI，MiniMax H3 的实际推理迁移到 **SGLang 4-GPU backend**，使用 TP/Ulysses 拆分同一个请求。

RTX 6000 Ada 的优势是 **48GB 显存、原生 FP8 Tensor Core**，更容易避免逐层 CPU offload；劣势是单卡算力低于 5090，而且四卡之间主要依赖 PCIe 4.0 P2P，不是 NVLink。因此，6000 Ada 的真正价值是“模型常驻＋四卡切分”，而不是单卡硬跑。NVIDIA 官方规格为 48GB GDDR6 ECC、PCIe Gen4 x16、91.1 TFLOPS FP32，并支持 FP8 Tensor Core；RTX 6000 Ada 本身不支持 NVLink。([NVIDIA][1])

SGLang 的原生 MiniMax H3 已经合并，包含 tensor parallel、sequence parallel、缓存和 OpenAI-compatible video API，因此当前比自己修改 ComfyUI 做分布式 H3 更合理。

---

# 一、你的四卡应该这样分

## 首选起点：TP4，不做 offload

```text
GPU 0 ─┐
GPU 1 ─┼─ TP4：每张卡保存约 1/4 DiT 和 1/4 Qwen 权重
GPU 2 ─┤
GPU 3 ─┘

Ulysses degree = 1
FSDP           = off
CPU offload    = off
layerwise      = off
```

MiniMax H3 原始 BF16 pipeline 大约包括：

```text
DiT:       66.3 GB
Qwen3-VL:  51.5 GB
plus video/audio VAE
```

TP4 后，粗略权重分布为：

```text
DiT per GPU:       66.3 / 4 ≈ 16.6 GB
Qwen per GPU:      51.5 / 4 ≈ 12.9 GB
基础权重合计:      ≈ 29.5 GB/GPU
剩余约 18 GB:      activations、VAE、workspace、NCCL buffers
```

这正适合 48GB 的 6000 Ada。H3 在 2×32GB 5090 上必须使用 distributed layerwise offload，是因为完整 BF16 pipeline 放不下；4×48GB 则应优先消除 offload。

### 启动配置

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_DEBUG=WARN \
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 4 \
  --ulysses-degree 1 \
  --performance-mode speed \
  --use-fsdp-inference false \
  --enable-torch-compile false \
  --warmup-resolutions 864x480 \
  --host 0.0.0.0 \
  --port 30010
```

这里刻意没有：

```text
--layerwise-offload-components
--performance-mode memory
--use-fsdp-inference true
```

因为这些是“让模型跑得起来”的容量手段，不是 30 秒延迟目标下的加速手段。

当前 H3 cookbook 也明确指出，`torch.compile` 对已测试 H3 路径的稳态收益低于测量噪声，同时增加启动成本并改变数值输出，所以第一版应保持 eager。

---

# 二、第二个必须测试的拓扑：TP2＋Ulysses2

在高带宽 H100/H200 上，SGLang 的实测显示：

```text
TP2 + Ulysses2
```

通常比：

```text
TP4 + Ulysses1
```

更快，因为它减少了 tensor-parallel all-reduce，并把长视频 sequence 分到两张 GPU 上。SGLang 在四卡 H100 上也把 TP2＋Ulysses2 作为速度优先拓扑。

启动时只改：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --performance-mode speed \
  --use-fsdp-inference false \
  --enable-torch-compile false \
  --warmup-resolutions 864x480 \
  --port 30010
```

但 TP2 下，每张 GPU 需要持有约一半 DiT 权重：

```text
66.3 / 2 ≈ 33.2 GB
```

加上 Qwen、VAE、activation 后，BF16 可能在 48GB 上比较紧。

因此测试顺序应为：

```text
A1: TP4 + Ulysses1 + BF16
A2: TP2 + Ulysses2 + BF16
A3: TP2 + Ulysses2 + FP8
```

如果 A2 OOM，不要立即加 CPU offload，直接进入 FP8。

---

# 三、6000 Ada 应优先测试 FP8，而不是 INT8

6000 Ada 有原生第四代 Tensor Core FP8 支持。对于这张卡，我建议优先测试：

```text
BF16 baseline
    ↓
FP8 W8A8
    ↓
INT8 仅作为对照
```

SGLang 的实验命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --performance-mode speed \
  --quantization fp8 \
  --enable-torch-compile false \
  --warmup-resolutions 864x480 \
  --port 30010
```

Ada/Hopper 平台本身支持 FP8 W8A8 和 INT8 W8A8。([vLLM][2])

不过需要注意：

* SGLang H3 的公开 FP8 完整验证目前主要集中在 B200/B300；
* 6000 Ada 上应先做固定 seed 的 BF16/FP8 A/B；
* 检查首帧、尾帧、人物身份、手部、音频和唇形同步；
* 不要未经验证就把 FP8 与 aggressive cache 同时上线。

对于继续留在 ComfyUI 的单卡实验，则比较：

```text
minimax_h3_fl2va_pruned_fp8_scaled
vs
minimax_h3_fl2va_pruned_int8_convrot
```

6000 Ada 上 FP8 很可能是更合理的起点，但最终仍取决于 Comfy Kitchen 是否命中高效 FP8 kernel。

---

# 四、30 秒目标的请求参数

第一轮不要跑 768P，也不要跑 50 steps。

推荐起点：

```json
{
  "seconds": 5,
  "task": "fl2va",
  "target": {
    "short_edge": 480,
    "aspect_ratio": "16:9",
    "duration_seconds": 5.0
  },
  "num_inference_steps": 12,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 42,
  "quality": "lossless"
}
```

这会生成接近：

```text
864 × 480
124 frames
24 FPS
约 5.17 秒
```

测试顺序：

| Test | Resolution | Steps | Parallelism      | Precision |
| ---- | ---------: | ----: | ---------------- | --------- |
| A1   |    864×480 |    12 | TP4              | BF16      |
| A2   |    864×480 |    12 | TP2＋U2           | BF16      |
| A3   |    864×480 |    12 | TP2＋U2           | FP8       |
| A4   |    864×480 |    10 | A1–A3 winner     | winner    |
| A5   |    736×416 |    10 | winner           | winner    |
| A6   |    864×480 |    16 | winner＋Cache-DiT | winner    |

必须做一次相同分辨率 warmup。SGLang 的 H200 实测中，仅把 warmup resolution 改成实际服务分辨率，就把 1344×768 H3 的 E2E 从 84.14 秒降低到 74.38 秒，约减少 11.6% 的首次形状初始化成本。

---

# 五、Cache-DiT 应该这样加

当前公开 `quality=high` 只接受经过审计的特定 H200 部署；6000 Ada 上不能假设它会直接放行。当前正式 profile 使用较保守的缓存参数，并在 H200 上取得约 1.4×速度提升、SSIM 0.931、PSNR 28.16 dB。更激进的旧 profile 虽达到约 2.5–2.9×，但同 seed 轨迹偏差明显更大，因此已经从公共 quality tier 中移除。

6000 Ada 上使用 process-wide 手动配置：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.04
export SGLANG_CACHE_DIT_MC=1
```

然后启动 TP/sequence-parallel server。

### 参数递进

```text
C0 保守:
RDT=0.04
MC=1

C1 平衡:
RDT=0.06
MC=1

C2 较快:
RDT=0.08
MC=2

C3 aggressive:
RDT=0.12
MC=2
```

不要从 C3 开始。

一个更稳妥的比较是：

```text
12 steps，无 cache
vs
16 steps，C0 cache
vs
16 steps，C1 cache
```

16 steps 给 cache 留出更多中间步骤，同时保留前后真实计算，往往比“8 steps＋激进缓存”更稳定。

**Cache-DiT 不要与 FSDP inference 或 DiT layerwise offload 组合。**缓存需要跳过部分 block，而 FSDP/offload 需要按 block 执行权重 all-gather 或传输，两者的执行模型冲突。

---

# 六、四张 6000 Ada 的 PCIe 拓扑必须先检查

6000 Ada 没有 NVLink，因此四卡并行效果高度依赖主板拓扑和 GPU P2P。

运行：

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
```

再跑 CUDA P2P benchmark：

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make -j"$(nproc)" SMS=89
./bin/x86_64/linux/release/p2pBandwidthLatencyTest
```

以及 NCCL：

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j"$(nproc)" CUDA_HOME=/usr/local/cuda

./build/all_reduce_perf -b 8M -e 1G -f 2 -g 4
./build/alltoall_perf  -b 8M -e 1G -f 2 -g 4
```

重点看：

```text
PIX / PXB：最好
PHB：可以接受
SYS：跨 CPU socket，风险较高
P2P disabled：必须先修
```

如果拓扑类似：

```text
GPU0 ↔ GPU1 很快
GPU2 ↔ GPU3 很快
两组之间较慢
```

那么 `CUDA_VISIBLE_DEVICES` 必须让 TP pair 对应物理上最近的 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

其中：

```text
TP group 0 = GPU0, GPU1
TP group 1 = GPU2, GPU3
Ulysses 跨两个 group
```

不要让 TP pair 跨 NUMA socket。

---

# 七、FSDP 为什么不是你的速度主线

四卡 FSDP 会把 DiT 参数分片，但每个 block forward 前需要重新 all-gather 参数：

```text
block 0 weights all-gather → compute
block 1 weights all-gather → compute
...
block 49 weights all-gather → compute
× 每个 denoise step
```

在 H200/NVLink 上这已经不是最低延迟路径；在 6000 Ada 的 PCIe P2P 上，代价通常更明显。SGLang 的实测也把 FSDP 定义为 capacity profile，而不是 H100/H200 的首选 latency profile。

所以：

```text
TP4                   首个可运行速度基线
TP2 + Ulysses2 + FP8  目标速度路线
FSDP + Ulysses4       只有显存放不下时才用
CPU layerwise offload 最后手段
```

---

# 八、预计能否达到 30 秒

目前没有公开的 **4×RTX 6000 Ada MiniMax H3** 精确 benchmark，所以只能给工程估算，不能当作已测数据。

合理目标区间是：

| 配置                                |       预计 warm E2E |
| --------------------------------- | ----------------: |
| 单张 6000 Ada，864×480，12 steps      |          很难到 30 秒 |
| 4卡 TP4 BF16，864×480，12 steps      |     约 35–60 秒候选区间 |
| 4卡 TP2＋U2 FP8，864×480，10–12 steps |     约 25–45 秒候选区间 |
| 上述 winner＋保守 Cache-DiT            | **有机会进入 25–35 秒** |
| 1344×768、50 steps、lossless        |        不可能接近 30 秒 |

30 秒预算应控制为：

```text
Qwen + keyframe encode:    2–4 s
DiT denoise:              20–23 s
VAE decode + mux:          3–5 s
--------------------------------
Total:                    25–32 s
```

因此最终必须满足：

```text
864×480 或 736×416
10–12 nominal steps
约 8–10 次等效完整 DiT 计算
无 CPU weight streaming
精确分辨率 warmup
四卡 P2P 正常
```

---

# 最终执行顺序

```text
1. 检查四卡 PCIe/P2P/NUMA 拓扑
2. ComfyUI 保留为前端，SGLang 作为 H3 backend
3. 跑 TP4 + U1 + BF16 + 864×480 + 12 steps
4. 跑 TP2 + U2 + BF16
5. 若 OOM，跑 TP2 + U2 + FP8，而不是加 CPU offload
6. 对 winner 测试 12/10 steps
7. 再加入 C0/C1 Cache-DiT
8. 最后才降到 736×416
9. 固定 topology、resolution、steps 和 cache 后，再接回 Comfy workflow
```

对这台机器，**最可能实现约 30 秒的 production 配置是 `4×6000 Ada + TP2/Ulysses2 + FP8 + 864×480 + 10–12 steps + 保守 Cache-DiT`**；TP4 BF16 应作为稳定、可解释的第一条基线。

[1]: https://www.nvidia.com/en-us/products/workstations/rtx-6000/?utm_source=chatgpt.com "NVIDIA RTX 6000 Ada Generation Graphics Card"
[2]: https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/quantization/online/?utm_source=chatgpt.com "Online Quantization - vLLM-Omni"
