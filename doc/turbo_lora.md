# 可以，但不是“把 ComfyUI 的 Turbo LoRA 文件直接塞进去”那么简单

准确结论是：

| 部署引擎                 | H3 Turbo LoRA                                                      |
| -------------------- | ------------------------------------------------------------------ |
| **普通 vLLM**          | 不行；普通 vLLM 不提供 H3 diffusion pipeline                               |
| **vLLM-Omni**        | **有条件可用**：支持视频请求级 PEFT LoRA，但 H3 Turbo LoRA 需要格式和 fused-layer 键名适配 |
| **SGLang Diffusion** | **更适合作为起点**：H3 pipeline 已直接继承 LoRA pipeline，但同样需要确认或转换键名           |
| **ComfyUI**          | 当前最即插即用；已有 Comfy 转换和 8–10 步公开运行案例                                  |

截至 **2026 年 8 月 7 日**，我没有找到 `larryvrh/MiniMax-H3-Turbo-Lora` 在 vLLM-Omni 或 SGLang 上已经公开验证、可直接复制的 H3 专用 recipe。两个引擎都有通用 diffusion LoRA 基础设施，但“通用 LoRA 支持”不等于“这个 Turbo LoRA 文件原样兼容”。

---

# 一、Turbo LoRA 不只是一个普通 LoRA

公开的 H3 Turbo LoRA 对比测试使用：

```text
Base:
  LoRA: none
  steps: 20
  audio shift: 3

Turbo:
  LoRA: ckpt500 / EMA ckpt500
  LoRA strength: 1.0
  steps: 8 或 10
  audio shift: 6
```

测试是在 ComfyUI 中通过：

```text
LoraLoaderModelOnly
MiniMaxH3SigmaShift
```

完成的，并生成了 5.17 秒、960×544、带原生音频的视频。

所以 Turbo 的完整执行条件是：

[
\text{Turbo LoRA}
+
\text{8–10 step sigma schedule}
+
\text{audio flow shift}=6
]

只加载 LoRA、但继续使用 20 或 50 steps，不能认为仍是正确的 Turbo 路径，可能出现：

* 过度去噪；
* 运动节奏异常；
* 尾帧突然追赶；
* 音频轨迹不稳定；
* 画面和音频去噪时钟不一致。

---

# 二、SGLang：结构上已经支持，当前更值得优先改

SGLang 的 H3 pipeline 直接声明为：

```python
class MiniMaxH3Pipeline(LoRAPipeline, ComposedPipelineBase):
```

也就是说，H3 已经进入 SGLang 的通用 diffusion LoRA 注入路径，而不是完全没有 LoRA support。

SGLang 当前支持：

```text
--lora-path
--lora-nickname
--lora-scale
--lora-merge-mode
--lora-target-modules
```

其中 merge mode 有：

```text
auto
merge
dynamic
```

对于固定使用一个 Turbo LoRA 的生产服务，理想形式是：

```bash
sglang serve \
  --model-path /models/MiniMax-H3 \
  --model-variant fl2va \
  --lora-path /models/h3-turbo-sglang-peft \
  --lora-nickname h3-turbo \
  --lora-scale 1.0 \
  --lora-merge-mode merge \
  ...
```

`merge` 的意义是启动时将 LoRA delta 合入当前基础权重，避免每个 Transformer block、每个 denoise step 都额外执行一组 LoRA A/B GEMM。对 Turbo LoRA 这种长期固定开启的 adapter，通常比 runtime dynamic adapter 更合适。

## 但 SGLang 目前缺少 H3 专用键名转换

SGLang 的通用 LoRA format adapter 能处理：

* 标准 `lora_A / lora_B`；
  -常见 `lora_down / lora_up`；
* Wan；
* Qwen-Image；
* Flux Kohya；
* ai-toolkit Flux；
* XLabs Flux。

但没有专门的 `MiniMax H3 Turbo LoRA` converter。

而且 H3 配置里当前是：

```python
lora_param_names_mapping = {}
```

也就是没有给出“外部 H3 LoRA 名字 → SGLang H3 layer 名字”的专门映射。

因此 SGLang 的判断是：

> **LoRA execution path 已经存在，但 adapter 必须转换成 SGLang H3 原生 module names。**

---

# 三、vLLM-Omni：API 层已经支持，但 H3 fused projection 是障碍

vLLM-Omni 的 diffusion LoRA loader 当前要求 **PEFT directory format**：

```text
h3-turbo-peft/
├── adapter_config.json
└── adapter_model.safetensors
```

它的视频 API 也已经有 request-level LoRA 字段：

```json
{
  "lora": {
    "name": "h3-turbo",
    "local_path": "/models/h3-turbo-peft",
    "scale": 1.0
  }
}
```

并接受 `name/path/scale/int_id` 及其别名。

所以 API 能力层面，vLLM-Omni 可以把 LoRA 附加到 H3 video request。

## 问题在 H3 的内部 layer layout

vLLM-Omni 的 H3 使用 fused layer：

```text
blocks.N.attn.qkv_proj
blocks.N.attn.out_proj

blocks.N.mlp.fc1
blocks.N.mlp.fc2

blocks.N.adaln_proj.linear
```

其中：

* Q、K、V 被合在 `qkv_proj`；
* MLP gate 和 up 被合在 `fc1`；
* output/down 是独立 projection。

H3 源码甚至明确设置：

```python
packed_modules_mapping = {}
```

并注明 checkpoint 本身已经存储 fused QKV 和 fused gate/up，因此没有 unfused names 可自动映射，LoRA 需要直接针对 fused layers。

vLLM-Omni 的通用 LoRA manager 确实支持：

* `QKVParallelLinear`；
* `MergedColumnParallelLinear`；
* `RowParallelLinear`；
* PEFT adapter；
* runtime adapter cache。

但它依赖模型提供的 packed-layer mapping，或 adapter 已经直接使用 fused module names。

因此：

```text
Turbo LoRA 已经是 qkv_proj / fc1 格式
    → 可能只需 prefix rename 和 PEFT packaging

Turbo LoRA 是 to_q / to_k / to_v / gate_proj / up_proj 格式
    → 不能直接加载，需要 fusion converter

Turbo LoRA 是 ComfyUI 单文件格式
    → 不能直接作为 vLLM-Omni PEFT directory 使用
```

---

# 四、应该使用原始 Turbo LoRA，还是 ComfyUI 转换版

公开评测列出的两个来源是：

```text
原始:
larryvrh/MiniMax-H3-Turbo-Lora

ComfyUI 转换:
drbaph/MiniMax-H3-Turbo-Lora-ComfyUI
```

ComfyUI 转换版是为：

```text
LoraLoaderModelOnly
```

设计的。

对于 SGLang 或 vLLM-Omni，建议转换顺序是：

```text
优先从 larryvrh 原始 adapter 开始
        ↓
检查它是否是 PEFT / Diffusers module naming
        ↓
转换到 engine-native fused H3 module naming
        ↓
再生成 SGLang 或 vLLM-Omni adapter
```

不要从 ComfyUI 转换版开始，除非原始 adapter 不可用，因为你可能会经历两次转换：

```text
原始训练命名
  → ComfyUI 命名
  → SGLang/vLLM 命名
```

每多一次转换，就多一个：

* prefix 错误；
* QKV row order 错误；
* gate/up 顺序错误；
* alpha/rank scaling 丢失；
* transpose 错误；

的风险。

---

# 五、先检查你手上 LoRA 的真实键名

在下载的 LoRA 文件上运行：

```python
from pathlib import Path

from safetensors import safe_open


path = Path("/models/h3-turbo/adapter_model.safetensors")

with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())

print(f"Number of tensors: {len(keys)}")
for key in keys[:200]:
    print(key)
```

重点寻找以下形式。

## 情况 A：最容易

```text
blocks.0.attn.qkv_proj.lora_A.weight
blocks.0.attn.qkv_proj.lora_B.weight

blocks.0.mlp.fc1.lora_A.weight
blocks.0.mlp.fc1.lora_B.weight
```

这已经接近 engine-native naming，主要需要：

* 去掉或补上 `transformer.`；
* 调整 `base_model.model.` prefix；
* 创建正确的 `adapter_config.json`。

## 情况 B：需要 QKV/MLP fusion

```text
...attn.to_q.lora_A.weight
...attn.to_k.lora_A.weight
...attn.to_v.lora_A.weight

...ff.gate_proj.lora_A.weight
...ff.up_proj.lora_A.weight
```

这时需要将三组 attention LoRA 合为：

```text
qkv_proj
```

将两组 MLP LoRA 合为：

```text
fc1
```

## 情况 C：ComfyUI naming

```text
diffusion_model.blocks...
lora_unet_...
...lora_down.weight
...lora_up.weight
```

需要同时做：

* prefix conversion；
* A/B conversion；
* fused projection conversion；
* PEFT metadata reconstruction。

---

# 六、QKV LoRA 不能只靠简单地 `torch.cat`

假设原始 LoRA 分别是：

[
\Delta W_q = B_q A_q
]

[
\Delta W_k = B_k A_k
]

[
\Delta W_v = B_v A_v
]

目标 fused projection 是：

[
W_{qkv} =
\begin{bmatrix}
W_q \
W_k \
W_v
\end{bmatrix}
]

为了准确表达三个独立 LoRA，可以构造 rank `3r`：

[
A_{qkv} =
\begin{bmatrix}
A_q \
A_k \
A_v
\end{bmatrix}
]

[
B_{qkv} =
\begin{bmatrix}
B_q & 0 & 0 \
0 & B_k & 0 \
0 & 0 & B_v
\end{bmatrix}
]

于是：

[
B_{qkv} A_{qkv}
===============

\begin{bmatrix}
B_q A_q \
B_k A_k \
B_v A_v
\end{bmatrix}
]

MLP 的 gate/up fusion 同理，rank 从 `r` 变成 `2r`。

这会产生一个实际问题：

```text
原始 rank = 64

fused QKV rank = 192
fused gate/up rank = 128
```

某些 runtime LoRA manager 对可接受 rank 有固定集合或最大值；vLLM-Omni 的 LoRA manager确实使用 vLLM 的 `MaxLoRARanks` 限制。

因此，对固定 Turbo LoRA，**预合并进 BF16 权重通常比把 separate LoRA 转成高 rank fused runtime adapter 更简单、更快、更稳定。**

---

# 七、最稳的生产路线：先 merge，再部署

推荐流程：

```text
Official MiniMax H3 FL2VA BF16 DiT
        +
H3 Turbo LoRA, strength 1.0
        ↓
在 BF16/FP32 权重上完成 merge
        ↓
输出一个新的 Turbo FL2VA checkpoint
        ↓
用 8 或 10 steps 部署
        ↓
最后再做 FP8 / INT8 / engine-specific conversion
```

严格顺序是：

```text
BF16 base
→ merge LoRA
→ validate merged BF16
→ quantize merged checkpoint
→ validate quantized checkpoint
```

不要：

```text
FP8/INT8 base
→ 直接把 BF16 LoRA delta merge 进去
```

因为量化权重不是普通浮点矩阵，直接 merge 会破坏：

* scale；
* zero point；
* block-wise quantization；
* FP8 amax；
* packed kernel layout。

## SGLang 固定 adapter

转换后的 adapter 能直接加载时：

```bash
sglang serve \
  --model-path /models/MiniMax-H3 \
  --model-variant fl2va \
  --lora-path /models/h3-turbo-sglang \
  --lora-scale 1.0 \
  --lora-merge-mode merge \
  ...
```

SGLang 已提供 `merge`、`dynamic` 和 `auto` 三种模式。

## vLLM-Omni request-level adapter

转换为 PEFT 后，视频请求可以包含：

```json
{
  "prompt": "The subject moves naturally from the first frame to the last frame.",
  "width": 864,
  "height": 480,
  "num_inference_steps": 8,
  "flow_shift": 12,
  "lora": {
    "name": "h3-turbo",
    "local_path": "/models/h3-turbo-vllm-peft",
    "scale": 1.0
  },
  "extra_params": {
    "task": "fl2va",
    "duration": 5.0,
    "frame_indices": [0, -1],
    "audio_flow_shift": 6.0
  }
}
```

vLLM-Omni 的 video protocol 已经暴露 request-level LoRA。

---

# 八、8-step Turbo 必须同步修改 H3 参数

第一组 engine parity 参数建议固定为：

```yaml
lora:
  scale: 1.0
  checkpoint: ema-ckpt500

sampling:
  num_inference_steps: 8
  flow_shift_video: 12.0
  flow_shift_audio: 6.0
  guidance: none
  negative_prompt: none

video:
  frames: 124
  fps: 24
  duration: approximately 5.17 seconds
```

第二组只改：

```yaml
num_inference_steps: 10
```

公开对比正是测试了 EMA ckpt500 的 8-step 和 10-step，以及 audio shift 6。

不要第一轮同时加：

```text
Turbo LoRA
Cache-DiT
Sage
FP8
更低分辨率
```

否则质量下降时无法确定来源。

正确的 ablation：

```text
T0: Comfy base, 20 steps
T1: Comfy Turbo, 8 steps
T2: engine base, 20 steps
T3: engine Turbo BF16, 8 steps
T4: engine Turbo merged BF16, 8 steps
T5: engine Turbo merged FP8, 8 steps
```

---

# 九、针对你的 2×5090＋128GB，实际建议

Turbo LoRA 能显著减少 denoise forward 数：

[
20 \rightarrow 8
]

理论 NFE 降低为：

[
8/20 = 40%
]

也就是忽略其他阶段时，denoise compute 上限接近 **2.5× 加速**。

但 Turbo LoRA不会减少：

* 66GB 级 DiT 的基础权重；
* Qwen3-VL 权重；
* VAE 权重；
* 启动时 host RAM；
* layerwise offload 所需的 CPU-side weight storage。

所以它可以缓解：

```text
每个视频重复执行多少次 DiT
```

但不能解决：

```text
2×32GB VRAM + 128GB RAM 能否可靠加载完整 engine pipeline
```

SGLang 当前公开的两张 5090 layerwise-offload recipe是在约 377GiB RAM 主机上验证，并建议 384GiB 级系统。

因此针对你当前机器：

### 立即可用、风险最低

```text
ComfyUI
+ pruned FP8/INT8 H3
+ H3 Turbo LoRA
+ 8–10 steps
```

### Engine 研发优先级

```text
1. SGLang
2. Turbo LoRA key converter
3. BF16 merge parity
4. 再解决 pruned/quantized checkpoint loader
```

### vLLM-Omni

适合作为第二条验证路线，因为：

* 有 PEFT LoRA manager；
* 有视频 request-level LoRA；
* 有 TP linears；
* 但 H3 fused mapping 需要专门处理；
* 2×5090 下又受到 FP8 与 layerwise offload 当前不兼容的限制。

---

# 最终判断

**可以继续使用 H3 Turbo LoRA。**

但当前状态是：

```text
ComfyUI:
  直接可用

SGLang:
  框架层已经支持 H3 LoRA
  需要 H3 Turbo 专用 key converter
  最适合优先开发

vLLM-Omni:
  API 和通用 PEFT LoRA 已支持
  H3 fused QKV/MLP 映射需要适配
  不是直接加载 Comfy 单文件

普通 vLLM:
  不适用
```

对生产部署，最优目标不是长期保留 runtime LoRA，而是：

> **先把 Turbo LoRA 正确 merge 到官方 BF16 H3 FL2VA DiT，验证 8-step＋audio shift 6 的 parity，再将 merged Turbo checkpoint 转成目标引擎需要的量化和分片格式。**

这样既保留 Turbo 的少步优势，也消除每层 runtime LoRA 计算，并避免 fused QKV 高 rank adapter 带来的兼容性问题。
