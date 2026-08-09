# 总体评价

这份 plan 的**主方向是对的**，而且已经比普通“下载 LoRA、挂上去跑”严谨很多。最关键的路线选择——

```text
官方 BF16 H3
→ 正确 merge Turbo LoRA
→ 验证 merged BF16
→ 再由 vLLM-Omni / SGLang 做 online FP8
```

——是当前最适合生产的方案。

以你现有数据：

```text
864×480
vLLM FP8
engine steps=12
实际 11 次 DiT forward
E2E=37.4s
```

推算 6 NFE 大约有机会进入 **23–27 秒**，4 NFE 可能进入 **17–21 秒**。所以 **864×480、5 秒视频、低于 30 秒是现实目标**。

但我不会直接按当前版本开始执行。它还有 **4 个必须修正的问题**，以及若干质量评测和工程可靠性缺口。最重要的是：

1. SGLang 的 2D fused LoRA-B bug 在当前上游已经修复；
2. 当前 merge verification 有一定“自证循环”，不足以证明 QKV row order 和 scaling 真正正确；
3. `steps=5/7/9` 的语义依赖当前 engine commit，未来很容易被上游 off-by-one 修复改变；
4. dynamic-FP8 与 merged-FP8 不应期望输出一致。

---

# 一、我认同的核心判断

## 1. Merge 主线是正确的

H3 原始磁盘 checkpoint 的 QKV 权重是 grouped/interleaved layout；vLLM-Omni 在加载时会把它重排成运行时的连续：

```text
[Q_all | K_all | V_all]
```

然后才交给 TP/FP8 loader。因此，若 LoRA 的 `qkv_proj.lora_B` 已经是运行时连续布局，那么离线 merge 回磁盘 checkpoint 时，确实必须先对：

[
\Delta W=B A
]

做 inverse reorder，再加到磁盘权重。vLLM-Omni 当前加载路径也明确是在进入权重 loader 之前执行这个 QKV reorder。

所以：

```text
qkv_proj:
    delta_runtime = B @ A
    delta_disk = inverse_qkv_reorder(delta_runtime)
    W_disk_merged = bf16(W_disk_fp32 + delta_disk_fp32)

其他层:
    W_merged = bf16(W_fp32 + B_fp32 @ A_fp32)
```

这个设计是正确的。

## 2. “先 merge，再 online FP8”是正确顺序

这两条必须严格区分：

```text
正确：
BF16 base + BF16 LoRA delta
→ merged BF16
→ online FP8 quantization

错误：
已经量化的 FP8 tensor
→ 直接 add BF16 LoRA delta
```

SGLang 的 `auto` LoRA merge mode 在非 DTensor 情况下会默认选择 merge；而 merge 实现会直接修改 `base_layer.weight.data`，最后 cast 回原来的 target dtype。对于已有独立 scale/quantization state 的在线 FP8 权重，这不等价于重新量化 `W+ΔW`。因此你要求 runtime FP8 路线显式使用：

```bash
--lora-merge-mode dynamic
```

是正确的。

## 3. vLLM-Omni TP4 runtime LoRA 的问题确实存在

当前 vLLM-Omni diffusion LoRA manager 对 fused multi-slice layer 会检查：

```python
lora_B.shape[0] == sum(output_slices)
```

TP4 下，`output_slices` 是本 rank 的 local output rows，但 Turbo LoRA 提供的是 full global `lora_B`：

```text
qkv:
full B rows  = 21504
TP4 local    = 5376

fc1:
full B rows  = 28672
TP4 local    = 7168
```

因此 qkv/fc1 会被 reset，Turbo LoRA 的关键部分不会生效。当前代码不是完全“静默”，而是会输出 warning，然后跳过该 layer。

所以对于你当前 TP4 topology：

> **vLLM-Omni 生产路径先只做 merged checkpoint，是合理决策。**

但应把文档里的“静默跳过”改成“warning 后跳过”。

---

# 二、必须修正：SGLang 的 2D fused-B bug 已经在上游修复

你计划里准备修改：

```text
MergedColumnParallelLinearWithLoRA.slice_lora_b_weights
```

来支持 H3 的 2D fused `lora_B`。

但当前 SGLang 上游代码已经有这一逻辑：

```python
if B.dim() == 3:
    ...

# Native fused checkpoints (MiniMax H3, etc.)
shards = []
row_offset = 0

for full_size, part_size in zip(
    output_sizes,
    output_partition_sizes,
):
    local_start = tp_rank * part_size
    local_end = (tp_rank + 1) * part_size

    shards.append(
        B[row_offset + local_start:
          row_offset + local_end]
    )

    row_offset += full_size

return torch.cat(shards, dim=0)
```

这正是你的计划打算补的功能。

所以 Phase 3 应改为：

```text
1. 记录本地 SGLang commit
2. 检查本地 linear.py 是否已有 2D fused B 分支
3. 如果已有：不修改源码
4. 如果没有：
   - 优先 backport 上游实现
   - 不要重新手写一个语义相似但未经测试的版本
5. 无论是否 backport，都补本地 TP 单元测试
```

建议的测试矩阵：

```text
TP = 1, 2, 4

qkv full sections:
[7168, 7168, 7168]

fc1 full sections:
[14336, 14336]

检查：
每个 rank 的输出
=
每个 full section 的 rank-local 行切片
按 section 顺序重新 concatenate
```

再做一个数值 forward test：

[
xA^TB_{\text{local}}^T
]

应与完整 LoRA delta 对应 rank 的输出 slice 一致。

你不应该继续保留一个名为 `turbo-lora-dynamic` 的永久私有补丁分支，除非当前生产版本无法升级。否则未来 upstream merge 时会产生重复实现和维护负担。

---

# 三、步数语义是对的，但必须 pin commit

你现在的推导：

```text
engine num_inference_steps = 5 → 4 forwards
engine num_inference_steps = 7 → 6 forwards
engine num_inference_steps = 9 → 8 forwards
```

对你目前核验的 vLLM-Omni 和 SGLang 版本是正确的。

两个引擎当前都生成 `num_steps` 个 sigma points：

```python
torch.linspace(1.0, 0.0, num_steps)
```

而 denoise loop 执行：

```python
num_steps = len(sigmas) - 1
```

因此会少一次 forward。

问题是，SGLang 已经有一个开放 PR，准备把 scheduler 改成：

```python
torch.linspace(..., num_steps + 1)
```

使 `num_inference_steps=N` 真正执行 N 次 forward。

因此绝不能把生产 profile 简单保存为：

```text
steps=7
```

建议在你自己的客户端/API 中暴露：

```text
nfe=4
nfe=6
nfe=8
```

然后根据被 pin 的 engine semantics 转换：

```python
if engine_uses_sigma_points_semantics:
    num_inference_steps = nfe + 1
else:
    num_inference_steps = nfe
```

每个输出的 manifest 同时记录：

```json
{
  "requested_nfe": 6,
  "engine_num_inference_steps": 7,
  "observed_dit_forwards": 6,
  "video_flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "engine_commit": "...",
  "sigma_schedule_sha256": "..."
}
```

## Phase 0 应新增 schedule unit test

不能只靠 server log 数步数。应该在两套 engine 环境中直接 import scheduler function，然后验证：

```text
NFE 4 → schedule length 5
NFE 6 → schedule length 7
NFE 8 → schedule length 9
```

同时比较：

```text
vLLM video sigma array
SGLang video sigma array
作者 generate.py video sigma array

vLLM audio sigma array
SGLang audio sigma array
作者 generate.py audio sigma array
```

要求：

```text
shape 完全一致
max_abs_diff <= 1e-7
```

这样才能证明不是“forward 数量一样，但 sigma grid 不一样”。

---

# 四、当前 G1 verification 不足以证明 merge 真正正确

你当前准备验证：

```python
reorder(inverse(x)) == x
inverse(reorder(x)) == x
```

以及：

```python
reorder(merged)
==
bf16(reorder(original) + delta)
```

这些测试有价值，但存在一个问题：

> 它们使用了同一套关于 LoRA row order 的假设来生成和验证结果。

即使你错误地假设了 LoRA B 是 `[Q_all|K_all|V_all]`，只要 merge 和 verify 都使用相同的错误假设，这些测试仍可能全部通过。

## G1 必须增加一个独立 oracle

最理想的验证链是：

```text
A. 官方 BF16 base + Comfy runtime Turbo LoRA
B. merged BF16 checkpoint，无 runtime LoRA
```

固定：

```text
同一 prompt
同一 first/last frame
同一 seed
同一 4 NFE schedule
同一 flow shifts
最低可行分辨率
```

比较：

```text
最终 video latent
最终 audio latent
或至少每个代表性 frame/audio waveform
```

如果不能直接拿 latent，则比较：

```text
frame PSNR / SSIM / LPIPS
DINO cosine
audio spectral cosine
audio RMS ratio
```

还应该增加一个单层独立数值测试：

```python
W_runtime = reorder(W_disk)

y_dynamic = x @ W_runtime.T
y_dynamic += (x @ A.T) @ B.T

W_disk_merged = bf16(
    W_disk.float()
    + inverse_reorder((B.float() @ A.float()))
)

W_runtime_merged = reorder(W_disk_merged)
y_merged = x @ W_runtime_merged.T
```

预期不是 bit-exact，因为 merged BF16 和 dynamic BF16 的 rounding 路径不同，但应达到很高的一致性，例如：

```text
cosine > 0.9999
relative L2 很小
无系统性 row permutation
```

这会直接发现：

* QKV row order 错误；
* A/B transpose 错误；
* alpha/rank scaling 错误；
* gate/up 顺序错误；
* merge 到错误 layer。

---

# 五、merge 脚本还需要四项增强

## 1. 不要把 scale=1.0 写死在数学里

即使当前 LoRA 满足：

```text
alpha = rank
strength = 1.0
```

脚本仍应实现完整公式：

[
\Delta W
========

\frac{\alpha}{r}
\cdot \text{strength}
\cdot B A
]

并在当前 checkpoint 上 assert：

```python
effective_scale == 1.0
```

而不是省略这个乘数。这样未来换 ckpt850 或新的 checkpoint 不会静默 merge 错误。

若 safetensors 没有 alpha metadata，应明确记录：

```text
alpha absent
→ 按作者/Comfy convention 使用 alpha=rank
```

而不是声称“metadata alpha==rank”。

## 2. 不要硬编码 architecture

这两个值可以 assert，但应从 `config.json` 读取：

```text
num_attention_heads = 56
attention_head_dim = 128
```

脚本可以：

```python
assert config["num_attention_heads"] == 56
assert config["attention_head_dim"] == 128
```

然后用 config 的值运行 reorder，而不是把 `56/128` 当作永久常量。

## 3. 输出必须原子化和可恢复

建议目录：

```text
MiniMax-H3-Turbo-v4s600ema.building/
MiniMax-H3-Turbo-v4s600ema/
```

每个 shard：

```text
model-00001-of-00013.safetensors.tmp
→ fsync
→ rename 到正式 shard 名
```

全部完成并验证后：

```text
写 merge_manifest.json
写 .complete
rename 整个 .building 目录
```

`h3_switch.sh` 必须拒绝启动没有 `.complete` 的模型目录。

## 4. 增加 provenance manifest

至少记录：

```json
{
  "base_model": "MiniMaxAI/MiniMax-H3",
  "base_revision": "<HF revision>",
  "base_transformer_index_sha256": "...",
  "lora_repo": "larryvrh/MiniMax-H3-Turbo-Lora",
  "lora_revision": "...",
  "lora_file": "minimax_h3_turbo_v4_step600_ema.safetensors",
  "lora_sha256": "...",
  "strength": 1.0,
  "merge_dtype": "float32",
  "output_dtype": "bfloat16",
  "qkv_disk_layout": "group-interleaved",
  "qkv_runtime_layout": "Q_all_K_all_V_all",
  "merge_script_git_sha": "...",
  "modified_tensor_count": 259
}
```

不要硬编码：

```text
总 tensor 数必须是 535
```

更稳妥的是：

```text
总 tensor 数由 index 动态读取
modified == expected target set
unchanged == all_weights - modified
```

---

# 六、audio shift 的修正不要写得过于绝对

如果作者当前 `v4_step600_ema` 的 `generate.py` 使用：

```text
video shift 12
audio shift 3
```

那么这应该成为 **v4 step600 checkpoint 的 canonical recipe**。

但旧的公开 ckpt500 对比实验确实使用过：

```text
audio shift 6
```

并配合 8/10-step 推理。

所以文档不要写：

> “audio shift 6 是错的。”

建议写：

> `v4_step600_ema` 按当前作者脚本使用 video/audio shifts `12/3`；较早的 ckpt500 社区实验曾使用 audio shift 6，不能把旧 checkpoint 的参数直接沿用到 v4。

另外，仍建议花很小成本做一次：

```text
v4 step600
6 NFE
audio shift 3 vs 6
2–3 个带明显声音事件的 prompt
```

检查：

* 音频 RMS；
* clipping；
* 瞬态；
* A/V event timing；
* 最后 0.5 秒音频是否断裂。

之后才能正式删除 shift 6 分支。

---

# 七、benchmark 设计需要明显加强

当前 plan 主要使用一个：

```text
ToS t=147s 首尾帧
```

这足以做 smoke test，但不足以判断 H3 是否“又快又好”。

Turbo LoRA 最容易在这些场景暴露问题：

* 首尾帧差异很大；
* 全身大动作；
* 多人；
* 手部；
* 镜头运动；
* 末尾突然加速追 last frame；
* 人脸 identity 中途漂移；
* 音频和动作不同步。

## 建议两阶段评测

### Stage A：快速筛选

用 8–10 个 case，每个一个固定 seed：

| 类别             |  样本 |
| -------------- | --: |
| 面部近景、小动作       |   1 |
| 全身、大姿态变化       |   2 |
| 双人交互           |   1 |
| 快速运动           |   1 |
| 静态相机、复杂背景      |   1 |
| pan/zoom/orbit |   1 |
| 首尾构图差异大        |   1 |
| 明显音频事件         | 1–2 |

筛选：

```text
step600: 4 / 6 / 8 NFE
ckpt850: 4 / 6 NFE
```

淘汰明显不稳定的 checkpoint/NFE 组合。

### Stage B：finalist qualification

对最终 2–3 个配置：

```text
12–20 个 case
每个 seed 0/1/2
```

再做完整质量统计。

## 必须保留两个 baseline

```text
Quality oracle:
Base BF16，20 NFE
当前 engine 语义下传 num_inference_steps=21

Production baseline:
Base FP8，11 NFE
当前 engine 语义下传 num_inference_steps=12
```

同时还要做：

```text
Base FP8，4/6/8 NFE
```

这能证明：

> 质量提升来自 Turbo LoRA，而不是单纯把 base model 步数降低。

---

# 八、性能 benchmark 和质量 benchmark 必须分离

你当前写的是：

```text
warmup 1
timed 2
取中位
seed 0/1/2 轮换
```

这里有两个问题。

## 1. 两次 timed run 没有可靠 median

两个值的 median 实际上只是二者中间值/平均意义，不能过滤抖动。

建议最低：

```text
1 次 exact-shape warmup
3 次 timed
取 median
```

更稳：

```text
1 次 warmup
5 次 timed
报告 median + min/max
```

## 2. 性能测试不要轮换 seed

性能测试固定：

```text
同 prompt
同首尾帧
同 seed
同 token length
同 shape
同 NFE
```

质量测试才使用多个 seed。

每个不同组合都需要自己的 warmup：

```text
分辨率
NFE
精度
checkpoint
engine
```

因为不同 NFE 可能触发不同 scheduler/AdaLN schedule cache，不应让第一个请求承担初始化成本。

## 应记录的 stage timing

```text
text/vision encode
first/last-frame VAE encode
DiT total
observed DiT forward count
median time per DiT forward
video VAE decode
audio VAE decode
mux
client E2E
server E2E
peak VRAM per GPU
peak host RAM
```

---

# 九、G3 不能要求 merged-FP8 和 dynamic-FP8 一致

需要分成四种路径：

```text
A. merged BF16
B. dynamic LoRA + BF16 base

C. merged BF16 → online FP8
D. dynamic LoRA + online FP8 base
```

A 与 B：

```text
理论目标相同
只存在计算/rounding 路径差异
应高度接近
```

C 与 D：

```text
C = Q(W + ΔW)
D = Q(W)x + ΔWx
```

量化是非线性的，因此：

[
Q(W+\Delta W)\neq Q(W)+\Delta W
]

所以 merged-FP8 和 dynamic-FP8 **不应被要求 latent 或像素一致**。

正确 Gate 是：

```text
merged BF16 vs dynamic BF16：
高数值/感知一致性

merged FP8 vs merged BF16：
量化质量门槛

dynamic FP8 vs merged BF16：
独立质量门槛

merged FP8 vs dynamic FP8：
只比较感知质量与速度，不要求相同轨迹
```

SGLang runtime LoRA 应只定位为：

```text
checkpoint A/B
hot swap
debug
merge oracle
```

生产默认仍应使用 merged checkpoint，因为 dynamic LoRA 每个适配 layer 都会增加额外 GEMM，而且 SGLang LoRA wrapper 初始化时还会保留 base-weight CPU snapshot。

如果要跑 dynamic，建议限制 wrapper target：

```bash
--lora-target-modules \
  qkv_proj \
  out_proj \
  fc1 \
  fc2 \
  adaln_proj.linear
```

这样不会给不相关的 H3 linears 建立 LoRA wrapper 和 CPU backup。

---

# 十、为 FP8 准备一个质量救援分支

Turbo LoRA 改了大量：

```text
qkv
attention out
fc1
fc2
AdaLN
final AdaLN
```

few-step student 对误差通常比 20-step base 更敏感。尤其 AdaLN 的 delta 直接改变每一步各模态的 scale、shift 和 gate。

所以如果：

```text
merged BF16 Turbo 很好
merged FP8 Turbo 明显变差
```

不要立即否定 Turbo LoRA。

应先测试 selective BF16：

```text
FP8 主干
但保留：
blocks.*.adaln_proj.linear
final_layer.adaln_proj.linear
```

若仍有问题，再试：

```text
首 2 block BF16
末 2 block BF16
AdaLN BF16
其余 FP8
```

你的 `‖ΔW‖F / ‖W‖F` 报告非常有价值，可以优先识别 LoRA delta 相对较大的模块，把这些模块作为 FP8 ignored-layer candidates。

---

# 十一、两套引擎必须做 apples-to-apples

目前计划是：

```text
vLLM：主要测 864×480
SGLang：主要测 768P
```

这样最终无法判断：

```text
速度差异来自 engine
还是来自 resolution
```

至少对最终候选做：

| Engine            | 864×480 | 1344×768 |
| ----------------- | ------: | -------: |
| vLLM merged FP8   |   6 NFE |    6 NFE |
| SGLang merged FP8 |   6 NFE |    6 NFE |

然后再选生产主线。

不建议长期维护两个“同功能、不同 engine”的一等生产服务。最终应该是：

```text
主生产 engine
+
另一 engine 作为验证/fallback
+
SGLang dynamic LoRA 作为研究服务
```

否则每次换 checkpoint、量化策略和 scheduler 都需要维护三份行为。

---

# 十二、更新后的 Gate 建议

| Gate                           | 必须满足                                                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **G0 — Provenance**            | LoRA exact key set 259；base/LoRA revision 与 SHA256；license/usage 条款；4/6/8 NFE sigma schedule 与作者脚本一致 |
| **G1 — Merge correctness**     | 259/259 应用；未修改 tensor 相同；qkv permutation round-trip；完整 scale 公式；独立 Comfy/runtime LoRA oracle 通过      |
| **G2 — BF16 quality frontier** | step600/ckpt850 与 4/6/8 NFE 筛选；与 20-NFE quality oracle 和 11-NFE production baseline 比较               |
| **G3 — FP8 qualification**     | merged FP8 相对 merged BF16 无严重回退；若失败，执行 AdaLN/边界 block selective-BF16 ablation                        |
| **G4 — Engine qualification**  | vLLM 与 SGLang 在同分辨率、同 NFE 下比较；observed forward 数正确；SGLang dynamic BF16 与 merged BF16 高度接近            |
| **G5 — Production soak**       | 最终 profile 连续跑至少一批请求；VRAM/RAM 无持续增长；无 OOM；报告 p50/p95；server switch/rollback 正常                       |

质量门槛可以先设置成：

```text
严重 anatomy / identity / scene-cut failure:
不高于当前 production baseline

first/last DINO similarity:
绝对下降不超过约 0.02

ArcFace track p10:
绝对下降不超过约 0.03

人工 blind judge:
Turbo finalist 的 win + tie >= 80%

音频：
无无声、爆音、截断、时长错误
```

这些不是学术标准，但足够作为第一版生产 gate。

---

# 十三、对延迟目标的现实判断

根据你给出的 864×480 数据：

```text
11 NFE → 37.4s
```

若固定 encode/decode/mux overhead 约 5–8 秒，粗略推算：

| NFE |   预计 E2E |
| --: | -------: |
|   4 | 约 17–21s |
|   6 | 约 23–27s |
|   8 | 约 29–33s |

因此最可能的定档是：

```text
turbo-fast:
v4 step600 EMA
4 NFE
merged FP8

turbo-balanced:
v4 step600 EMA
6 NFE
merged FP8

turbo-quality:
v4 step600 EMA
8 NFE
merged BF16 或 selective FP8

fallback:
base FP8
11 NFE
```

**6 NFE 是最值得押注的 production default。**

对于当前 768P 的 126.2 秒 baseline，即便 NFE 近似线性下降：

| NFE |    粗略 E2E |
| --: | --------: |
|   4 |  约 50–60s |
|   6 |  约 70–80s |
|   8 | 约 90–100s |

因此 Turbo LoRA 会让 native 768P **大幅变快**，但单靠它不太可能把 native 768P 压到 30 秒。

如果最终产品要求：

```text
768P 成品
且接近 30 秒
```

更现实的 production path 是：

```text
864×480 H3 Turbo，6 NFE
→ temporal-consistent video upscale 到 768P
```

而不是直接让 H3 在 1344×768 的长 packed sequence 上完成全部 denoise。

---

# 最终建议

这份 action plan 可以保留约 80% 内容，但执行前应做以下变更：

1. **删除“直接手写 SGLang 2D fused-B patch”步骤**，先检查并 backport 当前上游实现；
2. 将所有 profile 从“steps”改为 **NFE**，内部再映射成 engine sigma-point count；
3. 在 G1 加入 **独立 Comfy/runtime-LoRA oracle**，避免 merge 和 verify 使用同一错误假设；
4. merge 脚本实现完整 `alpha/rank × strength`，并输出 immutable manifest；
5. 将 quality benchmark 扩展到多种 motion/identity/endpoint 场景；
6. 性能测试固定 seed，warmup 1＋timed 至少 3；质量测试再换 seed；
7. 将 **merged BF16、dynamic BF16、merged FP8、dynamic FP8** 分开评估；
8. 最终只保留一个 production engine，另一套用于 fallback/verification；
9. 默认生产候选先锁定在 **v4 step600 EMA、6 NFE、merged FP8、864×480**。

完成这些修正后，这条路线不仅有很大概率把 864×480 压进 30 秒，也能避免“速度达标，但实际上 QKV merge、schedule 或 FP8 量化已经悄悄破坏 Turbo LoRA”的风险。
