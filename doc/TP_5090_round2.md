## 我的判断

这个阶段性结论可以归档为：

> **双 5090 的 H3 TP2＋global FP8＋DLO 容量与调度架构已经成立，但 correctness gate 失败；故障已收敛到 FP8 Linear 路径，attention 基本排除。**

不过我会修改你结论中的一句话：

> “根子在 sm_120 的 FP8 数值路径”

目前还应写成：

> **问题被限制在 sm_120 上的 FP8 Linear execution stack，但尚未区分是 ChannelWiseTorch / `torch._scaled_mm`、DLO 权重恢复、TP2 分片，还是 TE FP8。**

现在还不能把 **DLO/TP bug** 定为唯一最强假设。`ChannelWiseTorch` 自己仍然是并列第一嫌疑人：它不是简单调用一个完整的 channel-wise cuBLAS GEMM，而是先用单位 scale 执行 `torch._scaled_mm`，再在 Python/Torch 侧乘 activation scale 和 weight scale。这个执行路径和 B300 上验证过的 Cutlass path 并不相同。

另外，PR #5910 **目前仍是 draft**，当前 head 是 `b18eeff22a30b426060f44f734d8b16341b02959`；当前公开验证是 single-B300，并没有 SM120 TP2 correctness evidence。若你跑的是先前计划里的 `9f921476...`，应先对当前 head 做一次低成本复测。

---

# 最优先实验：不要先补 Marlin，先打开 `VLLM_BATCH_INVARIANT`

这是目前最干净、成本最低、信息量最大的实验。

vLLM 的 online FP8 实现里，当：

```text
VLLM_BATCH_INVARIANT=1
且当前不是 Cutlass FP8 kernel
```

PTPC/channel-wise FP8 会：

1. 保留完全相同的 FP8 weight；
2. 保留完全相同的 weight scale；
3. 保留完全相同的 TP2 分片；
4. 保留完全相同的 DLO offload/prefetch；
5. 但绕过 `torch._scaled_mm`；
6. 把权重反量化到 BF16，然后用 `torch.nn.functional.linear` 执行。

也就是说，它几乎完美地把：

```text
kernel 数值路径
```

和：

```text
权重、scale、DLO、TP 分片
```

分开了。

## 启动方式

保持当前全部参数不变，只增加：

```bash
export VLLM_BATCH_INVARIANT=1
```

并明确固定 Torch FP8 backend：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_BATCH_INVARIANT=1 \
vllm serve "${MODEL}" \
  --omni \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 \
  --quantization fp8 \
  --linear-backend torch \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager \
  ...
```

建议在 FP8 method 的 fallback branch 临时增加一条 `logger.warning_once(...)`，确认它确实进入了 BF16 dequant path，而不是环境变量没有被当前 worker 继承。

不要用 2 NFE 的最终视频判断，因为正常模型在 2 NFE 下也可能像噪声。使用你已经知道应当生成正常视频的：

```text
Turbo 4 NFE
或
Turbo 6 NFE
或
base 11 NFE
```

并保持完全相同的 seed、首尾帧和 sigma schedule。

## 结果解释

| 结果                                 | 结论                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------- |
| 原 ChannelWise＝噪声；BatchInvariant＝正常 | **DLO、TP、量化权重和 scale 基本正确；SM120 `torch._scaled_mm` / ChannelWise scale epilogue 是主因** |
| 两者都是噪声                             | 不是单纯 kernel 问题；继续查 **TE、DLO 数据恢复、TP2 分片或量化权重构造**                                      |
| BatchInvariant 输出不再是纯噪声，但明显异常      | 权重大体正确；可能同时存在 kernel 数值问题和某个量化敏感模块                                                    |
| BatchInvariant OOM                 | 只是诊断路径的临时 BF16 dequant workspace 太大；改用单层 microtest，不说明权重有问题                           |

**这是我认为你下一步必须最先做的实验。它比补 Marlin 更直接。**

---

# 第二个实验：用 per-component quantization 排除 TE FP8

你当前使用 global FP8，所以：

```text
DiT FP8
TE FP8
```

同时开启。切换 attention backend 只能排除 DiT attention kernel，不能排除 Qwen3-VL TE 已经产生错误 conditioning embedding。

当前 H3 pipeline 会单独 resolve：

```text
transformer
```

的 component quantization，而 TE 只有在拿到 `get_name() == "fp8"` 的全局 config 时才会执行它自己的 online FP8。因此，可以使用 component config 做：

```text
DiT FP8
TE BF16
```

隔离。

配置逻辑：

```python
quantization_config = {
    "transformer": {"method": "fp8"},
    "text_encoder": None,
}
```

在你的 server/stage config 中把该 dict 传入 `quantization_config`，不要再传单一字符串 `"fp8"`。

## 解释

| DiT FP8 / TE BF16 结果 | 判断                               |
| -------------------- | -------------------------------- |
| 正常                   | H3 TE online FP8 是问题源            |
| 仍是噪声                 | TE 基本排除，问题位于 DiT FP8 / DLO / TP2 |
| RAM 超限               | 改做 TE standalone 对齐测试，不要开 swap   |

TE standalone 对齐可以只加载 Qwen3-VL：

```text
TE BF16 TP2
vs
TE FP8 TP2
```

同一 prompt 和输入图像，比较最终 `[seq, 5120]` embedding：

```text
all finite
cosine similarity
relative L2
norm ratio
mean/std
max absolute error
```

由于只加载 TE，不加载完整 DiT 和 VAE，BF16 TP2 更有机会放进两张 5090。

建议门槛：

```text
cosine > 0.995
norm ratio 0.98–1.02
无 NaN/Inf
```

若 embedding 已经出现数量级漂移，视频纯噪声就完全可以由 TE 解释。

---

# 第三个实验：直接测试 DLO 前后同一个真实 FP8 Linear

PR #5910 当前对 DLO 的测试只验证了：

```text
普通 transposed tensor
值不变
stride 不变
```

它没有验证：

```text
真实 online-FP8 weight
weight_scale
bias
TP shard metadata
实际 FP8 Linear 输出
```

所以现在应该增加一个真正的 `H3 FP8 linear DLO round-trip test`。

## 至少覆盖这些层

```text
blocks.0.attn.qkv_proj        # fused column-parallel
blocks.0.attn.out_proj        # row-parallel
blocks.0.mlp.fc1              # fused gate/up column-parallel
blocks.0.mlp.fc2              # row-parallel
blocks.0.adaln_proj.linear    # 大型、极敏感、gather_output
token_refiner.blocks.0...
final_layer.adaln_proj.linear
```

## 每个 tensor 检查

DLO 前和 prefetch 后记录：

```python
shape
dtype
stride
storage_offset
numel
is_contiguous
weight value equality
weight_scale value equality
bias equality
quant_method class
kernel class
logical_widths
input_size_per_partition
output_size_per_partition
```

尤其不能只比较 logical tensor values。还要比较同一个输入上的输出。

## 反量化 reference

online PTPC FP8 通常把 weight 保存为：

```text
[K, N] FP8
```

weight scale 为：

```text
[N, 1] 或 [N]
```

可以用：

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def dequant_reference(
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    weight_kn = layer.weight.float()
    scale = layer.weight_scale.float()

    if scale.numel() == 1:
        weight_kn = weight_kn * scale
    elif scale.ndim == 2 and scale.shape[0] == weight_kn.shape[1]:
        weight_kn = weight_kn * scale.T
    elif scale.ndim == 1 and scale.shape[0] == weight_kn.shape[1]:
        weight_kn = weight_kn * scale.unsqueeze(0)
    else:
        raise RuntimeError(
            f"Unexpected scale={tuple(scale.shape)}, "
            f"weight={tuple(weight_kn.shape)}"
        )

    output = x.float() @ weight_kn
    bias = getattr(layer, "bias", None)
    if bias is not None:
        output = output + bias.float()

    return output.to(torch.bfloat16)


def compare(name: str, actual: torch.Tensor, reference: torch.Tensor) -> None:
    a = actual.float()
    b = reference.float()
    diff = a - b

    rel_l2 = float(diff.norm() / b.norm().clamp_min(1e-12))
    cosine = float(
        F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    )
    print(
        name,
        {
            "finite": bool(torch.isfinite(a).all()),
            "rel_l2": rel_l2,
            "cosine": cosine,
            "max_abs": float(diff.abs().max()),
            "a_std": float(a.std()),
            "b_std": float(b.std()),
        },
    )
```

对每个目标层执行三次：

```text
A. 刚完成 online quantization、尚未 DLO
B. DLO 写入 CPU/pinned buffer 后
C. DLO prefetch 回 GPU 后
```

判断：

```text
A 正常，C 错误 → DLO reconstruction
A 就错误            → quantization / kernel / TP
A/C reference 都正确，但整模型噪声 → 层间 TP collective、TE 或其他非 Linear 路径
```

---

# 第四个实验：TP2 数学重建测试

只测每个 rank 上的 local output 还不够。要验证 TP2 合成后的结果是否等价于 full layer。

## QKV column-parallel

每个 rank 的 local 输出布局通常是：

```text
rank 0: [q0 | k0 | v0]
rank 1: [q1 | k1 | v1]
```

完整 global QKV 不能简单做：

```python
torch.cat([rank0, rank1], dim=-1)
```

那会变成：

```text
[q0 | k0 | v0 | q1 | k1 | v1]
```

正确重建应是：

```text
[q0 | q1 | k0 | k1 | v0 | v1]
```

也就是：

```python
q = torch.cat([q0, q1], dim=-1)
k = torch.cat([k0, k1], dim=-1)
v = torch.cat([v0, v1], dim=-1)
qkv_global = torch.cat([q, k, v], dim=-1)
```

## FC1 fused gate/up

同理：

```text
rank 0: [gate0 | up0]
rank 1: [gate1 | up1]
```

正确 global：

```text
[gate0 | gate1 | up0 | up1]
```

## Row-parallel

`out_proj` 和 `fc2` 应满足：

[
Y_{\text{global}} =
Y_{\text{rank0}} + Y_{\text{rank1}}
]

并比较 all-reduce 前后：

```text
FP32 sum
BF16 sum
当前实际 collective
```

## AdaLN

AdaLN 的 `ColumnParallelLinear(... gather_output=True)` 必须验证最终 gather 顺序。AdaLN 一旦 row ordering 错误，shift/scale/gate 会被整体错配，最终表现非常容易变成纯噪声。

**从症状上看，AdaLN/qkv/fc1 的 layout 或 scale 错误，比普通“FP8 精度略差”更符合纯噪声。**

---

# 第五个实验：6000 Ada 交叉验证要强制相同 kernel

你计划中的 6000 Ada 交叉验证是正确的，但不要只比较默认 backend，否则：

```text
5090 → ChannelWiseTorch
6000 Ada → Cutlass
```

同时改变了硬件和 kernel，结论不够干净。

正确矩阵是：

| Hardware        | Kernel                   | DLO              | 用途                     |
| --------------- | ------------------------ | ---------------- | ---------------------- |
| 2×5090 SM120    | Torch/ChannelWise        | On               | 当前失败基线                 |
| 2×6000 Ada SM89 | **强制 Torch/ChannelWise** | On               | 检查 ChannelWise 是否跨硬件都错 |
| 2×6000 Ada SM89 | Cutlass                  | On               | 已知高性能 kernel 控制组       |
| 2×6000 Ada SM89 | Torch/ChannelWise        | Off 或尽量 resident | 分离 ChannelWise 与 DLO   |

强制 Torch：

```bash
--linear-backend torch
```

强制 Cutlass：

```bash
--linear-backend cutlass
```

## 结果解释

| 结果                                       | 结论                                                 |
| ---------------------------------------- | -------------------------------------------------- |
| 5090 ChannelWise 噪声；6000A ChannelWise 正常 | SM120 `torch._scaled_mm` / cuBLASLt 路径             |
| 两边 ChannelWise 都噪声；6000A Cutlass 正常      | ChannelWise implementation 或 scale/layout contract |
| 两边所有 kernel 都噪声                          | DLO、TP2 或量化权重构造                                    |
| 6000A Cutlass 正常且关 DLO正常、开 DLO噪声         | DLO reconstruction                                 |
| 6000A TP1 正常、TP2噪声                       | TP sharding/collective                             |

---

# Marlin 补丁应该排在什么位置

你提出的“给自定义 Linear 补属性让 Marlin 跑”是有价值的，但应放在 `VLLM_BATCH_INVARIANT` 之后。

## BatchInvariant 变正常

这时 Marlin 是很好的第二 kernel oracle：

```text
ChannelWiseTorch：噪声
BF16 dequant fallback：正常
Marlin：？
```

若 Marlin 正常，结论非常强：

```text
weight/scales/DLO/TP 正常
ChannelWiseTorch / scaled_mm 错
```

需要补齐的常见 metadata 包括：

```text
tp_size
input_size
input_size_per_partition
output_size
output_size_per_partition
logical_widths
orig_dtype
weight_block_size
```

补丁必须只增加 metadata contract，不得重新排列或重新量化 weight，否则会失去“只隔离 kernel”的意义。

## BatchInvariant 仍然噪声

这时先补 Marlin意义很小，因为换 kernel无法修复已经错误的 weight、scale 或 shard。

---

# 一个更快的 DLO 全模型初筛

你现在 `resident=20` 峰值只有约 15GB/卡，说明还有很大显存余量。

可以快速扫：

```text
resident=20
resident=32
resident=40
resident=46
resident=50
```

每次记录 peak VRAM。

如果 `resident=50` 能启动，做一次完全相同请求：

|           resident | 结果                                                         |
| -----------------: | ---------------------------------------------------------- |
|        20 噪声，50 正常 | 非 resident block 的 DLO prefetch/repoint 高度可疑               |
|          20、50 都噪声 | 不是“每步 streaming”本身；继续看 kernel/initial reconstruction/TP/TE |
| resident 越高，异常逐渐减轻 | 可能只有部分被 offload 的模块恢复错误                                    |

这不是严格 oracle，因为 resident layer 在启动期间仍可能经过 DLO 初始化，但它非常便宜，值得同时跑。

---

# 我会怎样重排下一步优先级

## P0：确认当前 commit

```text
vLLM-Omni PR #5910 current head:
b18eeff22a30b426060f44f734d8b16341b02959
```

若当前环境不是这个 commit，先做一次 current-head smoke。PR 当前仍是 draft，而且上游当前没有 SM120 TP2 验证，所以“等 merge”不是技术解决方案，只是行政等待。

## P1：`VLLM_BATCH_INVARIANT=1`

这是最决定性的 kernel bypass。

## P2：DiT FP8＋TE BF16 component isolation

排除 TE conditioning。

## P3：真实 FP8 Linear 的 DLO 前后输出等价测试

不要再只测普通 transpose tensor。

## P4：resident layers 20→50

快速判断 per-step DLO 是否相关。

## P5：6000 Ada 强制相同 ChannelWise kernel

之后再用 Cutlass 作控制组。

## P6：Marlin metadata 补丁

只在 P1 指向 kernel 后执行。

---

# 对最终“又快又好”目标的影响

好消息是：**容量架构已经成功。**

你现在已经证明：

```text
2×5090
TP2
global FP8
DLO no-AllGather
resident=20
约15GB/卡
128GB RAM安全
warm约47s
```

这意味着 correctness 修好后还有三个明确加速空间：

1. `resident=20 → 40–50`，减少 PCIe block streaming；
2. Turbo LoRA 降到 4–6 NFE；
3. 固定 864×480×124 shape，做完整 warmup。

最可能的生产档位仍然是：

```yaml
hardware: 2x RTX 5090 32GB
engine: vLLM-Omni
dit_parallel: TP2
text_encoder_parallel: TP2
precision: global FP8
turbo_lora: merged before quantization
nfe: 6
resolution: 864x480
frames: 124
dlo:
  no_allgather: true
  resident_layers: highest safe value, likely 40–50
attention: CUDNN_ATTN or verified fallback
compile: disabled until correctness qualified
```

如果 6 NFE＋更高 residency 仍超过 30 秒，再考虑 cache；当前不要加入 Cache-DiT、TeaCache 或 compile，因为它们会扩大 correctness 搜索空间。

## 最终阶段性结论

你的工程并没有失败。现在已经完成了最困难的容量与分布式部署部分：

> **H3 global FP8 TP2 能在 2×5090＋128GB RAM 上稳定运行。**

剩下的不是再设计一套部署架构，而是定位一个明确的 numerical correctness bug。最先运行的不是 Marlin，也不是等待 PR merge，而是：

```text
VLLM_BATCH_INVARIANT=1
```

这一个实验就能把当前搜索空间直接砍掉一半。
