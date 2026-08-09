# 5090 双卡 serving 结论(2026-08-08)

## 结论:vLLM-Omni 与 SGLang 在本机(2×RTX 5090 32GB + **125GB RAM**)均不可行,瓶颈是主机内存

尝试与失败根因(全部实测):

| 尝试 | 配置 | 失败根因 |
|---|---|---|
| vLLM #1 | TP2 + 在线 FP8 + cpu-offload + TE-TP2 | 运行时算不过:TE-TP 常驻 25.7G/卡 + DiT FP8 16.6G + VAE 5.8G = 48G > 31.3G 可用;实际在 pipeline 构造(VAE init)时 OOM |
| vLLM #2 | 同上去 TE-TP | TE 整体 51.5G BF16,单卡任何时刻都放不下,模式不成立 |
| vLLM #3 | TP2 + distributed layerwise offload(官方 2×5090 实验路径) | H3 模型不支持 mmap → 常规加载器把全量权重读进 RAM → 125G 主机 OOM(rank0 被内核杀) |
| SGLang | TP2 + FP8 + TE/VAE layerwise(memory 模式) | TP2 两个 rank **各自**在 RAM 暂存整份 TE(2×51.5G=103G)+ DiT 流式缓冲 → RAM 打爆 |

官方佐证:SGLang cookbook 的 2×5090 验证配方明确标注 *"validated on … a 377 GiB host; use a 384 GiB-class machine"* —— 本机 125GB 是官方要求的 1/3。无免密 sudo,无法加 swap 兜底(现 swap 仅 8G)。

## 实际影响:很小

5090 的 **ComfyUI turbo(TeaCache, 35s)本来就快于 6000a 的 vLLM FP8(37s)** —— serving 栈在 864×480 上对这台机器没有增益;它们的价值(768P 高清、无损多卡)已由 6000a 承担。

## 分工定型

```text
864×480 低延迟   → 5090 turbo(35s, TeaCache 有损)或 6000a turbo(37s, 仅 FP8 近似)
864×480 无损     → 6000a vllm(44s, BF16 TP4)
1376×768 高清    → 6000a sglang(126s, FP8)
质量参照         → 6000a bf16(400s, 零量化)
```

## 若要在 5090 恢复 serving

1. **加内存到 384GB 级**(官方配方门槛)→ `h3_switch_5090.sh vllm|sglang` 的启动逻辑已备好,直接可用
2. 或获得 sudo 加 ≥128G swapfile(NVMe 上流式读取速度与 RunAI streamer 同级,理论可行,未验证)
3. 相关安装坑均已修好并入档(vllm-omni/sglang 源码 editable、setuptools_scm、nvcc/ninja/libcudart):
   见 `claude_history/05_5090_serving/`,重试时环境即用

## 附带产出(已就绪,非浪费)

- 官方 FL2VA 权重 145G 已在 5090(`models_official/`)+ HF cache 桥接 —— 未来恢复/训练可用
- 两个 conda env(h3_vllm/h3_sglang)装好可 import;serving 客户端与切换脚本部署完毕
- CLI:5090 的 vllm/sglang profile 已标注不可用并回退 baseline(帮助文本说明原因)

---

# TP2 攻坚(2026-08-08~09,按 doc/TP2_5090.md 执行)

## 结果:基础设施跑通(47s),但输出为噪声 —— sm_120 的 FP8 数值路径全线不可用

### 路线 1:SGLang PR#33681(TE FP8 fold + DiT resident FP8)
❌ 加载期 OOM:sglang 在线 FP8 是「先整载 BF16 shard 再量化」,TP2 每 rank BF16 shard 33G > 31.3G。
(6000a TP4 能过是因为 16.6G shard 装得下 48G 卡。)

### 路线 2:vLLM-Omni PR#5910(全局 FP8 + rank-local DLO)
✅ 服务端可跑:TP2+DLO resident=20/50 都稳定出片,warm 47s,峰值 15G/24G 每卡,RAM 安全
❌ **输出为完全噪声**。排查证据链(每步单变量):

| # | 变量 | 结果 |
|---|---|---|
| 1 | Cutlass FP8 kernel | 请求期崩:`scaled_mm_entry.cu:209` 无 sm_120 kernel |
| 2 | ChannelWiseTorch FP8 kernel | 能跑,输出噪声 |
| 3 | attention CUDNN→TORCH_SDPA | 噪声不变 → attention 排除 |
| 4 | attention FLASH_ATTN | 崩:FA3 是 Hopper 编译,无 sm_120 kernel |
| 5 | Humming FP8 kernel | 权重处理崩(stride 与 DLO 布局冲突) |
| 6 | Marlin(FORCE_FP8_MARLIN=1) | 崩:PR 自定义 `MiniMaxH3Qwen3VLQKVParallelLinear` 缺 `output_size_per_partition`(PR 集成缺口) |
| 7 | FlashInfer | 未装(未测) |

### 当前最强假设(已被 round-2 推翻,见下章)
torch 的 `_scaled_mm`(cuBLAS)在 sm_120 数值上不应出错 → 噪声疑似来自 **GEMM 上游**:
PR5910 的 DLO no-allgather 权重重建或 H3 TP2 分片在该路径的 bug(作者验证硬件未知,PR 未合并)。

### 下一步的两个决定性实验(留给后续 session)
1. **Marlin 属性 patch**:给 PR 的自定义 Linear 补 `output/input_size_per_partition` 属性(几行),
   Marlin 若也噪声 → 实锤上游损坏;若干净 → ChannelWiseTorch sm_120 数值锅
2. **6000a 交叉验证**:同 PR 同配置跑 6000a 的 2 卡(sm_89,cutlass 可用)——
   噪声→PR 本身坏;干净→sm_120 特有。(注意会占用 6000a,需协调)
3. 或等 PR 合并/修复后重试;资产全部保留(权重/env/切换脚本),重试成本≈0

### 现场状态
comfy 双 worker 已恢复 ✓;CLI 的 5090 vllm profile 已标注不可用(防止误用噪声产线)

---

# TP2 攻坚 round-2(2026-08-09,按 doc/TP_5090_round2.md 执行):已修复,产线上线

## 结论:root cause 是 PR5910 的一个一行 bug,与 sm_120 完全无关;修复后 5090 TP2 产线可用

**`PinnedResidentLayerGroup.load()`(DLO 常驻组)重建权重时用 `.view(shape)` 而非
`torch.as_strided(..., stride=meta["stride"])`,丢弃了在线 FP8 权重的转置视图布局。**
`_shard_and_pin` 明确按物理列主序打包并保存 stride(streamed 路径 `prefetch_layer` 用得对),
常驻组却把列主序字节按行主序重解释 → 常驻 block 的所有 linear 权重值级乱序:
严格 kernel(Cutlass/Humming/cuBLASLt)拒收崩溃,宽松路径(ChannelWiseTorch、BF16-dequant)
照算出**纯噪声**。round-1 的「sm_120 FP8 数值路径全线不可用」是错误归因。

## round-2 实验链(每步单变量)

| # | 实验 | 结果 | 排除/定位 |
|---|---|---|---|
| P0 | PR head b18eeff2 复测 | 仍噪声 | 上游未修 |
| P1 | `VLLM_BATCH_INVARIANT=1`(BF16-dequant 绕过 `_scaled_mm`;哨兵实证分支生效) | 仍噪声、同花纹 | **kernel 数值排除** |
| — | 哨兵埋点发现 | H3 全链路走 `Fp8PerTensorOnlineLinearMethod`(vllm `online/fp8.py`),不走 `Fp8LinearMethod` | note 的 P1 机制在 online 模块,已验证 |
| P4 | resident=20/50(=全 50 块)均噪声(round-1 数据) | — | **运行期 DLO streaming 排除** |
| — | TP1 对照 | 两机都装不下(TE 构造期需 51.5G BF16 整载) | 不可行,弃 |
| P5a | 6000a TP2 强制 ChannelWiseTorch | cuBLASLt `NOT_SUPPORTED` 拒收 | 同 kernel 跨硬件不可达 |
| P5b | 6000a TP2 Cutlass | **与 5090 一模一样的 `scaled_mm_entry.cu:209` 崩** | 跨硬件同症 → 非 sm_120 |
| P3-lite | 逐调用 dump(210 调用) | 209 个 OK(stride `(1,K)` 转置视图);第 210 个 `blocks.0.adaln_proj.linear` **contiguous 行主序** → 崩 | **常驻块权重布局损坏实锤**(恰是 note 点名的 AdaLN) |
| 修复 | resident repoint 改 `as_strided`(一行) | 6000a Cutlass 出片干净;5090 ChannelWise 出片干净 | **root cause 确认** |

## 生产配置与数据(5090)

```text
2×RTX5090 TP2 + TE-TP2 | global online FP8(per-tensor)| DLO no-allgather resident=50(全常驻)
CUDNN_ATTN | enforce-eager | 864×480×124帧 12步
warm 46.9s / 46.8s / 46.6s(E2E),峰值 23.95G/卡(31.3G 内安全),音频 aac 立体声正常
质量: 帧对比通过(与 6000a Cutlass 输出一致;seed 0/1/7 多籽验证)
注: TORCH_SDPA 是 round-1 排查遗留,会慢 ~18s(64.8s),已回 CUDNN
```

- 补丁: `scripts/pr5910_resident_stride_fix.patch`(两机 src 均已打;git reset 后重打即可;
  可作为 upstream PR#5910 的修复贡献)
- CLI: `--host 5090 --profile vllm` 已恢复暴露,E2E 验证通过
- 分工更新: 5090 现有 turbo(TeaCache 35s 有损)与 vllm(TP2 FP8 47s,更接近无损)两档 serving 能力
- 后续加速空间(note 的路线,未做): Turbo LoRA merge 进 FP8(NFE6 预计 ~25s)、固定 shape 完整 warmup
