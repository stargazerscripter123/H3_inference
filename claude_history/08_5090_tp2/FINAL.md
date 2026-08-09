# FINAL · 5090 TP2(状态: **已修复上线**, 2026-08-09 by 3195aa2f)

## 一句话
TP2 噪声的 root cause 是 **PR5910 `PinnedResidentLayerGroup.load()` 用 `.view()` 重建
DLO 常驻块权重、丢弃在线 FP8 转置视图的 stride**(一行 bug,与 sm_120 无关);
改为 `as_strided(..., stride=meta["stride"])` 后 5090 TP2+FP8+DLO 产线可用:
**warm 46.9s,峰值 23.95G/卡,画质与 6000a Cutlass 输出一致**。
完整证据链: doc/speedup_5090_serving_results.md「TP2 攻坚 round-2」章节。

## 历史结论修订
- round-1「sm_120 FP8 数值路径全线不可用」被 round-2 推翻(2026-08-09):
  cutlass `scaled_mm_entry.cu:209` 在 sm_89 上报**同一行**错 → 是布局契约检查拒收
  损坏权重,不是缺 kernel。崩×3(Cutlass/Humming/cuBLASLt=严格 kernel 拒收)与
  噪声(ChannelWise/BF16-dequant=宽松路径照算乱序字节)是**同一个 bug 的两种表象**。
- Marlin 补丁 / FlashInfer / ModelOpt 三条备选路线全部不再需要。

## 生产配置(h3_switch_5090.sh vllm 即当前最佳)
```text
PR5910 @ b18eeff2 + scripts/pr5910_resident_stride_fix.patch(两机 src 已打)
TP2 + TE-TP2 | global online FP8(per-tensor)| DLO no-allgather resident=50(H3 恰 50 层=全常驻)
CUDNN_ATTN(勿用 TORCH_SDPA,慢 ~18s)| enforce-eager | ChannelWiseTorch(sm_120 上禁 Cutlass)
CLI: h3_generate.py --host 5090 --profile vllm(E2E 已验证)
```

## 若 src 被 reset/升级
1. **补丁已固化(2026-08-09)**:两机 `src/vllm-omni-pr5910` 现在分支
   `pr5910-stride-fix` @ `070096bd`(在 b18eeff2 之上),不再是未提交的工作区改动。
   误 reset 后:`git switch pr5910-stride-fix` 即可;patch 文件仍留作备份。
2. 若 upstream 已合并含修复的版本,先跑 seed0/12步 帧检查再撤 patch
3. 该 patch 值得贡献回 upstream PR#5910(作者只在单卡 B300 验证过,未踩到常驻+转置组合)
4. 判断补丁在不在,**不要用 `git diff`**(已 commit 时它会报"干净"=误判),
   用 `grep as_strided vllm_omni/diffusion/offloader/distributed_layerwise_backend.py`

## switch 脚本正确性(2026-08-09 补,由 1ed2dca3 修)
`h3_switch_5090.sh` 曾可**静默服错 checkpoint**:`vllm)`/`vllm-turbo)` 都不
`stop_one vllm`,而 `start_vllm` 首行"端口健康就 return"→ 互切成空操作。且
**延迟无法自证**(拟合 F=8.0s、p=3.5s/forward ⇒ base NFE6 也是 ~29.0s,与 turbo
NFE6 重合)。已修:变体追踪(`run/vllm.variant`+`.model`)、半死进程清理、端口占用
拒启、`flock` 串行化、启动存活确认、日志轮转+启动头;后门脚本
`tp2_r2_launch_{p2,tp1}.sh` 补上各自变体声明。决策表测试
`scripts/test_h3_switch_5090.sh` 9/9。

**本主题历史数字未受影响**:对 `outputs/vllm_turbo_5090/Q_nfe6_seed0.mp4`(seed 0)
与 6000a 的 turbo/base 锚点做了逐帧比对——PSNR 31.94(turbo)/29.70(base),
而"已知不同 ckpt"基准是 30.86;目视亦一致。**确认那次 29.0s 用的是 merged
Turbo-LoRA**。详见 `doc/turbo_lora_results.md` 第八b节。

## 资产台账
- TE FP8: models_te_fp8/Qwen3-VL-32B-Instruct-FP8(sglang 路线用;vllm 路线在线量化不需要)
- src/vllm-omni-pr5910 @ b18eeff2 + stride patch(5090/6000a 均有;6000a 用 PYTHONPATH
  影子加载即可跑,生产 env 无污染)
- src/sglang-pr33681(60b9e51): TP2 加载期 OOM 路线已死(在线 FP8 先整载 BF16 shard 33G>31.3G),
  维持不可行结论
- 后续加速空间(doc/TP_5090_round2.md 尾章): Turbo LoRA merge→FP8(NFE6 预计 ~25s)、
  固定 shape warmup;当前先不加 TeaCache/compile(避免扩大 correctness 面)

## 增补(2026-08-09):Turbo LoRA profile
TP2 产线现有两档:`h3_switch_5090.sh vllm`(官方 base,NFE11≈47s,近无损)与
`vllm-turbo`(Turbo LoRA merged,NFE6 29.0s / NFE4 21.9s,见
doc/turbo_lora_results.md 第八节)。同端口 8091 互斥,start_vllm 已参数化 model root。
