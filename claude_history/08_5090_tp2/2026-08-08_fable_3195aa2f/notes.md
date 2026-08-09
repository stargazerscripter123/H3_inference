# 08 · 5090 TP2 — 原始记录(2026-08-08 深夜~09 凌晨)

时间线:
1. ModelOpt(feizhai123 Mixed-FP8)路线启动后被 doc/TP2_5090.md 取代(下载 5G/85G 中止保留)
2. 资产: Qwen官方 TE FP8(HF 限速卡死 13G → ModelScope 换源 4MB/s 完成 34G);两 PR clone+editable
3. sglang PR33681: TP2 加载期 OOM(在线 FP8 先整载 BF16 shard 33G>31.3G)——路线死
4. vllm PR5910: resident=20 跑通 47.1s(峰值15G/卡)、resident=50 46.7s(24G/卡,流式非瓶颈)
   —— 但质量检查发现输出全噪声(T2/T3 均是)
5. 排查(单变量): CUDNN→TORCH_SDPA 噪声不变(attention 排除);FLASH_ATTN=FA3 Hopper 无 sm_120 崩;
   cutlass 崩(scaled_mm_entry.cu:209);Humming stride 崩;Marlin 缺属性崩(PR 集成缺口)
6. 收档: comfy 产线恢复 ✓,CLI 标注不可用 ✓,文档/证据链入 doc
教训: benchmark 数字必须配质量检查后才算数(47s 白高兴一场);
      新 PR 的"能跑"与"数值对"是两回事,首帧抽查应该在第一次出片就做(实际做了——T3 warmup 后立刻查的)

## round-2(2026-08-09,同 session 续)— 修复完成

按 doc/TP_5090_round2.md 阶梯执行,root cause 定位并修复:
1. P0 head b18eeff2 复测仍噪声;P1 BATCH_INVARIANT(BF16-dequant)仍同花纹噪声 → kernel 排除
   (中途发现 note 引用的 fp8.py::Fp8LinearMethod 不在 H3 链路上——哨兵埋点证实全走
   online/fp8.py::Fp8PerTensorOnlineLinearMethod,BI 分支在 online 模块同样存在且生效)
2. TP1 对照两机都 OOM 弃(TE 构造期整载 51.5G);H3 恰 50 层 → round-1 的 resident=50
   数据即"全常驻仍噪声" → 运行期 streaming 排除
3. 6000a 交叉:ChannelWise 被 sm_89 cuBLASLt 拒;**Cutlass 报与 5090 完全相同的
   scaled_mm_entry.cu:209** → 跨硬件同症,sm_120 归因崩塌
4. 逐调用 dump:209 个 linear OK(转置视图 stride (1,K)),第 210 个
   blocks.0.adaln_proj.linear 是 contiguous 行主序 → 常驻块布局损坏实锤
5. 源码对读:PinnedResidentLayerGroup.load() 用 .view(shape) 重建,而 streamed 的
   prefetch_layer 用 as_strided(...stride=meta["stride"]) —— 一行差异就是全部
6. 修复(as_strided 化)→ 6000a Cutlass 干净、5090 ChannelWise 干净、E2E 干净

产线化: CUDNN_ATTN(TORCH_SDPA 是排查遗留,慢 18s)+ resident=50,warm 46.9/46.8/46.6s,
峰值 23.95G/卡;CLI 5090 vllm profile 恢复;patch 存 scripts/pr5910_resident_stride_fix.patch
(两机已打;此 patch 可贡献 upstream PR#5910)。
现场: 5090 comfy 双 worker 恢复 ✓;6000a turbo-fp8 vllm + comfy :8288 恢复 ✓;
      两机 vllm site-packages 的诊断埋点已全部还原 ✓

教训: (1) "崩溃 kernel 的报错行号"≠root cause——cutlass :209 在两种硬件上是同一契约检查,
round-1 却按"sm_120 缺 kernel"归档;交叉验证前不要写死硬件归因。
(2) 逐调用 dump(80 行埋点)比五次 kernel 轮换的信息量都大——早做省 4 小时。
