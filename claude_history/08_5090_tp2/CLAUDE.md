# 主题: 5090 双卡 Tensor Parallel(TP2)H3

范围: 按 doc/TP2_5090.md + doc/TP_5090_round2.md 在 2×5090 跑 TP2 serving。
状态: **已修复上线(2026-08-09)** —— round-1 的噪声 root cause 是 PR5910
`PinnedResidentLayerGroup.load()` 丢 stride 的一行 bug(与 sm_120 无关),
补丁 `scripts/pr5910_resident_stride_fix.patch` 两机已打;产线 warm ~47s。
配置/复原/历史修订全在 FINAL.md;CLI `--host 5090 --profile vllm` 可用。
注意: 若升级/reset src/vllm-omni-pr5910,必须重打 patch 并帧检查后再上线。
