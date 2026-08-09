# FINAL · 5090 turbo(TeaCache)产线(整理版, 2026-08-08 by 3195aa2f)

## 生产配置(权威)
864×480 · 12 步 · pruned INT8 DiT + NVFP4 TE · TeaCache(pin 4cbb50d)thresh 0.10 start 2 end -2 ·
worker GPU0 :8189(`launch_comfy_5090_turbo.sh`: --preview-method none --async-offload 2 --reserve-vram 1.5)

## 数据(warm 中位;全表 doc/speedup_5090_results.md)
35s(=1.91s/it 有效)vs baseline 100s ≈ 2.9×;冷启动首单 +1min。
FP8_scaled A/B 落败(3.96 vs 3.29 s/it,慢 20%)→ 保持 INT8。

## 必须知道
- TeaCache 有损(跳 ~40% forward);runner 对 thresh 做 1e-9 抖动破 ComfyUI 节点缓存,
  没有抖动时同参数第二单会静默退化为无缓存速度 —— 改 runner 时保留该逻辑
- 消费级 5090 无 P2P:ComfyUI-MultiGPU 跨卡在 comfy-kitchen 环境必崩(已验证),别再试
