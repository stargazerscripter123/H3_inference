# 5090 turbo 产线 benchmark 结果(2026-08-05)

依据 [speedup_5090.md](speedup_5090.md) §九矩阵的核心子集执行。

## 环境

- popos-5090 GPU0(独立 turbo worker,port 8189),baseline 产线(GPU1:8188)未动
- worker 启动参数:`--preview-method none --async-offload 2 --reserve-vram 1.5`
- ComfyUI `57500fc` + TeaCache 节点 `Icyoung/ComfyUI-MiniMaxH3-TeaCache` @ `4cbb50d`
- 素材:Tears of Steel t=147s 首尾帧,864×480 原生裁切,124 帧,seed 0
- 计时协议:每配置 1 次 quality run(seed 0,留档)+ 2 次 timed run(变 seed+prompt 后缀,强制全链路重执行 = 真实 warm latency,含 TE 编码/denoise/VAE decode/mux)

## 结果

| Run | 配置 | warm wall(中位) | s/it | 峰值 VRAM |
|---|---|---|---|---|
| 参照 | baseline 产线(832×480, 30 步, 无缓存) | ~95-110s | 3.03 | 31.5G |
| B1 | 864×480, 20 步, INT8, 无缓存 | 77.5s | 3.28 | 30.8G |
| B2 | 864×480, 12 步, INT8, 无缓存 | ~45s | 3.29 | 30.9G |
| B4 | 864×480, 12 步, INT8, TeaCache 0.08 | 37.6s | 2.16 | 30.8G |
| **B5** | **864×480, 12 步, INT8, TeaCache 0.10** | **35.0s** | **1.91** | 30.8G |

- B5 有效 s/it 1.91(vs 裸跑 3.29)→ TeaCache 实际跳过约 40% forward,与作者 benchmark 一致
- **B5 = turbo 生产配置,warm latency 35s,达到 doc 的 30-35s 目标区间**(vs baseline ~3× 提速)
- 延迟构成(B5):TE 编码+VAE encode ~4s,denoise 12×1.91≈23s,VAE decode+mux ~8s

## B3 补测结果(FP8 A/B,2026-08-07)

| Run | 配置 | warm wall | s/it |
|---|---|---|---|
| B3 | 864×480, 12 步, **FP8_scaled**, 无缓存 | 60s | 3.96 |
| B3tc | 同上 + TeaCache 0.10 | 40s | 2.34 |

**结论:INT8 convrot 明显胜出**(3.29 vs 3.96 s/it,快 ~20%;加缓存后 35s vs 40s)。
comfy-kitchen 的 FP8_scaled 路径在 5090 上未命中最优 kernel(印证 doc §三"不要假设 FP8 一定比 INT8 快")。**turbo 生产配置维持 INT8 不变**。

## 未完成 / 后续
- B6-B8(736×416 降档)未跑:B5 已达标,无需降档
- 再往下压(<30s)按 doc §十三:few-step distillation LoRA,属训练路线
- 质量注意:TeaCache 为有损加速(跳过的 forward 直接复用上次输出);0.10 档在测试素材上肉眼未见明显退化,生产接入前建议对代表性内容抽查 endpoint 保真/音频连续性

## TeaCache 使用注意(实现层)

- `total_steps` 必须等于 sampler steps(runner 已自动同步)
- 节点 state 不跨 run 重置 + ComfyUI 节点缓存 → 相同参数二次提交时缓存静默失效;runner 对 `rel_l1_thresh` 做 1e-9 级 per-run 抖动破缓存(已内置)
- 仅在 CFG-free(BasicGuider)下步数计数正确 —— H3 本来就无 CFG,天然安全

## 产线拓扑(最终)

```text
popos-5090
├── GPU1 :8188  baseline(832×480 / 30 步 / 无缓存)   ← 质量优先
└── GPU0 :8189  turbo   (864×480 / 12 步 / TeaCache 0.10) ← 延迟优先, warm 35s

Mac CLI: scripts/h3_generate.py --profile baseline|turbo|bf16
```
