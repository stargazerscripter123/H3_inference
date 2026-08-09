# 02 · 5090 TeaCache 加速产线(2026-08-05~07)

## 目标
按 `doc/speedup_5090.md`:不动 baseline,5090 上出 warm ~30-35s 的加速产线 + CLI 可选。

## 做了什么
- 独立 turbo worker:GPU0 :8189,`--preview-method none --async-offload 2 --reserve-vram 1.5`
  (`scripts/launch_comfy_5090_turbo.sh`;baseline GPU1:8188 未动)
- TeaCache 节点 `Icyoung/ComfyUI-MiniMaxH3-TeaCache` pin `4cbb50d`
- runner 加 `--teacache-thresh/start/end`(total_steps 自动=steps;
  **rel_l1_thresh 每次提交加 1e-9 级抖动破 ComfyUI 节点缓存** —— 不抖动第二次提交缓存静默失效)
- benchmark 协议:quality run(seed0 留档)+ timed×2(变 seed+prompt 后缀强制全链路重执行)

## 关键数据(864×480 · 124帧,详表 `doc/speedup_5090_results.md`)
| 配置 | warm | s/it |
|---|---|---|
| 20 步无缓存(B1) | 77.5s | 3.28 |
| 12 步无缓存(B2) | ~45s | 3.29 |
| **12 步+TC 0.10(B5=生产)** | **35s** | 1.91 |
| 12 步 FP8_scaled(B3) | 60s | 3.96 ← FP8 比 INT8 慢 20%,弃 |
| 12 步 FP8+TC(B3tc) | 40s | 2.34 |

## 踩坑与修法
- MultiGPU 分卡实验失败:两张消费级 5090 无 P2P,ComfyUI-MultiGPU 的 CPU 中转与
  comfy-kitchen dlpack 不兼容(illegal memory access 崩服务端),纯 fp16 VAE 也崩 → 弃,单卡稳定配置
- ComfyUI-MultiGPU 的 p2p 检测 `CDLL("libcudart.so")` 需 env 里做无版本 symlink(修过,虽最终未用)

## 产出
- CLI `--profile turbo`(5090 语义 = TeaCache 产线)
- benchmark 产物本地:`outputs/bench_5090_turbo/`;质量对比在 gallery

## 未竟
- 更低延迟(<30s)需 few-step LoRA(训练路线)
