# 6000a 三产线 benchmark 结果(2026-08-07)

依据 [speedup_6000ada.md](speedup_6000ada.md)(SGLang 路线)与 [speedup_vllm.md](speedup_vllm.md)(vLLM-Omni 路线)执行。
硬件:4× RTX 6000 Ada 48GB,PCIe P2P 全互通,单 NUMA,376GB RAM。
素材:ToS t=147s 首尾帧,12 步(serving)/30 步(baseline),seed 0/1/2,warm 取 2 次计时中位。

## 总表

| 产线 | 后端 / 配置 | 画布 | warm 延迟 | 峰值 VRAM/卡 | 精度 |
|---|---|---|---|---|---|
| baseline | ComfyUI, pruned INT8 单卡计算(30 步) | 832×480 | ~100s(热)/ 210s(冷) | 48.2G(单卡) | INT8 |
| bf16 oracle | ComfyUI DisTorch, 4 卡 BF16(30 步) | 832×480 | 400s | 47.6/7.4/47.3/0.7 | BF16 |
| **turbo** | **vLLM-Omni TP4 + 在线 FP8(12 步)** | **864×480** | **37.4s** | ~38G ×4 | FP8(W8,关键层 FP32) |
| vllm | vLLM-Omni TP4 BF16(12 步) | 864×480 | 44.2s | ~46G ×4 | BF16 无损 |
| sglang | SGLang TP4 + 在线 FP8(12 步) | **1376×768** | 126.2s | ~42G ×4 | FP8 |

- **6000a 加速产线 = vLLM FP8:warm 37s**,vs baseline(100s 热)**2.7×**,vs 同素材 5090 turbo(35s)几乎同级但**无 TeaCache 有损缓存**(FP8 量化是唯一近似)
- vLLM BF16(44s)是无损多卡路线:比 baseline 快 2.3× 且零量化,FP8 与 BF16 同 seed 输出肉眼无差异
- SGLang 126s 是 **768P 大画布**(像素量 4.3×),不能与 864×480 直接比;它是"高清路线"

## 关键工程结论

1. **SGLang 公共 API 锁死 `short_edge=768`**(`target.short_edge must be 768 for minimax_h3`)——
   小画布低延迟路线在当前 commit(407a65d)走不通,只能做 768P 高清档
2. **768P + TP4 BF16 在 48GB 卡上 OOM**(H100 实测该负载 49.8GB > 47.3GB 可用)→ SGLang 产线固化 `--quantization fp8`(在线 FP8 在 Ada 上实测可用)
3. vLLM-Omni 0.26.0 PyPI wheel 的 H3 缺首尾帧 `frame_indices` 支持(8-05 才 merge)→ **必须源码装 main**(vllm==0.26.0 base + vllm-omni editable)
4. sglang 0.5.16 wheel 无 multimodal_gen H3 → **源码装 main**(`SGLANG_BUILD_RUST_EXTS=none`)
5. sglang bring-up 三坑:4-rank 并发 CPU 暂存 OOM(`RUNAI_STREAMER_MEMORY_LIMIT=8G` 解决)、
   JIT kernel 需 nvcc+ninja+libcudart(pip wheel + symlink + `LIBRARY_PATH` 解决)
6. HF cache 桥接(local-dir → snapshot symlink)有效,SGLang 直接命中缓存,避免了 144GB 重复下载
7. 每日 ~300GB 出口流量会触发 HF 匿名限速;写入 HF token 后恢复

## 与 doc 预估对照

- speedup_vllm.md 预估"1.5-2.5×(已驻留时)" → 实测 2.3×(BF16)~2.7×(FP8) ✓
- speedup_6000ada.md 预估"TP2+U2 FP8 25-45s 候选区间" → TP4 FP8 实测 37s 落在区间 ✓(TP2+U2 因 48GB 放不下 BF16 未测,FP8 下 TP4 已达标)
- FP8 vs BF16 = 1.18×(vLLM,Ada)—— 低于 B300 的 1.35×,但方向一致

## 产线拓扑(最终)

```text
popos-6000a(三条 serving 产线互斥占 4 卡,h3_switch.sh 管理;ComfyUI 进程常驻仅卸模型)
├── ComfyUI :8288   baseline(INT8 单卡)/ bf16 oracle(DisTorch)
├── vLLM    :8091   turbo(FP8, 864×480, 37s)/ vllm(BF16, 44s) ← 变体切换需重启(分钟级)
└── SGLang  :30010  sglang(FP8, 1376×768 高清, 126s)

Mac CLI: scripts/h3_generate.py --host 6000a --profile baseline|turbo|vllm|sglang|bf16
```

## 未尽事项

- Cache-DiT(SGLang env vars)未测:SGLang 已因 API 锁 768 不承担低延迟角色,留待需要 768P 提速时再调
- vLLM TP2+USP2 拓扑未测(TP4 已达标;若未来上更大画布可补)
- SGLang 每次请求 wall 恒定 126.2s(两次计时完全一致)—— 调度器可能有固定节拍,深究价值低
