# FINAL · 6000a serving(整理版, 2026-08-08 by 3195aa2f)

## 产线(权威)
| CLI profile | 后端/精度 | 画布 | warm | 说明 |
|---|---|---|---|---|
| turbo | vLLM TP4 FP8 | 864×480/12步 | **37s** | 加速产线(唯一近似=FP8) |
| vllm | vLLM TP4 BF16 | 864×480/12步 | 44s | 无损多卡 |
| sglang | SGLang TP4 FP8 | 1376×768/12步 | 126s | 高清档(API 锁 768 短边) |
切换: `~/data/dropbox/CV/h3/scripts/h3_switch.sh {baseline|sglang|vllm [bf16|fp8]}`(互斥,冷启动分钟级)

## 硬事实(改配置前必读)
- 768P+TP4 BF16 在 48G 卡 OOM → sglang 必须 FP8
- 权重: HF 官方 FL2VA/ 145G 在第二块 NVMe;SGLang 走 HF cache symlink 桥接(sha b3c7290)
- 两栈都是**源码安装**(发行版缺 H3 关键支持);SGLang 运行期依赖 nvcc/ninja/libcudart(修法见 session notes)
- 客户端: run_fl2va_vllm.py(必须 curl -F multipart)/ run_fl2va_sglang.py
- 未测: Cache-DiT、vLLM TP2+USP2(TP4 已达标)
