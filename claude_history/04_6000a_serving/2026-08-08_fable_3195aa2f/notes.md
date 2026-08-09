# 04 · 6000a vLLM/SGLang serving — 原始记录(2026-08-06~07)

## 目标
按 doc/speedup_6000ada.md(SGLang)与 doc/speedup_vllm.md(vLLM-Omni),6000a 上两条 4 卡 serving 产线,baseline 不动。

## 做了什么(时序)
1. 官方权重 `MiniMaxAI/MiniMax-H3` 只下 FL2VA/ 子树(144G,自含 transformer+TE+VAE)→
   `/home/isaac/Data/h3_weights/MiniMax-H3`;HF 深夜限速,写入 token 后恢复
2. HF cache 桥接:`~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/b3c7290…/`
   全部 symlink 到 local-dir,refs/main 写 sha → SGLang 直接命中缓存(实测有效)
3. env ×2:`h3_vllm_NV_py312` / `h3_sglang_NV_py312`
   - vLLM:`vllm==0.26.0`(pip, CUDA13 wheel)+ **vllm-omni 源码 editable(main)**
     (0.26.0 wheel 的 H3 缺 frame_indices 首尾帧支持 —— 8-05 才 merge)
   - SGLang:**源码 editable(main, 407a65d)** + `SGLANG_BUILD_RUST_EXTS=none`
     (0.5.16 wheel 无 multimodal_gen H3;Rust 扩展不装)
4. `scripts/h3_switch.sh {baseline|sglang|vllm [bf16|fp8]}`:互斥切换,ComfyUI 用 /free 卸模型保活,
   两后端 pid 文件启停,vllm 记 variant 文件按需重启
5. 客户端:`run_fl2va_vllm.py`(curl multipart /v1/videos/sync,自建 multipart 会被拒,必须 curl -F)
   `run_fl2va_sglang.py`(JSON /v1/videos 异步+轮询+/content 下载)
6. benchmark(ToS 864×480 12步 warmup+2timed;sglang 为 768P)

## 关键数据
| 配置 | warm | 峰值/卡 |
|---|---|---|
| vLLM TP4 BF16(V1) | 44.2s | ~46G |
| vLLM TP4 FP8(V2)= turbo | 37.4s | ~38G |
| SGLang TP4 FP8 @1376×768(A3) | 126.2s | ~42G |

## 踩坑全记录(重装照抄)
- vLLM serve flags(4卡): `--omni --num-gpus 4 --tensor-parallel-size 4 --usp 1 --ring 1
  --text-encoder-tp-size 4 --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling
  --diffusion-attention-backend FLASH_ATTN [--quantization fp8]`,MODEL=…/FL2VA 目录
- SGLang serve: `--model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va --num-gpus 4 --tp-size 4
  --ulysses-degree 1 --performance-mode speed --enable-torch-compile false --quantization fp8`
- SGLang 环境依赖(缺一崩一):RUNAI_STREAMER_MEMORY_LIMIT=8G(4rank 并发 CPU 暂存否则打爆 RAM);
  CUDA_HOME→nvidia/cu13(pip 装 nvidia-cuda-nvcc-cu13/runtime);PATH 含 env/bin(ninja);
  libcudart.so 无版本 symlink + LIBRARY_PATH(JIT 链接 -lcudart)
- SGLang API 拒绝 short_edge≠768(minimax_h3 校验);768P TP4 BF16 在 48G 卡 OOM(gpu2 剩 88MB)
- 首单含 regional compile/JIT,是 warmup,不计时

## 产出
- doc/speedup_6000a_results.md(全表+结论);gallery 6000a serving 卡片
- CLI profile:turbo(vllm fp8)/vllm(bf16)/sglang;E2E 三连验证过互斥切换
