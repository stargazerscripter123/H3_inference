# 05 · 5090 双卡 serving — 原始记录(进行中, 断线恢复点)

## 目标
2×5090(32G, host RAM 125G)跑通 vLLM 与 SGLang,选双卡装得下的尺寸(结论:必须 FP8,BF16 TP2=33G/卡放不下)。

## 计划配置(已批准)
- vLLM: TP2 + 在线 FP8 + `--enable-cpu-offload`(TE 换出)+ text-encoder-tp-size 2 + VAE tile ×2,864×480
- SGLang: TP2 + FP8 + `--performance-mode memory --layerwise-offload-components text_encoder,vae`
  (DiT FP8 常驻 16.6G/卡,TE/VAE 从 RAM 流式),768P(API 锁)
- **serving 时必须 kill 两个 ComfyUI worker**(它们的 CPU 缓存占 ~90G RAM)

## 已完成
- [x] `h3_switch_5090.sh` 部署(comfy|vllm|sglang 互斥,含 worker 启停)
- [x] 两个 serving 客户端脚本部署到 5090 scripts/
- [x] CLI `h3_generate.py`:switch_script/switch_baseline 参数化 + 5090 serving map(vllm/sglang)
- [x] 权重 rsync 6000a→5090 已启动(setsid 分离,1Gbps ~25min)
- [x] env ×2 创建 + 两栈安装脚本已启动(自等待 src rsync)

## 恢复点(断线后先做这些判定)
```bash
# 权重到位?(应 145G;rsync 日志在 6000a)
ssh popos-6000a 'grep WEIGHTS_RSYNC_EXIT ~/data/dropbox/CV/h3/logs/rsync_to_5090.log'
ssh popos-5090 'du -sh ~/data/dropbox/CV/h3/models_official/MiniMax-H3'
# 安装完成?(应见 VLLM_5090_DONE 与 SGLANG_5090_DONE)
ssh popos-5090 'grep -E "DONE|error" ~/data/dropbox/CV/h3/logs/install_backends.log'
```

## 待办(按序)
1. 5090 HF cache 桥接(同 6000a 做法,sha b3c7290e66afdf293bef3b9077b7a266ef421f34,
   指向 ~/data/dropbox/CV/h3/models_official/MiniMax-H3)
2. `h3_switch_5090.sh vllm` → 盯 logs/vllm_server.log(**sm_120 FP8 + cpu-offload 兼容性是首要验证点**)
   → run_fl2va_vllm.py ToS 864×480 12步 warmup+2timed
3. `h3_switch_5090.sh sglang` → 768P 12步(盯 OOM;后备:官方 layerwise 配方 + RUNAI 限流,RAM 很紧)
4. E2E CLI 四连: vllm→sglang→turbo→baseline(验证 worker 重启回切)
5. doc/speedup_5090_serving_results.md + gallery + FINAL.md 终态

## 风险后备(原 plan)
- vLLM fp8×cpu-offload 若不兼容 → 无好后备(BF16 TP2 放不下),记录不可行
- SGLang OOM → layerwise 官方配方(需 ~123G RAM,只有 ~115G 可用,RUNAI 限流硬试)

## 终态执行记录(2026-08-08 下午)
- 权重 rsync ✅ 145G(LAN);env 安装经三轮修复 ✅(nvcc wheel --only-binary;
  vllm-omni: wheel先行→rm冲突→SETUPTOOLS_SCM_PRETEND_VERSION+--no-build-isolation 终装成功)
- vLLM 三次 bring-up 全败、SGLang 一次败(根因见 FINAL.md / doc/speedup_5090_serving_results.md)
- 判定不可行(host RAM 结构性不足),恢复 comfy 双 worker ✅,turbo E2E 35s 完好 ✅
- CLI: 5090 vllm/sglang profile 移除并注释原因
