# 07 · 5090 侧 Turbo LoRA 落地(继承自 2026-08-08_opus_1ed2dca3 的未竟项)

by 3195aa2f(同 session 亦完成 08 主题 TP2 stride 修复,这是前置依赖)。

1. 素材:merge/check/test 脚本 + LoRA(780M)经 Mac 中转 6000a→5090
   (两机 ~/data/dropbox 目录**不互相同步**,勿假设)
2. 本地 merge:~9min(CPU),L1/L3 过;与 6000a manifest 溯源哈希
   (base index/lora/script sha256、259/535 计数)完全一致 → 位级等价,
   免 62G 传输(--comfy-single /nonexistent 跳过 L2,由哈希对比替代)
3. 集成:h3_switch_5090.sh 的 start_vllm 参数化(model root + label),
   新增 vllm-turbo target(.complete 契约校验);run_fl2va_vllm.py 升级为
   6000a 的 NFE 版(向后兼容 --steps)
4. 数据:NFE6 warm 29.0s(×3 离散 0.1s)、NFE4 21.9s;峰值 23.95G/卡;
   质量帧干净;Mac CLI E2E 过(--profile turbo-lora)
5. 现场:comfy 双 worker 已回切;vllm/vllm-turbo 同端口互斥(既有约定)
