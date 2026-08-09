# 09 · RunPods 4×5090 建站 + TP4/TP2×U2 提速(原始记录)

by 3195aa2f(同 session 先完成了 08 主题的 TP2 stride 修复与 07 主题的 5090 Turbo LoRA)。

## 时间线
1. 探机:4×5090 32.6G / 1007G RAM / /workspace 503G(664MB/s)+ / 1TB(2.3GB/s) /
   Ubuntu24.04 py3.12 / 无 conda 无 ComfyUI。拓扑: GPU0,1@NUMA0 GPU2,3@NUMA1,
   跨对 SYS,**P2P 全对 CNS**
2. 下载:HF_HUB_ENABLE_HF_TRANSFER 已弃用 → 退化 28MB/s;换 HF_XET_HIGH_PERFORMANCE
   峰值 247MB/s(全程波动,168G 约 1 小时)。LoRA 误拉整仓 22G,改 --include 单文件
3. merge:本地重做,溯源哈希与 6000a/5090 三机完全一致(L1 535 张位级 + L3 cos≥0.999999)
4. **阻塞**:vllm 0.26.0(PyPI)= torch cu130,pod 驱动 570 只到 CUDA 12.8 → 启动即崩。
   开 3-agent workflow 研究 + 自查同时进行,结论一致:装官方 `+cu129` wheel
   (cu128 wheel 根本不存在;CUDA 12.9 与 12.8 同大版本可 minor-compat)
5. 副作用坑:先建 env129 再 mv 成 env → venv 不可重定位,bin/* shebang 还指旧路径,
   报 "No such file or directory"(说的是解释器)。sed 修 shebang + pyvenv.cfg
6. 矩阵(每格先抽帧过质量关再计时,warmup1+timed3 取中位):
   bf16-tp4-r40 20.9 / bf16-tp4-r50 21.1 / bf16-tp2u2-r40 20.6 / fp8-tp4-r50 20.4 /
   **fp8-tp2u2-r50 18.5** / fp8-tp4-nfe4 17.4 / **fp8-tp2u2-nfe4 16.3**
7. tp2u2 初次全崩于 `_build_text_encoder_group` 的 `assert cpu_group is not None`
   —— 我把 --text-encoder-tp-size 传成了 TP(2),而世界大小是 4。改 ${H3_TE_TP:-4} 后通
8. E2E: Mac CLI --host runpods --profile vllm-fp8-turbo-tp2u2 → 启动 196s + 推理 22.7s

## 教训
- **纸面显存估算会误判**:BF16 TP2 按 33G/卡 判"放不下",实测 r40 只用 28.4G 能跑
  (DLO 常驻数才是决定项)。同理"resident 冲到 50"没收益 —— 先测再信 note。
- **拓扑决定并行策略**:无 P2P 时 Ulysses > TP。先跑 `topo -p2p` 再排实验顺序,
  比按 note 顺序盲跑省了至少一轮。
- pgrep -f "pip install" 会匹配到我自己那些含该字符串的 watcher 循环 → post_setup
  永远等下去。**bracket 模式这条铁律不只适用于 kill,也适用于 wait**。
