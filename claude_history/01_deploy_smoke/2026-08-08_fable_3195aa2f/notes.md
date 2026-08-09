# 01 · 双机 ComfyUI 部署 + FL2VA 冒烟(2026-08-05)

## 目标
按 `doc/high_level.md` Phase 2:两台机器 `~/data/dropbox/CV/h3` 部署 ComfyUI + H3 开放权重,跑通首条 FL2VA。

## 做了什么
- conda env `h3_comfy_NV_py312`(两机,python3.12 + torch 2.11.0+cu130)
- ComfyUI pin `57500fc5bc92566a63f2046824f522cd55c335ca`(H3 支持 merge commit,v0.29.0;
  requirements 内 comfy-kitchen==0.2.26 / comfy-aimdo==0.4.11 是量化算子,必装)
- 权重(Comfy-Org/MiniMax-H3 单文件版):pruned INT8 DiT 21G + TE(6000a INT8 27.1G / 5090 NVFP4 15.7G)
  + video VAE fp16 5.2G + audio VAE fp32 0.6G(fp32 硬性要求,BF16 音量 -20dB)
- 测试素材:Tears of Steel(CC-BY)t=147s 连续镜头抽首尾帧(间隔恰 123 帧),832×480 cover-crop
- headless 提交:`scripts/run_fl2va.py`(API 格式 graph:UNETLoader+CLIPLoader(type=minimax)
  +MiniMaxH3ImageToVideo+BasicScheduler(simple)+res_multistep+BasicGuider+SamplerCustomAdvanced
  +VAEDecode/VAEDecodeAudio+CreateVideo(24fps)+SaveVideo)

## 关键数据
| 机器 | 权重组合 | 总耗时 | s/step | 峰值 VRAM |
|---|---|---|---|---|
| 5090 GPU1 | pruned INT8 + NVFP4 TE | 110s | 3.03 | 31.5/32.6G(贴顶) |
| 6000a GPU3 | pruned INT8 + INT8 TE | 210s(冷) | ~3.3 | 48.2/49.1G |
输出契约均验证:832×480·124帧·5.167s·AAC 立体声 32kHz;首尾帧对齐、身份保持 ✓
同 seed 下两机高度相似 → NVFP4 TE 无可见质量损失。

## 踩坑与修法
- 帧数必须 `n%17==5`(124=5.17s);画布 32 倍数
- 首帧 stretch / 尾帧 center-crop 不对称 → 预处理统一画布(CLI 已内置)
- HF 匿名下载当日 ~300G 后被限速 → 认证 token 解决(见 06_infra)
- ssh 远程命令含 pkill/pgrep 模式会自杀会话 → 一律方括号 `[m]ain.py` 防自匹配

## 未竟
- 无(任务闭环;benchmark 矩阵/API 对比属 doc Phase 0/1,未开始)
