# FINAL · 双机部署与冒烟(整理版, 2026-08-08 by 3195aa2f)

## 现行部署(权威)
- env: `h3_comfy_NV_py312`(两机)= py3.12 + torch 2.11.0+cu130
- ComfyUI pin `57500fc5bc9`(H3 merge, v0.29.0);comfy-kitchen 0.2.26 / comfy-aimdo 0.4.11 必装
- 权重(Comfy 单文件): pruned INT8 DiT 21G;TE 6000a=INT8 27.1G / 5090=NVFP4 15.7G;
  video VAE fp16;**audio VAE 必须 fp32**(BF16 音量 -20dB)
- 提交路径: `scripts/run_fl2va.py`(headless API graph, BasicGuider 无 CFG, res_multistep/simple)

## 已验证事实
- 冒烟 PASS(832×480/30步/seed0): 5090 110s@3.03s/it 峰值31.5G;6000a 210s冷@3.3s/it 峰值48.2G
- NVFP4 TE vs INT8 TE 同 seed 无可见差异
- 输出契约: 124帧/5.167s/24fps/AAC 立体声 32kHz;帧数 n%17==5;画布 32 倍数
- 首帧 stretch/尾帧 cover-crop 不对称 → 必须预处理统一画布(CLI 已内置)
