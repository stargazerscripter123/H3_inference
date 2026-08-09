# 06 · 基建 — 原始记录(2026-08-05~08)

本 session 建立了 FINAL.md 中的全部基建。演进要点:
- h3_generate.py: 冒烟版 → +--quality → +--profile(5 档)→ serving 分发(switch+health+远端client+回传)
  → switch_script 参数化(6000a/5090 各自切换脚本)
- runner run_fl2va.py: 基础版 → +prompt-file → +MultiGPU/DisTorch2 参数 → +TeaCache(防缓存抖动)
- 切换脚本: h3_switch.sh(6000a)/ h3_switch_5090.sh(5090, 含 ComfyUI worker 启停)
- HF token 事件: 匿名日流量 ~300G 触发限速(3MB/s→认证后恢复);token 在 credentials/HF.md
- 内容红线事件: 用户提供 `data/` 下一份受限素材(未授权的真实可识别人物影像)要求生成,
  已拒绝(生成与调通命令均拒),用户接受;之后全部用 ToS(CC-BY)素材
