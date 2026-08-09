# 主题: RunPods 4×5090 云机 TP4 H3

范围: 按 doc/TP4_5090.md 在云端 4×5090(32.6G×4、1007G RAM、/workspace 503G 持久卷)
部署 H3 Turbo TP4 serving,先 BF16 主线再 FP8 提速。

## 机器特性(与两台本地机不同,务必先读)

- **非 dropbox 布局**: 工作根 `/workspace/h3`(持久卷),`/` 是临时盘(pod 重建即失);
  python 用 venv `/workspace/h3/env/bin/python`,**没有 conda、没有 ComfyUI**
- **pod 可能被销毁**: 一键恢复脚本 `scripts/setup_runpods.sh`(Mac 项目内),
  重建后把 Mac 项目 scripts/ 里的 h3_switch_runpods.sh / probe.sh /
  run_fl2va_vllm.py / merge_turbo_lora.py 推上去即可
- **拓扑(实测 2026-08-09)**: GPU0/1 在 NUMA0,GPU2/3 在 NUMA1,跨对为 SYS;
  `nvidia-smi topo -p2p r/w` **全对 CNS = 无 P2P**,所有 NCCL 走 host 中转。
  → TP2×U2 分组必须 (0,1)(2,3);TP4 的每层 all-reduce 会跨 NUMA,是主要性能风险
- **下载坑**: `HF_HUB_ENABLE_HF_TRANSFER=1` 已被 hub 弃用且会退化到 ~28MB/s;
  必须用 `HF_XET_HIGH_PERFORMANCE=1`(实测 247MB/s)

## 现状指针

- 权威结论与 benchmark: 本主题 `FINAL.md` + `doc/speedup_runpods_tp4_results.md`
- Mac CLI: `h3_generate.py --host runpods --profile turbo-lora [--nfe N]`
  (该 host 无 ComfyUI,只接受 serving profile;`--host both` 不含 runpods)
- 服务切换: `/workspace/h3/scripts/h3_switch_runpods.sh {bf16|fp8|tp2u2|stop} [resident]`
