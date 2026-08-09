# FINAL · 5090 双卡 serving(终态: 不可行, 2026-08-08 by 3195aa2f)

## 结论
vLLM-Omni 与 SGLang 在 2×5090 32G + **125G RAM** 上均不可行,瓶颈是**主机内存**:
- vLLM: TE-TP+offload 运行时 48G/卡 算不过;dist-layerwise 因 H3 无 mmap 支持把 RAM 打爆
- SGLang: TP2 双 rank 各暂存整份 TE(103G)+ 流式缓冲 → RAM 打爆
- 官方 2×5090 配方明确要求 384GB 级主机;无免密 sudo 无法加 swap
完整根因表: doc/speedup_5090_serving_results.md

## 影响小: 5090 ComfyUI turbo(35s)本就快于 6000a vLLM FP8(37s),serving 价值由 6000a 承担

## 恢复条件(满足其一即可重试, 环境全部就绪)
1. 主机内存升级到 384GB 级;2. sudo 加 ≥128G swap(理论可行未验证)
权重 145G/env×2/切换脚本/客户端已全部就位,h3_switch_5090.sh vllm|sglang 直接可用。

## 本主题沉淀的通用坑(其它机器也适用)
- rsync 源码 `--exclude .git` → setuptools_scm 版本回退 'dev' → 构建失败;
  修法: SETUPTOOLS_SCM_PRETEND_VERSION=<版本> pip install -e . --no-build-isolation
- vllm-omni 源码 editable 需先装 wheel 拿依赖(构建脚本 import 自身),或 --no-build-isolation + setuptools_scm
- 强 rm site-packages 包体会残留断损元数据 → `--omni requires vllm-omni` 假报;规范 uninstall+reinstall

## 更新(2026-08-09): TP2 攻坚续篇见 08_5090_tp2/
doc/TP2_5090.md 两条 PR 路线已尝试:基础设施跑通但 sm_120 FP8 数值全线不可用,暂停待上游。
