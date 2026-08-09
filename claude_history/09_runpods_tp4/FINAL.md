# FINAL · RunPods 4×5090 TP4/TP2×U2 H3(状态: **已上线**, 2026-08-09 by 3195aa2f)

## 一句话
4×5090 云机跑通 H3 Turbo serving,**最优 = FP8 + TP2×Ulysses2 + DLO resident50**:
**NFE6 18.5s / NFE4 16.3s**(864×480×124 帧),比 6000a 同 profile 快 1.27×、
比双 5090 快 1.57×。全表与踩坑见 `doc/speedup_runpods_tp4_results.md`。

## 对 doc/TP4_5090.md 的三条实测修正
1. **TP2×U2 快于 TP4**(18.5 vs 20.4s)。note 让"先 TP4"是基于有 P2P 的机器;
   这台 `nvidia-smi topo -p2p r/w` 全 CNS,NCCL 走 host 中转,逐层 all-reduce 是主成本,
   Ulysses 只做 qkv all-to-all 因而占优。
2. **resident 不必冲到 50**:TP4 BF16 r40 20.9s vs r50 21.1s(噪声内),1007G RAM 下
   剩余块的 PCIe 流式被计算掩盖;r40 还省 3G/卡。
3. **BF16 TP2×U2 装得下**(28.4G/卡 @r40),按"33G/卡 > 32G"纸面估算会误判为不可行。

## 两个必须记住的部署事实
- **驱动 570 = CUDA 12.8 → 必须用 vLLM 的 `+cu129` wheel**(PyPI 默认是 cu130,
  直接 `driver too old (found version 12080)`)。cu128 wheel **从来不存在**,
  官方文档那句是过期文本。命令见结果文档第二节。
- **PR#5910 新 head `1a9b9c2c` 仍带 resident-repoint 丢 stride 的 bug**,
  必须打 `pr5910_resident_stride_fix.patch`(根因见 `../08_5090_tp2/FINAL.md`),
  否则 FP8 路径出纯噪声。

## 现场
```text
/workspace/h3/                     持久卷 503G(用掉约 300G)
├── base/MiniMax-H3/FL2VA          官方 BF16(168G)
├── merged/MiniMax-H3-Turbo-v4s600ema   Turbo merged(溯源哈希与 6000a/5090 逐位一致)
├── env/                           venv: vllm 0.26.0+cu129 / torch 2.11.0+cu129
├── env_cu130_unusable/            旧 venv,确认无用后可删(约 15G)
├── src-vllm-omni/                 PR#5910 @1a9b9c2c + stride patch(editable)
└── scripts/                       h3_switch_runpods.sh bench_matrix.sh run_matrix*.sh …
```
入口:`h3_switch_runpods.sh <original|turbo> <bf16|fp8> <tp4|tp2u2> [resident]`;
Mac CLI `--host runpods --profile vllm-fp8-turbo-tp2u2 [--nfe 4]`(E2E 已验证)。
pod 重建:`scripts/setup_runpods.sh`(Mac 项目内)+ 把 scripts/ 推上去。

## 未竟
- 768P 原生档未测(note 预估 30-55s);当前只验证了 864×480。
- `--text-encoder-tp-size` 对 tp2u2 必须=4(=世界大小),传 2 会在
  `_build_text_encoder_group` 断言 `cpu_group is not None` 崩 —— 已在 switch
  脚本里固定为 `${H3_TE_TP:-4}`,但上游这个报错很不友好,值得给 PR 提一句。
- Cache-DiT / torch.compile 未开(按 note 建议先不扩大 correctness 面)。

## 增补(2026-08-09 夜):BF16 基座格失败 = 引擎 30 秒超时,不是能力上限
用户的 eval sweep 在 runpods 上 `vllm-bf16-original-tp4` 与 `-tp2u2` 两格
返回 500(103 字节错误 JSON)。根因是 vllm-omni 写死的
`_ASYNC_OUTPUT_TIMEOUT = 30.0`(`diffusion_engine.py:58`),不是显存/模型/配置问题
—— 同一 pod 上 FP8 基座 NFE11 与全部 Turbo NFE6 格都正常。详见
`../06_infra/FINAL.md` 同日增补。**结论文字里不要写成"4×5090 跑不了 BF16 基座"。**

## 增补(2026-08-09 夜, session 1ed2dca3): switch 无条件重启导致速度数字虚高

`h3_switch_runpods.sh` 原本缺变体追踪,**每次调用都重启服务**。因此这台机器上
一切"调一次 CLI 测一次"的做法测到的都是**重启后的冷态首推**,比 warm 慢约 20%:

| 拓扑 | 冷启首推 | warm(复用) |
|---|---|---|
| FP8 Turbo TP2×U2 NFE6 | 22.1 / 22.3 / 22.7s | **18.5 / 18.8s** |
| FP8 Turbo TP4 NFE6 | 24.2s | **20.3s** |

补齐铁律后启动耗时 214s → **5s**,归档的 18.5s 当场复现;
"TP2×U2 比 TP4 快"的结论也在同一组对照里再次成立(18.5 vs 20.3s)。

顺带堵上安全缺口:没有 `run/vllm.variant` 时,`:8091` 被 base 与 merged-Turbo 两个
checkpoint 复用却无法判断在服的是哪个 —— 与 5090 上修掉的"静默服错 checkpoint"
同源。现在变体不匹配会打印
`vllm variant mismatch (turbo-fp8-tp2u2-r50 -> turbo-fp8-tp4-r50), restarting` 并重启。
