# MiniMax H3 生产化 · Roadmap 与待办

最后更新:2026-08-09 夜(by session 3195aa2f)。
**这份文件是给"下次坐下来的人"看的:先读这里,再按指针进 `claude_history/`。**
权威结论永远以各主题 `FINAL.md` 为准,这里只做导航与排期。

---

## 一、现在到哪儿了(一句话)

三台机器都跑通了 H3 FL2VA serving,**最快 18.5s / NFE6**(RunPods 4×5090,
FP8 + TP2×Ulysses2)。质量档、速度档、高清档都有产线;剩下的主要是
**质量定档(需要你本人参与)** 和 **几处工程收尾**。

### 产线速查(864×480×124 帧,warm 中位)

| 机器 | profile | NFE | 耗时 | 用途 |
|---|---|---:|---:|---|
| **RunPods 4×5090** | `vllm-fp8-turbo-tp2u2` | 6 | **18.5s** | 最快;当前速度冠军 |
| RunPods | 同上 | 4 | **16.3s** | 极速档(大运动内容慎用) |
| RunPods | `vllm-fp8-turbo-tp4` | 6 | 20.4s | 备选拓扑 |
| 6000a 4×RTX6000Ada | `vllm-fp8-turbo-tp4` | 6 | 23.5s | 本地最快 |
| 6000a | `sglang-fp8-turbo-tp4` | 6 | 72.1s | **1376×768 高清档** |
| popos 2×5090 | `vllm-fp8-turbo-tp2` | 6 | 29.0s | 本地备份 |
| popos 2×5090 | `comfy-int8-teacache-1c` | — | 35s | ComfyUI 有损档 |
| 6000a | `comfy-bf16-original-4c` | — | 400s | 零量化质量参照(oracle) |

入口一律走 `scripts/h3_generate.py --host {5090|6000a|runpods} --profile <名> [--nfe N]`;
profile 命名规范见该文件顶部注释(`<引擎>-<精度>-<变体>-<拓扑>`)。

---

## 二、下一步该干嘛(按优先级)

### P0 · 需要你本人做的:Turbo LoRA 质量定档(Stage B 盲评)
这是唯一卡在"人"这一环的事,做完才能把 `turbo` 转正为默认档。

- 候选:`NFE6 FP8`(均衡)/ `NFE4`(极速)/ `ckpt850 NFE4`(大运动专用)
- 规模:12-20 个 case × seed 0/1/2,**盲评**(别看文件名)
- 素材:必须用 Tears of Steel(CC-BY);`data/` 目录整体不入库,详见
  `claude_history/CLAUDE.md` 的内容红线
- 已知边界:8 个 Stage A case 里 7 个 NFE6 帧级全优,唯一例外是极端运动
  (fight)——v4 NFE4 会融毁、NFE6 强拖影,ckpt850 NFE4 明显更好
- 背景与全部数据:`doc/turbo_lora_results.md` 第六节 + `claude_history/07_turbo_lora/FINAL.md`

**做完之后**:把 `turbo-lora` 提升为各机默认 turbo 档,更新 CLI 别名表与 gallery。

### P1 · 引擎 30 秒硬超时(会污染你正在跑的 eval 表)
`vllm_omni/diffusion/diffusion_engine.py:58` 的 `_ASYNC_OUTPUT_TIMEOUT = 30.0`
把所有慢于 ~30s 的请求掐成 500(报错信息是空的)。三台机器都一样。

- **已被它掐掉的格**:RunPods 的 `vllm-bf16-original-tp4` 与 `-tp2u2`
- 修法:改大该常量(如 600.0)→ 重启引擎。editable 安装,一行;
  **源码 reset/升级后要重打**
- 修完补跑那两格,并在结果表里给"失败"格加脚注:是被超时掐掉,不是跑不动
- 细节与未解疑点(5090 上 47s 的格反而能过)见 `claude_history/06_infra/FINAL.md` 同日增补

### P2 · 上游贡献:PR#5910 的 stride bug 补丁
`scripts/pr5910_resident_stride_fix.patch` 修的是 DLO 常驻组重建权重时用
`.view()` 丢掉在线 FP8 转置视图 stride 的问题 —— 不修则 FP8 输出纯噪声。

- **PR 当前 head `1a9b9c2c` 仍然带着这个 bug**(2026-08-09 核实)
- 作者只在单卡 B300 验过,没踩到"常驻 + 转置"这个组合
- 顺带可以提一句:`--text-encoder-tp-size` 与世界大小不一致时,报错是
  `assert cpu_group is not None`,极难定位
- 根因证据链:`claude_history/08_5090_tp2/FINAL.md`

### P3 · RunPods 成本决策(**在计费,别忘了**)
pod 一直开着就一直烧钱。权重与 env 都在 `/workspace` 持久卷上(用 263G / 共 503G),
**销毁 pod 不会丢权重**,重建用 `scripts/setup_runpods.sh` + 把 `scripts/` 推上去。

- 短期还要跑评测 → 保留,但空闲时 `h3_switch_runpods.sh stop` 释放显存
- 评测做完 → 决定是"留着当生产机"还是"销毁,需要时重建"(重建约 1 小时,主要是下载)

### P4 · 工程收尾(有空再做)
- **768P 原生档在 RunPods 没测过**(note 预估 30-55s)。若能压到 30s 内,
  高清档就能从 6000a 的 72s 迁走
- `gallery/index.html` 还没加 RunPods 区块(媒体文件已经放进 `gallery/media/`)
- Cache-DiT / `torch.compile` 一直没开(此前刻意不开,避免扩大 correctness 排查面);
  correctness 现在稳了,可以试,但**每次都要先抽帧过质量关再看速度**
- FP4:**已结论"无工程路径"**,不用再花时间(vllm-omni 的 mxfp4 是 NPU-only,
  comfy 的 NVFP4 引擎加载不了;而且 FP8 下显存只用掉 19G/32G,容量早不是瓶颈)

---

## 三、几条会反复咬人的规矩

1. **benchmark 数字必须配质量抽帧才算数**。有过 47s 的漂亮数字配一屏噪声的教训。
2. **"失败"要分清 *跑不动* 还是 *被超时/被拒绝掐掉***(见 P1)。
3. **500 + 空 message 一律去读服务端 traceback**,别从客户端反推。
4. **ssh 里的 `pkill`/`pgrep` 一律用括号模式**(`[v]llm serve`),否则会匹配到自己
   —— 这坑不只在 kill,`while pgrep ...; do` 等待循环里同样会自匹配、永远等下去。
5. **不覆盖已跑通的产物**:新结果进新目录/新文件名。
6. **多 session 并行**:`claude_history/` 二级 session 目录只归自己写,FINAL.md 才是共享可变的。
7. **步数一律用 NFE 表述**(引擎语义 `num_inference_steps = NFE + 1`),别把裸 steps 写进 profile。

---

## 四、当前现场(2026-08-09 夜)

- **RunPods**:server UP,你的 eval sweep 正在跑(已到 `fp8-turbo-tp2u2` 那几格)
- **popos-6000a**:vLLM(turbo-fp8)+ ComfyUI :8288 都在,产线正常
- **popos-5090**:有另一个 session 的 vLLM 在跑 —— 动这台机器前先协调
- 未读的研究笔记:`doc/` 下若有新 note,先读再动手(本项目的惯例是 note 先行)
