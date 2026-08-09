# FINAL · 基建台账(整理版, 2026-08-08 by 3195aa2f)

## 机器/端口/env
| 机器 | GPU | 端口 | env |
|---|---|---|---|
| popos-5090 | 2×5090 32G(GPU0=turbo/serving, GPU1=baseline) | 8188 base / 8189 turbo / 8091 vllm / 30010 sglang | h3_comfy / h3_vllm / h3_sglang(_NV_py312) |
| popos-6000a | 4×6000Ada 48G | 8288 comfy / 8091 vllm / 30010 sglang | 同上 |
远端根: `~/data/dropbox/CV/h3/`(ComfyUI, scripts/, logs/, inputs/, outputs/, run/, src/)
权重: 6000a `/home/isaac/Data/h3_weights/`(第二 NVMe);5090 `~/data/dropbox/CV/h3/models_official/`

## Mac 侧
- CLI: `scripts/h3_generate.py`(预处理→上传→生成→回传;--host/--profile/--seconds/--steps/--seed)
- benchmark 档: doc/speedup_*.md;可视化: gallery/index.html;输出: outputs/<name>/
- HF token: `credentials/HF.md`(已写入两机 ~/.cache/huggingface/token;匿名日流量 ~300G 会被限速)

## 通用踩坑(所有 session 适用)
1. ssh 远程 pkill/pgrep:模式必须方括号防自匹配(`[m]ain.py`),否则杀死自己的会话
2. 远端长任务:`setsid nohup … < /dev/null &` 分离,日志落 logs/,terminal 断线不受影响
3. ComfyUI 会缓存整图:同参数重跑需变 seed/prompt(计时)或参数抖动(TeaCache)
4. 6000a kreaid(GPU0/1)已批准停用,恢复见 /home/isaac/workdir/kreaid/
5. 内容红线见根 CLAUDE.md

## 增补(2026-08-08, 5090 serving 结案时沉淀)
6. rsync 源码目录若 `--exclude .git`,setuptools_scm 会回退版本 'dev' 致构建失败;
   修法: `SETUPTOOLS_SCM_PRETEND_VERSION=<ver> pip install -e . --no-build-isolation`
7. vllm-omni 源码 editable 前先装同版 wheel(构建脚本 import 自身需依赖在场)
8. H3 在 vLLM-Omni 的加载器不支持 mmap → 大内存需求无法靠页缓存缓解(125G 主机跑不了 layerwise)

## 增补(2026-08-08, Turbo LoRA 落地, 详见 07_turbo_lora)
9. 6000a `h3_switch.sh` 新模式: `vllm-turbo [bf16|fp8]` / `sglang-turbo` / `sglang-lora`
   (Turbo-LoRA merged checkpoint 与 dynamic LoRA 产线;merged 目录缺 `.complete` 拒绝启动)
10. 新权重目录(6000a Data 盘): `h3_weights/loras/`(LoRA 文件 ~0.78G×2) 与
    `h3_weights/MiniMax-H3-Turbo-v4s600ema/`(merged checkpoint +62G, 含
    merge_manifest.json/.complete/delta_norms.csv;transformer 为真实文件,其余 symlink)
11. SGLang 用本地目录跑 H3 的两个硬条件: 根目录完整 HF 布局 + 路径 basename 必须为
    `MiniMax-H3`(registry 短名匹配)→ 别名目录 `turbo_v4s600ema_alias/MiniMax-H3`
12. sglang 源码在分支 `turbo-lora-backport`(cherry-pick 上游 914644e81c9b, 2D fused
    lora_B TP 切片修复;下次升级 sglang 时消化该分支)
13. 步数语义: 两引擎 `num_inference_steps=N` → N−1 次 DiT forward;客户端
    `run_fl2va_*.py` 用 `--nfe` 表述(steps=NFE+1),Mac CLI 新 profile
    `turbo-lora`/`sglang-turbo` 默认 NFE6;引擎计时协议=固定 seed+warmup1+timed3
    (引擎无 ComfyUI 图缓存,与旧"换 seed"口径不同)
14. `/health` 在两引擎都早于权重加载就绪 → 切换后首单必为 warmup,不计时

## 增补(2026-08-09, switch 脚本正确性加固, 详见 07 主题 doc 第八b节)
15. **共用端口 = 必须有变体追踪**。vllm 与 vllm-turbo 都绑 :8091 但服务不同 ckpt,
    而 `start_vllm` 的"端口健康就 return"会让互切变成空操作 → 静默服错权重,
    且**延迟无法自证**(base NFE6 与 turbo NFE6 耗时几乎相同)。两台机器现均以
    `run/vllm.variant` + `run/vllm.model` 记录变体,互切必重启。
16. **写 run/*.pid 的入口必须同时写 .variant**。5090 的 `tp2_r2_launch_{p2,tp1}.sh`
    是绕过 switcher 的后门(直接在 :8091 起 base 并改写 pid),已补上各自的变体声明
    (`p2-base`/`tp1-base`);今后任何新加的直启脚本同此要求。
17. 其余加固(两机对齐): 半死进程先 `stop_one` 再起;清理后端口仍占则**拒绝启动**
    (不撞端口);`flock` 串行化并发切换(两个 session 共用机器);启动后 `kill -0`
    确认存活再落变体声明(`setsid nohup` 即使 exec 失败也返回 0);
    日志**轮转不截断**(保留 10 份)并写启动头(model/label/src HEAD/补丁状态)。
18. **补丁状态判据用代码内容,不用 `git diff`**。补丁一旦 commit,worktree 干净,
    `git diff --quiet` 会反过来报"未打补丁"。stride 补丁现已在两机固化为分支
    `pr5910-stride-fix`(commit 070096bd),日志头改用 `grep as_strided` 判定。
19. 产物审计闭环: `run_fl2va_vllm.py` / `run_fl2va_sglang.py` 的 manifest 新增
    `served_model`(向 `/v1/models` 取服务端实际加载的权重路径),使单条 mp4 可回溯
    到 base 还是 merged turbo —— 不依赖文件名。
20. 决策表测试 `scripts/test_h3_switch_5090.sh`(5090): `H3_SWITCH_LIB=1` source
    脚本只取函数并 stub 掉启动,9 个用例覆盖变体匹配/不匹配/半死/端口占用;
    改 switcher 后必跑。注意该变量会被 export 继承 —— 测完别用同一个 shell 直接
    调 switcher,否则 TARGET 变成 `__lib__`。

## 增补(2026-08-09 晚, profile 改名 + ComfyUI Turbo 产线)
21. **profile 命名规范**: `<引擎>-<精度>-<变体…>-<拓扑>`。精度=DiT 权重精度
    (int8/fp8/bf16/nvfp4);变体=**会改变输出或步数**的技术选择,可叠加,按
    `<加速方法…>-turbo` 排序(original / teacache|sage|flash / turbo);拓扑=
    ComfyUI 用计算卡数 1c/4c、引擎用 tp2/tp4。纯内存调度 flag 与 vLLM/SGLang 的
    attention backend **不进名字**(前者不改输出,后者是引擎必选配置)。
    旧名(baseline/turbo/vllm/sglang/bf16/turbo-lora/sglang-turbo)**按机器**映射为
    永久别名 —— 旧 `turbo` 在 5090 指 ComfyUI TeaCache、在 6000a 指 vLLM FP8,
    这正是新命名要消除的歧义。家族简称按 token 子序列匹配(`vllm-fp8-turbo` 两机
    各自解析成 tp2/tp4),有歧义则报错列候选。全量: `h3_generate.py --list`。
22. **新产线 comfy-int8-turbo-1c**(5090): 作者节点
    `Larryvrh/ComfyUI-MiniMax-H3-Turbo` @ `55fee864`(零 pip 依赖)+ GPU0:8190
    独立 worker(`h3_switch_5090.sh comfy-tlora`,与 :8188/:8189 **互斥**,
    host RAM 125G 装不下三个)。NFE4/6/8 = 25.0/30.0/40.0s,单卡。
    **不能用 ComfyUI 核心 LoraLoaderModelOnly**(裸键 0/518 命中且不报错);
    pruned 基座还需作者节点在运行时注入 51 个 adaln。详见 doc 第九节。
23. **flock + 守护进程 = fd 泄漏死锁(踩过)**。`exec 9>lock` 的 fd 9 会被
    `setsid nohup` 的子进程继承,而 flock 锁绑定在 open file description 上 →
    **守护进程活着就一直持锁**,后续每次切换干等 600s。修法: 所有守护进程启动处
    加 `9>&-`(两机共 7 处)。加锁时务必同时处理 fd 继承。
24. ComfyUI 会缓存节点输出: LoRA 节点输入不变时只在首次执行并打印一次日志,
    权重 patch 随缓存的 MODEL 带下去 —— 别把"日志只出现 1 次"误判为 LoRA 失效;
    用采样器的 per-step 日志计数交叉验证。另: ComfyUI 是 steps==forwards,
    与引擎侧 `num_inference_steps=NFE+1` 不同,两边都由 `--nfe` 统一表述。

## 增补(2026-08-09 夜, 引擎级 30 秒硬超时 —— 三台机器通用)

**`vllm_omni/diffusion/diffusion_engine.py:58` 有一个写死的
`_ASYNC_OUTPUT_TIMEOUT = 30.0`**,在 `step_streaming`(同文件 :326)里作为
"等引擎吐下一个输出"的上限。超时后抛 `TimeoutError`,经
`inline_stage_diffusion_client.py` 包成 HTTP 500,body 是**一句没有内容的**
`{"error":{"message":"Video generation failed:"}}`。

- 三台机器(5090 / 6000a / runpods)的 vllm-omni 源码里都是 30.0 —— **不是某台机器配错**。
  它与 `VLLM_OMNI_VIDEO_SYNC_TIMEOUT`(我们设 1800)是两回事,后者管不到它。
- 实测表现(runpods,2026-08-09 eval sweep):FP8 基座 NFE11(~28s)过、
  Turbo NFE6(~21s)过、**BF16 基座 NFE11(~30s 出头)必挂**,两种拓扑都挂。
- 未解矛盾(如实记录):5090 的 `vllm-fp8-original-tp2` 实测 ~47s 却能过 ——
  说明这 30 秒卡的是两次 yield 之间的间隔而非整次请求,不同拓扑下分段节奏不同。
  想彻底钉死机制需要再挖 `step_streaming` 的 yield 粒度。
- 处置:editable 安装,直接把 58 行改大(如 600.0)后重启引擎即可;
  改动属于本地 patch,升级/reset 源码后要重打。

**两条通用教训**
1. **"500 + 空 message" 一律去读服务端 traceback**,不要从客户端错误反推。
   这次客户端只给了 `Video generation failed:`,真因在服务端日志里写得清清楚楚。
2. **别把引擎的等待上限当成硬件/模型的能力上限**。差点把"4×5090 跑不了 BF16 基座"
   写进结论 —— 实际它只是比 30 秒慢了一点点。benchmark 表里凡是"失败"格,
   都要先分清是 *跑不动* 还是 *被超时掐掉*。

## 增补(2026-08-09, session 1ed2dca3): 代码进 git,以及一条被漏掉的运维铁律

**四处代码(Mac + 三机)已收敛进一个公开仓库**,每台机器的项目根就是仓库工作区,
`git pull` 即更新。结构决定、清洗门禁、接管流程、Phase 2 排期全部在
`../10_repo/FINAL.md`,本条只记与基建台账直接相关的两点。

### 25. 运维铁律补第 0 条:**同变体且健康则复用,不重启**

原先记录的铁律(变体追踪 / 端口拒启 / flock+`9>&-` / 验活 / 日志轮转)漏了这一条,
结果 `h3_switch_runpods.sh` 无条件重启也没被发现。两个后果:

- 每次生成白付一次冷启动(runpods 210s)
- **测到的推理耗时虚高约 20%** —— 重启后首推 22.3s,真 warm 18.5s。
  归档的 18.5s 一度复现不出来,原因就在这里。

推论(比这条铁律本身更重要):**benchmark 数字必须声明冷热态**。同一条产线
冷启首推与 warm 的差距,runpods 是 22.3 → 18.5s,5090 ComfyUI 是 45.0 → 25.0s
(将近一倍)。`h3_eval.py` 的"1 warmup 不计 + N timed"协议就是为这个设计的,
手工 benchmark 也要照做。

### 26. 三份 switch 的铁律对齐状态(2026-08-09 后)

`h3_switch_5090.sh` / `h3_switch.sh`(6000a) / `h3_switch_runpods.sh` 现在**全部**
具备:变体追踪(`run/vllm.variant` + `.model`)、同变体复用、半死进程清理、
端口占用拒启、`flock` 串行化且守护进程 `9>&-`、启动后 `kill -0` 验活、
日志轮转不截断、日志头记录 model/label/src head/stride 补丁状态。

改任何一份都要保持这八条 —— 它们不是风格问题,每一条都对应一次真实事故。
