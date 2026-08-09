# 10 · 仓库化 — 原始记录(2026-08-09, fable, 1ed2dca3)

## 目标

用户要求:把 runpods 5090 / popos-5090 / popos-6000a 三处的代码推到
`https://github.com/stargazerscripter123/H3_inference.git`(用户当场改为 public),
"把几个不同的代码 fraction 汇总成一个整体,以后在不同 machine 即插即用"。

## 做了什么

1. **侦查**:逐文件 md5 比对四处(含 Mac)的 `scripts/`,证明代码已经**交叉分叉** ——
   5090 有最新 ComfyUI 客户端、6000a 有最新 SGLang 客户端、runpods 有最旧 vLLM 客户端。
   四处都没有 git,唯一的版本控制替代品是 7 个 `.orig_2026*` 手工备份。
2. **骨架**:`.gitignore` / `README.md` / `LICENSE`(MIT) / `scripts/README.md` /
   目录 `.gitkeep`。
3. **收敛**:每个文件取最新那份(逐文件记录来源),修两处"假一致"死代码。
4. **清洗门禁**:密钥正则、红线素材、媒体/权重文件三项全 0 命中后才首次推送
   (`d66049a`,101 文件,`.git` 720K)。
5. **三机原地接管**:runpods → 5090 → 6000a,每台 `reset --mixed` 后逐文件审差异
   再 `checkout`。
6. **验证**:三机 `scripts/` 对仓库 `LC_ALL=C` 聚合 md5 一致;三条产线各出片一条。

## 关键数据

- 三机 48 个 `scripts/` 文件聚合 md5 `18109749d526238b`,与仓库完全一致
- 产线实测(ToS 素材,864×480/124 帧/带音频):
  runpods `vllm-fp8-turbo-tp2u2` NFE6 **18.5s** /
  6000a `vllm-fp8-turbo-tp4` NFE6 **22.5s** /
  5090 `comfy-int8-turbo-1c` NFE4 **25.0s**(冷启首推 45.0s)

## 踩坑

- **提示词文件被 `.gitignore` 的 `workflows/*` 挡住**。它们是**输入**不是产物,
  三个归档脚本直接引用;漏掉就等于归档 benchmark 不可复现。加负向规则放行
  `smoke_prompt.txt` 与 `stage_a/`,生成的 `*_prompt.txt` 继续忽略。
- **`git ls-files | sort` 依赖 locale**,Mac 与容器对 `README.md` 排序不同,
  害我一度以为 runpods 的 `scripts/` 与另外两台不一致。必须 `LC_ALL=C sort`。
- **`git -c credential.helper='!f(){...}' push` 会挂住**(超时),改用一次性
  tokenized URL,推完确认 `.git/config` 里没有 token。
- **本地未跟踪同名文件会让 `git pull` abort**;先证明与 `origin/main` 版本逐位相同
  再删。我第一次写的检查把 `git fetch` 放在比对之后,结果比的是旧 `origin/main`,
  三台全都误报"不同"(无害,但逻辑是错的)。

## 顺手修掉的真问题

**runpods 的 switch 缺了另外两台都有的全部运维铁律**(只写 `run/vllm.pid`)。
发现路径:连续两次同 profile 生成,每次都付 210s 启动 + 22.3s 推理 —— 归档的
18.5s 怎么都复现不出来。读脚本发现它**无条件重启**,所以每次测的都是重启后的
冷态首推。补齐变体追踪 / 同变体复用 / 端口拒启 / flock+`9>&-` / 启动验活 /
日志轮转后:启动 214s → **5s**,推理 → **18.8s / 18.5s**,归档值复现。

顺带堵上的是安全缺口:没有 `run/vllm.variant` 时,`:8091` 被不同 checkpoint 复用
无法判断在服的是哪个模型 —— 与 2026-08-09 在 5090 上修掉的"静默服错 checkpoint"
是同一个洞。变体不匹配现在会打印
`vllm variant mismatch (turbo-fp8-tp2u2-r50 -> turbo-fp8-tp4-r50), restarting` 并重启,
已实测。

**`test_h3_schedule.py` 改为所有 checkout 全部参测**:5090 上同时有 `sglang` 与
`sglang-pr33681`,原逻辑"命中多个就跳过"直接把 SGLang 那一路测没了;而这两份的
区别往往正是打没打补丁。

## 未竟

- Phase 2 全部(`machines.yaml` / 统一 switch / 统一 launcher / `bootstrap.sh` /
  修 `setup_runpods.sh` 三缺陷),见本主题 FINAL.md 第六节
- 仓库公开后 `credentials/` 里三个明文 token 建议轮换(它们从未入库,但躺在
  Dropbox 同步目录)
- Turbo LoRA 的 Stage B 人工盲评(需用户参与)
