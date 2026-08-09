# 10 · 仓库化与跨机同步 — 最终结论

建立于 2026-08-09(session 1ed2dca3)。

## 一、结构决定

**仓库根 = 项目根**。每台机器的项目根原地 `git init`,不新建目录、不搬文件。
选它的唯一理由是**所有路径引用零改动** —— 三条产线、三份 switch、五个 launcher、
所有归档文档里的路径全部原样有效,这是入库这一步最大的风险控制。

| 机器 | 项目根 = 仓库工作区 |
|---|---|
| Mac(编排) | `~/Library/CloudStorage/Dropbox/Working_Directory/CV/minimax-h3` |
| popos-5090 | `~/data/dropbox/CV/h3` |
| popos-6000a | `~/data/dropbox/CV/h3` |
| 5090-Runpods | `/workspace/h3` |

入库的:`scripts/`(48)、`doc/`、`claude_history/`、`gallery/index.html`、
`workflows/smoke_prompt.txt` 与 `workflows/stage_a/*.txt`、`README.md`、`LICENSE`。
排除的(各机自行获取/生成):权重(`models*/ base/ merged/ loras/`)、第三方源码
(`src*/ ComfyUI/`)、`env/`、`outputs/ logs/ run/ inputs/*`、`gallery/media/`、
`credentials/`、`data/`。`.git` 约 720K。

**提示词文件是输入不是产物**:`bench_matrix.sh`、`stage_a_run.sh`、
`ab_ckpt850_dynamic.sh` 都直接引用 `workflows/` 下的 txt。首次提交漏了它们,
归档 benchmark 当场不可复现。`.gitignore` 用 `workflows/*` + 负向规则放行这两处,
生成的 `*_prompt.txt` 仍然忽略。

## 二、清洗门禁(仓库公开,每次新增内容前都要跑)

`bash scripts/check_repo_clean.sh` —— 三项全 0 才 exit 0。等价的裸命令:

```bash
git ls-files | xargs grep -nE \
  '(hf|ghp|gho|ghs|github_pat)_[A-Za-z0-9]{16,}|sk-or-v1-[a-f0-9]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY'
git ls-files | grep -i <红线素材文件名>
git ls-files | grep -E '\.(mp4|safetensors|pt|bin)$'
```

基线:三项全部 **0 命中**。

⚠️ 密钥正则必须写成"**前缀 + 至少 16 位随机体**",不能只匹配裸前缀。
最初写的版本含 `sk-or-v1-` 这样的裸前缀,结果**本文件记录门禁命令的这一行自己就会
命中**,门禁永远报 1 —— 一个恒假警报的门禁比没有门禁更危险,因为它训练人去忽略它。
(2026-08-09 实际踩到:那次命中经逐行核对确认是本文件自匹配,无真实泄漏。)
门禁本身要有反向对照 —— 塞一个假 token 进去确认它会响,否则"全 0"可能只是正则失效。

`credentials/` 里有三个明文有效 token
(HF / GitHub / OpenRouter),它们躺在 Dropbox 同步目录里 —— **建议全部轮换**。

推送凭据用一次性 tokenized URL,不写进 `.git/config`:

```bash
git push "https://x-access-token:${T}@github.com/<owner>/<repo>.git" main:main
```

用 `git -c credential.helper='!f(){...}'` 会挂住(实测超时),不要用。

## 三、分叉收敛(入库时做的一次性合并)

四处代码交叉分叉,**三边都不是"正确版本"**:

| 文件 | 取自 | 依据 |
|---|---|---|
| `run_fl2va.py` | 5090 | 含 Turbo LoRA 节点;6000a 那份是它的旧备份 |
| `run_fl2va_sglang.py` | 6000a | 含 NFE 语义/manifest/`--flow-shift`;5090 那份旧 |
| `run_fl2va_vllm.py` | 5090/6000a | runpods 那份缺 `served_model()` 审计 |
| `tp2_r2_probe.sh` | 5090 | 三份合一 |

同时修掉两个**"假一致"**(三机 md5 相同,但内含 6000a 专属绝对路径,在别的机器上
是 100% 死代码):

- `test_h3_schedule.py` 写死 `/home/isaac/...` → 改读 `H3_SRC_ROOT` / `H3_ROOT`;
  SGLang 变为可选(runpods 没装,缺了就少测一路不算失败)。
- `merge_turbo_lora.py` 的 `--base` / `--bf16-single` 默认值写死 → 改必填或读
  `H3_BASE_FL2VA`。

`test_h3_schedule.py` 后续又改成**所有 checkout 全部参测**:一台机器上常同时存在
`sglang` 与 `sglang-pr33681`、`vllm-omni` 与 `vllm-omni-pr5910`,只测其中一份等于
放过另一份 —— 而这两份的区别往往正是"打没打 stride 补丁"。

## 四、接管既有机器的安全流程

```bash
tar czf ~/h3_scripts_backup_$(date +%F_%H%M%S).tgz scripts/   # 1 备份,不入库
git init -b main && git remote add origin <url> && git fetch origin
git reset --mixed origin/main        # 2 只设索引不动工作区
git diff                             # 3 逐文件审阅"机器现状 vs 仓库版本"
git checkout -- .                    # 4 确认每一处都是预期收敛后才落地
```

`reset --mixed` 是关键:直接 `checkout`/`pull` 会覆盖机器上的改动而看不到差异。
实测三台的差异全部是预期内的收敛(runpods 3 个文件、5090 3 个、6000a 6 个,
其中两个只是丢了可执行位)。

### 两个会绊人的小坑

- **`git ls-files | sort` 的结果依赖 locale**:Mac(en_US)与容器(C)对大写
  `README.md` 排序不同,导致"逐文件 md5 聚合摘要"假报不一致。要比对必须
  `LC_ALL=C sort`。
- **本地已有同名未跟踪文件时 `git pull` 会 abort**("would be overwritten")。
  先确认与 `git show origin/main:<path>` 逐位相同再 `rm`,不要盲删。

## 五、验证基线(2026-08-09 实测,ToS 素材 + `workflows/smoke_prompt.txt`)

三机 `scripts/`(48 文件)对仓库 `LC_ALL=C` 聚合 md5 **完全一致**;
各机至少一条产线出片(864×480 / 124 帧 / 5.17s / 带音频):

| 机器 | profile | NFE | warm 推理 | 备注 |
|---|---|---|---|---|
| runpods | `vllm-fp8-turbo-tp2u2` | 6 | **18.5–18.8s** | 复现归档值 |
| 6000a | `vllm-fp8-turbo-tp4` | 6 | 22.5s | 归档 23.5s |
| 5090 | `comfy-int8-turbo-1c` | 4 | 25.0s | 归档 25.0s;冷启首推 45.0s |

## 六、Phase 2(未做,按优先级)

1. `machines.yaml` 外置机器台账 —— 现在还在 `scripts/h3_generate.py` 的 `HOSTS` 里
2. 三份 `h3_switch*.sh` 合成一份参数化版(建议采用 runpods 的三维正交 CLI)
3. 5 个 `launch_comfy*.sh` 合 1
4. `bootstrap.sh` 一键装机 + 机器自识别
5. 修 `setup_runpods.sh` 的三个已知缺陷(stride 补丁探测恒真、用了已弃用的
   `HF_HUB_ENABLE_HF_TRANSFER`、依赖尚未推送的 `scripts/*`)
