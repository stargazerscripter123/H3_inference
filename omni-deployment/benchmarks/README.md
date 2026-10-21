# FastH3 Omni：32 次正式测量的筛选计划

本目录保存测量计划及素材生成命令。2026-10-21 主任务已在 popos-5090 生成素材并核验哈希/媒体元数据；尚未运行本计划的模型测量，也未打开图片或视频。
`generate_plans.py` 通过当前 `omni-deployment/scripts/run_omni.py` 的
`build_request(..., server_paths=True)` 校验每行，同时核对 `bench_omni.py`
接受的缓存标签。远端验证清单为 `run/omni/bench32-assets-verified.json`，覆盖 21 个衍生文件及完整官方影片 SHA-256。

## 测量矩阵与顺序

两份计划内容完全相同；文件名和结果目录区分 SP1/SP2。
每种配置 4 个场景 × 2 种缓存条件 × 2 个 seed = 16 次正式测量。
SP1/SP2 合计 **32 次 measured，另加 8 次显式 warmup，共 40 次提交**。
输出固定为 832×480、362 帧、24 fps、8 次 PDD forward、CFG 1，使用发布版 36L decoder。
362 帧对应约 15.083 秒输出视频；15 秒阈值是生成耗时目标，不是输出时长。

| case_id | 有序参考 | 每配置正式测量 |
| --- | --- | ---: |
| S1-single-image | 一张图 | 4 |
| S2-multiple-images | 两张图 | 4 |
| S3-image-audio | 一张图、4 秒显式音频 | 4 |
| S4-image-video | 一张图、4 秒无音轨视频 | 4 |

每个场景连续执行以下五行，完成后再进入下一场景：

| 顺序 | phase / cache_condition | 素材 | seed | 用途 |
| --- | --- | --- | ---: | --- |
| 1 | warmup / unspecified | 独立 W | 9000 | 同场景显式预热，排除统计 |
| 2 | measured / warm-new | 独立 A | 1234 | 新路径、新内容、新 prompt |
| 3 | measured / warm-new | 独立 B | 5678 | 新路径、新内容、新 prompt，同时给后两行提供参考预热 |
| 4 | measured / warm-fixed-ref | 固定 B | 1234 | 参考不变，换 prompt/seed，部分缓存条件 |
| 5 | measured / warm-fixed-ref | 固定 B | 5678 | 参考不变，再换 prompt/seed，部分缓存条件 |

一个进程内所有 20 行 prompt 均不同。不同场景的 W/A/B 素材也各自独立。
SP1/SP2 使用相同素材、prompt、seed、顺序；两种配置分别运行一个服务进程。
不在同一已跑过这些素材的进程里重跑整份计划并继续称为 `warm-new`。

引用编号已对照固定 FastVideo commit
`cdda1b7535fe32699335e39db82c1d4db21ceb88` 的
`fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_conditioning.py:72–87`
核对：图片、视频、音频分别计数，不按混合列表的全局位置编号。
因此 S3 使用 `<Picture 1>` + `<Audio 1>`，S4 使用 `<Picture 1>` + `<Video 1>`，
S2 使用 `<Picture 1>` + `<Picture 2>` 均正确。
音频计数的特殊点是任何 `has_audio` 的参考都会增加计数，包括带音轨的视频；
以后若同时加入带音轨视频和显式音频，应按实际参考顺序重新检查 `<Audio n>`。
本计划 S4 的视频去掉音轨，S3 不含视频，避开了该歧义。

`warm-fixed-ref` 只表示**部分缓存实验条件**，不是缓存命中证明。
每行换 prompt，因此不会重用完整 conditioning；上游 video conditioning memo
本身禁用，S4 尤其不能据此宣称完整缓存命中。文件系统缓存、JIT、参考编码缓存等
也不能从这个 HTTP API 分别观测。前两条正式测量为固定参考组提供 priming，
没有额外隐藏提交或缓存清理。

## 素材衍生清单

`derivatives.json` 列出 **21 个**衍生文件，以及每个文件的完整影片时间戳、类型、
目标路径、预期 ffprobe 元数据。统一放在：

`/home/isaac/workdir/h3/inputs/tos/bench32-v1/`

W/A/B 的影片锚点分别是 240/55/90 秒；场景依次加 0/1/2/3 秒。
S2 的第二张图再加 0.5 秒。图片保持宽高比缩放到 480 高；音频为 4 秒
PCM 32-kHz 双声道；视频为 4 秒、96 帧、24 fps、480 高 H.264，去除音轨，
让 S4 聚焦图像与视频参考，显式音频由 S3 覆盖。输入宽度由影片宽高比决定，
与模型输出的 832 宽分开记录。

素材来自已获授权的 Tears of Steel。命令使用完整影片的时间轴；不要直接把
已截短的片段当作输入，否则 55/90/240 秒会指向错误位置。
官方影片源为：

[Tears of Steel 720p MOV](https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov)

授权证据：[Blender 官方说明，CC BY 3.0](https://mango.blender.org/about/)。
署名保留为 `Tears of Steel — Blender Foundation | mango.blender.org`，并记录截取、
缩放、视频去音轨及音频重采样等改动。这里没有重新验证下载 URL 或 Range 支持。
源内容没有经过本轮视觉/听觉检查，因此这些时间段不能被当成已确认的身份、对白或
动作测试用例。

在远端使用已经确认的完整影片路径，或由执行者确认可 Range 读取的 URL：

```bash
bash /home/isaac/workdir/h3/omni-deployment/benchmarks/derive_refs.sh \
  /absolute/path/to/authorized-full-tears-of-steel.mov
```

也可传上面的官方 URL。脚本生成命令已由主任务在远端执行并验证。
默认使用部署目录 `.tools/media/bin/ffmpeg` 和 `ffprobe`，可用
`OMNI_FFMPEG`、`OMNI_FFPROBE` 指定其他已核实的工具。
`-n` 拒绝覆盖已有文件；中途失败后应核对已生成文件再按清单续做，不直接覆盖。
每个输出都附带 `.sha256` 和 `.ffprobe.json`。

运行前必须逐项核对素材元数据、生成命令成功状态和 SHA-256，并保留证据。
所有同类型 W/A/B 文件的 SHA-256 必须互不相同；新文件名不等于新内容。
如重复，应选取另一实际帧/片段、更新衍生记录并重新核对，再运行计划。
SHA-256 不同也不证明视觉或音频语义有显著区别；本轮只保证实验条件可追溯。
如整片已落盘，记录整片 SHA-256；HTTP Range 取段时记录源 URL、HTTP 证据、
时间段和衍生文件哈希，不声称验证了完整源文件哈希。

## 本地再生成与结构校验

```bash
python3 omni-deployment/benchmarks/generate_plans.py
```

默认从相邻 `omni-deployment/scripts/` 导入现有标准库客户端。
`--scripts-dir` 可指向实际将执行的同版本脚本。
输出 `screen32-sp1.json`、`screen32-sp2.json`、`derivatives.json`、
`derive_refs.sh` 和 `dry-validation.json`。
此命令不读取素材、不执行 shell 中的命令、不进行 HTTP 请求。
校验结果记录实际导入的客户端/benchmark SHA-256；它只证明请求结构和计划矩阵，
不能证明远端素材存在、GPU 运行成功或 reference 内容有效。

## 远端运行命令

以下假设本目录已由主任务分发至 `/home/isaac/workdir/h3/omni-deployment/benchmarks`，
模型、table、decoder、环境变量已经按部署 README 验证，媒体清单核对通过。
配置切换由 `omni_control.py` 完成，JSON plan 本身不选择 GPU。

```bash
export H3_OMNI_ROOT=/home/isaac/workdir/h3
export OMNI_PYTHON="$H3_OMNI_ROOT/envs/omni/bin/python"
export PATH="$H3_OMNI_ROOT/.tools/media/bin:$PATH"

"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/omni_control.py" stop
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/omni_control.py" start sp1
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/omni_control.py" status
```

确认 status 为 READY 后，才执行：

```bash
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/bench_omni.py" \
  --plan "$H3_OMNI_ROOT/omni-deployment/benchmarks/screen32-sp1.json" \
  --server http://127.0.0.1:8108 \
  --output-root "$H3_OMNI_ROOT/outputs/omni-screen32-sp1" \
  --threshold 15 --poll-interval 0.25 --timeout 1800
```

SP1 完成且没有未决任务后，再切换：

```bash
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/omni_control.py" stop
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/omni_control.py" start sp2
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/omni_control.py" status
```

确认 READY 后：

```bash
"$OMNI_PYTHON" "$H3_OMNI_ROOT/omni-deployment/scripts/bench_omni.py" \
  --plan "$H3_OMNI_ROOT/omni-deployment/benchmarks/screen32-sp2.json" \
  --server http://127.0.0.1:8108 \
  --output-root "$H3_OMNI_ROOT/outputs/omni-screen32-sp2" \
  --threshold 15 --poll-interval 0.25 --timeout 1800
```

在远端同机运行时刻意不加 `--server-paths`，让现有客户端检查路径并记录真实参考
SHA-256。输入 hashing 在 client timer 之前发生；两配置保持相同操作。
若从别处运行，才考虑 `--server-paths`，并另行保存远端素材完整性证据。

## 结果边界

- 每个 SP/case/cache 只有两条正式样本；p95 在这里就是最大值，不能用于稳定 SLA
  验收或置信区间。后续应根据筛选结果增加独立重复，并考虑交叉执行顺序。
- 固定顺序存在温度、时钟、JIT 和缓存先后效应；warm-new A/B 与 fixed B/B 的素材
  构成也不同，不能把耗时差全部归因于缓存。这套筛选更适合相同请求下 SP1/SP2 比较。
- 单次同场景预热未证明所有 kernel 都稳定；应保留全部 warmup 记录，异常波动另查。
- 默认 SP1 是 GPU1，SP2 用 GPU0/1；两卡 PCIe 链路不同，GPU0 桌面占用也可能干扰。
  服务配置、每卡显存/利用率、主机内存、频率和日志由主任务独立记录。
- `client_to_media_s` 包含提交、排队、polling 和完整下载，排除输入哈希及下载后的
  ffprobe；0.25 秒 polling 会影响观测粒度。API 的 inference time 不是完整服务耗时。
- 现有 benchmark 遇失败会停止，不自动重试或取消服务任务。失败保留在已执行组的
  分母中，未执行行单独报告；必须按完整矩阵核对实际完成数。
- 自动媒体校验只覆盖编码、帧数、尺寸、音轨等；未做身份、动作、对白、口型或美学
  评判，不读取任何 `data/`，本轮也没有浏览图片内容。
