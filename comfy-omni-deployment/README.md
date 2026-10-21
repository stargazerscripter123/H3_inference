# FastH3 Omni 原生 ComfyUI 对照

这是独立单 GPU 部署包，使用重装后的 `popos-5090`、GPU1 和 loopback `127.0.0.1:8188`。截至2026-10-21 UTC，固定环境已安装，三条业务请求各1首推+3warm，共12/12通过执行与完整MP4元数据检查；人工画面、声音和参考一致性验收仍待完成。此状态取代早期“仅本地模拟验证”的描述，不代表服务此后持续运行；查当前状态使用下方controller。本包不会启动或停止FastVideo。

## 固定配方

- ComfyUI `0.42.0`，commit `b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`；`comfy-kitchen==0.2.39`。不需要自定义节点。
- 独立源码 `src/ComfyUI-Omni`，独立环境 `envs/comfy-omni`；已验证Python3.12.15、Torch2.12.0+cu130、Torchvision0.27.0+cu130。
- 本pin的H3 attention执行 **dense forward**；源码虽保留VSA gate参数，但该forward不使用它，实际调用`optimized_attention`。本部署未启用VSA稀疏补丁，且controller强制`--disable-all-custom-nodes`。不能将FastVideo的VSA加速配方视为此原生Comfy图已经具备的功能。
- 四个模型共35,274,204,555 bytes。具体 Hugging Face revision、文件名、目标目录、大小及 SHA256 全部在 [pins.json](pins.json)。DiT 是 Comfy 专用 NVFP4 QAD；文本编码器是含 vision 的 Qwen3-VL-32B NVFP4 AWQ；视频 VAE 是完整 FP16，音频 VAE 是 FP32。
- 输出宽480×高832、362帧、24fps；Euler / simple、8步PDD配方、BasicGuider（CFG1）、video/audio shift12/3；无额外LoRA。客户端证明采样节点执行且非整图缓存；未对每次CUDA forward单独插桩计数。

[官方 QAD 验证](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD-Comfy/blob/36dbf15cd11113471114b1040c4f38646c031549/provenance/VALIDATION.md)使用同一 Comfy pin，但硬件为GB200、124帧，Torch为2.9开发版。下面是本机实测，不能用官方数据替代。此处使用单GPU、dense attention和完整FP16视频VAE；FastVideo对照使用双GPU SP2、VSA与36L NVFP4 decoder。比较的是整套部署配方，不能把差值归因于单一框架、GPU数量或attention实现。音频VAE保留FP32；不要用全局`--fp16-vae`覆盖音频精度。

## 已验证环境与结果（2026-10-21 UTC）

项目根为`/home/isaac/workdir/h3`。四个模型放在`src/ComfyUI-Omni/models`的对应子目录；源码commit、模型revision及完整SHA256按[pins.json](pins.json)固定。该文件记录上游静态来源与所选配方，其`evidence_scope`不是GPU实测回执，本次运行结果单独记在本节。

| 项目 | 已验证版本 |
| --- | --- |
| Python | 3.12.15 |
| Torch / Torchvision | 2.12.0+cu130 / 0.27.0+cu130 |
| comfy-kitchen / comfy-aimdo | 0.2.39 / 0.5.6 |
| transformers / huggingface_hub | 5.20.2 / 2.2.0 |
| PyAV | 19.0.2 |
| comfyui-frontend-package | 1.55.18 |
| comfyui-workflow-templates | 0.11.89 |
| comfyui-embedded-docs | 0.5.14 |

该环境已通过`uv pip check`（104个包）；完整已安装版本见[环境快照](environment/installed-packages-20261021.json)。快照记录已安装状态，不是从全新环境重新安装的验收记录。准确的分步安装命令尚未从当前证据逐条核实，本页不把推导出的重建命令写成已执行历史。现有环境可核对：

```bash
cd /home/isaac/workdir/h3
.bootstrap/bin/uv pip check --python envs/comfy-omni/bin/python
envs/comfy-omni/bin/python comfy-omni-deployment/scripts/comfy_control.py validate \
  --project-root /home/isaac/workdir/h3
```

三条业务各保留其原prompt和有序参考，均使用cap480；每case启动独立服务，seed依次为9000（首推）及1234/5678/9012（warm）。三条case的prompt彼此不同；每个case的warm请求重复同一prompt与参考、只更换seed，衡量完整conditioning缓存的暖态，不代表新prompt的服务时延。

| 用例 | 首推秒 | warm中位秒 | warm N | 成功请求 |
| --- | ---: | ---: | ---: | ---: |
| case-01 | 147.803 | 143.452 | 3 | 4/4 |
| case-02 | 176.994 | 169.320 | 3 | 4/4 |
| case-03 | 172.428 | 164.584 | 3 | 4/4 |

时延从GPU主机客户端提交请求起，到history成功、完整MP4复制到客户端输出目录并`fsync`完成；包含生成、视频/音频解码、编码与该次本地写盘，不含服务启动、提交前参考处理/模型SHA核验、完成后的媒体校验和Mac回传。首推含首次权重加载，但不等于OS/GPU缓存全冷；每格仅3个warm，不据此给稳定p95或SLA。

12/12指采样执行证据、成功终态和媒体契约检查通过，不代表人工质量通过。case-01生成四条均成功后，初始即时空闲检查未通过，具体原因未确认；随后独立两次只读确认服务STOPPED、端口/GPU空闲并保留原失败快照与恢复证据，再继续case-02/03，没有重跑case-01。最终三case均确认关停；之后服务状态仍需实时查询。

## 生命周期

以下命令在GPU主机项目根运行；代码通过项目Git分发。controller检查固定源码及模型大小，客户端在生成前校验四个模型的完整SHA256。开始新case前先确认FastVideo和Comfy均已停止、8108/8188无监听且无GPU计算进程；有未知占用时停止操作，不强杀。默认单卡GPU1，不能与FastVideo双卡服务同时运行。

```bash
envs/comfy-omni/bin/python comfy-omni-deployment/scripts/comfy_control.py start \
  --project-root /home/isaac/workdir/h3 --gpu 1

envs/comfy-omni/bin/python comfy-omni-deployment/scripts/comfy_control.py status \
  --project-root /home/isaac/workdir/h3

envs/comfy-omni/bin/python comfy-omni-deployment/scripts/comfy_control.py stop \
  --project-root /home/isaac/workdir/h3 \
  --expected-owner-token TOKEN_FROM_START_OR_STATUS
```

默认输入目录 `run/comfy-omni/input`，输出目录 `outputs/comfy-omni`。`start --input-dir` 可选择项目的 `inputs` 或默认输入树内的目录；`--output-dir` 可选择 `outputs/comfy-omni` 的子目录。临时文件使用每个owner独立目录，日志写入新的 `logs/comfy-omni/<UTC>-<owner>.log`，状态保存在 `run/comfy-omni/service.json`。既有产物不覆盖。

启动固定关闭自定义节点、浏览器自动打开、采样预览和文件prompt元数据。环境不继承FastVideo、Ray、NCCL、Python/CUDA搜索路径或allocator开关。`READY`要求进程身份、监听socket归属、Comfy版本、单卡可见性及原生Reference节点全部匹配；权重仍在第一条请求加载，READY不代表warm。

停止必须匹配owner token、UID、PID启动时刻、进程组和wrapper参数。正常停止只向自己的wrapper发一次TERM；超时返回`STOPPING`，可以继续查状态。显式`--force`也必须通过身份门禁，才会向该进程组发KILL。身份未建立、PID复用或leader消失但进程组残留时不发信号，并阻止自动替换服务。没有全局`pkill`或自动强杀。

`status --sanitized`输出可公开的状态白名单，省略owner token及日志/配置路径。业务客户端的`--service-receipt`必须使用未脱敏的READY快照；该文件保持私有，客户端报告只保留其hash与必要元数据。

## 同输入对照

[comfy_business_client.py](scripts/comfy_business_client.py)读取已密封的业务协议，只选择cap480计划，保留原prompt、有序参考及配对seed；一组1首推+3warm，每次换seed。它核对采样节点实际执行、未命中整图缓存、history成功终态及完整MP4媒体契约；不自动重试失败请求。实际入口为：

```bash
envs/comfy-omni/bin/python comfy-omni-deployment/scripts/comfy_business_client.py --help
```

执行客户端时必须提供已密封协议及SHA、case ID、[pins.json](pins.json)、本次未脱敏READY receipt和独立输入/输出根；协议及原prompt不进入公开仓库。完整参数以该入口为准，业务原文不作为README示例。

Comfy原生`ref_image_size=match`按输出像素面积缩放，不能等同短边480上限。客户端先按现有FastVideo规则准备短边最多480、保持比例、小图不放大、向32网格下取整的参考图，再用`max`传入原生节点；已处理且32对齐的尺寸不会再次改变。原请求的独立硬首帧能力没有迁移，Picture1参考不能宣称与之等价。

默认是单卡原生对照，没有启用FastVideo SP2。Comfy和FastVideo之间，统一参考几何与seed并不能证明条件编码、颜色处理、数值精度或最终画面完全等价。所有质量图像与音视频只供用户人工审阅，agent不打开媒体。

## 固定pin的API契约

客户端的`SaveVideo`使用展开后的DynamicCombo字段：`format="mp4"`、`format.codec="h264"`、`format.codec.encoding="auto"`；不是`SaveMP4`节点，也不是向API直接传codec嵌套字典。固定pin的history结果位于`outputs[save_node_id].images[]`，条目提供`filename/subfolder/type`并带animated标志。客户端兼容读取`images/videos/gifs`中的MP4，但要求恰好一个、路径归属正确、history成功、完整文件存在及媒体检查通过；多键重复条目不会被默认为多份成功输出。

[pins.json](pins.json)的SaveVideo字段与实际客户端一致。上游Reference节点提供`ref_images.ref_image_0`至`ref_image_8`共9槽；本业务客户端主动限定1–8张图片，按顺序填入前8槽。此限制是客户端契约，不意味着上游只有8槽。

## 本地验证

```bash
python3 -m unittest discover -s comfy-omni-deployment/tests -v
```

生命周期测试使用模拟进程、信号和HTTP接口，不导入Torch或运行GPU任务；业务客户端测试使用模拟协议、事件与媒体metadata。这些测试的证据范围仍是CPU模拟；本页的12/12 GPU运行结果来自独立业务实测，不由单元测试推断。
