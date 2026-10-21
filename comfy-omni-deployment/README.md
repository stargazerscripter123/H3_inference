# FastH3 Omni 原生 ComfyUI 对照

这是独立单 GPU 部署包，默认使用 `popos-5090` 的 GPU1 和 loopback `127.0.0.1:8188`。代码与本地 CPU 模拟验证已完成；远端安装、首次生成及性能结果以本轮实际回执为准。本包不会启动或停止 FastVideo。

## 固定配方

- ComfyUI `0.42.0`，commit `b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`；`comfy-kitchen==0.2.39`。不需要自定义节点。
- 独立源码 `src/ComfyUI-Omni`，独立环境 `envs/comfy-omni`；本轮选择 Python3.12.15、Torch2.12.0+cu130、Torchvision0.27.0+cu130。
- 四个模型共35,274,204,555 bytes。具体 Hugging Face revision、文件名、目标目录、大小及 SHA256 全部在 [pins.json](pins.json)。DiT 是 Comfy 专用 NVFP4 QAD；文本编码器是含 vision 的 Qwen3-VL-32B NVFP4 AWQ；视频 VAE 是完整 FP16，音频 VAE 是 FP32。
- 输出宽480×高832、362帧、24fps；Euler / simple、8次 forward、BasicGuider、video/audio shift12/3；无额外 LoRA。

[官方 QAD 验证](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD-Comfy/blob/36dbf15cd11113471114b1040c4f38646c031549/provenance/VALIDATION.md)使用同一 Comfy pin，但硬件为GB200、124帧，Torch为2.9开发版。本轮5090环境必须实际验收，不能引用官方数据作为本机成绩。此处FP16 VAE与FastVideo的36L NVFP4 decoder不同；比较的是完整部署配方，不能把差值全部归因于推理框架。

## 生命周期

以下命令在GPU主机项目根运行；代码通过项目 Git 分发。controller只检查固定源码及模型大小，客户端在生成前校验四个模型的完整SHA256。

```bash
python3 comfy-omni-deployment/scripts/comfy_control.py start \
  --project-root /home/isaac/workdir/h3 --gpu 1

python3 comfy-omni-deployment/scripts/comfy_control.py status \
  --project-root /home/isaac/workdir/h3

python3 comfy-omni-deployment/scripts/comfy_control.py stop \
  --project-root /home/isaac/workdir/h3 \
  --expected-owner-token TOKEN_FROM_START_OR_STATUS
```

默认输入目录 `run/comfy-omni/input`，输出目录 `outputs/comfy-omni`。`start --input-dir` 可选择项目的 `inputs` 或默认输入树内的目录；`--output-dir` 可选择 `outputs/comfy-omni` 的子目录。临时文件使用每个owner独立目录，日志写入新的 `logs/comfy-omni/<UTC>-<owner>.log`，状态保存在 `run/comfy-omni/service.json`。既有产物不覆盖。

启动固定关闭自定义节点、浏览器自动打开、采样预览和文件prompt元数据。环境不继承FastVideo、Ray、NCCL、Python/CUDA搜索路径或allocator开关。`READY`要求进程身份、监听socket归属、Comfy版本、单卡可见性及原生Reference节点全部匹配；权重仍在第一条请求加载，READY不代表warm。

停止必须匹配owner token、UID、PID启动时刻、进程组和wrapper参数。正常停止只向自己的wrapper发一次TERM；超时返回`STOPPING`，可以继续查状态。显式`--force`也必须通过身份门禁，才会向该进程组发KILL。身份未建立、PID复用或leader消失但进程组残留时不发信号，并阻止自动替换服务。没有全局`pkill`或自动强杀。

`status --sanitized`输出可公开的状态白名单，省略owner token及日志/配置路径。业务客户端的`--service-receipt`必须使用未脱敏的READY快照；该文件保持私有，客户端报告只保留其hash与必要元数据。

## 同输入对照

[comfy_business_client.py](scripts/comfy_business_client.py)读取已密封的业务协议，只选择cap480计划，保留原prompt、有序参考及配对seed；一组1首推+3warm，每次换seed。它核对采样节点实际执行、未命中整图缓存、history成功终态及完整MP4媒体契约；不自动重试失败请求。命令参数见 `python3 comfy-omni-deployment/scripts/comfy_business_client.py --help`。

Comfy原生`ref_image_size=match`按输出像素面积缩放，不能等同短边480上限。客户端先按现有FastVideo规则准备短边最多480、保持比例、小图不放大、向32网格下取整的参考图，再用`max`传入原生节点；已处理且32对齐的尺寸不会再次改变。原请求的独立硬首帧能力没有迁移，Picture1参考不能宣称与之等价。

默认是单卡原生对照，没有启用FastVideo SP2。所有质量图像与音视频只供用户人工审阅，agent不打开媒体。速度需分开报告首推和warm，并注明VAE、参考尺寸及完整MP4写盘口径。

## 本地验证

```bash
python3 -m unittest discover -s comfy-omni-deployment/tests -v
```

生命周期测试使用模拟进程、信号和HTTP接口，不导入Torch或运行GPU任务；业务客户端测试使用模拟协议、事件与媒体metadata。通过这些检查不等于GPU生成通过。
