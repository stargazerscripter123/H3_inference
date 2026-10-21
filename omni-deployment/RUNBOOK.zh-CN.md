# popos-5090 操作入口

当前实际根目录为 `/home/isaac/workdir/h3`。这是 FastVideo 原生 FastH3
Omni NVFP4 QAD，独立于旧 ComfyUI/vLLM 产线。最终验收范围和限制见
[README](README.md) 与 [validation](validation/)。人工画面/声音检查仍需用户完成。

目标机env文件默认启用已验证的 `OMNI_VAE_OFFLOAD=1` 和
`OMNI_EXPANDABLE_SEGMENTS=1`，用于多参考长视频。要复现实验中的原生常驻VAE，
请在source前显式把两者export为0；切换时先stop。

## 启动双卡服务

```bash
ssh popos-5090
cd /home/isaac/workdir/h3
source omni-deployment/env.popos-5090.sh
"$OMNI_PYTHON" omni-deployment/scripts/omni_control.py start sp2
"$OMNI_PYTHON" omni-deployment/scripts/omni_control.py status
```

`start` 返回 `STARTING`，必须在 `status` 的 JSON 中确认 `READY`。READY
表示两个 worker 的存活和模型身份已核对，首次生成仍可能需要加载/JIT。
同配置健康服务会复用；切换配置前先用下方 stop。API 仅监听本机 `127.0.0.1:8108`。

## 一条图像参考请求

下面使用已校验的 Tears of Steel 授权素材。8 次 PDD forward、24 fps、
832×480、32 kHz 双声道；`--frames 124` 对应约 5.17 秒视频。

```bash
"$OMNI_PYTHON" omni-deployment/scripts/run_omni.py \
  --image /home/isaac/workdir/h3/inputs/tos/tos_first.png \
  --prompt 'Keep the appearance of <Picture 1>. The character turns slowly toward the camera. Natural quiet ambient sound.' \
  --frames 124 --seed 1234 \
  --output-root /home/isaac/workdir/h3/outputs/omni-manual
```

客户端会输出唯一目录、完整 MP4 和 `manifest.json`，并校验视频帧数、画布、
帧率、音频采样率与声道。数值校验不代表人工质量通过。再次测试请换 seed。
图像、视频、音频参数按类型依次排列，同类保持顺序；可重复 `--image`，
再传 `--video`，最后 `--audio`。参考文件必须位于当前允许的 `inputs/` 内。
最长 362 帧约 15.08 秒；它是输出时长，不能与生成延迟混淆。多参考和长视频
的显存边界以实测验收记录为准。

## 停止和排查

```bash
"$OMNI_PYTHON" omni-deployment/scripts/omni_control.py stop
```

控制器只停止本服务拥有的 PID/进程组；不要使用全局 `ray stop`。
`run/omni/service.json` 指向当前独立日志。正常停止超时后先查看日志和进程身份，
必要时才使用控制器的 `stop --force`；强停后应核对私有 Ray worker 和显存已释放。
客户端超时不会自动取消服务器任务，不能直接重发同一请求。

## 测量和人工检查

正式测试关闭 `OMNI_SP_PROFILE`（默认 0），从新服务开始，执行
[benchmarks](benchmarks/README.md) 中显式预热/测量计划。诊断模式 1 会增加
CUDA event/synchronize，不能和正式成绩混用。失败、未运行和成功分开记录。

质量抽帧只保存于目标机，由用户查看；agent 不打开图片。已有准备/SP1产物的
抽帧索引为 `outputs/omni-quality-002/manifest.json`，其中标明源视频和帧路径。
