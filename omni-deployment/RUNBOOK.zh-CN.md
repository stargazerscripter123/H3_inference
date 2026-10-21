# popos-5090 操作入口

当前实际根目录为 `/home/isaac/workdir/h3`。这是 FastVideo 原生 FastH3
Omni NVFP4 QAD，独立于旧 ComfyUI/vLLM 产线。最终验收范围和限制见
[README](README.md) 与 [业务评测协议](benchmarks/README.md)。人工画面/声音检查仍需用户完成。

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

## 业务请求与三组参考图

旧测试用例已移除。真实请求和素材保存在私有、Git忽略的运行目录，不提交到公开仓库。
保持原prompt与参考图顺序；三组分别设置cap480、cap720和original，均输出
480×832、362帧/24fps、8次PDD、32kHz双声道。720上限经32像素对齐实际为704；
小图不放大。original保留源尺寸，仅做必要对齐，不采用官方2048放大。

首次使用新策略前，停止服务并显式应用[版本补丁](SERVICE.md#reference-image-size-policy)。
启动前设置`OMNI_REFERENCE_IMAGE_RESIZE_POLICY=cap`与
`OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE=480`或`720`；原图组使用`original`与`0`。
默认target env为cap480，改变策略必须stop/start。

每个case/策略fresh服务，1条首推+3条warm，原prompt固定、seed轮换。
计时至完整MP4写盘，模型启动和回传Mac另计；失败、未运行不计为零延迟。
独立首帧conditioning与普通reference不等价，迁移请求时明确记录这个能力差异。

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

质量抽帧只保存于目标机，由用户查看；agent 不打开图片。新业务视频与统计另存新目录，旧用例不再参与评测。
