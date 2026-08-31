# ORT-GAIT

ORT-GAIT 是面向径向多相机与 Quest VR 的实时全景系统。工控机采集 RealSense 彩色图像并通过 UDP 分片发送；PC 端根据 Quest 头部朝向选择相机、进行动态区域更新和羽化拼接，再以 Unity 已有协议通过 TCP 发送 JPEG 全景。

本次工程化重写保留了原有的 H0–H5、offset、中心到边缘的 6.5 次幂权重、帧差二值化、Shannon entropy 和加权融合公式。`Pseudo-Event` 目录未改动。

## 快速运行

先按平台完成环境配置：

- [Windows 配置说明](README-Windows.md)
- [Ubuntu 配置说明](README-Ubuntu.md)

修改 `config.yaml` 中的 PC 相机网卡地址、RealSense 序列号和标定矩阵。工控机和 PC 必须使用同一份相机顺序与相同的 `--cam_no`。

六相机：

```bash
python main.py --role ipc --cam_no 6
python main.py --role pc --cam_no 6
```

两相机：

```bash
python main.py --role ipc --cam_no 2
python main.py --role pc --cam_no 2
```

PC 端默认 `--device auto`：CUDA 可用时输出 `当前使用：GPU`，否则输出 `当前使用：CPU`。也可显式指定：

```bash
python main.py --role pc --cam_no 2 --device cpu
python main.py --role pc --cam_no 2 --device gpu
```

`--device gpu` 在 CUDA 不可用时会直接报错；`auto` 才会自动选择 CPU。电脑调试窗口默认关闭，按需增加 `--preview`，它不会改变发往 Quest 的图像。

## 四条网络链路

| 方向 | 协议 | 默认端口 | 用途 |
|---|---:|---:|---|
| 工控机 → PC | UDP | 7000 | 相机 JPEG 应用层分片 |
| PC → 工控机 | UDP | 8001 | 当前可见相机列表 |
| Quest → PC | UDP | 8084 | Unity 绝对 yaw |
| PC → Quest | TCP | 8082 | `4-byte big-endian length + JPEG` |

PC 会从 7000 端口收到的图像包自动学习工控机 IP，并将 8001 反馈发回该 IP，因此不再写死回传地址。`config.yaml` 的 `network.pc_host` 只表示工控机发送相机数据时使用的 PC 相机网卡地址。

QuestDemo 无需修改。当前场景中的 `ImageReceiver` 连接 `192.168.137.100:8082`，`HeadPoseLogger` 向 `192.168.137.100:8084` 发送 yaw；PC 端绑定 `0.0.0.0`，只要该 PC 网卡仍使用 `192.168.137.100` 即可连接。

## 协议边界

工控机相机链路将 JPEG 切成默认 1200 字节的数据块，每块带有相机号、帧号、总块数和块序号。这避免单个大 UDP 数据报被常见 MTU 再次分片；不完整帧在 0.25 秒后丢弃，在线系统继续追最新帧。分片本身不提供重传或 FEC，块数增加也不意味着丢包概率必然下降。

Quest 图像链路是 TCP 字节流。Unity 现有 `ImageReceiver.cs` 在 4 字节长度后立即读取 JPEG，因此这里不能插入 8 字节 timestamp。TCP 会自行进行网络分段，不需要改变应用层帧格式。

## 文档与测试

- [工程架构与算法说明](docs/ARCHITECTURE.md)
- 运行测试：`python -m pytest -q`
- 检查配置：`python -c "from ort_gait.config import load_config; print(load_config('config.yaml', 2))"`
