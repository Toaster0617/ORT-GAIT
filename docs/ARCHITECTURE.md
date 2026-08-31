# ORT-GAIT 工程架构与算法说明

## 1. 设计边界

本工程将原有脚本重构为配置驱动的类化运行时，修复状态共享、首帧死锁、协议错位、无界 UDP 重组和 GPU/CPU 反复传输等工程问题。以下算法保持不变：

1. H0–H5 和 offset 的几何含义及数值；
2. 中心高、边缘低的二维羽化权重，指数 6.5；
3. 灰度帧差、阈值 30、二值图 Shannon entropy；
4. `old_weight / (old_weight + new_weight + ε)` 的融合关系；
5. 按头部 FOV 与相机角度覆盖区求交来选择可见相机。

原 `dynamic_threshold=0` 会让静态帧也通过，因此仅把这个参数修正为配置中的 0.10；检测算法本身没有替换。该阈值必须用实际水下噪声、照明频闪和运动速度数据复标。

## 2. 端到端数据流

```text
RealSense cam0..camN
        │ JPEG(Q=90)
        ▼
IPC: UDP 1200-byte application chunks ──7000/UDP──► PC reassembly/latest frame
        ▲                                                   │
        │                                                   ▼
        └──────── visible camera names ◄──8001/UDP── yaw/FOV selector
                                                            ▲
Quest HeadPoseLogger ────────────────────8084/UDP────────────┘

PC CPU/CUDA stitcher ─► JPEG(Q=80) ─► 4-byte length + JPEG ─8082/TCP─► Quest ImageReceiver
```

工控机启动时默认所有已配置相机可见，因此 PC 能收到全部首帧。PC 从首个合法相机包记录工控机源 IP，随后把可见列表发到该 IP 的 8001 端口。这形成确定的启动握手，不再依赖两个模块中各自写死的全局地址。

## 3. 三个核心问题

### 3.1 径向排列相机的视差与接缝体感

物理上，径向相机的光心不同。近距离目标在相邻相机中的投影差随深度变化，单个平面 H 无法同时对齐不同深度，所以当前算法不能真正“消除视差”。代码实际做了两层缓解：

1. `config.yaml` 中每台相机的 H 将图像放进统一全景坐标。当前 H0–H5 是以水平平移为主的既有标定参数，并没有新增几何模型。
2. `WeightFactory` 生成中心到边缘衰减的二维权重，`CpuBackend`/`CudaBackend` 在重叠区按累计权重羽化。相机边缘通常畸变和标定误差更大，降低边缘贡献可把硬切缝变成渐变，减轻用户转头时的突变感。

因此“无缝”的来源是接缝能量被平滑，而不是三维近景被重新投影到正确深度。若近景双影仍明显，需要另一个经确认的算法项目（深度分层、光流接缝或 3D 重建）；本次按要求没有引入。

另一个必须知晓的既有参数是：H5 的水平起点约 3750，而全景宽度为 4350，所以 cam5 右侧会被裁剪。这两个数值都按原算法保留在 YAML；若这不是预期，需要重新确认标定坐标系和 Quest 所需全景宽度，不能只把画布随意改大。

### 3.2 Quest 观看意图与资源倾斜

`HeadPoseLogger.cs` 每帧发送 Unity 的绝对 yaw。`HeadYawTracker` 第一次收到 yaw 时记录它作为参考方向，逻辑画面从 `base_yaw_deg=-30` 开始：

```text
Unity 初始值 0°       → 逻辑 -30°
相对初始向左 -5°     → 逻辑 -35°
随后向右转 10°，即相对初始 +5° → 逻辑 -25°
```

这个定义避免了原先硬减 114° 的设备相关常量。首次报文是参考零点；如果启动程序前头部已经偏转，系统无法仅凭绝对 yaw 推断那部分历史运动。

`determine_visible_cameras` 用逻辑 yaw 和 80° FOV 与各相机角度区间求交。结果有两处资源作用：

1. PC 的 `VisibilityPublisher` 把列表反馈给工控机，`RealSenseCameraWorker` 只对可见相机取帧、JPEG 编码和发包；
2. PC 的 `PanoramaStitcher.update` 只对可见相机做运动检测和拼接更新。

列表变化时立即发送，相同时每 0.5 秒心跳重发，兼顾转头响应和 UDP 偶发丢失。Quest 图像只在全景实际更新后发送，不重复编码和传输同一帧。用户的注意方向因此同时控制采集、网络、计算和输出刷新。

### 3.3 动态场景的计算资源

对每台可见相机，`MotionDetector` 执行：

1. BGR 转灰度；
2. 与该相机上一帧做绝对差；
3. 差值大于 30 的像素置 255，其余置 0；
4. 计算二值图 Shannon entropy；
5. entropy 达到 0.10 才调用拼接后端。

静态相机不会重复 warp/blend，动态相机才更新其全景区域；不可见相机在工控机端已经停止编码和发送。这是“空间注意（视口）× 时间注意（运动）”的两级资源筛选。它不是目标检测：水体悬浮物、自动曝光和灯光闪烁也可能提高 entropy，所以阈值必须通过现场统计确定。

## 4. CPU/CUDA 等价路径

`create_backend` 在 PC 启动时用 CuPy 分配一个 CUDA 探针：成功则创建 `CudaBackend` 并打印 `当前使用：GPU`，否则创建 `CpuBackend` 并打印 `当前使用：CPU`。

- `CpuBackend`：OpenCV `warpPerspective` + NumPy 羽化。
- `CudaBackend`：CuPy + `cupyx.scipy.ndimage.map_coordinates` 的双线性逆映射，随后在 CuPy 数组上完成权重与融合。

CUDA 路径中，全景、累计权重、相机权重和坐标图均驻留显存。每个新相机帧只做一次 host-to-device 上传；完成全景需要 JPEG 编码时只做一次完整 panorama 的 device-to-host 下载。旧代码对每个 ROI 做多次 `download → cp.asarray → get → upload` 的循环已经移除。CPU 与 GPU 保持相同的 H、ROI、权重和融合公式，但不同库的双线性插值舍入不保证逐像素 bit-exact。

## 5. 模块与类职责

| 文件 / 类 | 职责与物理含义 |
|---|---|
| `main.py` | 唯一入口；选择工控机/PC、相机数量和计算设备。 |
| `config.py` / `AppConfig` 等 dataclass | 从 YAML 加载并校验端口、相机内参外的外部标定参数、采集与阈值；防止奇异 H、重复相机和端口冲突。 |
| `geometry.py` / `HeadYawTracker` | 把 Quest 绝对姿态变成以 −30° 为起点的相对观看方向。 |
| `geometry.py` / `determine_visible_cameras` | 将头部 FOV 投影到 360° 环形相机覆盖区，输出需要服务当前视口的相机。 |
| `motion.py` / `MotionDetector` | 用原始帧差熵度量当前相机画面的时间变化量。 |
| `packet.py` / `FrameReassembler` | 对相机 JPEG 做有界 UDP 分片重组；超时帧被丢弃，避免旧帧拖慢实时链路和内存无限增长。 |
| `state.py` / `SenderState` | 工控机线程安全地共享当前可见相机集合；初值为全部相机。 |
| `state.py` / `ReceiverState` | PC 线程安全地共享最新相机帧、逻辑 yaw、工控机 IP 和最新 Quest JPEG；替代失效的跨模块全局变量。 |
| `network.py` / `UdpCameraSender` | 将单帧 JPEG 切成带序号的小 UDP 数据报。 |
| `network.py` / `UdpCameraReceiver` | 重组并解码最新相机帧，同时学习工控机源 IP。 |
| `network.py` / `VisibilityReceiver` | 工控机接收 PC 的观看意图，只允许配置中存在的相机名。 |
| `network.py` / `VisibilityPublisher` | PC 将可见相机列表反馈给实际图像来源 IP。 |
| `network.py` / `YawUdpReceiver` | 在 8084/UDP 接收、校验 Quest yaw。 |
| `network.py` / `PanoramaTcpServer` | 在 8082/TCP 支持 Quest 重连，严格发送 4 字节长度和 JPEG。 |
| `camera.py` / `RealSenseCameraWorker` | 一台 RealSense 对应一个采集线程；仅在可见时编码和传输。 |
| `backends.py` / `WeightFactory` | 生成并缓存原始 6.5 次幂羽化权重。 |
| `backends.py` / `CpuBackend` | CPU 版固定 H warp、ROI 权重融合和全景累计。 |
| `backends.py` / `CudaBackend` | 数学等价的 CUDA 常驻显存实现。 |
| `stitching.py` / `PanoramaStitcher` | 按配置顺序构建首帧全景，随后只更新“可见且动态”的相机。 |
| `app.py` / `IpcApplication` | 管理反馈监听、RealSense 工作线程和有序退出。 |
| `app.py` / `PcApplication` | 管理四条 PC 服务线程、首帧屏障、在线拼接和 Quest 发布。 |

## 6. 配置与部署约束

1. 两端的 `config.yaml` 相机顺序必须一致，因为 UDP 头发送的是紧凑相机索引。
2. 两端的 `--cam_no` 必须相同；`--cam_no 2` 选择 YAML 中前两台相机及其 H。
3. QuestDemo 当前接收缓冲为 1 MiB。PC 不会发送超过 `max_output_jpeg_bytes` 的帧，以免 Unity 读完长度后不消费 payload 而造成 TCP 帧边界永久错位；超限会记录错误，需调整全景/质量或 Unity 缓冲后再部署。
4. UDP 分片没有可靠传输保证。若链路实测丢包仍高，应先测 MTU、抖动、吞吐和交换机队列，再决定是否设计重传/FEC；那属于协议算法变更，不在本次重构范围。
