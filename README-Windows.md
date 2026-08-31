# ORT-GAIT Windows 环境配置

推荐 Windows 10/11、64 位 Miniconda/Anaconda、Python 3.11。RealSense 相机所在机器需要安装对应设备驱动。GPU 模式需要 NVIDIA 驱动；requirements 中 CuPy 的 `[ctk]` 组件会安装 CUDA 12 用户态运行库。CPU 模式不需要 NVIDIA GPU。

## 1. 创建 Conda 环境

在仓库根目录打开 Anaconda Prompt 或 PowerShell：

```powershell
conda env create -f environment-windows.yml
conda activate ort-gait
```

该环境文件会读取 `requirements.txt` 并安装固定版本的 NumPy、OpenCV、PyYAML、RealSense Python SDK、CuPy 与 CUDA 12 用户态组件以及 pytest。

验证：

```powershell
python -c "import cv2, numpy, yaml, pyrealsense2; print('runtime OK')"
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
python -m pytest -q
```

第二条命令在纯 CPU 机器上可能报告 CUDA 驱动不可用，这是正常的；程序以 `--device auto` 启动时会选择 CPU。

## 2. 网络与防火墙

确保 `config.yaml` 的 `network.pc_host` 是 PC 面向工控机的网卡 IP。QuestDemo 当前连接 PC 的 `192.168.137.100:8082` 并向 `192.168.137.100:8084` 发 yaw，所以 PC 的 Quest 网卡应保留该地址。

按机器职责放行：

- PC 入站：UDP 7000、UDP 8084、TCP 8082。
- 工控机入站：UDP 8001。

PC 会自动从图像包来源识别工控机 IP，无需另配回传 IP。

## 3. 运行

工控机：

```powershell
conda activate ort-gait
python main.py --role ipc --cam_no 6
```

PC：

```powershell
conda activate ort-gait
python main.py --role pc --cam_no 6 --device auto
```

两相机测试时，两端均改为 `--cam_no 2`。仅 PC 调试时可加 `--preview`。

## 4. GPU 检查

GPU 模式由 CuPy 执行 CUDA warp 与融合，不依赖 PyPI OpenCV 是否带 CUDA。程序启动后必须看到 `当前使用：GPU` 才说明实际进入 GPU 路径。若使用 `--device gpu`，驱动或 CuPy 不可用会明确终止，不会静默切到 CPU。
