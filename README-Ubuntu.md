# ORT-GAIT Ubuntu 环境配置

推荐 Ubuntu 22.04/24.04 LTS、Python 3.11。GPU 模式需要 NVIDIA 驱动；requirements 中 CuPy 的 `[ctk]` 组件会安装 CUDA 12 用户态运行库。CPU 模式不需要 NVIDIA GPU。

## 1. RealSense 系统支持

工控机需要先安装 RealSense 设备权限与系统驱动。按 [librealsense 官方 Linux 安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md) 安装 `librealsense2-dkms` 和 `librealsense2-utils`，重新插拔相机后用 `realsense-viewer` 验证。PC 若不接 RealSense，可跳过这一系统步骤。

OpenCV 桌面 wheel 需要系统 OpenGL 运行库：

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

## 2. 创建 Conda 环境

```bash
conda env create -f environment-ubuntu.yml
conda activate ort-gait
```

该环境文件会读取 `requirements.txt`，安装全部固定版本 Python 依赖。

验证：

```bash
python -c "import cv2, numpy, yaml, pyrealsense2; print('runtime OK')"
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
python -m pytest -q
```

纯 CPU 机器上的第二条命令可能报告 CUDA 驱动不可用；程序以 `--device auto` 运行时会正常选择 CPU。CuPy 官方提供 Linux/Windows 的 `cupy-cuda12x` wheel，详见 [CuPy 安装说明](https://docs.cupy.dev/en/stable/install.html)。

## 3. 网络与运行

防火墙方向：

- PC 入站：UDP 7000、UDP 8084、TCP 8082。
- 工控机入站：UDP 8001。

工控机：

```bash
conda activate ort-gait
python main.py --role ipc --cam_no 6
```

PC：

```bash
conda activate ort-gait
python main.py --role pc --cam_no 6 --device auto
```

两相机测试时，两端均使用 `--cam_no 2`。无桌面的 Ubuntu PC 不要启用 `--preview`；这不影响 Quest 图像输出。
