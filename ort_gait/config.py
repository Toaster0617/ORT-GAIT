from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised when config.yaml cannot describe a valid runtime."""


@dataclass(frozen=True)
class CameraConfig:
    name: str
    serial: str
    angle_range_deg: tuple[float, float]
    homography: np.ndarray
    offset: np.ndarray


@dataclass(frozen=True)
class NetworkConfig:
    pc_host: str
    bind_host: str
    camera_udp_port: int
    visibility_udp_port: int
    quest_image_tcp_port: int
    quest_yaw_udp_port: int
    udp_chunk_bytes: int
    reassembly_timeout_s: float
    max_camera_jpeg_bytes: int


@dataclass(frozen=True)
class CaptureConfig:
    width: int
    height: int
    fps: int
    jpeg_quality: int


@dataclass(frozen=True)
class StitchConfig:
    panorama_size: tuple[int, int]
    weight_exponent: float
    output_jpeg_quality: int
    max_output_jpeg_bytes: int


@dataclass(frozen=True)
class DynamicConfig:
    pixel_difference_threshold: int
    entropy_threshold: float


@dataclass
class RuntimeConfig:
    base_yaw_deg: float
    headset_fov_deg: float
    processing_fps: float
    visibility_heartbeat_s: float
    preview: bool


@dataclass(frozen=True)
class AppConfig:
    network: NetworkConfig
    capture: CaptureConfig
    stitch: StitchConfig
    dynamic: DynamicConfig
    runtime: RuntimeConfig
    cameras: tuple[CameraConfig, ...]


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"配置项 {key!r} 必须是映射。")
    return value


def _matrix(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ConfigError(f"{label} 必须是有限数值组成的 3x3 矩阵。")
    if abs(float(np.linalg.det(array))) < 1e-8:
        raise ConfigError(f"{label} 不可逆。")
    return array


def _port(value: Any, label: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ConfigError(f"{label} 必须在 1..65535。")
    return port


def _quality(value: Any, label: str) -> int:
    quality = int(value)
    if not 1 <= quality <= 100:
        raise ConfigError(f"{label} 必须在 1..100。")
    return quality


def load_config(path: str | Path, cam_no: int | None = None) -> AppConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 格式错误：{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml 顶层必须是映射。")

    n = _mapping(raw, "network")
    c = _mapping(raw, "capture")
    s = _mapping(raw, "stitch")
    d = _mapping(raw, "dynamic")
    r = _mapping(raw, "runtime")

    ports = {
        "camera_udp_port": _port(n["camera_udp_port"], "camera_udp_port"),
        "visibility_udp_port": _port(
            n["visibility_udp_port"], "visibility_udp_port"
        ),
        "quest_image_tcp_port": _port(
            n["quest_image_tcp_port"], "quest_image_tcp_port"
        ),
        "quest_yaw_udp_port": _port(
            n["quest_yaw_udp_port"], "quest_yaw_udp_port"
        ),
    }
    if len(set(ports.values())) != 4:
        raise ConfigError("四条网络链路必须使用四个不同端口。")

    network = NetworkConfig(
        pc_host=str(n["pc_host"]),
        bind_host=str(n["bind_host"]),
        **ports,
        udp_chunk_bytes=int(n["udp_chunk_bytes"]),
        reassembly_timeout_s=float(n["reassembly_timeout_s"]),
        max_camera_jpeg_bytes=int(n["max_camera_jpeg_bytes"]),
    )
    if not 256 <= network.udp_chunk_bytes <= 60_000:
        raise ConfigError("udp_chunk_bytes 必须在 256..60000。")
    if network.reassembly_timeout_s <= 0:
        raise ConfigError("reassembly_timeout_s 必须大于 0。")

    capture = CaptureConfig(
        width=int(c["width"]),
        height=int(c["height"]),
        fps=int(c["fps"]),
        jpeg_quality=_quality(c["jpeg_quality"], "capture.jpeg_quality"),
    )
    if min(capture.width, capture.height, capture.fps) <= 0:
        raise ConfigError("采集宽、高和帧率必须大于 0。")

    pano_size = tuple(int(v) for v in s["panorama_size"])
    if len(pano_size) != 2 or min(pano_size) <= 0:
        raise ConfigError("stitch.panorama_size 必须是正数 [width, height]。")
    stitch = StitchConfig(
        panorama_size=pano_size,
        weight_exponent=float(s["weight_exponent"]),
        output_jpeg_quality=_quality(
            s["output_jpeg_quality"], "stitch.output_jpeg_quality"
        ),
        max_output_jpeg_bytes=int(s["max_output_jpeg_bytes"]),
    )
    if stitch.weight_exponent <= 0 or stitch.max_output_jpeg_bytes <= 0:
        raise ConfigError("拼接权重指数和最大输出字节数必须大于 0。")

    dynamic = DynamicConfig(
        pixel_difference_threshold=int(d["pixel_difference_threshold"]),
        entropy_threshold=float(d["entropy_threshold"]),
    )
    if not 0 <= dynamic.pixel_difference_threshold <= 255:
        raise ConfigError("pixel_difference_threshold 必须在 0..255。")
    if dynamic.entropy_threshold < 0:
        raise ConfigError("entropy_threshold 不能为负数。")

    runtime = RuntimeConfig(
        base_yaw_deg=float(r["base_yaw_deg"]),
        headset_fov_deg=float(r["headset_fov_deg"]),
        processing_fps=float(r["processing_fps"]),
        visibility_heartbeat_s=float(r["visibility_heartbeat_s"]),
        preview=bool(r["preview"]),
    )
    if not 0 < runtime.headset_fov_deg <= 360:
        raise ConfigError("headset_fov_deg 必须在 (0, 360]。")
    if runtime.processing_fps <= 0 or runtime.visibility_heartbeat_s <= 0:
        raise ConfigError("处理帧率和反馈心跳周期必须大于 0。")

    camera_rows = raw.get("cameras")
    if not isinstance(camera_rows, list) or not camera_rows:
        raise ConfigError("cameras 必须是非空列表。")
    cameras: list[CameraConfig] = []
    for index, row in enumerate(camera_rows):
        if not isinstance(row, dict):
            raise ConfigError(f"cameras[{index}] 必须是映射。")
        angle = tuple(float(v) for v in row["angle_range_deg"])
        if len(angle) != 2:
            raise ConfigError(f"cameras[{index}].angle_range_deg 必须有两个值。")
        cameras.append(
            CameraConfig(
                name=str(row["name"]),
                serial=str(row["serial"]),
                angle_range_deg=angle,
                homography=_matrix(row["homography"], f"{row['name']}.homography"),
                offset=_matrix(row["offset"], f"{row['name']}.offset"),
            )
        )

    names = [camera.name for camera in cameras]
    serials = [camera.serial for camera in cameras]
    if len(set(names)) != len(names) or len(set(serials)) != len(serials):
        raise ConfigError("相机 name 和 serial 必须分别唯一。")
    if cam_no is not None:
        if not 1 <= cam_no <= len(cameras):
            raise ConfigError(f"--cam_no 必须在 1..{len(cameras)}。")
        cameras = cameras[:cam_no]

    return AppConfig(
        network=network,
        capture=capture,
        stitch=stitch,
        dynamic=dynamic,
        runtime=runtime,
        cameras=tuple(cameras),
    )
