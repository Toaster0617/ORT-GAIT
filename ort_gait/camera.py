from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ort_gait.config import AppConfig, CameraConfig
from ort_gait.network import UdpCameraSender
from ort_gait.state import SenderState


LOGGER = logging.getLogger(__name__)


class RealSenseCameraWorker:
    def __init__(
        self,
        camera_index: int,
        camera: CameraConfig,
        config: AppConfig,
        state: SenderState,
        sender: UdpCameraSender,
    ) -> None:
        self._index = camera_index
        self._camera = camera
        self._config = config
        self._state = state
        self._sender = sender

    def run(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError:
            LOGGER.exception("缺少 pyrealsense2，无法启动相机。")
            return

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self._camera.serial)
        rs_config.enable_stream(
            rs.stream.color,
            self._config.capture.width,
            self._config.capture.height,
            rs.format.bgr8,
            self._config.capture.fps,
        )
        try:
            pipeline.start(rs_config)
        except Exception:
            LOGGER.exception(
                "RealSense %s (%s) 启动失败。",
                self._camera.name,
                self._camera.serial,
            )
            return

        frame_id = 0
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY,
            self._config.capture.jpeg_quality,
        ]
        LOGGER.info("相机已启动：%s (%s)", self._camera.name, self._camera.serial)
        try:
            while not self._state.stop_event.is_set():
                if not self._state.is_visible(self._camera.name):
                    time.sleep(0.01)
                    continue
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as exc:
                    LOGGER.warning("%s 采集超时：%s", self._camera.name, exc)
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                image = np.asanyarray(color_frame.get_data())
                ok, encoded = cv2.imencode(".jpg", image, encode_params)
                if not ok:
                    LOGGER.warning("%s JPEG 编码失败。", self._camera.name)
                    continue
                jpeg = encoded.tobytes()
                if len(jpeg) > self._config.network.max_camera_jpeg_bytes:
                    LOGGER.error("%s JPEG 超过安全上限，已丢弃。", self._camera.name)
                    continue
                self._sender.send(self._index, frame_id, jpeg)
                frame_id = (frame_id + 1) & 0xFFFFFFFF
        finally:
            pipeline.stop()
            LOGGER.info("相机已停止：%s", self._camera.name)
