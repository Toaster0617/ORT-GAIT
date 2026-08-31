from __future__ import annotations

import logging

import cv2
import numpy as np

from ort_gait.backends import PanoramaBackend
from ort_gait.config import AppConfig
from ort_gait.motion import MotionDetector


LOGGER = logging.getLogger(__name__)


class PanoramaStitcher:
    def __init__(self, config: AppConfig, backend: PanoramaBackend) -> None:
        self._config = config
        self._backend = backend
        self._motion = MotionDetector(config.dynamic)
        self._cameras = {camera.name: camera for camera in config.cameras}
        self._initialized = False

    def initialize(self, images: dict[str, np.ndarray]) -> None:
        for camera in self._config.cameras:
            self._backend.add_image(
                images[camera.name], camera.homography, camera.offset
            )
        self._initialized = True

    def update(
        self,
        images: dict[str, np.ndarray],
        visible_cameras: list[str],
    ) -> list[str]:
        if not self._initialized:
            raise RuntimeError("PanoramaStitcher must be initialized first")
        updated: list[str] = []
        for name in visible_cameras:
            image = images[name]
            is_dynamic, _ = self._motion.is_dynamic(name, image)
            if not is_dynamic:
                continue
            camera = self._cameras[name]
            self._backend.add_image(image, camera.homography, camera.offset)
            updated.append(name)
        return updated

    def snapshot(self) -> np.ndarray:
        return self._backend.snapshot()

    def encode_for_quest(self, panorama: np.ndarray) -> bytes | None:
        ok, encoded = cv2.imencode(
            ".jpg",
            panorama,
            [cv2.IMWRITE_JPEG_QUALITY, self._config.stitch.output_jpeg_quality],
        )
        if not ok:
            LOGGER.error("全景 JPEG 编码失败。")
            return None
        jpeg = encoded.tobytes()
        if len(jpeg) > self._config.stitch.max_output_jpeg_bytes:
            LOGGER.error(
                "全景 JPEG 为 %d 字节，超过 QuestDemo 当前 1 MiB 缓冲限制；该帧未发送。",
                len(jpeg),
            )
            return None
        return jpeg
