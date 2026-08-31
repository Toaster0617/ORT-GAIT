from __future__ import annotations

import threading
import time

import numpy as np

from ort_gait.geometry import HeadYawTracker


class SenderState:
    def __init__(self, camera_names: set[str]) -> None:
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        # All cameras stream until PC feedback arrives, avoiding startup deadlock.
        self._visible = set(camera_names)

    def set_visible(self, camera_names: set[str]) -> None:
        with self._lock:
            self._visible = set(camera_names)

    def is_visible(self, camera_name: str) -> bool:
        with self._lock:
            return camera_name in self._visible

    def visible_snapshot(self) -> set[str]:
        with self._lock:
            return set(self._visible)


class ReceiverState:
    def __init__(self, camera_names: tuple[str, ...], base_yaw_deg: float) -> None:
        self.stop_event = threading.Event()
        self._camera_names = camera_names
        self._images: dict[str, np.ndarray | None] = {
            name: None for name in camera_names
        }
        self._image_condition = threading.Condition()
        self._jpeg_condition = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._jpeg_version = 0
        self._yaw_lock = threading.Lock()
        self._yaw = HeadYawTracker(base_yaw_deg)
        self._peer_lock = threading.Lock()
        self._ipc_ip: str | None = None

    def update_image(self, camera_name: str, image: np.ndarray, peer_ip: str) -> None:
        with self._image_condition:
            self._images[camera_name] = image
            self._image_condition.notify_all()
        with self._peer_lock:
            self._ipc_ip = peer_ip

    def wait_for_all_images(self, timeout_s: float) -> dict[str, np.ndarray] | None:
        deadline = time.monotonic() + timeout_s
        with self._image_condition:
            while not all(image is not None for image in self._images.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.stop_event.is_set():
                    return None
                self._image_condition.wait(remaining)
            return {name: image for name, image in self._images.items() if image is not None}

    def image_snapshot(self) -> dict[str, np.ndarray] | None:
        with self._image_condition:
            if not all(image is not None for image in self._images.values()):
                return None
            return {name: image for name, image in self._images.items() if image is not None}

    def update_unity_yaw(self, unity_yaw_deg: float) -> float:
        with self._yaw_lock:
            return self._yaw.update(unity_yaw_deg)

    def yaw_snapshot(self) -> float:
        with self._yaw_lock:
            assert self._yaw.current_yaw_deg is not None
            return self._yaw.current_yaw_deg

    def ipc_ip_snapshot(self) -> str | None:
        with self._peer_lock:
            return self._ipc_ip

    def publish_jpeg(self, jpeg: bytes) -> None:
        with self._jpeg_condition:
            self._latest_jpeg = jpeg
            self._jpeg_version += 1
            self._jpeg_condition.notify_all()

    def wait_for_jpeg(
        self,
        after_version: int,
        timeout_s: float,
    ) -> tuple[int, bytes | None]:
        with self._jpeg_condition:
            if (
                self._latest_jpeg is None or self._jpeg_version <= after_version
            ) and not self.stop_event.is_set():
                self._jpeg_condition.wait(timeout_s)
            return self._jpeg_version, self._latest_jpeg
