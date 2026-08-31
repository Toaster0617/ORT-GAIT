from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2

from ort_gait.backends import PanoramaBackend
from ort_gait.camera import RealSenseCameraWorker
from ort_gait.config import AppConfig
from ort_gait.geometry import determine_visible_cameras
from ort_gait.network import (
    PanoramaTcpServer,
    UdpCameraReceiver,
    UdpCameraSender,
    VisibilityPublisher,
    VisibilityReceiver,
    YawUdpReceiver,
)
from ort_gait.state import ReceiverState, SenderState
from ort_gait.stitching import PanoramaStitcher


LOGGER = logging.getLogger(__name__)


def _start_thread(name: str, target: Callable[[], None]) -> threading.Thread:
    thread = threading.Thread(name=name, target=target, daemon=True)
    thread.start()
    return thread


class IpcApplication:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._state = SenderState({camera.name for camera in config.cameras})
        self._sender = UdpCameraSender(config)

    def run(self) -> None:
        threads = [
            _start_thread(
                "visibility-receiver",
                VisibilityReceiver(self._config, self._state).run,
            )
        ]
        for index, camera in enumerate(self._config.cameras):
            worker = RealSenseCameraWorker(
                index, camera, self._config, self._state, self._sender
            )
            threads.append(_start_thread(f"camera-{camera.name}", worker.run))
        LOGGER.info("IPC 已启动，共 %d 台相机。", len(self._config.cameras))
        try:
            while not self._state.stop_event.wait(1.0):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self._state.stop_event.set()
            for thread in threads:
                thread.join(timeout=2.0)
            self._sender.close()


class PcApplication:
    def __init__(self, config: AppConfig, backend: PanoramaBackend) -> None:
        self._config = config
        names = tuple(camera.name for camera in config.cameras)
        self._state = ReceiverState(names, config.runtime.base_yaw_deg)
        self._stitcher = PanoramaStitcher(config, backend)

    def run(self) -> None:
        receiver = UdpCameraReceiver(self._config, self._state)
        yaw_receiver = YawUdpReceiver(self._config, self._state)
        tcp_server = PanoramaTcpServer(self._config, self._state)
        visibility = VisibilityPublisher(self._config, self._state)
        threads = [
            _start_thread("camera-receiver", receiver.run),
            _start_thread("yaw-receiver", yaw_receiver.run),
            _start_thread("quest-tcp-server", tcp_server.run),
        ]
        try:
            images = self._wait_for_initial_images()
            if images is None:
                return
            self._stitcher.initialize(images)
            self._publish_panorama()
            LOGGER.info("初始全景已完成，进入在线更新。")
            self._processing_loop(visibility)
        except KeyboardInterrupt:
            pass
        finally:
            self._state.stop_event.set()
            visibility.close()
            for thread in threads:
                thread.join(timeout=2.0)
            if self._config.runtime.preview:
                cv2.destroyAllWindows()

    def _wait_for_initial_images(self) -> dict | None:
        LOGGER.info("等待所有已配置相机的首帧……")
        while not self._state.stop_event.is_set():
            images = self._state.wait_for_all_images(timeout_s=1.0)
            if images is not None:
                return images
            LOGGER.info("仍在等待相机首帧。")
        return None

    def _processing_loop(self, visibility: VisibilityPublisher) -> None:
        period = 1.0 / self._config.runtime.processing_fps
        while not self._state.stop_event.is_set():
            started = time.monotonic()
            yaw = self._state.yaw_snapshot()
            visible = determine_visible_cameras(
                yaw, self._config.runtime.headset_fov_deg, self._config.cameras
            )
            visibility.publish(visible)
            images = self._state.image_snapshot()
            if images is not None:
                updated = self._stitcher.update(images, visible)
                if updated:
                    self._publish_panorama()

            if self._config.runtime.preview:
                cv2.waitKey(1)
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                self._state.stop_event.wait(remaining)

    def _publish_panorama(self) -> None:
        panorama = self._stitcher.snapshot()
        if self._config.runtime.preview:
            cv2.imshow("ORT-GAIT panorama (debug only)", panorama)
        jpeg = self._stitcher.encode_for_quest(panorama)
        if jpeg is not None:
            self._state.publish_jpeg(jpeg)
