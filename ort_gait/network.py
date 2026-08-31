from __future__ import annotations

import logging
import math
import socket
import time

import cv2
import numpy as np

from ort_gait.config import AppConfig
from ort_gait.packet import (
    FrameReassembler,
    PacketError,
    TCP_LENGTH,
    fragment_jpeg,
)
from ort_gait.state import ReceiverState, SenderState


LOGGER = logging.getLogger(__name__)


class UdpCameraSender:
    def __init__(self, config: AppConfig) -> None:
        self._target = (
            config.network.pc_host,
            config.network.camera_udp_port,
        )
        self._chunk_bytes = config.network.udp_chunk_bytes
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, camera_index: int, frame_id: int, jpeg: bytes) -> None:
        for packet in fragment_jpeg(
            camera_index, frame_id, jpeg, self._chunk_bytes
        ):
            self._socket.sendto(packet, self._target)

    def close(self) -> None:
        self._socket.close()


class UdpCameraReceiver:
    def __init__(self, config: AppConfig, state: ReceiverState) -> None:
        self._config = config
        self._state = state
        self._camera_names = tuple(camera.name for camera in config.cameras)
        self._reassembler = FrameReassembler(
            camera_count=len(config.cameras),
            timeout_s=config.network.reassembly_timeout_s,
            max_frame_bytes=config.network.max_camera_jpeg_bytes,
        )

    def run(self) -> None:
        address = (
            self._config.network.bind_host,
            self._config.network.camera_udp_port,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
        sock.settimeout(0.2)
        LOGGER.info("工控机图像 UDP 监听：%s:%d", *address)
        try:
            while not self._state.stop_event.is_set():
                try:
                    packet, peer = sock.recvfrom(65535)
                    completed = self._reassembler.push(packet, peer[0])
                    if completed is None:
                        continue
                    camera_index, jpeg = completed
                    image = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if image is None:
                        LOGGER.warning("丢弃无法解码的 cam%d JPEG。", camera_index)
                        continue
                    self._state.update_image(
                        self._camera_names[camera_index], image, peer[0]
                    )
                except socket.timeout:
                    continue
                except PacketError as exc:
                    LOGGER.debug("丢弃非法 UDP 图像包：%s", exc)
        finally:
            sock.close()


class VisibilityReceiver:
    def __init__(self, config: AppConfig, state: SenderState) -> None:
        self._config = config
        self._state = state
        self._allowed = {camera.name for camera in config.cameras}

    def run(self) -> None:
        address = (
            self._config.network.bind_host,
            self._config.network.visibility_udp_port,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
        sock.settimeout(0.2)
        LOGGER.info("可见相机反馈 UDP 监听：%s:%d", *address)
        try:
            while not self._state.stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(1024)
                except socket.timeout:
                    continue
                try:
                    decoded = data.decode("ascii").strip()
                except UnicodeDecodeError:
                    continue
                visible = set(filter(None, decoded.split(",")))
                if not visible.issubset(self._allowed):
                    LOGGER.warning("忽略含未知相机的反馈：%s", decoded)
                    continue
                self._state.set_visible(visible)
        finally:
            sock.close()


class VisibilityPublisher:
    """Publish view intent to the IPC IP learned from incoming camera packets."""

    def __init__(self, config: AppConfig, state: ReceiverState) -> None:
        self._port = config.network.visibility_udp_port
        self._heartbeat_s = config.runtime.visibility_heartbeat_s
        self._state = state
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_message: bytes | None = None
        self._last_sent_at = 0.0

    def publish(self, camera_names: list[str]) -> None:
        target_ip = self._state.ipc_ip_snapshot()
        if target_ip is None:
            return
        message = ",".join(camera_names).encode("ascii")
        now = time.monotonic()
        if message == self._last_message and now - self._last_sent_at < self._heartbeat_s:
            return
        self._socket.sendto(message, (target_ip, self._port))
        self._last_message = message
        self._last_sent_at = now

    def close(self) -> None:
        self._socket.close()


class YawUdpReceiver:
    def __init__(self, config: AppConfig, state: ReceiverState) -> None:
        self._config = config
        self._state = state

    def run(self) -> None:
        address = (
            self._config.network.bind_host,
            self._config.network.quest_yaw_udp_port,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
        sock.settimeout(0.2)
        LOGGER.info("Quest 头姿 UDP 监听：%s:%d", *address)
        try:
            while not self._state.stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(64)
                except socket.timeout:
                    continue
                try:
                    yaw = float(data.decode("ascii"))
                    if math.isfinite(yaw):
                        self._state.update_unity_yaw(yaw)
                except (UnicodeDecodeError, ValueError):
                    LOGGER.debug("忽略非法 Quest yaw 数据。")
        finally:
            sock.close()


class PanoramaTcpServer:
    """Unity-compatible 4-byte length + JPEG TCP stream."""

    def __init__(self, config: AppConfig, state: ReceiverState) -> None:
        self._config = config
        self._state = state

    def run(self) -> None:
        address = (
            self._config.network.bind_host,
            self._config.network.quest_image_tcp_port,
        )
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(address)
        server.listen(1)
        server.settimeout(0.5)
        LOGGER.info("Quest 图像 TCP 服务：%s:%d", *address)
        try:
            while not self._state.stop_event.is_set():
                try:
                    connection, peer = server.accept()
                except socket.timeout:
                    continue
                LOGGER.info("Quest 已连接：%s:%d", *peer)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    self._serve_connection(connection)
                except (ConnectionError, OSError) as exc:
                    LOGGER.warning("Quest 连接结束：%s", exc)
                finally:
                    connection.close()
        finally:
            server.close()

    def _serve_connection(self, connection: socket.socket) -> None:
        version = -1
        while not self._state.stop_event.is_set():
            new_version, jpeg = self._state.wait_for_jpeg(version, 0.5)
            if new_version == version or jpeg is None:
                continue
            if len(jpeg) > self._config.stitch.max_output_jpeg_bytes:
                LOGGER.error("拒绝发送超过 Quest 缓冲限制的 JPEG。")
                version = new_version
                continue
            # Do not insert timestamps here: ImageReceiver.cs expects JPEG now.
            connection.sendall(TCP_LENGTH.pack(len(jpeg)))
            connection.sendall(jpeg)
            version = new_version
