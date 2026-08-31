from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import socket
import threading
import time

import cv2
import numpy as np

from ort_gait.config import AppConfig, load_config
from ort_gait.network import (
    PanoramaTcpServer,
    UdpCameraReceiver,
    UdpCameraSender,
    VisibilityPublisher,
    VisibilityReceiver,
    YawUdpReceiver,
)
from ort_gait.state import ReceiverState, SenderState


CONFIG = Path(__file__).parents[1] / "config.yaml"


def _free_port(socket_type: int) -> int:
    with socket.socket(socket.AF_INET, socket_type) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_config() -> AppConfig:
    config = load_config(CONFIG, cam_no=1)
    network = replace(
        config.network,
        pc_host="127.0.0.1",
        bind_host="127.0.0.1",
        camera_udp_port=_free_port(socket.SOCK_DGRAM),
        visibility_udp_port=_free_port(socket.SOCK_DGRAM),
        quest_image_tcp_port=_free_port(socket.SOCK_STREAM),
        quest_yaw_udp_port=_free_port(socket.SOCK_DGRAM),
    )
    return replace(config, network=network)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _wait_until(predicate, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not reached")
        time.sleep(0.01)


def test_udp_camera_sender_to_receiver_chain() -> None:
    config = _local_config()
    state = ReceiverState(("cam0",), -30.0)
    receiver = UdpCameraReceiver(config, state)
    thread = threading.Thread(target=receiver.run, daemon=True)
    thread.start()
    time.sleep(0.05)

    sender = UdpCameraSender(config)
    image = np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    sender.send(0, 7, encoded.tobytes())

    images = state.wait_for_all_images(1.0)
    state.stop_event.set()
    thread.join(1.0)
    sender.close()
    assert images is not None
    assert images["cam0"].shape == image.shape
    assert state.ipc_ip_snapshot() == "127.0.0.1"


def test_panorama_tcp_server_matches_unity_stream_contract() -> None:
    config = _local_config()
    state = ReceiverState(("cam0",), -30.0)
    server = PanoramaTcpServer(config, state)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    deadline = time.monotonic() + 1.0
    while True:
        try:
            client.connect(("127.0.0.1", config.network.quest_image_tcp_port))
            break
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)

    jpeg = b"\xff\xd8quest-frame\xff\xd9"
    state.publish_jpeg(jpeg)
    length = int.from_bytes(_recv_exact(client, 4), "big")
    assert length == len(jpeg)
    assert _recv_exact(client, length) == jpeg

    state.stop_event.set()
    client.close()
    thread.join(1.0)


def test_yaw_udp_receiver_applies_initial_reference() -> None:
    config = _local_config()
    state = ReceiverState(("cam0",), -30.0)
    receiver = YawUdpReceiver(config, state)
    thread = threading.Thread(target=receiver.run, daemon=True)
    thread.start()
    time.sleep(0.05)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        target = ("127.0.0.1", config.network.quest_yaw_udp_port)
        sender.sendto(b"0", target)
        time.sleep(0.05)
        sender.sendto(b"355", target)
        _wait_until(lambda: state.yaw_snapshot() == -35.0)
        sender.sendto(b"5", target)
        _wait_until(lambda: state.yaw_snapshot() == -25.0)

    state.stop_event.set()
    thread.join(1.0)


def test_visibility_feedback_returns_to_learned_ip() -> None:
    config = _local_config()
    ipc_state = SenderState({"cam0"})
    pc_state = ReceiverState(("cam0",), -30.0)
    fake_image = np.zeros((2, 2, 3), dtype=np.uint8)
    pc_state.update_image("cam0", fake_image, "127.0.0.1")

    receiver = VisibilityReceiver(config, ipc_state)
    publisher = VisibilityPublisher(config, pc_state)
    thread = threading.Thread(target=receiver.run, daemon=True)
    thread.start()
    time.sleep(0.05)
    publisher.publish([])
    _wait_until(lambda: ipc_state.visible_snapshot() == set())

    ipc_state.stop_event.set()
    thread.join(1.0)
    publisher.close()
