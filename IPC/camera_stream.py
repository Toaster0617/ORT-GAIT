"""
camera_stream.py
----------------
Defines per-camera threads for capturing RealSense color frames,
JPEG encoding, fragmentation, and UDP transmission to the PC.
"""

import time
import socket
import threading
import numpy as np
import cv2
import pyrealsense2 as rs

from config.camera_configs import (
    CAMERAS, JPEG_QUALITY, ENCODE_PARAMS, MAX_DGRAM_PAYLOAD
)
from config.network_params import DEST_IP, DEST_PORT

from visibility_listener import get_current_visible

# Shared UDP socket for all camera threads
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def camera_thread(serial, cam_key):
    """
    Captures frames from a RealSense camera, encodes JPEG,
    splits into UDP packets, and sends to PC.
    """

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    frame_id = 0
    print(f"[IPC] Camera thread started: {cam_key}")

    try:
        while True:

            # Skip cameras that are not in the visible set
            visible = get_current_visible()
            if cam_key not in visible:
                time.sleep(0.01)
                continue

            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                print(f"[IPC] {cam_key} frame capture error: {e}")
                continue

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            # JPEG encoding
            ok, buf = cv2.imencode('.jpg', img, ENCODE_PARAMS)
            if not ok:
                continue

            data = buf.tobytes()
            total_len = len(data)
            num_chunks = (total_len + MAX_DGRAM_PAYLOAD - 1) // MAX_DGRAM_PAYLOAD

            # Fragment and send
            for idx in range(num_chunks):
                start = idx * MAX_DGRAM_PAYLOAD
                end   = min(start + MAX_DGRAM_PAYLOAD, total_len)
                chunk = data[start:end]

                # Header format: "cam_key,frame_id,num_chunks,idx|"
                header = f"{cam_key},{frame_id},{num_chunks},{idx}|".encode('utf-8')
                udp_sock.sendto(header + chunk, (DEST_IP, DEST_PORT))

            frame_id = (frame_id + 1) & 0xFFFFFFFF

    finally:
        pipeline.stop()
        print(f"[IPC] Camera stopped: {cam_key}")
