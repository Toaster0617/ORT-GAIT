"""
udp_receiver.py
----------------
Receives fragmented JPEG packets from IPC for all six cameras,
reassembles them, and stores them into global_images buffer.
"""

import socket
import numpy as np
import cv2
from config.system_params import global_images, global_lock
from config.camera_configs import IMAGE_RECV_PORT
from config.system_params import stop_flag

def udp_image_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', IMAGE_RECV_PORT))
    print(f"[PC] Listening for camera frames on UDP port {IMAGE_RECV_PORT}")

    buffer_dict = {}

    while not stop_flag:
        packet, _ = sock.recvfrom(65536)

        try:
            header, chunk = packet.split(b'|', 1)
            cam_key, frame_id_s, total_s, idx_s = header.decode().split(',')
            frame_id = int(frame_id_s)
            total    = int(total_s)
            idx      = int(idx_s)

            key = (cam_key, frame_id)

            entry = buffer_dict.get(key)
            if entry is None:
                entry = {"total": total, "chunks": {}, "count": 0}
                buffer_dict[key] = entry

            # Avoid duplicated chunks
            if idx not in entry["chunks"]:
                entry["chunks"][idx] = chunk
                entry["count"] += 1

            # Reassemble full JPEG
            if entry["count"] == entry["total"]:
                jpeg_bytes = b''.join(entry["chunks"][i] for i in range(total))
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if img is not None:
                    with global_lock:
                        global_images[cam_key] = img

                del buffer_dict[key]

        except Exception as e:
            print("[PC] UDP receive error:", e)

    sock.close()
