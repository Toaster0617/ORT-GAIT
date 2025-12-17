"""
image_tcp_server.py
-------------------
Streams the latest stitched panorama to the VR/XR client via TCP.
"""

import socket
import time
import struct
from config.system_params import latest_image, image_lock, stop_flag
from config.camera_configs import XR_IP, XR_PORT

def image_tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((XR_IP, XR_PORT))
    server.listen(1)

    print(f"[PC] XR image TCP server listening on port {XR_PORT}")

    conn, _ = server.accept()

    while not stop_flag:
        with image_lock:
            if latest_image is not None:
                header = struct.pack('!I', len(latest_image))
                try:
                    conn.sendall(header + latest_image)
                except:
                    break
        time.sleep(1/30)

    conn.close()
    server.close()
