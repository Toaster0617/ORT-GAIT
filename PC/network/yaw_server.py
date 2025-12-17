"""
yaw_server.py
--------------
Receives yaw angle from Unity/VR client through UDP.
Updates the global_yaw shared variable.
"""

import socket
from config.camera_configs import YAW_IP, YAW_PORT
from config.system_params import global_yaw, global_lock, stop_flag

def yaw_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((YAW_IP, YAW_PORT))
    print(f"[PC] Listening for yaw on UDP port {YAW_PORT}")

    while not stop_flag:
        data, _ = sock.recvfrom(64)
        try:
            deg = float(data.decode())
            with global_lock:
                global global_yaw
                global_yaw = deg - 114  # Same logic as original PC.py
                if global_yaw > 180:
                    global_yaw -= 360
                if global_yaw < -180:
                    global_yaw += 360
        except:
            pass

    sock.close()
