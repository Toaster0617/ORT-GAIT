"""
visibility_feedback.py
-----------------------
Sends visible camera IDs back to IPC over UDP.
"""

import socket
from config.camera_configs import SEND_BACK_IP, SEND_BACK_PORT

def send_visible_cams(cam_list):
    msg = ",".join(cam_list).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg, (SEND_BACK_IP, SEND_BACK_PORT))
    sock.close()
