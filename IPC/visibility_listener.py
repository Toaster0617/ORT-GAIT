"""
visibility_listener.py
----------------------
Listens on a UDP port for visible camera IDs sent from the PC.
The IPC uses this information to decide which cameras should stream.
"""

import socket
import threading
from config.network_params import VISIBILITY_PORT

# Shared state for visible cameras
current_visible = set()
current_visible_lock = threading.Lock()

def get_current_visible():
    """Safely return a copy of the current visible camera set."""
    with current_visible_lock:
        return set(current_visible)

def visibility_listener():
    """UDP server to listen for visible cam IDs from PC."""
    global current_visible
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', VISIBILITY_PORT))

    print(f"[IPC] Listening for visibility feedback on UDP port {VISIBILITY_PORT}")

    while True:
        data, _ = sock.recvfrom(1024)
        decoded = data.decode('utf-8').strip()
        if not decoded:
            continue

        new_set = set(decoded.split(','))
        with current_visible_lock:
            current_visible = new_set

    sock.close()
