"""
main_sender.py
---------------
Main entry for IPC-side application.
Starts:
1. Visibility feedback listener thread
2. Camera streaming threads for all RealSense devices
"""

import threading
import time

from config.camera_configs import CAMERAS
from visibility_listener import visibility_listener
from camera_stream import camera_thread

def main():
    # Start visibility listener
    threading.Thread(target=visibility_listener, daemon=True).start()

    # Start camera threads
    threads = []
    for serial, cam_key in CAMERAS:
        t = threading.Thread(target=camera_thread, args=(serial, cam_key), daemon=True)
        t.start()
        threads.append(t)

    print("[IPC] All camera threads running.")
    print("[IPC] Sending images to PC...")

    # Main loop (keep alive)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[IPC] Interrupted. Closing IPC sender...")

if __name__ == "__main__":
    main()
