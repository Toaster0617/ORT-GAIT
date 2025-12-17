"""
main_receiver.py
----------------
Entry point of the PC-side system.
Starts:
1. UDP receiver for incoming camera images
2. Yaw server
3. XR image TCP server
4. Main stitching pipeline
"""

import threading
from network.udp_receiver import udp_image_receiver
from network.yaw_server import yaw_server
from network.image_tcp_server import image_tcp_server
from stitching.stitch import main_processing

def main():
    threading.Thread(target=udp_image_receiver, daemon=True).start()
    threading.Thread(target=yaw_server, daemon=True).start()
    threading.Thread(target=image_tcp_server, daemon=True).start()

    main_processing()

if __name__ == "__main__":
    main()
