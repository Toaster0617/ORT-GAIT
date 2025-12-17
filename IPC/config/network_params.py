"""
network_params.py
---------------------
Stores all networking parameters for the IPC(device):
1. Destination PC IP and ports for sending images
2. UDP port for receiving visibility feedback
"""

# -------------------------------------------------------------------
# PC IP address and port to send camera JPEG images
# -------------------------------------------------------------------
DEST_IP   = '10.192.17.244'
DEST_PORT = 7000

# -------------------------------------------------------------------
# IPC listens to visibility feedback from PC
# -------------------------------------------------------------------
VISIBILITY_PORT = 8001
