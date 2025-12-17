"""
camera_configs.py
--------------------
This file stores camera-related parameters for the PC(receiver side),
including:
1. Homography matrices for 6 cameras
2. Per-camera offset transforms
3. Panorama output size
4. Camera angular ranges (used for visible cam selection)
5. PC-side networking parameters
"""

import numpy as np
import math

# -------------------------------------------------------------------
# Homography matrices
# -------------------------------------------------------------------
H0 = np.array([[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]], dtype=np.float32)

H1 = np.array([[1.0, 0.0, 760.0],
               [0.0, 1.0, 10.0],
               [0.0, 0.0, 1.0]], dtype=np.float32)

H2 = np.array([[1.0, 0.0, 1510.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]], dtype=np.float32)

H3 = np.array([[1.0, 0.0, 2250.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]], dtype=np.float32)

H4 = np.array([[1.0, 0.0, 2969.2424],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]], dtype=np.float32)

H5 = np.array([[1.0, 0.0, 3750.2424],
               [0.0, 1.0, -10.0],
               [0.0, 0.0, 1.0]], dtype=np.float32)

OFFSET = np.eye(3, dtype=np.float32)

# camera parameter dictionary
camera_params = {
    "cam0": {"H": H0, "offset": OFFSET, "size": (6500, 810)},
    "cam1": {"H": H1, "offset": OFFSET, "size": (6500, 810)},
    "cam2": {"H": H2, "offset": OFFSET, "size": (6500, 810)},
    "cam3": {"H": H3, "offset": OFFSET, "size": (6500, 810)},
    "cam4": {"H": H4, "offset": OFFSET, "size": (6500, 810)},
    "cam5": {"H": H5, "offset": OFFSET, "size": (6500, 810)},
}

# -------------------------------------------------------------------
# Panorama output size (width, height)
# -------------------------------------------------------------------
PANORAMA_SIZE = (4350, 740)

# -------------------------------------------------------------------
# Camera angular ranges for visible camera selection
# -------------------------------------------------------------------
cam_ranges = {
    "cam0": (math.radians(-180), math.radians(-120)),
    "cam1": (math.radians(-120), math.radians(-60)),
    "cam2": (math.radians(-60),  math.radians(0)),
    "cam3": (math.radians(0),    math.radians(60)),
    "cam4": (math.radians(60),   math.radians(120)),
    "cam5": (math.radians(120),  math.radians(180)),
}

# -------------------------------------------------------------------
# PC Networking Parameters
# -------------------------------------------------------------------
IMAGE_RECV_PORT = 7000   # Receives camera JPEG chunks
SEND_BACK_IP    = '10.192.17.244'
SEND_BACK_PORT  = 8001   # Sends visible cam list

YAW_IP   = '0.0.0.0'
YAW_PORT = 8083

XR_IP   = '0.0.0.0'
XR_PORT = 8082

# FOV for visible camera selection
DEFAULT_FOV = 80
