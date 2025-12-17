"""
camera_configs.py
---------------------
This file stores all camera-related configurations for the IPC(device),
including:
1. RealSense camera serial numbers
2. Assigned cam keys (cam0~cam5)
3. JPEG encoding quality and UDP packet size
"""

# -------------------------------------------------------------------
# RealSense camera serial numbers and cam keys
# These are required ONLY on IPC (sender side)
# -------------------------------------------------------------------
CAMERAS = [
    ('338122301918', 'cam0'),
    ('338522301241', 'cam1'),
    ('309622301481', 'cam2'),
    ('309622301066', 'cam3'),
    ('242422304700', 'cam4'),
    ('338122300640', 'cam5'),
]

# -------------------------------------------------------------------
# JPEG Encoding quality (sender side)
# -------------------------------------------------------------------
JPEG_QUALITY = 90

# cv2.imencode parameters
ENCODE_PARAMS = [int(1), JPEG_QUALITY]  # 1 = cv2.IMWRITE_JPEG_QUALITY

# -------------------------------------------------------------------
# Max UDP payload (per datagram)
# -------------------------------------------------------------------
MAX_DGRAM_PAYLOAD = 60000
