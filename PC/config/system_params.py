"""
system_params.py
-------------------
Stores global runtime variables for the PC side:
1. Global yaw value
2. Dynamic update entropy threshold
3. Shared image buffers for 6 cameras
4. Locks
5. Global stop flag
"""

import threading

# -------------------------------------------------------------------
# Global yaw (updated by yaw server)
# -------------------------------------------------------------------
global_yaw = -30.0

# -------------------------------------------------------------------
# Dynamic shannon entropy threshold
# -------------------------------------------------------------------
dynamic_threshold = 0

# -------------------------------------------------------------------
# Shared buffers
# -------------------------------------------------------------------
latest_image = None
global_images = {f"cam{i}": None for i in range(6)}

# -------------------------------------------------------------------
# Locks
# -------------------------------------------------------------------
image_lock = threading.Lock()
global_lock = threading.Lock()

# -------------------------------------------------------------------
# Stop flag (for all PC-side threads)
# -------------------------------------------------------------------
stop_flag = False
