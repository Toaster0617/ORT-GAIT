"""
warp.py
-----------
GPU warpPerspective wrapper using OpenCV CUDA.
"""

import cv2
import numpy as np

def warp_perspective_gpu(image, H, size):
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    H32 = np.array(H, dtype=np.float32)
    return cv2.cuda.warpPerspective(gpu_img, H32, size)
