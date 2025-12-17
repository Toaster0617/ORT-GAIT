"""
blender.py
----------
Handles CuPy-based GPU blending of warped images into the panorama.
"""

import cv2
import cupy as cp
import numpy as np
from stitching.weights import get_gpu_weight, single_weights_matrix

def add_image_gpu(pano_gpu, img, H, offset, weights_gpu, pano_size):
    if pano_gpu is None:
        pano_gpu = cv2.cuda_GpuMat()
        pano_gpu.upload(np.zeros((pano_size[1], pano_size[0], 3), np.uint8))
        weights_gpu = cv2.cuda_GpuMat()
        weights_gpu.upload(np.zeros((pano_size[1], pano_size[0], 3), np.float32))
    else:
        pano_gpu = cv2.cuda.warpPerspective(pano_gpu, offset, pano_size)
        weights_gpu = cv2.cuda.warpPerspective(weights_gpu, offset, pano_size)

    h, w = img.shape[:2]
    corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]]).T
    warped = H @ corners
    warped /= warped[2]

    xs, ys = warped[0], warped[1]
    xmin = max(int(xs.min())-1, 0)
    ymin = max(int(ys.min())-1, 0)
    xmax = min(int(xs.max())+1, pano_size[0])
    ymax = min(int(ys.max())+1, pano_size[1])

    roi_w, roi_h = xmax-xmin, ymax-ymin
    if roi_w <= 0 or roi_h <= 0:
        return pano_gpu, weights_gpu

    T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)
    H_local = T @ H

    pano_roi = pano_gpu.rowRange(ymin, ymax).colRange(xmin, xmax)
    w_roi = weights_gpu.rowRange(ymin, ymax).colRange(xmin, xmax)

    gimg = cv2.cuda_GpuMat(); gimg.upload(img)
    warped_img = cv2.cuda.warpPerspective(gimg, H_local, (roi_w, roi_h))

    shape = img.shape[:2]
    w_gpu = get_gpu_weight(shape)
    if w_gpu is None:
        w_mat = single_weights_matrix(shape)
        w3 = np.repeat(w_mat[:,:,None], 3, axis=2).astype(np.float32)
        w_gpu = cv2.cuda_GpuMat(); w_gpu.upload(w3)

    warped_w = cv2.cuda.warpPerspective(w_gpu, H_local, (roi_w, roi_h))

    pano_cp = cp.asarray(pano_roi.download())
    new_cp = cp.asarray(warped_img.download())
    w_cp = cp.asarray(w_roi.download())
    w_new = cp.asarray(warped_w.download())

    norm = w_cp[:,:,0] / (w_cp[:,:,0] + w_new[:,:,0] + 1e-6)
    norm = norm[...,None]

    blended = new_cp*(1-norm) + pano_cp*norm
    blended = cp.clip(blended, 0,255).astype(cp.uint8)

    combined = w_cp + w_new
    mx = combined.max()
    combined = combined/mx if mx != 0 else combined
    combined = combined.astype(cp.float32)

    pano_roi.upload(blended.get())
    w_roi.upload(combined.get())

    return pano_gpu, weights_gpu
