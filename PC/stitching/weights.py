"""
weights.py
----------
Computes and caches blending weight matrices used for image stitching.
"""

import numpy as np
import cv2
from config.system_params import stop_flag

_weights_cache = {}
_gpu_weights_cache = {}

def single_weights_array(size):
    """Generates a 1D weight array from center to edges."""
    if size % 2 == 1:
        return np.concatenate([
            np.linspace(0,1,(size+1)//2),
            np.linspace(1,0,(size+1)//2)[1:]
        ])
    else:
        return np.concatenate([
            np.linspace(0,1,size//2),
            np.linspace(1,0,size//2)
        ])

def single_weights_matrix(shape, method='exponent', exponent=6.5):
    """2D weight map for image blending."""
    if shape not in _weights_cache:
        h, w = shape
        wr = single_weights_array(h)[:, None]
        wc = single_weights_array(w)[None, :]
        mat = (wr @ wc) ** exponent
        _weights_cache[shape] = mat.astype(np.float32)
    return _weights_cache[shape]

def precompute_all_weights(shapes):
    """Uploads all weight matrices to GPU."""
    for shape in shapes:
        if shape not in _gpu_weights_cache:
            mat = single_weights_matrix(shape)
            w3 = np.repeat(mat[:, :, None], 3, axis=2).astype(np.float32)
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(w3)
            _gpu_weights_cache[shape] = gpu_mat

def get_gpu_weight(shape):
    """Returns cached GPU weight map."""
    return _gpu_weights_cache.get(shape)
