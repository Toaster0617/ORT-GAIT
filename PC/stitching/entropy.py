"""
entropy.py
----------
Computes Shannon entropy with CuPy for dynamic update filtering.
"""

import cupy as cp

def shannon_entropy_gpu(image):
    flat = cp.asarray(image.flatten())
    hist, _ = cp.histogram(flat, bins=256, range=(0,256), density=True)
    hist = hist[hist > 0]
    return float((-hist * cp.log2(hist)).sum())
