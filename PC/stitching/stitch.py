"""
stitch.py
-----------
Main processing pipeline: waits for frames, precomputes weights,
warps & blends images, handles visible cams, and updates panorama.
"""

import time
import numpy as np
import cv2
import math

from config.camera_configs import camera_params, cam_ranges, PANORAMA_SIZE, DEFAULT_FOV
from config.system_params import global_images, global_lock, latest_image, image_lock, global_yaw, dynamic_threshold
from network.visibility_feedback import send_visible_cams
from utils.angle_utils import determine_visible_cams
from stitching.weights import precompute_all_weights
from stitching.blender import add_image_gpu
from stitching.entropy import shannon_entropy_gpu

def main_processing():
    print("[PC] Waiting for first frames...")

    # Wait for all cameras
    while True:
        with global_lock:
            ready = all(global_images[cam] is not None for cam in global_images)
        if ready:
            break
        time.sleep(0.05)

    with global_lock:
        init_imgs = {cam: global_images[cam] for cam in global_images}

    shapes = [img.shape[:2] for img in init_imgs.values()]
    precompute_all_weights(shapes)

    # First warping
    pano_gpu, w_gpu = None, None

    for cam, params in camera_params.items():
        pano_gpu, w_gpu = add_image_gpu(
            pano_gpu,
            init_imgs[cam],
            params["H"],
            params["offset"],
            w_gpu,
            PANORAMA_SIZE
        )

    print("[PC] Main stitching pipeline started.")

    prev_frames = {cam: None for cam in camera_params}

    while True:
        with global_lock:
            cur_imgs = {cam: global_images[cam] for cam in global_images}
            yaw_rad = math.radians(global_yaw)

        if any(img is None for img in cur_imgs.values()):
            time.sleep(0.01)
            continue

        visible = determine_visible_cams(yaw_rad, DEFAULT_FOV, cam_ranges)
        send_visible_cams(visible)

        # Dynamic blending only visible cams
        for cam in visible:
            img = cur_imgs[cam]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if prev_frames[cam] is None:
                prev_frames[cam] = gray
                continue

            diff = cv2.absdiff(prev_frames[cam], gray)
            _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            ent = shannon_entropy_gpu(mask)
            prev_frames[cam] = gray

            if ent < dynamic_threshold:
                continue

            params = camera_params[cam]
            pano_gpu, w_gpu = add_image_gpu(
                pano_gpu, img, params["H"], params["offset"], w_gpu, PANORAMA_SIZE
            )

        pano = pano_gpu.download()
        cv2.imshow("panorama", pano)
        ret, buf = cv2.imencode('.jpg', pano, [1, 80])
        if ret:
            with image_lock:
                latest_image = buf.tobytes()

        cv2.waitKey(1)
