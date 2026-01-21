"""
main_offline.py
----------------
Offline stitching entry point.
Features:
1. Allows direct modification of Homography matrices (H0-H5) in this file.
2. Reads input images from the 'figure' subdirectory.
3. SAVES the result image into the 'figure' subdirectory.
4. Uses English comments.
"""

import cv2
import os
import numpy as np

# 1. Import necessary configuration and modules
# We import camera_params to get default offsets and sizes, 
# but we will OVERRIDE the H matrices with the ones defined below.
from config.camera_configs import camera_params, PANORAMA_SIZE

# 2. Import core stitching algorithms
from stitching.blender import add_image_gpu
from stitching.weights import precompute_all_weights


PANORAMA_SIZE = (5040, 740)

def main_offline():
    print("[Offline] Starting offline stitching process...")

    # =========================================================================
    # PART 1: Custom Homography Matrices
    # You can modify the values here directly to tune the stitching.
    # =========================================================================
    
    # Cam 0
    custom_H0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Cam 1
    custom_H1 = np.array([
        [1.0, 0.0, 760.0],
        [0.0, 1.0, 10.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Cam 2
    custom_H2 = np.array([
        [1.0, 0.0, 1510.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Cam 3
    custom_H3 = np.array([
        [1.0, 0.0, 2250.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Cam 4
    custom_H4 = np.array([
        [1.0, 0.0, 2969.2424],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Cam 5
    custom_H5 = np.array([
        [1.0, 0.0, 3750.2424],
        [0.0, 1.0, -10.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Store them in a list for easy iteration
    custom_matrices = [custom_H0, custom_H1, custom_H2, custom_H3, custom_H4, custom_H5]

    # =========================================================================
    # PART 2: Load Images from 'figure' Folder
    # =========================================================================
    
    # Define the image directory
    img_dir = "figure"
    
    # Check if directory exists
    if not os.path.exists(img_dir):
        print(f"[Error] Directory '{img_dir}' not found. Please create it and add cam0.jpg ~ cam5.jpg.")
        return

    images = {}
    shapes = []
    
    for i in range(6):
        filename = f"cam{i}.jpg"
        file_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[Error] File {file_path} not found.")
            return
        
        print(f"[Offline] Loading {file_path}...")
        img = cv2.imread(file_path)
        
        if img is None:
            print(f"[Error] Failed to read {file_path}.")
            return
            
        key = f"cam{i}"
        images[key] = img
        shapes.append(img.shape[:2])

    # =========================================================================
    # PART 3: Precompute GPU Weights
    # =========================================================================
    # This step generates the alpha blending masks on the GPU.
    print("[Offline] Precomputing GPU weights...")
    precompute_all_weights(shapes)

    # =========================================================================
    # PART 4: Stitching Loop
    # =========================================================================
    pano_gpu = None
    weights_gpu = None

    # Iterate through cameras 0 to 5
    for i in range(6):
        cam_key = f"cam{i}"
        print(f"[Offline] Stitching {cam_key} using CUSTOM matrix...")
        
        img = images[cam_key]
        
        # Retrieve default params (like offset) from config
        default_params = camera_params[cam_key]
        
        # Use our LOCALLY defined H matrix instead of the one in config
        current_H = custom_matrices[i]

        # Call the GPU blending function
        # Arguments: (current_pano, new_image, H_matrix, offset_matrix, current_weights, output_size)
        pano_gpu, weights_gpu = add_image_gpu(
            pano_gpu,
            img,
            current_H,              # <--- Using the custom H defined in this file
            default_params["offset"],
            weights_gpu,
            PANORAMA_SIZE
        )

    # =========================================================================
    # PART 5: Save and Display Result
    # =========================================================================
    if pano_gpu is not None:
        print("[Offline] Downloading result from GPU...")
        final_pano = pano_gpu.download()
        
        # Construct the output path inside the 'figure' directory
        output_filename = "offline_result_custom.jpg"
        output_path = os.path.join(img_dir, output_filename)
        
        # Save the image
        cv2.imwrite(output_path, final_pano)
        print(f"[Success] Panorama saved to {output_path}")
        
        # Optional: Display the result window
        # Resize for display if the screen is too small
        display_h, display_w = final_pano.shape[:2]
        if display_w > 1920:
             scale = 1920 / display_w
             display_img = cv2.resize(final_pano, None, fx=scale, fy=scale)
        else:
             display_img = final_pano
             
        cv2.imshow("Offline Panorama (Custom H)", display_img)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[Error] Stitching failed, result is None.")

if __name__ == "__main__":
    main_offline()