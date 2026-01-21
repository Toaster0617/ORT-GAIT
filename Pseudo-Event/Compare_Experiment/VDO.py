# -*- coding: utf-8 -*-
"""
Baseline 2: Dense Flow-based Detection (Mimicking VDO-SLAM Motion Estimation)
 - Uses Dense Optical Flow (Farneback) instead of Sparse KLT.
 - Compensates Ego-motion using global affine transform.
 - Output: Processing time per frame (ms).Average Processing Time: 77.44 ms (12.9 FPS)
"""

import os
import csv
import argparse
import time
import cv2
import numpy as np

# Parameters
PYR_SCALE = 0.5
LEVELS = 3
WINSZ = 15
ITERATIONS = 3
POLY_N = 5
POLY_SIGMA = 1.2
FLAGS = 0

MAG_THRESHOLD = 2.0  # Pixel movement threshold for dynamic objects
MIN_AREA = 100       # Minimum area to keep

def load_frames_csv(frames_csv):
    mapping = []
    with open(frames_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            idx = int(row[0])
            ts = float(row[1])
            fname = row[2]
            mapping.append((idx, ts, fname))
    mapping.sort()
    return mapping

def main(imdir, frames_csv, outdir, visualize=False):
    os.makedirs(outdir, exist_ok=True)
    mapping = load_frames_csv(frames_csv)

    prev_gray = None
    
    total_time = 0
    valid_frames = 0
    
    print(f"Processing {len(mapping)} frames...")

    for i, ts, fname in mapping:
        path = os.path.join(imdir, fname)
        img = cv2.imread(path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        pred_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # --- TIMER START ---
        start_t = time.time()

        if prev_gray is None:
            prev_gray = gray.copy()
            continue

        # 1. Compute Dense Optical Flow (Simulating PWC-Net in VDO-SLAM)
        # This is usually the bottleneck
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                          PYR_SCALE, LEVELS, WINSZ, 
                                          ITERATIONS, POLY_N, POLY_SIGMA, FLAGS)

        # 2. Estimate Global Ego-motion (Background Model)
        # Grid sample to speed up affine estimation (using all pixels is too slow)
        h, w = flow.shape[:2]
        step = 10
        y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1)
        fx, fy = flow[y, x].T
        
        # Select valid flows (remove extreme outliers before fitting)
        mag = np.hypot(fx, fy)
        valid = mag < 50 # Ignore crazy large flows
        if np.sum(valid) > 10:
            src_pts = np.stack([x[valid], y[valid]], axis=1).astype(np.float32)
            dst_pts = src_pts + np.stack([fx[valid], fy[valid]], axis=1).astype(np.float32)
            
            # Estimate Affine Transform (Background Motion)
            H_aff, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            
            if H_aff is None: H_aff = np.eye(2, 3)
            
            # 3. Create Predicted Background Flow Field
            # Create a grid of all pixels
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            grid_pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)
            
            # Transform all pixels using H_aff
            warped_pts = cv2.transform(grid_pts, H_aff).reshape(h, w, 2)
            
            # Background Flow = Warped_Pos - Original_Pos
            bg_flow = warped_pts - np.stack([grid_x, grid_y], axis=-1)
            
            # 4. Compute Residual Flow (Object Motion)
            # Residual = Measured_Flow - Background_Flow
            diff_flow = flow - bg_flow
            diff_mag = np.sqrt(diff_flow[..., 0]**2 + diff_flow[..., 1]**2)
            
            # 5. Thresholding
            pred_mask[diff_mag > MAG_THRESHOLD] = 255
            
            # Post-processing (Open & Close)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # --- TIMER END ---
        dt = (time.time() - start_t) * 1000 # ms
        total_time += dt
        valid_frames += 1
        print(f"Frame {i}: {dt:.2f} ms")

        # Save and Visualize
        out_path = os.path.join(outdir, os.path.basename(fname))
        cv2.imwrite(out_path, pred_mask)

        if visualize:
            disp = img.copy()
            disp[pred_mask > 0] = (0, 0, 255)
            cv2.putText(disp, f"{dt:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Dense Baseline', disp)
            if cv2.waitKey(1) & 0xFF == 27: break

        prev_gray = gray.copy()

    if valid_frames > 0:
        print(f"\nAverage Processing Time: {total_time/valid_frames:.2f} ms ({1000/(total_time/valid_frames):.1f} FPS)")

    if visualize:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdir', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    main(args.imdir, args.frames, args.outdir, visualize=args.vis)