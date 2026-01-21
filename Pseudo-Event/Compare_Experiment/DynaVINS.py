# -*- coding: utf-8 -*-
"""
Baseline 1: Geometry-based Detection (Mimicking DynaVINS / Visual SLAM Front-end)
 - Uses Epipolar Geometry (Fundamental Matrix) to reject dynamic outliers.
 - Does NOT rely on IMU (Pure Vision).
 - Output: Processing time per frame (ms).Average Processing Time: 2.38 ms (420.8 FPS)
"""

import os
import csv
import argparse
import time
import cv2
import numpy as np

# Parameters
MAX_CORNERS = 600
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
LK_WIN = (21, 21)
RANSAC_THRESH = 1.0  # Pixel error threshold for epipolar constraint
CONFIDENCE = 0.99
DILATE_KERNEL = (5, 5) # Dilate more to create a mask from sparse points

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
    prev_pts = None
    
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
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, 
                                             qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)
            # Init frame doesn't count for processing speed
            continue

        # 1. Optical Flow Tracking
        if prev_pts is None or len(prev_pts) < 8:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, 
                                             qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)
            prev_gray = gray.copy()
            continue

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=LK_WIN)
        
        status = status.reshape(-1)
        good_prev = prev_pts.reshape(-1, 2)[status == 1]
        good_next = next_pts.reshape(-1, 2)[status == 1]

        if len(good_prev) < 8:
            prev_gray = gray.copy()
            continue

        # 2. Epipolar Geometry Constraint (The Core of "Geometry Baseline")
        # Compute Fundamental Matrix F using RANSAC
        F, mask = cv2.findFundamentalMat(good_prev, good_next, cv2.FM_RANSAC, 
                                         RANSAC_THRESH, CONFIDENCE)

        if mask is not None:
            mask = mask.ravel()
            # Inliers (1) = Static Background
            # Outliers (0) = Dynamic Objects
            outlier_indices = np.where(mask == 0)[0]
            
            # 3. Generate Mask from Outliers
            # Draw circles at outlier positions
            for idx in outlier_indices:
                pt = good_next[idx]
                # Draw a filled circle for the dynamic point
                cv2.circle(pred_mask, (int(pt[0]), int(pt[1])), 15, 255, -1)
            
            # Post-processing to fill gaps
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, 
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)))
        
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
            disp[pred_mask > 0] = (0, 0, 255) # Red mask
            cv2.putText(disp, f"{dt:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Geometry Baseline', disp)
            if cv2.waitKey(1) & 0xFF == 27: break

        # Update for next frame
        prev_gray = gray.copy()
        # Redetect features regularly to maintain density
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, 
                                         qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)

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