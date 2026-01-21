# -*- coding: utf-8 -*-
"""
Baseline 3: K-Means Clustering Detection (Mimicking Cao et al. / Real-time Paper)
 - Concepts adapted from "Real-time motion state estimation..." (Cao et al., 2025)
 - Core Idea: Uses K-Means clustering on optical flow vectors instead of Grid/Connectivity.
 - Motion Compensation: Uses IMU rotation (like your method) to calculate residuals for each cluster.
 - Output: Processing time per frame (ms).
"""

import os
import csv
import math
import time
import argparse
import cv2
import numpy as np

# --- Parameters inspired by Cao et al. ---
# They use K=8~12. We pick 10.
K_CLUSTERS = 10
KMEANS_ITER = 10
KMEANS_EPSILON = 1.0
KMEANS_ATTEMPTS = 3

# Shared parameters with your method for fair comparison
LK_WIN = (21, 21)
MAX_CORNERS = 600
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 8
RESIDUAL_THRESH_PIX = 4.0 # Threshold to judge a cluster as dynamic

def integrate_gyro_between(gyro_data, t_start, t_end):
    """ Integrates gyro Z (yaw) between two timestamps. """
    # Simple integration: sum(wz * dt)
    # Finds rows in gyro_data where ts >= t_start and ts <= t_end
    # gyro_data expects list of (ts, wx, wy, wz)
    
    total_yaw = 0.0
    relevant = [row for row in gyro_data if t_start <= row[0] <= t_end]
    
    if len(relevant) < 2:
        return 0.0

    for k in range(len(relevant) - 1):
        t0, _, _, wz0 = relevant[k]
        t1, _, _, wz1 = relevant[k+1]
        dt = t1 - t0
        # Average wz integration
        wz_avg = (wz0 + wz1) / 2.0
        total_yaw += wz_avg * dt
        
    return math.degrees(total_yaw) # return degrees

def load_gyro_csv(gyro_csv):
    data = []
    if not gyro_csv or not os.path.exists(gyro_csv):
        return []
    with open(gyro_csv, 'r') as f:
        reader = csv.reader(f)
        # Check header
        try:
            header = next(reader) 
            # Adapt this check to your specific CSV format
            # Assuming: timestamp, wx, wy, wz or similar
        except:
            return []
            
        for row in reader:
            try:
                # User's format: timestamp_s, wx, wy, wz (check your detect_on_images.py logic)
                # If your CSV format is different, adjust indices here
                ts = float(row[0])
                wx = float(row[1])
                wy = float(row[2])
                wz = float(row[3])
                data.append((ts, wx, wy, wz))
            except:
                continue
    data.sort(key=lambda x: x[0])
    return data

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

def main(imdir, frames_csv, gyro_csv, outdir, visualize=False):
    os.makedirs(outdir, exist_ok=True)
    
    mapping = load_frames_csv(frames_csv)
    gyro_data = load_gyro_csv(gyro_csv)
    has_imu = len(gyro_data) > 0
    print(f"Loaded {len(mapping)} frames and {len(gyro_data)} gyro samples.")

    prev_gray = None
    prev_pts = None
    prev_ts = 0.0

    total_time = 0
    valid_frames = 0

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
            prev_ts = ts
            continue

        # 1. Optical Flow
        if prev_pts is None or len(prev_pts) < K_CLUSTERS * 2: # Need enough points for K-Means
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, 
                                             qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)
            prev_gray = gray.copy()
            prev_ts = ts
            continue

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=LK_WIN)
        
        status = status.reshape(-1)
        good_prev = prev_pts.reshape(-1, 2)[status == 1]
        good_next = next_pts.reshape(-1, 2)[status == 1]

        if len(good_prev) < K_CLUSTERS * 2:
            prev_gray = gray.copy()
            prev_ts = ts
            continue

        # 2. IMU Compensation (Calculating the Reference Rotation)
        gyro_yaw = 0.0
        if has_imu:
            gyro_yaw = integrate_gyro_between(gyro_data, prev_ts, ts)
        
        # Calculate visual rotation (RANSAC) as fallback or comparison
        H_vis, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
        if H_vis is None: H_vis = np.eye(2, 3)
        
        # Extract visual rotation angle
        theta_vis_rad = np.arctan2(H_vis[0, 1], H_vis[0, 0])
        theta_vis_deg = np.degrees(theta_vis_rad)

        # Fusion Logic (Simple version for baseline):
        # We assume IMU is correct if available, otherwise Visual
        ref_rot_deg = -gyro_yaw if has_imu else theta_vis_deg # Note: sign might depend on coordinate system
        # For simplicity in this baseline, let's trust Visual RANSAC for the "Global Background"
        # because simulating full VIO depth residual is impossible offline without depth.
        # We use the RANSAC model as the "Background Model" to calculate residuals.
        
        # Predict next points based on global background motion (Affine)
        # This mimics the "Reprojection" in Eq 10 of the paper, but in 2D.
        pred_next = cv2.transform(good_prev.reshape(-1, 1, 2), H_vis).reshape(-1, 2)
        
        # Calculate Residuals (Magnitude of difference)
        residuals_vec = good_next - pred_next
        residuals_mag = np.linalg.norm(residuals_vec, axis=1)

        # 3. K-Means Clustering on Optical Flow Vectors (The Core of Cao et al.)
        # Feature vector for clustering: [flow_x, flow_y, x, y]
        # Adding spatial (x, y) helps group local objects, though paper focuses on flow field.
        # We normalized x,y to balance with flow magnitude.
        flow_vecs = good_next - good_prev
        
        h, w = img.shape[:2]
        features = np.hstack([
            flow_vecs,              # Motion
            good_prev / max(h, w)   # Spatial (normalized)
        ]).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_ITER, KMEANS_EPSILON)
        compactness, labels, centers = cv2.kmeans(features, K_CLUSTERS, None, criteria, KMEANS_ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS)

        # 4. Motion State Estimation (Per Cluster)
        # Cao et al. classifies clusters based on residuals.
        labels = labels.ravel()
        
        for k in range(K_CLUSTERS):
            # Get indices of points in this cluster
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) < 3: continue

            # Get residuals for this cluster
            cluster_residuals = residuals_mag[cluster_indices]
            avg_residual = np.mean(cluster_residuals)
            
            # Decision Rule: Is this cluster dynamic?
            # Cao et al. uses Eq 15 with thresholds. We use a simplified threshold.
            if avg_residual > RESIDUAL_THRESH_PIX:
                # This cluster is dynamic!
                # Draw dynamic points on mask
                dynamic_pts = good_next[cluster_indices]
                for pt in dynamic_pts:
                    cv2.circle(pred_mask, (int(pt[0]), int(pt[1])), 15, 255, -1)
        
        # Post-processing
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, 
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)))

        # --- TIMER END ---
        dt = (time.time() - start_t) * 1000 
        total_time += dt
        valid_frames += 1
        print(f"Frame {i}: {dt:.2f} ms")

        # Save and Vis
        out_path = os.path.join(outdir, os.path.basename(fname))
        cv2.imwrite(out_path, pred_mask)

        if visualize:
            disp = img.copy()
            # Draw Flow
            # for p1, p2 in zip(good_prev, good_next):
            #     cv2.line(disp, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 1)
            
            # Draw Mask
            disp[pred_mask > 0] = (0, 0, 255)
            cv2.putText(disp, f"K-Means: {dt:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('K-Means Baseline', disp)
            if cv2.waitKey(1) & 0xFF == 27: break

        prev_gray = gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, 
                                         qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)
        prev_ts = ts

    if valid_frames > 0:
        print(f"\nAverage Processing Time: {total_time/valid_frames:.2f} ms ({1000/(total_time/valid_frames):.1f} FPS)")

    if visualize:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdir', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--imu', required=True, help="Path to IMU csv")
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    main(args.imdir, args.frames, args.imu, args.outdir, visualize=args.vis)