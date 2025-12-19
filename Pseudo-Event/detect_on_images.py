# -*- coding: utf-8 -*-
"""
Offline detector that runs the motion_fused pipeline on saved image frames and IMU log.
 - Reads frames from folder and timestamps from frames.csv (produced by record_realsense.py)
 - Reads IMU gyro CSV (timestamp_s, wx, wy, wz) and integrates wz between frames to get gyro yaw
 - Runs the same clustering+fusion logic and saves predicted binary masks to an output folder

Usage:
  python detect_on_images.py --imdir ./rec --imu ./rec/imu_gyro.csv --frames ./rec/frames.csv --outdir ./rec_preds

Note: frames.csv should contain rows: frame_idx, timestamp_s, filename
"""

import os
import csv
import math
import time
import argparse
import cv2
import numpy as np

# copy parameters from motion_fused
MAG_THRESH_PIX = 0.2
CELL_MAG_THRESH = 2.0
MIN_CELL_AREA = 2
GRID_W, GRID_H = 40, 30

LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
RANSAC_REPROJ_THRESH = 5.0
QUALITY_LEVEL = 0.1
MAX_CORNERS = 600
MIN_DISTANCE = 7
DILATE_KERNEL = (2, 2)

RESIDUAL_THRESH_PIX = 2.1
IMU_ANGLE_DIFF_THRESH_DEG = 7
TRUST_ROTATION_DISAGREE_DEG = 12
ALPHA_GYRO = 0.6
INLIER_RATIO_THRESH = 0.45
MIN_POINTS_IN_CLUSTER = 7
MIN_MAG_FOR_ANGLE = 0.4

def wrap_angle_deg(a):
    return (a + 180) % 360 - 180

def angle_diff_deg(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)

def avg_angle_rad(a):
    if len(a) == 0:
        return 0.0
    s = np.sum(np.sin(a))
    c = np.sum(np.cos(a))
    return math.atan2(s, c)

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

def load_imu_csv(imu_csv):
    imu = []
    with open(imu_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            ts = float(row[0])
            wx = float(row[1]); wy = float(row[2]); wz = float(row[3])
            imu.append((ts, wx, wy, wz))
    imu.sort()
    return imu

def integrate_gyro_between(imu, t0, t1):
    # integrate wz over samples between t0 and t1 (simple trapezoid approx)
    if t1 <= t0:
        return 0.0
    total = 0.0
    # find samples in interval
    prev_t = None
    prev_wz = None
    for ts, wx, wy, wz in imu:
        if ts < t0:
            prev_t = ts; prev_wz = wz
            continue
        if ts > t1:
            # handle last segment
            if prev_t is not None:
                dt = t1 - prev_t
                total += prev_wz * dt
            break
        if prev_t is None:
            prev_t = ts; prev_wz = wz
            continue
        # integrate segment prev_t -> ts with prev_wz
        dt = ts - prev_t
        total += prev_wz * dt
        prev_t = ts; prev_wz = wz
    return total  # in rad*s -> but actually wz is rad/s, integration yields rad

def main(imdir, frames_csv, imu_csv, outdir, visualize=False):
    os.makedirs(outdir, exist_ok=True)
    mapping = load_frames_csv(frames_csv)
    imu = load_imu_csv(imu_csv) if imu_csv is not None else []

    prev_gray = None
    prev_pts = None
    gyro_yaw = 0.0
    gyro_yaw_filt = 0.0

    for i, ts, fname in mapping:
        path = os.path.join(imdir, fname)
        img = cv2.imread(path) 
        if img is None:
            print('missing', path); continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # compute gyro yaw delta between previous frame time and this frame
        if imu and prev_gray is not None:
            # need previous timestamp
            prev_ts = prev_ts_global
            delta_rad = integrate_gyro_between(imu, prev_ts, ts)
            gyro_yaw += math.degrees(delta_rad)
            gyro_yaw = wrap_angle_deg(gyro_yaw)
            gyro_yaw_filt = ALPHA_GYRO * gyro_yaw_filt + (1.0 - ALPHA_GYRO) * gyro_yaw

        if prev_gray is None:
            prev_gray = gray.copy()
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7)
            prev_ts_global = ts
            continue

        if prev_pts is None or len(prev_pts) < 10:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7)
            prev_gray = gray.copy(); prev_ts_global = ts
            continue

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=LK_WIN, maxLevel=LK_MAX_LEVEL,
                                                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if next_pts is None:
            prev_gray = gray.copy(); prev_ts_global = ts
            continue

        status = status.reshape(-1)
        good_prev = prev_pts.reshape(-1,2)[status==1]
        good_next = next_pts.reshape(-1,2)[status==1]
        if len(good_prev) < 8:
            prev_gray = gray.copy(); prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7); prev_ts_global = ts
            continue

        H, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESH)
        if H is None: H = np.eye(2,3)
        inlier_ratio = 0.0
        if inliers is not None:
            inlier_ratio = float(np.count_nonzero(inliers)) / inliers.size

        theta_vis_rad = math.atan2(H[1,0], H[0,0])
        theta_vis_deg = math.degrees(theta_vis_rad)
        use_visual_rot = (angle_diff_deg(theta_vis_deg, gyro_yaw_filt) > TRUST_ROTATION_DISAGREE_DEG) and (inlier_ratio > INLIER_RATIO_THRESH)
        ref_rot = theta_vis_deg if use_visual_rot else gyro_yaw_filt

        pred_next = cv2.transform(good_prev.reshape(1,-1,2), H)[0]
        residuals = good_next - pred_next
        mags = np.linalg.norm(residuals, axis=1)
        angs = np.arctan2(residuals[:,1], residuals[:,0])

        valid_idx = np.where(mags > MAG_THRESH_PIX)[0]
        if len(valid_idx) < 8:
            prev_gray = gray.copy(); prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7); prev_ts_global = ts
            continue

        # grid aggregation
        sum_vx = np.zeros((GRID_H, GRID_W), np.float32)
        sum_vy = np.zeros_like(sum_vx)
        sum_residual_mag = np.zeros_like(sum_vx)
        res_max_grid = np.zeros_like(sum_vx)
        count = np.zeros_like(sum_vx, np.int32)
        res_vals = [[[] for _ in range(GRID_W)] for _ in range(GRID_H)]

        for idx in valid_idx:
            x,y = good_prev[idx]
            vx,vy = residuals[idx]
            mag = math.hypot(vx,vy)
            gx = int(min(GRID_W-1, max(0, math.floor(x / (img.shape[1]/GRID_W)))))
            gy = int(min(GRID_H-1, max(0, math.floor(y / (img.shape[0]/GRID_H)))))
            sum_vx[gy,gx] += vx; sum_vy[gy,gx] += vy; sum_residual_mag[gy,gx] += mag
            res_max_grid[gy,gx] = max(res_max_grid[gy,gx], mag)
            res_vals[gy][gx].append(mag); count[gy,gx] += 1

        avg_vx = np.divide(sum_vx, np.maximum(count,1)); avg_vy = np.divide(sum_vy, np.maximum(count,1))
        avg_mag = np.hypot(avg_vx, avg_vy); avg_residual_mag = np.divide(sum_residual_mag, np.maximum(count,1))
        avg_ang = np.arctan2(avg_vy, avg_vx)
        cell_mask = ((avg_residual_mag >= CELL_MAG_THRESH) & (count >= 1)).astype(np.uint8)
        cell_mask = cv2.dilate(cell_mask, cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL), iterations=1)

        num_labels, labels = cv2.connectedComponents(cell_mask, connectivity=8)
        comps = []
        h_cell = img.shape[0] / GRID_H; w_cell = img.shape[1] / GRID_W
        for lab in range(1, num_labels):
            comp_mask = (labels == lab)
            if np.sum(comp_mask) < MIN_CELL_AREA: continue
            ys, xs = np.where(comp_mask)
            gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
            ang_mean = avg_angle_rad(avg_ang[comp_mask].ravel())
            mag_mean = np.mean(avg_mag[comp_mask])
            res_mean = np.mean(avg_residual_mag[comp_mask])
            pts_in_comp = int(np.sum(count[comp_mask]))
            all_vals = []
            for gy,gx in zip(ys,xs):
                vals = res_vals[gy][gx]
                if vals: all_vals.extend(vals)
            res_p95 = float(np.percentile(np.array(all_vals),95)) if len(all_vals)>0 else 0.0
            res_max = np.max([res_max_grid[gy,gx] for gy,gx in zip(ys,xs)])
            comps.append({'bbox':(gx0,gy0,gx1,gy1),'ang_mean':ang_mean,'mag_mean':mag_mean,'res_mean':res_mean,'res_p95':res_p95,'res_max':res_max,'pts_in_comp':pts_in_comp,'mask':comp_mask})

        pred_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for c in comps:
            gx0,gy0,gx1,gy1 = c['bbox']
            x0,y0 = int(gx0 * w_cell), int(gy0 * h_cell)
            x1,y1 = int((gx1+1) * w_cell), int((gy1+1)*h_cell)
            cluster_angle_deg = math.degrees(c['ang_mean'])
            mag_mean = c['mag_mean']
            is_strong_magnitude = (c['res_p95'] >= RESIDUAL_THRESH_PIX) or (c['res_mean'] >= 0.5 * RESIDUAL_THRESH_PIX)
            if c['pts_in_comp'] >= MIN_POINTS_IN_CLUSTER:
                if is_strong_magnitude:
                    pred_mask[y0:y1, x0:x1] = 255
                else:
                    diff = angle_diff_deg(cluster_angle_deg, ref_rot)
                    if (diff >= IMU_ANGLE_DIFF_THRESH_DEG) and (mag_mean >= MIN_MAG_FOR_ANGLE):
                        pred_mask[y0:y1, x0:x1] = 255

        out_path = os.path.join(outdir, fname)
        cv2.imwrite(out_path, pred_mask)

        if visualize:
            disp = img.copy()
            disp[pred_mask>0] = (0,0,255)
            cv2.imshow('det', disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        prev_gray = gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7)
        prev_ts_global = ts

    if visualize:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdir', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--imu', required=False, default=None)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    main(args.imdir, args.frames, args.imu, args.outdir, visualize=args.vis)
