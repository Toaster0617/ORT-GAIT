# -*- coding: utf-8 -*-
"""
RealSense Motion Detection — Visual + IMU (gyro) Fusion (with Fundamental-matrix filtering)
- Adds cv2.findFundamentalMat RANSAC to separate background correspondences (inliers)
  from potential moving-object correspondences (outliers).

This is a minimally invasive variant of `motion_fused.py` that first uses the
fundamental matrix to identify background matches and then estimates the
affine/camera motion from those background inliers. Residuals that are
fundamental-outliers are treated preferentially as moving candidates.
"""

import time
import math
import cv2
import numpy as np
import pyrealsense2 as rs
import argparse
import os

# ==============================
# Parameters (mostly same as original)
# ==============================
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

# Residual threshold for classifying pseudo motion vs true motion
RESIDUAL_THRESH_PIX = 2.1

# Fusion / trust thresholds
IMU_ANGLE_DIFF_THRESH_DEG = 7
TRUST_ROTATION_DISAGREE_DEG = 12
ALPHA_GYRO = 0.6
INLIER_RATIO_THRESH = 0.45

# Minimum counts to consider cluster valid
MIN_POINTS_IN_CLUSTER = 7
MIN_POINTS_ABOVE_THRESH = 2
MIN_MAG_FOR_ANGLE = 0.4

# Accel-based translation hint (no integration)
ACCEL_ALPHA = 0.6               # smoothing for accel magnitude
ACCEL_TRANS_THRESH = 0.6        # m/s^2 change threshold to consider translation (tunable)
ACCEL_SUPPRESS_CELL_MULT = 1.8  # multiply CELL_MAG_THRESH when translation detected
ACCEL_SUPPRESS_RES_MULT = 1.5   # multiply RESIDUAL_THRESH_PIX when translation detected


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


# ==============================
# RealSense Init
# ==============================
pipeline = rs.pipeline()
config = rs.config()

# enable IMU (we only use gyro z)
config.enable_stream(rs.stream.gyro)
config.enable_stream(rs.stream.accel)

# enable color
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

IMG_W, IMG_H = intr.width, intr.height
cell_w, cell_h = IMG_W / GRID_W, IMG_H / GRID_H

prev_gray = None
prev_pts = None

# IMU state
prev_time = time.time()
gyro_yaw = 0.0
gyro_yaw_filt = 0.0
prev_accel_filt = None


def main(save_pred_dir=None):
    global prev_gray, prev_pts, prev_time, gyro_yaw, gyro_yaw_filt, prev_accel_filt
    frame_idx = 0
    if save_pred_dir:
        os.makedirs(save_pred_dir, exist_ok=True)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # ----- Read IMU (gyro z only) -----
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            if gyro_frame:
                g = gyro_frame.as_motion_frame().get_motion_data()
                wx, wy, wz = g.x, g.y, g.z

                now = time.time()
                dt = now - prev_time if prev_time is not None else 0.0
                prev_time = now

                gyro_yaw += math.degrees(wz * dt)
                gyro_yaw = wrap_angle_deg(gyro_yaw)
                gyro_yaw_filt = ALPHA_GYRO * gyro_yaw_filt + (1.0 - ALPHA_GYRO) * gyro_yaw

            # ----- Read Accel (short-term translation hint) -----
            translation_flag = False
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                a = accel_frame.as_motion_frame().get_motion_data()
                ax, ay, az = a.x, a.y, a.z
                accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
                if prev_accel_filt is None:
                    prev_accel_filt = accel_norm
                # smoothed accel magnitude (low-pass)
                prev_accel_filt = ACCEL_ALPHA * prev_accel_filt + (1.0 - ACCEL_ALPHA) * accel_norm
                accel_delta = abs(accel_norm - prev_accel_filt)
                if accel_delta >= ACCEL_TRANS_THRESH:
                    translation_flag = True

            # ----- Read Color Frame -----
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            # Init first frame
            if prev_gray is None:
                prev_gray = gray
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                                   maxCorners=MAX_CORNERS,
                                                   qualityLevel=QUALITY_LEVEL,
                                                   minDistance=MIN_DISTANCE,
                                                   blockSize=7)
                continue

            if prev_pts is None or len(prev_pts) < 10:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                                   maxCorners=MAX_CORNERS,
                                                   qualityLevel=QUALITY_LEVEL,
                                                   minDistance=MIN_DISTANCE,
                                                   blockSize=7)
                prev_gray = gray
                continue

            # LK optical flow
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None,
                                                           winSize=LK_WIN,
                                                           maxLevel=LK_MAX_LEVEL,
                                                           criteria=(cv2.TERM_CRITERIA_EPS |
                                                                     cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            if next_pts is None:
                prev_gray = gray
                continue

            status = status.reshape(-1)
            good_prev = prev_pts.reshape(-1, 2)[status == 1]
            good_next = next_pts.reshape(-1, 2)[status == 1]

            if len(good_prev) < 8:
                prev_gray = gray
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                                   maxCorners=MAX_CORNERS,
                                                   qualityLevel=QUALITY_LEVEL,
                                                   minDistance=MIN_DISTANCE,
                                                   blockSize=7)
                continue

            # ----- Fundamental-matrix filtering (RANSAC) -----
            try:
                F, fm_mask = cv2.findFundamentalMat(good_prev, good_next, cv2.FM_RANSAC, 3.0)
            except Exception:
                F = None
                fm_mask = None

            if fm_mask is None:
                # assume all background (conservative fallback)
                fm_mask_flat = np.ones((len(good_prev),), dtype=np.uint8)
            else:
                fm_mask_flat = fm_mask.reshape(-1).astype(np.uint8)

            # Use background inliers (fm_mask==1) to estimate camera affine robustly
            bg_idx = np.where(fm_mask_flat == 1)[0]
            if len(bg_idx) >= 8:
                H, inliers = cv2.estimateAffinePartial2D(good_prev[bg_idx], good_next[bg_idx], method=cv2.RANSAC,
                                                         ransacReprojThreshold=RANSAC_REPROJ_THRESH)
            else:
                # fallback: use all matches to estimate H
                H, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC,
                                                         ransacReprojThreshold=RANSAC_REPROJ_THRESH)

            if H is None:
                H = np.eye(2, 3)

            # compute RANSAC inlier ratio to judge visual estimate reliability
            inlier_ratio = 0.0
            if inliers is not None:
                try:
                    inlier_ratio = float(np.count_nonzero(inliers)) / inliers.size
                except Exception:
                    inlier_ratio = 0.0

            # extract visual rotation (degrees) from affine H
            theta_vis_rad = math.atan2(H[1, 0], H[0, 0])
            theta_vis_deg = math.degrees(theta_vis_rad)

            # decide which rotation source to trust
            use_visual_rot = False
            if (angle_diff_deg(theta_vis_deg, gyro_yaw_filt) > TRUST_ROTATION_DISAGREE_DEG) and (inlier_ratio > INLIER_RATIO_THRESH):
                use_visual_rot = True

            ref_rot = theta_vis_deg if use_visual_rot else gyro_yaw_filt

            pred_next = cv2.transform(good_prev.reshape(1, -1, 2), H)[0]
            residuals = good_next - pred_next
            mags = np.linalg.norm(residuals, axis=1)
            angs = np.arctan2(residuals[:, 1], residuals[:, 0])

            # Preferentially treat fundamental-outliers as moving candidates
            valid_idx = np.where((mags > MAG_THRESH_PIX) & (fm_mask_flat == 0))[0]

            # fallback: if no f-outliers but many residuals, use residual threshold as before
            if len(valid_idx) < 8:
                valid_idx = np.where(mags > MAG_THRESH_PIX)[0]

            if len(valid_idx) < 8:
                cv2.putText(vis, f"FM_outliers={np.sum(fm_mask_flat==0)}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0),1)
                cv2.imshow("MotionClusters", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                prev_gray = gray
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                                   maxCorners=MAX_CORNERS,
                                                   qualityLevel=QUALITY_LEVEL,
                                                   minDistance=MIN_DISTANCE,
                                                   blockSize=7)
                continue

            # ----- Grid aggregation -----
            sum_vx = np.zeros((GRID_H, GRID_W), np.float32)
            sum_vy = np.zeros_like(sum_vx)
            sum_residual_mag = np.zeros_like(sum_vx)
            res_max_grid = np.zeros_like(sum_vx)
            count = np.zeros_like(sum_vx, np.int32)
            res_vals = [[[] for _ in range(GRID_W)] for _ in range(GRID_H)]

            for idx in valid_idx:
                x, y = good_prev[idx]
                vx, vy = residuals[idx]
                mag = math.hypot(vx, vy)
                gx = int(min(GRID_W - 1, max(0, math.floor(x / cell_w))))
                gy = int(min(GRID_H - 1, max(0, math.floor(y / cell_h))))
                sum_vx[gy, gx] += vx
                sum_vy[gy, gx] += vy
                sum_residual_mag[gy, gx] += mag
                if mag > res_max_grid[gy, gx]:
                    res_max_grid[gy, gx] = mag
                res_vals[gy][gx].append(mag)
                count[gy, gx] += 1

            avg_vx = np.divide(sum_vx, np.maximum(count, 1))
            avg_vy = np.divide(sum_vy, np.maximum(count, 1))
            avg_mag = np.hypot(avg_vx, avg_vy)
            avg_residual_mag = np.divide(sum_residual_mag, np.maximum(count, 1))
            avg_ang = np.arctan2(avg_vy, avg_vx)

            # adjust thresholds if accel indicates translation
            cell_mag_thresh = CELL_MAG_THRESH * (ACCEL_SUPPRESS_CELL_MULT if translation_flag else 1.0)
            residual_thresh = RESIDUAL_THRESH_PIX * (ACCEL_SUPPRESS_RES_MULT if translation_flag else 1.0)

            cell_mask = ((avg_residual_mag >= cell_mag_thresh) & (count >= 1)).astype(np.uint8)
            cell_mask = cv2.dilate(cell_mask, cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL), iterations=1)

            num_labels, labels = cv2.connectedComponents(cell_mask, connectivity=8)
            comps = []

            for lab in range(1, num_labels):
                comp_mask = (labels == lab)
                if np.sum(comp_mask) < MIN_CELL_AREA:
                    continue
                ys, xs = np.where(comp_mask)
                gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
                ang_mean = avg_angle_rad(avg_ang[comp_mask].ravel())
                mag_mean = np.mean(avg_mag[comp_mask])
                res_mean = np.mean(avg_residual_mag[comp_mask])
                pts_in_comp = int(np.sum(count[comp_mask]))
                all_vals = []
                for gy, gx in zip(ys, xs):
                    vals = res_vals[gy][gx]
                    if vals:
                        all_vals.extend(vals)
                if len(all_vals) > 0:
                    res_p95 = float(np.percentile(np.array(all_vals), 95))
                else:
                    res_p95 = 0.0
                res_max = np.max([res_max_grid[gy, gx] for gy, gx in zip(ys, xs)])

                comps.append({
                    'bbox': (gx0, gy0, gx1, gy1),
                    'ang_mean': ang_mean,
                    'mag_mean': mag_mean,
                    'res_mean': res_mean,
                    'res_max': res_max,
                    'res_p95': res_p95,
                    'pts_in_comp': pts_in_comp,
                    'mask': comp_mask
                })

            # ----- Draw clusters & classify (fusion rules) -----
            # Only draw clusters that are classified as moving (red). Leave pred_mask logic unchanged.
            moving_count = 0
            for c in comps:
                gx0, gy0, gx1, gy1 = c['bbox']
                x0, y0 = int(gx0 * cell_w), int(gy0 * cell_h)
                x1, y1 = int((gx1 + 1) * cell_w), int((gy1 + 1) * cell_h)

                cluster_angle_deg = math.degrees(c['ang_mean'])
                mag_mean = c['mag_mean']

                is_strong_magnitude = (c['res_p95'] >= residual_thresh) or (c['res_mean'] >= 0.5 * residual_thresh)

                # Determine if this cluster should be considered moving (red)
                is_moving = False
                reason = ""
                if c['pts_in_comp'] < MIN_POINTS_IN_CLUSTER:
                    is_moving = False
                else:
                    if is_strong_magnitude:
                        is_moving = True
                        reason = f"res_p95={c['res_p95']:.2f} max={c['res_max']:.2f}"
                    else:
                        diff = angle_diff_deg(cluster_angle_deg, ref_rot)
                        if (diff >= IMU_ANGLE_DIFF_THRESH_DEG) and (mag_mean >= MIN_MAG_FOR_ANGLE):
                            is_moving = True
                            reason = f"ang_diff={diff:.1f}deg"

                if is_moving:
                    color = (0, 0, 255)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(vis, reason, (x0 + 3, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    moving_count += 1

            # produce a prediction mask
            pred_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            for c in comps:
                gx0, gy0, gx1, gy1 = c['bbox']
                x0, y0 = int(gx0 * cell_w), int(gy0 * cell_h)
                x1, y1 = int((gx1 + 1) * cell_w), int((gy1 + 1) * cell_h)
                cluster_angle_deg = math.degrees(c['ang_mean'])
                mag_mean = c['mag_mean']
                is_strong_magnitude = (c['res_p95'] >= residual_thresh) or (c['res_mean'] >= 0.5 * residual_thresh)
                if c['pts_in_comp'] >= MIN_POINTS_IN_CLUSTER:
                    if is_strong_magnitude:
                        pred_mask[y0:y1, x0:x1] = 255
                    else:
                        diff = angle_diff_deg(cluster_angle_deg, ref_rot)
                        if (diff >= IMU_ANGLE_DIFF_THRESH_DEG) and (mag_mean >= MIN_MAG_FOR_ANGLE):
                            pred_mask[y0:y1, x0:x1] = 255

            if save_pred_dir is not None:
                out_name = f"{frame_idx:06d}.png"
                out_path = os.path.join(save_pred_dir, out_name)
                cv2.imwrite(out_path, pred_mask)
                frame_idx += 1

            src = 'VIS' if use_visual_rot else 'IMU'
            trans_tag = 'TRANS' if translation_flag else 'OK'
            # include count of moving (red) clusters
            hud = f"rot_ref={src} vis_rot={theta_vis_deg:+.1f} imu_rot={gyro_yaw_filt:+.1f} FM_outliers={np.sum(fm_mask_flat==0)} trans={trans_tag} moving={moving_count}"
            cv2.putText(vis, hud, (6, IMG_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("MotionClusters", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                               maxCorners=MAX_CORNERS,
                                               qualityLevel=QUALITY_LEVEL,
                                               minDistance=MIN_DISTANCE,
                                               blockSize=7)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-pred-dir', default=None, help='directory to save predicted binary masks (PNG)')
    args = parser.parse_args()
    main(save_pred_dir=args.save_pred_dir)
