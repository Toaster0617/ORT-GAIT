# -*- coding: utf-8 -*-
"""
RealSense Motion Detection — Academic Final Edition
[FEATURES]
1. Photometric Invariant Pre-processing (CLAHE): Explicitly handles illumination changes.
2. Visual-Inertial Fusion (Gyro+Accel): Handles rapid rotation and translation.
3. Semantic Gating (YOLOv8): Differentiates between 'Person' (Sensitive) and 'Background' (Strict).
4. Zero-Latency Response: Frame-by-frame detection without temporal lag.
"""

import time
import math
import cv2
import numpy as np
import pyrealsense2 as rs
import argparse
import os
from ultralytics import YOLO

# ==============================
# 1. Photometric Pre-processing
# ==============================
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Theory: Normalizes local contrast to satisfy Brightness Constancy Assumption 
# even under specular highlights or shadows.
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def photometric_invariant_process(frame_bgr):
    """
    Convert BGR to a 'Photometric Invariant' Grayscale representation.
    Uses L-channel equalization from LAB space.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel (Luminance)
    # This suppresses low-frequency lighting variations (glare)
    # and enhances high-frequency texture (edges).
    l_clahe = clahe.apply(l)
    
    return l_clahe

# ==============================
# 2. Algorithm Parameters
# ==============================
GRID_W, GRID_H = 40, 30
IMG_W, IMG_H = 640, 480
cell_w, cell_h = IMG_W / GRID_W, IMG_H / GRID_H

# Optical Flow & Geometry
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
RANSAC_REPROJ_THRESH = 5.0
QUALITY_LEVEL = 0.1
MAX_CORNERS = 600
MIN_DISTANCE = 7
DILATE_KERNEL = (3, 3) 

# Visual-Inertial Fusion
IMU_ANGLE_DIFF_THRESH_DEG = 7
TRUST_ROTATION_DISAGREE_DEG = 12
ALPHA_GYRO = 0.6
INLIER_RATIO_THRESH = 0.45

# Rapid Motion Adaptation
# A. Translation (Accel)
ACCEL_ALPHA = 0.6               
ACCEL_TRANS_THRESH = 0.3       
ACCEL_SUPPRESS_MULT = 4.0      

# B. Rotation (Gyro) - Anti-Shake
GYRO_SHAKE_THRESH = 0.8         # Sensitivity to shaking
GYRO_SUPPRESS_MULT = 4.0        # Penalty multiplier during shake

# Semantic Gating (YOLO)
# Background (Strict)
BG_CELL_THRESH = 2.0      
BG_RESIDUAL_THRESH = 2.0  

# Foreground (Person - Sensitive)
FG_CELL_THRESH = 0.5      
FG_RESIDUAL_THRESH = 0.6  
MIN_POINTS_IN_CLUSTER = 3 

# YOLO Model
YOLO_CONF_THRESH = 0.45       
YOLO_OVERLAP_MATCH_THRESH = 0.12 
YOLO_BOOST_CLASSES = ['person'] 

# ==============================
# Helper Functions
# ==============================
def wrap_angle_deg(a):
    return (a + 180) % 360 - 180

def angle_diff_deg(a, b):
    return abs((a - b + 180) % 360 - 180)

def avg_angle_rad(a):
    if len(a) == 0: return 0.0
    s = np.sum(np.sin(a))
    c = np.sum(np.cos(a))
    return math.atan2(s, c)

def compute_overlap_ratio(motion_box, yolo_box):
    xA = max(motion_box[0], yolo_box[0])
    yA = max(motion_box[1], yolo_box[1])
    xB = min(motion_box[2], yolo_box[2])
    yB = min(motion_box[3], yolo_box[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    motionArea = (motion_box[2] - motion_box[0]) * (motion_box[3] - motion_box[1])
    if motionArea < 1e-6: return 0.0
    return interArea / float(motionArea)

# ==============================
# Main Pipeline
# ==============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.gyro)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

prev_gray = None
prev_pts = None

prev_time = time.time()
gyro_yaw = 0.0
gyro_yaw_filt = 0.0
prev_accel_filt = None
prev_gyro_mag_filt = 0.0 

def main(save_pred_dir=None):
    global prev_gray, prev_pts, prev_time, gyro_yaw, gyro_yaw_filt, prev_accel_filt, prev_gyro_mag_filt
    
    if save_pred_dir: os.makedirs(save_pred_dir, exist_ok=True)

    print("Loading YOLOv8 model...")
    yolo_model = YOLO('yolov8n.pt') 
    print(f"Model loaded. System Active with CLAHE Pre-processing.")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # -----------------------------------------------------------
            # 1. IMU Fusion: Rotation (Gyro) & Translation (Accel)
            # -----------------------------------------------------------
            shake_flag = False
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            if gyro_frame:
                g = gyro_frame.as_motion_frame().get_motion_data()
                wz = g.z
                now = time.time()
                dt = now - prev_time if prev_time is not None else 0.0
                prev_time = now
                gyro_yaw += math.degrees(wz * dt)
                gyro_yaw = wrap_angle_deg(gyro_yaw)
                gyro_yaw_filt = ALPHA_GYRO * gyro_yaw_filt + (1.0 - ALPHA_GYRO) * gyro_yaw

                # Detect Shake
                gyro_mag = math.sqrt(g.x**2 + g.y**2 + g.z**2)
                prev_gyro_mag_filt = 0.7 * prev_gyro_mag_filt + 0.3 * gyro_mag
                if prev_gyro_mag_filt > GYRO_SHAKE_THRESH:
                    shake_flag = True

            translation_flag = False
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                a = accel_frame.as_motion_frame().get_motion_data()
                ax, ay, az = a.x, a.y, a.z
                accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
                if prev_accel_filt is None:
                    prev_accel_filt = accel_norm
                prev_accel_filt = ACCEL_ALPHA * prev_accel_filt + (1.0 - ACCEL_ALPHA) * accel_norm
                accel_delta = abs(accel_norm - prev_accel_filt)
                if accel_delta >= ACCEL_TRANS_THRESH:
                    translation_flag = True

            # -----------------------------------------------------------
            # 2. Visual & YOLO (with Photometric Invariance)
            # -----------------------------------------------------------
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            frame = np.asanyarray(color_frame.get_data())
            
            # [NEW] Apply CLAHE instead of standard gray conversion
            gray = photometric_invariant_process(frame)
            
            vis = frame.copy()

            # YOLO Inference (Uses Color Frame)
            yolo_results = yolo_model(frame, conf=YOLO_CONF_THRESH, verbose=False)[0]
            
            yolo_objects = []
            
            # Init Maps with Background Defaults (Strict)
            thresh_map_cell = np.full((GRID_H, GRID_W), BG_CELL_THRESH, dtype=np.float32)
            thresh_map_res  = np.full((GRID_H, GRID_W), BG_RESIDUAL_THRESH, dtype=np.float32)
            
            for box in yolo_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                
                yolo_objects.append({
                    'box': (x1, y1, x2, y2), 
                    'label': label, 
                    'is_moving': False 
                })
                
                # Semantic Gating: Person gets High Sensitivity
                if label in YOLO_BOOST_CLASSES:
                    gx1, gy1 = int(max(0, x1/cell_w)), int(max(0, y1/cell_h))
                    gx2, gy2 = int(min(GRID_W, x2/cell_w)), int(min(GRID_H, y2/cell_h))
                    thresh_map_cell[gy1:gy2+1, gx1:gx2+1] = FG_CELL_THRESH
                    thresh_map_res[gy1:gy2+1, gx1:gx2+1]  = FG_RESIDUAL_THRESH

            # -----------------------------------------------------------
            # 3. Optical Flow & Geometry (Fusion Decision)
            # -----------------------------------------------------------
            if prev_gray is None:
                prev_gray = gray; prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7)
                continue
            
            if prev_pts is None or len(prev_pts) < 10:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7)
                prev_gray = gray; continue

            # LK Flow now operates on CLAHE-enhanced gray images
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=LK_WIN, maxLevel=LK_MAX_LEVEL, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if next_pts is None: prev_gray = gray; continue
            
            status = status.reshape(-1)
            good_prev = prev_pts.reshape(-1, 2)[status == 1]
            good_next = next_pts.reshape(-1, 2)[status == 1]

            if len(good_prev) < 8: prev_gray = gray; continue

            try: F, fm_mask = cv2.findFundamentalMat(good_prev, good_next, cv2.FM_RANSAC, 3.0)
            except: F = None; fm_mask = None
            fm_mask_flat = fm_mask.reshape(-1).astype(np.uint8) if fm_mask is not None else np.ones(len(good_prev), dtype=np.uint8)

            bg_idx = np.where(fm_mask_flat == 1)[0]
            if len(bg_idx) >= 8: H, inliers = cv2.estimateAffinePartial2D(good_prev[bg_idx], good_next[bg_idx], method=cv2.RANSAC)
            else: H, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
            if H is None: H = np.eye(2,3)

            theta_vis_rad = math.atan2(H[1, 0], H[0, 0])
            theta_vis_deg = math.degrees(theta_vis_rad)
            inlier_ratio = float(np.count_nonzero(inliers))/inliers.size if inliers is not None else 0.0
            
            # Visual-Inertial Decision
            use_visual_rot = (angle_diff_deg(theta_vis_deg, gyro_yaw_filt) > TRUST_ROTATION_DISAGREE_DEG) and (inlier_ratio > INLIER_RATIO_THRESH)
            ref_rot = theta_vis_deg if use_visual_rot else gyro_yaw_filt

            pred_next = cv2.transform(good_prev.reshape(1, -1, 2), H)[0]
            residuals = good_next - pred_next
            mags = np.linalg.norm(residuals, axis=1)
            
            valid_idx = np.where((mags > 0.2) & (fm_mask_flat == 0))[0]
            if len(valid_idx) < 8: valid_idx = np.where(mags > 0.2)[0]

            # -----------------------------------------------------------
            # 4. Aggregation & Adaptive Thresholds
            # -----------------------------------------------------------
            sum_residual_mag = np.zeros((GRID_H, GRID_W), np.float32)
            count = np.zeros((GRID_H, GRID_W), np.int32)
            res_vals = [[[] for _ in range(GRID_W)] for _ in range(GRID_H)]
            
            for idx in valid_idx:
                vx, vy = residuals[idx]
                mag = math.hypot(vx, vy)
                x, y = good_prev[idx]
                gx = int(np.clip(x / cell_w, 0, GRID_W-1))
                gy = int(np.clip(y / cell_h, 0, GRID_H-1))
                sum_residual_mag[gy, gx] += mag
                count[gy, gx] += 1
                res_vals[gy][gx].append(mag)
            
            avg_residual_mag = np.divide(sum_residual_mag, np.maximum(count, 1))

            # Apply Physics-based Suppression
            suppress_factor = 1.0
            if translation_flag: suppress_factor *= ACCEL_SUPPRESS_MULT 
            if shake_flag: suppress_factor *= GYRO_SUPPRESS_MULT  
            
            thresh_map_cell *= suppress_factor
            thresh_map_res  *= suppress_factor

            # -----------------------------------------------------------
            # 5. Classification & Semantic Rendering
            # -----------------------------------------------------------
            cell_mask = ((avg_residual_mag >= thresh_map_cell) & (count >= 1)).astype(np.uint8)
            cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, DILATE_KERNEL)
            
            num_labels, labels = cv2.connectedComponents(cell_mask, connectivity=8)
            unknown_clusters = []

            for lab in range(1, num_labels):
                comp_mask = (labels == lab)
                if np.sum(comp_mask) < 2: continue
                
                ys, xs = np.where(comp_mask)
                gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
                
                all_vals = []
                for gy, gx in zip(ys, xs):
                    if res_vals[gy][gx]: all_vals.extend(res_vals[gy][gx])
                res_p95 = float(np.percentile(np.array(all_vals), 95)) if all_vals else 0.0
                pts_in_comp = int(np.sum(count[comp_mask]))

                local_res_thresh = np.min(thresh_map_res[ys, xs])
                
                is_moving = False
                if pts_in_comp >= MIN_POINTS_IN_CLUSTER:
                    if res_p95 >= local_res_thresh:
                        is_moving = True
                
                if is_moving:
                    x0, y0 = int(gx0 * cell_w), int(gy0 * cell_h)
                    x1, y1 = int((gx1 + 1) * cell_w), int((gy1 + 1) * cell_h)
                    motion_box = (x0, y0, x1, y1)

                    matched = False
                    for obj in yolo_objects:
                        overlap = compute_overlap_ratio(motion_box, obj['box'])
                        # If motion overlaps with YOLO box, mark YOLO object as Moving
                        if overlap > YOLO_OVERLAP_MATCH_THRESH:
                            obj['is_moving'] = True
                            matched = True
                    
                    if not matched:
                        unknown_clusters.append(motion_box)

            # Draw Logic
            # 1. YOLO Objects (Green=Static, Red=Moving)
            for obj in yolo_objects:
                x1, y1, x2, y2 = obj['box']
                if obj['is_moving']:
                    color = (0, 0, 255) # Red
                    label = f"{obj['label']} (Moving)"
                    thickness = 3 
                else:
                    color = (0, 255, 0) # Green
                    label = f"{obj['label']} (Static)"
                    thickness = 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 2. Unknown Clusters (Purple)
            for (mx0, my0, mx1, my1) in unknown_clusters:
                cv2.rectangle(vis, (mx0, my0), (mx1, my1), (255, 0, 255), 2)
                cv2.putText(vis, "Unknown", (mx0, my0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # HUD
            status_tags = []
            if translation_flag: status_tags.append("TRANS")
            if shake_flag: status_tags.append("SHAKE")
            if not status_tags: status_tags.append("STABLE")
            
            hud = f"AcademicFinal | {' '.join(status_tags)} | Gyro={prev_gyro_mag_filt:.1f}"
            cv2.putText(vis, hud, (10, IMG_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Academic Motion", vis)
            if cv2.waitKey(1) & 0xFF == 27: break
            
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=7)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()