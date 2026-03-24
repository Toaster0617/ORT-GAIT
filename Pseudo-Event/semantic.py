# -*- coding: utf-8 -*-
"""
RealSense Semantic-Motion Fusion (YOLO Prior + Adaptive Flow)
- Uses YOLOv8 as a semantic prior to create a "Sensitivity Mask".
- Lowers motion thresholds inside semantic regions (Person/Car) to detect micro-motions.
- Validates static vs. dynamic objects to solve the "Billboard/Static Car" problem.
- Retains original IMU/Fundamental Matrix logic.
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
# Parameters
# ==============================

# --- Motion Thresholds (Adaptive) ---
# 原始的 MAG_THRESH_PIX 是 0.2。
# 现在我们将它分级：
# HIGH: 用于普通背景，抑制噪声 (设为 0.5 或更高)
# LOW:  用于 YOLO 框内，捕捉微小运动 (设为 0.15 或 0.2)
THRESH_HIGH_BG = 2.5       # 框外：不仅要动，还要动得明显
THRESH_LOW_SEMANTIC = 1.2 # 框内：只要有一点点动，就认为是动

# Grid & Cluster
CELL_MAG_THRESH = 2.0
MIN_CELL_AREA = 2
GRID_W, GRID_H = 40, 30

# Optical Flow
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
RANSAC_REPROJ_THRESH = 5.0
QUALITY_LEVEL = 0.1
MAX_CORNERS = 600
MIN_DISTANCE = 7
DILATE_KERNEL = (2, 2)

# Residual threshold for classifying pseudo motion vs true motion
RESIDUAL_THRESH_PIX = 2.1

# Fusion / Trust thresholds
IMU_ANGLE_DIFF_THRESH_DEG = 7
TRUST_ROTATION_DISAGREE_DEG = 12
ALPHA_GYRO = 0.6
INLIER_RATIO_THRESH = 0.45

# Cluster validation
MIN_POINTS_IN_CLUSTER = 7
MIN_POINTS_ABOVE_THRESH = 2
MIN_MAG_FOR_ANGLE = 0.4

# Accel-based translation hint
ACCEL_ALPHA = 0.6
ACCEL_TRANS_THRESH = 0.6
ACCEL_SUPPRESS_CELL_MULT = 1.8
ACCEL_SUPPRESS_RES_MULT = 1.5

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

# enable IMU
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

# ==============================
# YOLO Model Init
# ==============================
print("Loading YOLO model... (First run may download weights)")
# 自动下载 yolov8n.pt
yolo_model = YOLO('yolov8n.pt') 
# 我们只关心可能移动的物体类 ID (COCO format)
# 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 15:cat, 16:dog
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16]
print("YOLO model loaded.")

def main(save_pred_dir=None):
    global prev_gray, prev_pts, prev_time, gyro_yaw, gyro_yaw_filt, prev_accel_filt
    frame_idx = 0
    if save_pred_dir:
        os.makedirs(save_pred_dir, exist_ok=True)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # ----- 1. Read IMU (gyro z only) -----
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

            # ----- 2. Read Accel (Translation Hint) -----
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

            # ----- 3. Read Color Frame & Run YOLO -----
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            
            # === YOLO Prior: Generate Sensitivity Mask ===
            # 创建一个敏感度掩码：0 = 普通区域, 1 = 语义敏感区域
            sensitivity_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            yolo_results = yolo_model(frame, stream=True, verbose=False, conf=0.4, classes=TARGET_CLASSES)
            
            # 存储 YOLO 框用于后续联合验证
            # 结构: {'bbox': [x1,y1,x2,y2], 'label': str, 'has_motion': bool}
            semantic_boxes = [] 

            for r in yolo_results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = b
                    
                    # 边界安全检查
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(IMG_W, x2), min(IMG_H, y2)
                    
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    
                    semantic_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': label,
                        'has_motion': False, # 稍后通过光流更新这个状态
                        'motion_pts': 0
                    })
                    
                    # 在敏感掩码上把这个区域涂成 1
                    cv2.rectangle(sensitivity_mask, (x1, y1), (x2, y2), 1, -1)

            # 图像预处理
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

            # ----- 4. Optical Flow (LK) -----
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

            # =================================================================
            # Step 5: Background-Locked Motion Estimation (完全修复版)
            # =================================================================
            
            # 1. 严格分离：只用“非 YOLO 区域”的点来计算相机的运动
            bg_indices = []
            for i, (px, py) in enumerate(good_prev.astype(int)):
                if 0 <= px < IMG_W and 0 <= py < IMG_H:
                    if sensitivity_mask[py, px] == 0: # 0 是背景
                        bg_indices.append(i)
            bg_indices = np.array(bg_indices)

            # 2. 计算背景运动模型 (Affine/Homography)
            # 这里的 H 矩阵代表“相机的背景运动”
            if len(bg_indices) >= 8:
                src_bg = good_prev[bg_indices]
                dst_bg = good_next[bg_indices]
                # 使用 RANSAC 抗噪
                H, inliers = cv2.estimateAffinePartial2D(src_bg, dst_bg, method=cv2.RANSAC, ransacReprojThreshold=2.0)
            else:
                # 备用：背景点不够时，退化为全局估计
                H, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)

            if H is None: H = np.eye(2, 3)

            # === [修复关键点] 定义旋转参考和标志位 ===
            # 从 H 矩阵中提取视觉旋转角度
            theta_vis_rad = math.atan2(H[1, 0], H[0, 0])
            theta_vis_deg = math.degrees(theta_vis_rad)
            
            # 在背景锁定模式下，我们优先信任视觉计算的背景旋转
            # 定义 use_visual_rot 防止 HUD 报错
            use_visual_rot = True 
            ref_rot = theta_vis_deg 
            # =========================================

            # 3. 计算“背景流速度” (用于抗抖动/视差抑制)
            bg_speed = 0.0
            if len(bg_indices) > 0:
                # 预测背景位置
                pred_bg = cv2.transform(good_prev[bg_indices].reshape(1, -1, 2), H)[0]
                # 计算实际光流模长
                actual_flow_bg = np.linalg.norm(good_next[bg_indices] - good_prev[bg_indices], axis=1)
                # 取中位数代表相机速度 (抗噪)
                bg_speed = np.median(actual_flow_bg)

            # =================================================================
            # Step 6: Dynamic Thresholding (解决晃动幅度不同的问题)
            # =================================================================
            
            valid_idx = []
            
            # 1. 预测：如果所有点都是静止的，它们下一帧应该在哪？
            pred_next = cv2.transform(good_prev.reshape(1, -1, 2), H)[0]
            
            # 2. 残差：实际位置 - 预测位置
            # 对于静止物体，Residual ≈ 0 (理想) 或 ≈ 视差误差 (现实)
            # 对于移动物体，Residual = 真实移动量
            residuals = good_next - pred_next
            mags = np.linalg.norm(residuals, axis=1)

            # 3. 设定“浮动阈值” (关键！)
            # 这里的逻辑是：相机晃得越快，容忍度越高。
            # base_thresh = 0.8 (处理传感器底噪)
            # parallax_factor = 0.15 * bg_speed (处理视差：相机动10px，允许1.5px的视差误差)
            dynamic_thresh = 0.8 + 0.15 * bg_speed

            for i, mag in enumerate(mags):
                px, py = int(good_prev[i][0]), int(good_prev[i][1])
                if px < 0 or px >= IMG_W or py < 0 or py >= IMG_H: continue

                is_sensitive = (sensitivity_mask[py, px] == 1)
                is_moving_pixel = False
                
                if is_sensitive:
                    # 【框内策略】
                    # 使用动态阈值。
                    # 如果相机静止(speed=0)，阈值=0.8 -> 抓住慢走的人。
                    # 如果相机狂晃(speed=20)，阈值=3.8 -> 放过视差大的静止人。
                    if mag > dynamic_thresh: 
                        is_moving_pixel = True
                else:
                    # 【框外策略】
                    # 框外我们可以沿用 motion_new.py 的严格逻辑，或者设定一个很高的阈值
                    if mag > max(2.5, dynamic_thresh * 1.5):
                        is_moving_pixel = True
                
                if is_moving_pixel:
                    valid_idx.append(i)
                    if is_sensitive:
                        for box in semantic_boxes:
                            bx1, by1, bx2, by2 = box['bbox']
                            if bx1 <= px <= bx2 and by1 <= py <= by2:
                                box['motion_pts'] += 1

            valid_idx = np.array(valid_idx)
            
            # Update YOLO box status based on collected points
            for box in semantic_boxes:
                # 如果框内动点超过一定数量，标记为 Moving
                if box['motion_pts'] >= MIN_POINTS_IN_CLUSTER:
                    box['has_motion'] = True

            # fallback check
            if len(valid_idx) < 5:
                # 即使没有动点，我们也应该画出 YOLO 框（显示为静态）
                # 这里不直接 continue，为了保证显示逻辑连贯，我们在下面处理
                pass

            # ----- 7. Grid Aggregation (Original Logic) -----
            sum_vx = np.zeros((GRID_H, GRID_W), np.float32)
            sum_vy = np.zeros_like(sum_vx)
            sum_residual_mag = np.zeros_like(sum_vx)
            res_max_grid = np.zeros_like(sum_vx)
            count = np.zeros_like(sum_vx, np.int32)
            res_vals = [[[] for _ in range(GRID_W)] for _ in range(GRID_H)]

            if len(valid_idx) > 0:
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

            # Adjust thresholds if accel indicates translation
            cell_mag_thresh = CELL_MAG_THRESH * (ACCEL_SUPPRESS_CELL_MULT if translation_flag else 1.0)
            
            # Mask generation
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
                
                # Get P95
                all_vals = []
                for gy, gx in zip(ys, xs):
                    vals = res_vals[gy][gx]
                    if vals: all_vals.extend(vals)
                res_p95 = float(np.percentile(np.array(all_vals), 95)) if len(all_vals) > 0 else 0.0
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

            # ----- 8. Visualization & Fusion Logic (The "Human Operator" View) -----
            
            # A. Draw YOLO Boxes (Red if Moving, Green if Static)
            for box in semantic_boxes:
                bx1, by1, bx2, by2 = box['bbox']
                label = box['label']
                is_moving = box['has_motion']
                pts_count = box['motion_pts']
                
                if is_moving:
                    # 动态目标：红色框
                    color = (0, 0, 255) 
                    text = f"{label}: Moving ({pts_count} pts)"
                else:
                    # 静态语义目标（如广告牌上的车，或者停着的车）：绿色框
                    # 这就是论文中提到的“解决误检”
                    color = (0, 255, 0)
                    text = f"{label}: Static"
                
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
                # 文字背景条
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (bx1, by1 - 20), (bx1 + text_w, by1), color, -1)
                cv2.putText(vis, text, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # B. Draw Unknown Motion Clusters (Purple)
            # 只有那些 不在 任何 YOLO 框内的运动聚类，才显示为“未知物体”
            for c in comps:
                gx0, gy0, gx1, gy1 = c['bbox']
                x0, y0 = int(gx0 * cell_w), int(gy0 * cell_h)
                x1, y1 = int((gx1 + 1) * cell_w), int((gy1 + 1) * cell_h)
                
                # Check overlap with any semantic box
                cx, cy = (x0+x1)/2, (y0+y1)/2
                is_inside_semantic = False
                for box in semantic_boxes:
                    bx1, by1, bx2, by2 = box['bbox']
                    if bx1 < cx < bx2 and by1 < cy < by2:
                        is_inside_semantic = True
                        break
                
                # 如果聚类有效，且不在语义框内 -> 未知障碍物 (Purple)
                # 依然需要基本的运动判定逻辑
                is_strong_magnitude = (c['res_p95'] >= RESIDUAL_THRESH_PIX)
                diff = angle_diff_deg(math.degrees(c['ang_mean']), ref_rot)
                is_angle_disagree = (diff >= IMU_ANGLE_DIFF_THRESH_DEG) and (c['mag_mean'] >= MIN_MAG_FOR_ANGLE)
                
                if (c['pts_in_comp'] >= MIN_POINTS_IN_CLUSTER) and (is_strong_magnitude or is_angle_disagree):
                    if not is_inside_semantic:
                        color = (255, 0, 255) # Purple
                        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                        cv2.putText(vis, "Unknown Motion", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # C. HUD Info
            src = 'VIS' if use_visual_rot else 'IMU'
            trans_tag = 'TRANS' if translation_flag else 'OK'
            hud = f"RotRef={src} Trans={trans_tag} YOLO_Objs={len(semantic_boxes)}"
            cv2.putText(vis, hud, (6, IMG_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("Semantic-Motion Fusion", vis)
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
    parser.add_argument('--save-pred-dir', default=None, help='directory to save predicted binary masks')
    args = parser.parse_args()
    main(save_pred_dir=args.save_pred_dir)