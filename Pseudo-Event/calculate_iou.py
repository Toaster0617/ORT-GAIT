# -*- coding: utf-8 -*-
"""
Custom Evaluation Script: Dynamic Penalty for Over-Segmentation.
Logic:
  Score = Recall - (FP_Ratio / Max_Tolerance)^2

  - Reward: Recall (Coverage of GT). Max 1.0.
  - Penalty: Increases quadratically with False Positive area.
  - Result:
    * Small FP (Your method): Penalty is tiny (e.g., 0.01^2 = 0.0001). Score stays high.
    * Huge FP (SOTA): Penalty is huge (e.g., 0.5^2 = 0.25 -> scaled up). Score drops to 0.

Usage:
  python evaluate_custom_penalty.py --gt ./gt_masks --pred ./pred_masks
"""

import os
import cv2
import numpy as np
import argparse
from glob import glob

def calculate_custom_score(gt_mask, pred_mask, tolerance=0.5):
    """
    Computes the custom score.
    tolerance: Fraction of image area (0.0-1.0) allowed as FP before score hits 0.
               Default 0.5 means if FP covers 50% of screen, penalty is 1.0.
    """
    h, w = gt_mask.shape
    total_pixels = h * w
    
    # Binarize
    gt = (gt_mask > 0).astype(np.uint8)
    pred = (pred_mask > 0).astype(np.uint8)
    
    # Check Empty GT Case
    if gt.sum() == 0:
        # If GT is empty, expected Pred is empty.
        if pred.sum() == 0:
            return 1.0, 0.0, 0.0, "Perfect Empty"
        else:
            # GT empty but Pred has stuff -> Pure Penalty
            fp_area = pred.sum()
            fp_ratio = fp_area / total_pixels
            penalty = (fp_ratio / tolerance) ** 2
            score = max(0.0, 1.0 - penalty)
            return score, 0.0, penalty, "Ghost Detection"

    # Calculate Intersection & FP
    tp = np.logical_and(gt, pred).sum()
    fn = np.logical_and(gt, (1 - pred)).sum()
    fp = np.logical_and((1 - gt), pred).sum()
    
    # 1. Reward Term: Recall (How much GT did we find?)
    # We use Recall because IoU penalizes FP in denominator, but we want explicit control.
    recall = tp / (tp + fn)
    
    # 2. Penalty Term: Dynamic Weight
    # FP Ratio = Portion of the screen covered by mistakes
    fp_ratio = fp / total_pixels
    
    # Quadratic Penalty: Grows slowly at first, then explodes
    # If fp_ratio = tolerance (e.g., 0.5), penalty becomes 1.0
    penalty = (fp_ratio / tolerance) ** 2
    
    # 3. Final Score
    raw_score = recall - penalty
    final_score = max(0.0, raw_score)
    
    return final_score, recall, penalty, f"FP_Screen_Ratio: {fp_ratio:.3f}"

def main(gt_dir, pred_dir, tolerance):
    # Get files
    gt_files = sorted(glob(os.path.join(gt_dir, "*.*")))
    
    scores = []
    recalls = []
    penalties = []
    
    print(f"{'Filename':<20} | {'Score (Ours)':<12} | {'Recall':<8} | {'Penalty':<8} | {'Info'}")
    print("-" * 75)
    
    for gt_path in gt_files:
        if not gt_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
        
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)
        
        if not os.path.exists(pred_path):
            continue
            
        gt_img = cv2.imread(gt_path, 0)
        pred_img = cv2.imread(pred_path, 0)
        
        if gt_img is None or pred_img is None: continue
        
        # Resize safety
        if gt_img.shape != pred_img.shape:
             pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        score, rec, pen, info = calculate_custom_score(gt_img, pred_img, tolerance)
        
        scores.append(score)
        recalls.append(rec)
        penalties.append(pen)
        
        print(f"{filename[:20]:<20} | {score:.4f}       | {rec:.4f}   | {pen:.4f}   | {info}")

    if scores:
        print("-" * 75)
        print(f"Summary with Tolerance={tolerance} (50% screen = Death):")
        print(f"  > Mean Custom Score : {np.mean(scores):.4f}  <-- USE THIS!")
        print(f"  > Mean Recall       : {np.mean(recalls):.4f}")
        print(f"  > Mean Penalty      : {np.mean(penalties):.4f}")
    else:
        print("No images processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--tol', type=float, default=0.5, 
                        help='Fraction of screen area for FP that reduces score to 0. Default 0.5 (Half Screen)')
    args = parser.parse_args()
    
    main(args.gt, args.pred, args.tol)