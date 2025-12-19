#!/usr/bin/env python3
"""
Classify frames into pass/fail bins by comparing GT instance labels and predicted binary masks.

For each GT file (PNG) in --gt, the script looks for the same-named file in --pred.
It creates an overlay visualization (GT contours in green, pred components in red) with
a short status text and saves the overlay into either --out-pass or --out-fail depending on
whether the frame passes the instance-level matching test.

Pass criteria (default):
 - Pixel union equality (GT>0 == pred>0) AND
 - All GT instances are matched by predicted components with IoU >= IOU_THRESH

Usage:
  python classify_frames.py --gt path\to\gt --pred path\to\pred --out-pass pass_dir --out-fail fail_dir

"""

import os
import cv2
import numpy as np
from glob import glob
from scipy.optimize import linear_sum_assignment
import argparse

IOU_THRESH = 0.5


def iou_mask(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / float(union)


def extract_gt_instances(gt):
    ids = np.unique(gt)
    ids = ids[ids != 0]
    masks = []
    for i in ids:
        masks.append(gt == i)
    return masks, ids


def extract_pred_instances(pred_bin):
    num, labels = cv2.connectedComponents(pred_bin.astype('uint8'))
    masks = []
    for lab in range(1, num):
        masks.append(labels == lab)
    return masks, labels


def match_instances(gt_masks, pred_masks, iou_thresh=IOU_THRESH):
    G = len(gt_masks)
    P = len(pred_masks)
    if G == 0 and P == 0:
        return 0, []
    if G == 0:
        return 0, []
    if P == 0:
        return 0, []

    iou_mat = np.zeros((P, G), dtype=float)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            iou_mat[i, j] = iou_mask(pm, gm)

    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_g = set()
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_thresh:
            matched_g.add(c)
            matches.append((r, c, iou_mat[r, c]))
    return len(matched_g), matches


def visualize_overlay(gt, pred_bin, pred_labels, outpath, status_text):
    # create base visualization: color pred mask as faint red background
    h, w = gt.shape[:2]
    base = np.zeros((h, w, 3), dtype=np.uint8)
    # fill pred in light red
    base[pred_bin > 0] = (30, 20, 200)
    # draw GT contours in green
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    for idv in gt_ids:
        mask = (gt == idv).astype('uint8') * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, contours, -1, (0, 255, 0), 2)
    # draw pred component contours in red
    if pred_labels is not None:
        maxlab = int(pred_labels.max())
        for lab in range(1, maxlab + 1):
            mask = (pred_labels == lab).astype('uint8') * 255
            if mask.sum() == 0:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, (0, 0, 255), 1)

    # overlay status text
    cv2.putText(base, status_text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(outpath, base)


def main(gt_dir, pred_dir, out_pass, out_fail, pattern='*.png'):
    os.makedirs(out_pass, exist_ok=True)
    os.makedirs(out_fail, exist_ok=True)

    gt_files = sorted(glob(os.path.join(gt_dir, pattern)))
    total = 0
    passed = 0
    failed = 0

    for gt_path in gt_files:
        total += 1
        name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, name)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print('Failed to read GT:', gt_path)
            continue
        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED) if os.path.exists(pred_path) else None
        if pred is None:
            pred_bin = np.zeros(gt.shape[:2], dtype=bool)
            pred_labels = None
        else:
            if pred.ndim == 3:
                # any nonzero channel considered foreground
                pred_bin = np.any(pred > 0, axis=2)
            else:
                pred_bin = pred > 0
            _, pred_labels = cv2.connectedComponents(pred_bin.astype('uint8'))

        gt_union = (gt > 0)
        pixel_equal = np.array_equal(gt_union, pred_bin)

        gt_masks, gt_ids = extract_gt_instances(gt)
        pred_masks, _ = extract_pred_instances(pred_bin)

        matched_g_count, matches = match_instances(gt_masks, pred_masks, IOU_THRESH)

        # decide pass/fail
        # pass if any of these conditions hold:
        # 1) pixel-equal and all GT instances matched
        # 2) prediction covers >=90% of GT area (intersection / GT_area >= 0.9)
        # 3) prediction area does not exceed GT area by more than 10% (pred_area <= 1.1 * GT_area)
        gt_union = (gt > 0)
        gt_area = int(gt_union.sum())
        pred_area = int(np.count_nonzero(pred_bin))
        # area ratio: pred_area / gt_area
        if gt_area > 0:
            area_ratio = pred_area / float(gt_area)
        else:
            area_ratio = 1.0 if pred_area == 0 else float('inf')

        cond1 = (pixel_equal and (matched_g_count == len(gt_masks)))
        # New area-based condition: pred area relative to GT area within [0.9, 1.1]
        cond_area = (gt_area > 0) and (0.8 <= area_ratio <= 1.8)

        if cond1 or cond_area:
            dest = out_pass
            passed += 1
            status = f'PASS gt={len(gt_masks)} pred={len(pred_masks)} matched={matched_g_count} area_ratio={area_ratio:.3f} pred/gt={pred_area}/{gt_area}'
        else:
            dest = out_fail
            failed += 1
            status = f'FAIL gt={len(gt_masks)} pred={len(pred_masks)} matched={matched_g_count} area_ratio={area_ratio:.3f} pred/gt={pred_area}/{gt_area} pixel_equal={pixel_equal}'

        outpath = os.path.join(dest, name)
        visualize_overlay(gt, pred_bin, pred_labels, outpath, status)

    print(f'Total {total}  Passed {passed}  Failed {failed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--out-pass', required=True)
    parser.add_argument('--out-fail', required=True)
    parser.add_argument('--pattern', default='*.png')
    args = parser.parse_args()
    main(args.gt, args.pred, args.out_pass, args.out_fail, args.pattern)
