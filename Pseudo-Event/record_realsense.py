# -*- coding: utf-8 -*-
"""
Record RealSense color frames (and IMU gyro) to disk.
Saves RGB frames as numbered PNGs and IMU gyro z (timestamped) to CSV.

Usage:
  python record_realsense.py --outdir ./rec --max-frames 1000
"""

import os
import time
import csv
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs

def main(outdir, max_frames=None):
    os.makedirs(outdir, exist_ok=True)
    imu_path = os.path.join(outdir, 'imu_gyro.csv')

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    prev_time = time.time()
    frame_idx = 0

    frames_csv = os.path.join(outdir, 'frames.csv')
    with open(imu_path, 'w', newline='') as fcsv, open(frames_csv, 'w', newline='') as fframes_csv:
        writer = csv.writer(fcsv)
        writer.writerow(['timestamp_s', 'wx', 'wy', 'wz'])
        fwriter = csv.writer(fframes_csv)
        fwriter.writerow(['frame_idx', 'timestamp_s', 'filename'])

        try:
            while True:
                frames = pipeline.wait_for_frames()
                # IMU
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                if gyro_frame:
                    g = gyro_frame.as_motion_frame().get_motion_data()
                    ts = time.time()
                    writer.writerow([f"{ts:.6f}", f"{g.x:.6f}", f"{g.y:.6f}", f"{g.z:.6f}"])

                color = frames.get_color_frame()
                if not color:
                    continue
                img = np.asanyarray(color.get_data())
                ts = time.time()
                fname_only = f"{frame_idx:06d}.png"
                fname = os.path.join(outdir, fname_only)
                cv2.imwrite(fname, img)
                # record frame timestamp and filename
                fwriter.writerow([frame_idx, f"{ts:.6f}", fname_only])
                frame_idx += 1

                cv2.imshow('rec', img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()
    main(args.outdir, args.max_frames)
