import threading
import socket
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import struct

# ---------------- Configuration ----------------
DEST_IP            = '192.168.137.1' # PC IP, Send images 主端IP
DEST_PORT          = 7000            # Send Images
VISIBILITY_PORT    = 8001            # Receive {visible_cam} id
CAMERAS = [
    ('338122301918', 'cam0'),
    ('338522301241', 'cam1'),
    ('309622301481', 'cam2'),
    ('309622301066', 'cam3'),
    ('242422304700', 'cam4'),
    ('338122300640', 'cam5'),
]
JPEG_QUALITY       = 90
ENCODE_PARAMS      = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
MAX_DGRAM_PAYLOAD  = 60000

stop_flag          = False
current_visible    = set(cam_key for _, cam_key in CAMERAS)

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def visibility_listener():
    global current_visible
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', VISIBILITY_PORT))
    print(f"[Sender] Listening visibility feedback on UDP port {VISIBILITY_PORT}")
    while not stop_flag:
        data, _ = sock.recvfrom(1024)
        s = data.decode('utf-8').strip()
        if not s:
            continue
        new_vis = set(s.split(','))
        with threading.Lock():
            current_visible = new_vis
    sock.close()

def camera_thread(serial, cam_key):
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    frame_id = 0
    print(f"[Sender] {cam_key} thread started")
    try:
        while not stop_flag:
            if cam_key not in current_visible:
                time.sleep(0.01)
                continue

            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                print(f"[Sender] {cam_key} frame error: {e}")
                continue

            aligned = align.process(frames)
            color = aligned.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())
            ok, buf = cv2.imencode('.jpg', img, ENCODE_PARAMS)
            if not ok:
                continue

            data = buf.tobytes()
            total_len = len(data)
            num_chunks = (total_len + MAX_DGRAM_PAYLOAD - 1) // MAX_DGRAM_PAYLOAD

            for idx in range(num_chunks):
                start = idx * MAX_DGRAM_PAYLOAD
                end   = min(start + MAX_DGRAM_PAYLOAD, total_len)
                chunk = data[start:end]

                header = f"{cam_key},{frame_id},{num_chunks},{idx}|".encode('utf-8')
                udp_sock.sendto(header + chunk, (DEST_IP, DEST_PORT))

            frame_id = (frame_id + 1) & 0xFFFFFFFF
            
    finally:
        pipeline.stop()
        print(f"[Sender] {cam_key} stopped")

def main():
    
    threading.Thread(target=visibility_listener, daemon=True).start()

    
    threads = []
    for serial, key in CAMERAS:
        t = threading.Thread(target=camera_thread, args=(serial, key), daemon=True)
        t.start()
        threads.append(t)

    print(f"[Sender] All camera threads running, sending to {DEST_IP}:{DEST_PORT}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Sender] Interrupted, stopping...")
    finally:
        global stop_flag
        stop_flag = True
        for t in threads:
            t.join()
        udp_sock.close()
        print("[Sender] Shutdown complete.")

if __name__ == '__main__':
    main()
