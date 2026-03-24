import threading
import socket
import time
import numpy as np
import pyrealsense2 as rs
import cv2

# ---------------- Configuration ----------------
DEST_IP            = '10.192.147.61' # PC IP, Send images 主端IP
DEST_PORT          = 7000            # Send Images
VISIBILITY_PORT    = 8001            # Receive {visible_cam} id

# 修改1：改为2个指定的摄像头
CAMERAS = [
    ('338522300974', 'cam0'),
    ('309622301066', 'cam1'),
]

JPEG_QUALITY       = 90
ENCODE_PARAMS      = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
MAX_DGRAM_PAYLOAD  = 60000

stop_flag          = False
current_visible    = set(cam_key for _, cam_key in CAMERAS)

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---------------- Shannon Entropy (CPU版) ----------------
def calc_shannon_entropy(mask):
    """
    使用 OpenCV 和 NumPy 计算香农熵 (替代 PC 端的 CuPy 版本)
    """
    hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist[hist > 0]
    hist = hist / hist.sum() # 归一化为概率密度
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)

# ---------------- Visibility Listener ----------------
def visibility_listener():
    """
    接收 PC 端发来的可见性 ID，但我们在当前修改中不再限制相机的发送逻辑
    保留此线程是为了防止 PC 端的反馈包无处投递导致端口报错。
    """
    global current_visible
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', VISIBILITY_PORT))
    print(f"[Sender] Listening visibility feedback on UDP port {VISIBILITY_PORT} (Currently ignored for forcing benchmark)")
    while not stop_flag:
        data, _ = sock.recvfrom(1024)
        s = data.decode('utf-8').strip()
        if not s:
            continue
        new_vis = set(s.split(','))
        with threading.Lock():
            current_visible = new_vis
    sock.close()

# ---------------- Camera Processing Thread ----------------
def camera_thread(serial, cam_key):
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    frame_id = 0
    prev_gray = None # 用于存储上一帧以计算差异
    
    print(f"[Sender] {cam_key} thread started")
    try:
        while not stop_flag:
            # 修改4：测算整个工控机单相机的处理周期耗时
            loop_start_time = time.time()

            # 注意：这里删除了 if cam_key not in current_visible: continue 的判断
            # 强制所有相机都进行推流

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

            # 修改3：将 Shannon Entropy 逻辑平移至工控机端
            ent_start_time = time.time()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            entropy_val = 0.0
            
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                entropy_val = calc_shannon_entropy(mask)
            
            prev_gray = gray # 更新上一帧
            ent_end_time = time.time()

            # JPEG 压缩
            ok, buf = cv2.imencode('.jpg', img, ENCODE_PARAMS)
            if not ok:
                continue

            data = buf.tobytes()
            total_len = len(data)
            num_chunks = (total_len + MAX_DGRAM_PAYLOAD - 1) // MAX_DGRAM_PAYLOAD

            # 修改4/5：打上发送时间戳，供 PC 端计算网络延时
            send_ts = time.time()

            for idx in range(num_chunks):
                start = idx * MAX_DGRAM_PAYLOAD
                end   = min(start + MAX_DGRAM_PAYLOAD, total_len)
                chunk = data[start:end]

                # Header 格式新增 send_ts: "cam_key,frame_id,num_chunks,idx,send_timestamp|"
                header = f"{cam_key},{frame_id},{num_chunks},{idx},{send_ts}|".encode('utf-8')
                udp_sock.sendto(header + chunk, (DEST_IP, DEST_PORT))

            frame_id = (frame_id + 1) & 0xFFFFFFFF
            
            loop_end_time = time.time()
            
            # 打印本帧各项耗时与熵值
            ent_time_ms = (ent_end_time - ent_start_time) * 1000
            loop_time_ms = (loop_end_time - loop_start_time) * 1000
            
            # 为了避免刷屏太快，这里仅打印信息，你可以重定向输出或稍作截流
            print(f"[{cam_key}] Entropy Time: {ent_time_ms:.2f} ms | Total Loop Time: {loop_time_ms:.2f} ms | Entropy: {entropy_val:.4f}")

    finally:
        pipeline.stop()
        print(f"[Sender] {cam_key} stopped")

# ---------------- Main ----------------
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