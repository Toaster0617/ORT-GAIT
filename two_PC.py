# ---------------- Import ----------------
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import threading
import math
import cupy as cp
import socket
import struct

# ---------------- Configuration ----------------
global_yaw             = -30.0                      # Head Orientation from Unity（angle）
fov                    = 80                         # Viewing angle
latest_image           = None                       # Latest Panorama Image
image_lock             = threading.Lock()           # Lock, Protect {latest_image}
lock                   = threading.Lock()           # Lock, Protect other thread parameters
_weights_cache         = {}                         # Blending weights parameters
_gpu_weights_gpu_cache = {}                         # Blending weights parameters in GPU
stop_flag              = False                      # Control threading

# 修改1：改为2摄像头
global_images = {f'cam{i}': None for i in range(2)} # 2 cams RGB images

# Angle of cams (仅保留前两个)
cam_ranges = {
    "cam0": (math.radians(-180), math.radians(-120)),
    "cam1": (math.radians(-120), math.radians(-60))
}

# ---------------- IP/PORT ----------------
# 修改5：修改IP
SEND_BACK_IP    = '10.192.147.61'  # IPC IP 工控机IP
IMAGE_RECV_PORT = 7000             # IPC PORT, Receive Images
SEND_BACK_PORT  = 8001             # IPC PORT, Send {visible_cam} id

# 2. Head Orientation Yaw Server
YAW_IP   = '0.0.0.0' # Local IP (default)
YAW_PORT = 8083

# 3. VR/XR Image Server
XR_IP   = '0.0.0.0' # Local IP (default)
XR_PORT = 8082

# ---------------- Image Receiver ----------------
def image_receiver():
    """
    UDP Receives:
    1. cam_key
    2. frame_id
    3. num_chunks
    4. idx
    5. send_timestamp (新增：用于计算网络传输时间)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', IMAGE_RECV_PORT))
    print(f"Listening for images on UDP port {IMAGE_RECV_PORT}")
 
    buffer_dict = {}

    while not stop_flag:
        packet, _ = sock.recvfrom(65536)
        try:
            header_bytes, chunk = packet.split(b'|', 1)
            # 修改4：Header格式增加时间戳 b"cam_key,frame_id,num_chunks,idx,send_timestamp"
            header_str = header_bytes.decode('utf-8')
            cam_key, frame_id_s, total_s, idx_s, ts_s = header_str.split(',')
            
            frame_id = int(frame_id_s)
            total    = int(total_s)
            idx      = int(idx_s)
            send_ts  = float(ts_s)

            key = (cam_key, frame_id)
            entry = buffer_dict.get(key)
            if entry is None:
                entry = {'total': total, 'chunks': {}, 'count': 0, 'send_ts': send_ts}
                buffer_dict[key] = entry

            if idx not in entry['chunks']:
                entry['chunks'][idx] = chunk
                entry['count'] += 1

            # Reassemble if all received
            if entry['count'] == entry['total']:
                recv_time = time.time()
                # 计算网络传输时间
                net_time_ms = (recv_time - entry['send_ts']) * 1000
                print(f"[Timing] Network Transmission ({cam_key} Frame {frame_id}): {net_time_ms:.2f} ms")

                data = b''.join(entry['chunks'][i] for i in range(entry['total']))
                arr  = np.frombuffer(data, dtype=np.uint8)
                img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    with lock:
                        global_images[cam_key] = img
                del buffer_dict[key]

        except Exception as e:
            # 屏蔽偶尔的丢包报错，保持整洁
            pass

    sock.close()

# ---------------- GPU Warping & Blending ----------------
def warp_perspective_gpu(image, H, size):
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    gpu_H = np.array(H, dtype=np.float32)
    warped_gpu = cv2.cuda.warpPerspective(gpu_img, gpu_H, size)
    return warped_gpu

def single_weights_array(size: int) -> np.ndarray:
    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])

def single_weights_matrix(
    shape: tuple,
    method: str = 'gaussian',
    exponent: float = 0.5,
    sigma_scale: float = 0.25
) -> np.ndarray:
    if shape not in _weights_cache:
        h, w = shape
        if method == 'exponent':
            weights_row = single_weights_array(h)[:, None]
            weights_col = single_weights_array(w)[None, :]
            mat = (weights_row @ weights_col) ** exponent
        elif method == 'gaussian':
            ys = np.linspace(-1, 1, h)[:, None]
            xs = np.linspace(-1, 1, w)[None, :]
            sigma_y = sigma_scale
            sigma_x = sigma_scale
            mat = np.exp(- (ys**2 / (2*sigma_y**2) + xs**2 / (2*sigma_x**2)))
            mat = (mat - mat.min()) / (mat.max() - mat.min())
        else:
            raise ValueError(f"unknown method '{method}'")
        _weights_cache[shape] = mat.astype(np.float32)

    return _weights_cache[shape]

def precompute_all_weights(image_shapes: list):
    for shape in image_shapes:
        if shape not in _gpu_weights_gpu_cache:
            w_mat = single_weights_matrix(shape,method='exponent', exponent=6.5)
            w3 = np.repeat(w_mat[:, :, None], 3, axis=2).astype(np.float32)
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(w3)
            _gpu_weights_gpu_cache[shape] = gpu_mat

def add_image_gpu(panorama_gpu, img, H_img, fixed_offset, weights_gpu, panorama_size):
    if panorama_gpu is None:
        panorama_gpu = cv2.cuda_GpuMat()
        panorama_gpu.upload(np.zeros((panorama_size[1], panorama_size[0], 3), np.uint8))
        weights_gpu = cv2.cuda_GpuMat()
        weights_gpu.upload(np.zeros((panorama_size[1], panorama_size[0], 3), np.float32))
    else:
        panorama_gpu = cv2.cuda.warpPerspective(panorama_gpu, fixed_offset, panorama_size)
        weights_gpu  = cv2.cuda.warpPerspective(weights_gpu,  fixed_offset, panorama_size)

    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [w, h, 1],
                        [0, h, 1]], dtype=np.float32).T 
    warped = H_img.dot(corners)
    warped /= warped[2:3, :]
    xs, ys = warped[0, :], warped[1, :]
    
    xmin = max(int(np.floor(xs.min())) - 1, 0)
    ymin = max(int(np.floor(ys.min())) - 1, 0)
    xmax = min(int(np.ceil (xs.max())) + 1, panorama_size[0])
    ymax = min(int(np.ceil (ys.max())) + 1, panorama_size[1])
    roi_w, roi_h = xmax - xmin, ymax - ymin
    if roi_w <= 0 or roi_h <= 0:
        return panorama_gpu, weights_gpu

    T = np.array([[1, 0, -xmin],
                  [0, 1, -ymin],
                  [0, 0,     1 ]], dtype=np.float32)
    H_local = T.dot(np.array(H_img, dtype=np.float32))

    pano_roi = panorama_gpu.rowRange(ymin, ymax).colRange(xmin, xmax)
    wts_roi  = weights_gpu .rowRange(ymin, ymax).colRange(xmin, xmax)

    gpu_img      = cv2.cuda_GpuMat(); gpu_img.upload(img)
    warped_img   = cv2.cuda.warpPerspective(gpu_img, H_local, (roi_w, roi_h))
    
    shape = img.shape[:2]
    if shape not in _gpu_weights_gpu_cache:
        w_mat = single_weights_matrix(shape,method='exponent', exponent=6.5)
        w3    = np.repeat(w_mat[:, :, None], 3, axis=2).astype(np.float32) 
        tmp   = cv2.cuda_GpuMat(); tmp.upload(w3) 
        _gpu_weights_gpu_cache[shape] = tmp
    warped_wmat = cv2.cuda.warpPerspective(_gpu_weights_gpu_cache[shape], H_local, (roi_w, roi_h)) 

    pano_cp      = cp.asarray(pano_roi.download())    
    new_cp       = cp.asarray(warped_img.download())  
    wts_cp       = cp.asarray(wts_roi.download())     
    wmat_cp      = cp.asarray(warped_wmat.download()) 

    norm_w = wts_cp[:, :, 0] / (wts_cp[:, :, 0] + wmat_cp[:, :, 0] + 1e-8)
    norm_w = norm_w[:, :, cp.newaxis]
    blended = new_cp * (1 - norm_w) + pano_cp * norm_w
    blended = cp.clip(blended, 0, 255).astype(cp.uint8)

    combined = wts_cp + wmat_cp
    mx = cp.max(combined)
    new_wts = cp.where(mx != 0,
                       combined / mx,
                       combined).astype(cp.float32)

    pano_roi.upload(blended.get())
    wts_roi.upload(new_wts.get())

    return panorama_gpu, weights_gpu

# ---------------- Visible Cams Calculation ----------------
def normalize_angle(angle):
    while angle < -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle

def determine_visible_cams(yaw, fov, cam_ranges):
    fov_rad = math.radians(fov)
    view_min = (normalize_angle(yaw - fov_rad/2)) % (2*math.pi)
    view_max = (normalize_angle(yaw + fov_rad/2)) % (2*math.pi)
    if view_min <= view_max:
        view_intervals = [(view_min, view_max)]
    else:
        view_intervals = [(view_min, 2*math.pi), (0, view_max)]
    visible = []
    for cam, (cam_min, cam_max) in cam_ranges.items():
        cam_min = cam_min % (2*math.pi)
        cam_max = cam_max % (2*math.pi)
        if cam_min <= cam_max:
            cam_intervals = [(cam_min, cam_max)]
        else:
            cam_intervals = [(cam_min, 2*math.pi), (0, cam_max)]
        found = False
        for (a, b) in view_intervals:
            for (c, d) in cam_intervals:
                if max(a, c) < min(b, d):
                    found = True
        if found:
            visible.append(cam)
    return visible

# ---------------- 主处理流程 ----------------
def main_processing():
    global stop_flag, global_yaw, latest_image

    print("Waiting for first frames...")
    while True:
        with lock:
            ready = all(global_images[cam] is not None for cam in global_images)
        if ready:
            break
        time.sleep(0.05)

    with lock:
        init_imgs = {cam: global_images[cam] for cam in global_images}
    shapes = [img.shape[:2] for img in init_imgs.values()]
    precompute_all_weights(shapes)

    panorama_gpu, weights_gpu = None, None
    # 仅保留 H0 和 H1 
    H0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H1 = np.array([[1.0, 0.0, 760], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    offset34 = np.eye(3, dtype=np.float32)
    
    camera_params = {
        "cam0": {"H": H0, "offset": offset34, "size": (6500,810)},
        "cam1": {"H": H1, "offset": offset34, "size": (6500,810)}
    }

    for cam, params in camera_params.items():
        panorama_gpu, weights_gpu = add_image_gpu(
            panorama_gpu,
            init_imgs[cam],
            params["H"],
            params["offset"],
            weights_gpu,
            (4350,740)
        )
    print("Main Process Started — 2 Cameras Active")

    while not stop_flag:
        loop_start_time = time.time()

        with lock:
            cur_imgs = {cam: global_images[cam] for cam in camera_params}
            yaw_rad = math.radians(global_yaw)

        if any(img is None for img in cur_imgs.values()):
            time.sleep(0.01)
            continue

        # 修改3 & 4：仅做测算用途的 Visible 模块
        vis_start_time = time.time()
        visible = determine_visible_cams(yaw_rad, fov, cam_ranges)
        vis_end_time = time.time()
        print(f"[Timing] Visibility Calculation: {(vis_end_time - vis_start_time)*1000:.2f} ms")

        try:
            msg = ",".join(visible).encode('utf-8')
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(msg, (SEND_BACK_IP, SEND_BACK_PORT))
            sock.close()
        except Exception as e:
            pass
        
        # 修改3：无视 Visible 列表，强制对所有（两台）相机进行处理
        for cam in ["cam0", "cam1"]:
            img = cur_imgs[cam]
            params = camera_params[cam]
            # 注意：香农熵检查已被移除，保证每一帧都直接进行 GPU 并行拼图以测算完整负载
            panorama_gpu, weights_gpu = add_image_gpu(
                panorama_gpu,
                img,
                params["H"],
                params["offset"],
                weights_gpu,
                (1600,740)
            )
        loop_end_time = time.time()
        print(f"[Timing] PC.py Total Loop Execution: {(loop_end_time - loop_start_time)*1000:.2f} ms")
        print("-" * 40)
        pano = panorama_gpu.download()
        cv2.imshow('pano', pano)
        ret, buf = cv2.imencode('.jpg', pano, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with image_lock:
                latest_image = buf.tobytes()
        
        cv2.waitKey(1) & 0xFF

        # 修改4：计算 PC 整体运行时间
        

    stop_flag = True

# ---------------- Yaw Server ----------------
def yaw_server():
    global global_yaw
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((YAW_IP, YAW_PORT))
    print("Listening yaw on UDP port 8083")
    while not stop_flag:
        data, _ = sock.recvfrom(64)
        try:
            deg = float(data.decode())
            with lock:
                global_yaw = deg - 114
                if global_yaw > 180: global_yaw -= 360
                if global_yaw < -180: global_yaw += 360
        except:
            pass
    sock.close()

# ---------------- Image Server ----------------
def image_server():
    global latest_image
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((XR_IP, XR_PORT))
    server.listen(1)
    print("Image server TCP port 8082 ready")
    conn, _ = server.accept()
    while not stop_flag:
        with image_lock:
            if latest_image is not None:
                try:
                    # 获取当前发送时间戳
                    send_ts = time.time()
                    
                    # 打包规则：
                    # '!I'  -> 大端存储的 4 字节整数 (表示图片的长度)
                    # '<d'  -> 小端存储的 8 字节浮点数 (双精度 Double，表示时间戳，兼容 C# 的 BitConverter)
                    prefix = struct.pack('!I', len(latest_image)) + struct.pack('<d', send_ts)
                    
                    # 发送: 长度(4) + 时间戳(8) + 图片数据
                    conn.sendall(prefix + latest_image)
                except:
                    break
        time.sleep(1/30)
    conn.close()
    server.close()

# ---------------- Threading ----------------
if __name__ == '__main__':
    threading.Thread(target=image_receiver, daemon=True).start()
    threading.Thread(target=yaw_server,     daemon=True).start()
    threading.Thread(target=image_server,   daemon=True).start()
    main_processing()