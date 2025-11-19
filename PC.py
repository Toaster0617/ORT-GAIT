# ---------------- Import ----------------
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import threading
import math
import cupy as cp
import socket
#import json
#import base64
import struct

# ---------------- Configuration ----------------
"""
Modified Configurations:
1. SEND_BACK_IP(47)
2. IMAGE_RECV_PORT(48)
3. SEND_BACK_PORT(49)
4. YAW_PORT(53)
5. XR_PORT(57)
6. fov
7. dynamic_threshold
"""
global_yaw             = -30.0                      # Head Orientation from Unity（angle）
fov                    = 80                         # Viewing angle
dynamic_threshold      = 0                          # Dynamic threshold for shannon entropy
latest_image           = None                       # Latest Panorama Image
image_lock             = threading.Lock()           # Lock, Protect {latest_image}
lock                   = threading.Lock()           # Lock, Protect other thread parameters
_weights_cache         = {}                         # Blending weights parameters
_gpu_weights_gpu_cache = {}                         # Blending weights parameters in GPU
stop_flag              = False                      # Control threading
global_images = {f'cam{i}': None for i in range(6)} # 6 cams RGB images

# Angle of cams
cam_ranges = {
    "cam0": (math.radians(-180), math.radians(-120)),
    "cam1": (math.radians(-120), math.radians(-60)),
    "cam2": (math.radians(-60), math.radians(0)),
    "cam3": (math.radians(0), math.radians(60)),
    "cam4": (math.radians(60), math.radians(120)),
    "cam5": (math.radians(120), math.radians(180))
}
# ---------------- IP/PORT ----------------
# 1. Image Receive
SEND_BACK_IP    = '10.192.17.244'  # IPC IP 工控机IP
IMAGE_RECV_PORT = 7000             # IPC PORT, Receive Images
SEND_BACK_PORT  = 8001             # IPC PORT, Send {visible_cam} id

# 2. Head Orientation Yaw Server
YAW_IP   = '0.0.0.0' # Local IP (default)
YAW_PORT = 8083

# 3. VR/XR Image Server
XR_IP   = '0.0.0.0' # Local IP (default)
XR_PORT = 8082

# ---------------- Six Cams Image Receiver ----------------
def image_receiver():
    """
    UDP Receives:
    1. cam_key
    2. frame_id
    3. num_chunks
    4. idx|JPEG_BYTES
    Stores to {global_images}
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock.bind(('', IMAGE_RECV_PORT))
    print(f"Listening for images on UDP port {IMAGE_RECV_PORT}")
 
    buffer_dict = {}

    while not stop_flag:
        packet, _ = sock.recvfrom(65536)
        try:
            # First '|' to separate the header and chunk
            header_bytes, chunk = packet.split(b'|', 1)
            # Header format：b"cam_key,frame_id,num_chunks,idx"
            header_str = header_bytes.decode('utf-8')
            cam_key, frame_id_s, total_s, idx_s = header_str.split(',')
            frame_id = int(frame_id_s)
            total    = int(total_s)
            idx      = int(idx_s)

            key = (cam_key, frame_id)
            entry = buffer_dict.get(key)
            if entry is None:
                entry = {'total': total, 'chunks': {}, 'count': 0}
                buffer_dict[key] = entry

            # Avoid repeated sending of the same fragment due to network problems
            if idx not in entry['chunks']:
                entry['chunks'][idx] = chunk
                entry['count'] += 1

            # Reassemble if all received
            if entry['count'] == entry['total']:
                data = b''.join(entry['chunks'][i] for i in range(entry['total']))
                arr  = np.frombuffer(data, dtype=np.uint8)
                img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    with lock:
                        global_images[cam_key] = img
                # Clear cache
                del buffer_dict[key]

        except Exception as e:
            print(f"Image recv error: {e}")

    sock.close()

# ---------------- GPU Warping & Blending ----------------
def warp_perspective_gpu(image, H, size):
    """
    Warp with homography matrix
    1. Upload image to GPU
    2. Warping using homography matrix
    3. Output: Warped image
    """
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    gpu_H = np.array(H, dtype=np.float32)
    warped_gpu = cv2.cuda.warpPerspective(gpu_img, gpu_H, size)
    return warped_gpu

def single_weights_array(size: int) -> np.ndarray:
    """
    Generate weights from the center to the edge of the image
    1. Weights are closer to 1 for image centers, and closer to 0 for edges
    2. Center pixels have a greater influence on the result, while edges gradually weaken the weight
    3. Output: Weight array
    """
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
    """
    Overlapping areas are weighted to avoid noticeable seams:
    1. Method = 'exponent': Separable linear weights are calculated first, then the exponent is raised to the power of the weights.
    2. Method = 'gaussian': Weights are generated using a two-dimensional Gaussian function.

    Parameters:
    1. exponent    - The exponent of the power function. Only valid when method = 'exponent' (< 1 for smoother results).
    2. sigma_scale - The ratio of sigma to h and w. Only valid when method = 'gaussian'
    """
    if shape not in _weights_cache:
        h, w = shape

        if method == 'exponent':
            # Separate Linear Weights
            weights_row = single_weights_array(h)[:, None]
            weights_col = single_weights_array(w)[None, :]
            mat = (weights_row @ weights_col) ** exponent

        elif method == 'gaussian':
            # 2D Gaussian falloff. Coordinates are normalized to [-1,1]
            ys = np.linspace(-1, 1, h)[:, None]
            xs = np.linspace(-1, 1, w)[None, :]
            sigma_y = sigma_scale
            sigma_x = sigma_scale
            # Gaussian eq
            mat = np.exp(- (ys**2 / (2*sigma_y**2) + xs**2 / (2*sigma_x**2)))
            # Normalized to [0,1]
            mat = (mat - mat.min()) / (mat.max() - mat.min())

        else:
            raise ValueError(f"unknown method '{method}'")
        # Save weights matrix
        _weights_cache[shape] = mat.astype(np.float32)

    return _weights_cache[shape]

def precompute_all_weights(image_shapes: list):
    """
    Run when the main process is initialized to calculate the GPU weight map corresponding to all images at once and cache it
    """
    for shape in image_shapes:
        if shape not in _gpu_weights_gpu_cache:
            w_mat = single_weights_matrix(shape,method='exponent', exponent=6.5)
            w3 = np.repeat(w_mat[:, :, None], 3, axis=2).astype(np.float32)
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(w3)
            _gpu_weights_gpu_cache[shape] = gpu_mat

def add_image_gpu(panorama_gpu, img, H_img, fixed_offset, weights_gpu, panorama_size):
    """
    Add a new image to the existing panorama scene in GPU
    1. Apply {fixed_offset} warp to the panorama
    2. Calculate the projection areas of the new image on the panorama
    3. Construct a local Homography and translate it
    """
    # ---- Initialize weights for new panorama, else warp onto existing panorama ----
    if panorama_gpu is None:
        # Initialize panorama and weights
        panorama_gpu = cv2.cuda_GpuMat()
        panorama_gpu.upload(np.zeros((panorama_size[1], panorama_size[0], 3), np.uint8))
        weights_gpu = cv2.cuda_GpuMat()
        weights_gpu.upload(np.zeros((panorama_size[1], panorama_size[0], 3), np.float32))
    else:
        # Warp onto existing panorama
        panorama_gpu = cv2.cuda.warpPerspective(panorama_gpu, fixed_offset, panorama_size)
        weights_gpu  = cv2.cuda.warpPerspective(weights_gpu,  fixed_offset, panorama_size)

    # ---- Calculate the projection areas of the new image on the panorama ----
    h, w = img.shape[:2]
    # Corners coor of image
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [w, h, 1],
                        [0, h, 1]], dtype=np.float32).T 
    warped = H_img.dot(corners)
    warped /= warped[2:3, :] # Normalized
    xs, ys = warped[0, :], warped[1, :]
    # Max & Min x,y values for the region 
    xmin = max(int(np.floor(xs.min())) - 1, 0)
    ymin = max(int(np.floor(ys.min())) - 1, 0)
    xmax = min(int(np.ceil (xs.max())) + 1, panorama_size[0])
    ymax = min(int(np.ceil (ys.max())) + 1, panorama_size[1])
    roi_w, roi_h = xmax - xmin, ymax - ymin
    if roi_w <= 0 or roi_h <= 0:
        # Check whether the region is valid in panorama
        return panorama_gpu, weights_gpu

    # ---- Construct a local Homography ----
    T = np.array([[1, 0, -xmin],
                  [0, 1, -ymin],
                  [0, 0,     1 ]], dtype=np.float32) # Translation matrix
    H_local = T.dot(np.array(H_img, dtype=np.float32))

    # ---- Crop out the panorama & weights of the area ----
    pano_roi = panorama_gpu.rowRange(ymin, ymax).colRange(xmin, xmax)
    wts_roi  = weights_gpu .rowRange(ymin, ymax).colRange(xmin, xmax)

    # ---- Perform local warp ----
    # Image warping
    gpu_img      = cv2.cuda_GpuMat(); gpu_img.upload(img)
    warped_img   = cv2.cuda.warpPerspective(gpu_img, H_local, (roi_w, roi_h))
    # Weights matrix warping
    shape = img.shape[:2]
    if shape not in _gpu_weights_gpu_cache:
        w_mat = single_weights_matrix(shape,method='exponent', exponent=6.5)
        w3    = np.repeat(w_mat[:, :, None], 3, axis=2).astype(np.float32) # Expand to tri-channels to align colors
        tmp   = cv2.cuda_GpuMat(); tmp.upload(w3) # Upload weight to GPU
        _gpu_weights_gpu_cache[shape] = tmp
    warped_wmat = cv2.cuda.warpPerspective(_gpu_weights_gpu_cache[shape], H_local, (roi_w, roi_h)) # Perform local warp

    # ---- Download to CuPy for blending ----
    pano_cp      = cp.asarray(pano_roi.download())    # Area of {visible_cam}
    new_cp       = cp.asarray(warped_img.download())  # Warped image
    wts_cp       = cp.asarray(wts_roi.download())     # Area of Weights
    wmat_cp      = cp.asarray(warped_wmat.download()) # Warped weights

    # Calculate blending weights
    norm_w = wts_cp[:, :, 0] / (wts_cp[:, :, 0] + wmat_cp[:, :, 0] + 1e-8)
    norm_w = norm_w[:, :, cp.newaxis]
    blended = new_cp * (1 - norm_w) + pano_cp * norm_w
    blended = cp.clip(blended, 0, 255).astype(cp.uint8)

    # Update weights
    combined = wts_cp + wmat_cp
    mx = cp.max(combined)
    new_wts = cp.where(mx != 0,
                       combined / mx,
                       combined).astype(cp.float32)

    # ---- Upload blending results to GPU ----
    pano_roi.upload(blended.get())
    wts_roi.upload(new_wts.get())

    return panorama_gpu, weights_gpu

# ---------------- Visible Cams Calculation ----------------
def normalize_angle(angle):
    # Normalize angle
    while angle < -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle

def determine_visible_cams(yaw, fov, cam_ranges):
    """
    Calculate the visible cam according to head orientation
    """
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

# ---------------- Shannon entropy calculation ----------------
def shannon_entropy_gpu(image):
    """
    1. Flatten the grayscale image and calculate its histogram
    2. Calculate dynamic entropy
    Output: Entropy (float)
    """
    img_cp = cp.asarray(image.flatten())
    hist, _ = cp.histogram(img_cp, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -cp.sum(hist * cp.log2(hist))
    return float(entropy.get())

# ---------------- 主处理流程 ----------------
def main_processing():
    global stop_flag, global_yaw, latest_image

    # ------ A. Wait for First Frames ----------
    print("Waiting for first frames...")
    while True:
        with lock:
            ready = all(global_images[cam] is not None for cam in global_images)
        if ready:
            break
        time.sleep(0.05)

    # ------ B. Initialize & Calculate Weight Matrix ------
    with lock:
        init_imgs = {cam: global_images[cam] for cam in global_images}
    shapes = [img.shape[:2] for img in init_imgs.values()]
    precompute_all_weights(shapes)

    # ------ C. First Warping ------
    panorama_gpu, weights_gpu = None, None
    # Homography matrix of six cam images
    H0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H1 = np.array([[1.0, 0.0, 760], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H2 = np.array([[1.0, 0.0, 1510], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H3 = np.array([[1.0, 0.0, 2250], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H4 = np.array([[1.0, 0.0, 2969.2424], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H5 = np.array([[1.0, 0.0, 3750.2424], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    #offset1 = np.array([[1.0,0.0,784.6212],[0.0,1.0, 10.0],[0.0,0.0,1.0]], dtype=np.float32)
    offset34 = np.eye(3, dtype=np.float32)
    camera_params = {
        "cam0": {"H": H0, "offset": offset34, "size": (6500,810)},
        "cam1": {"H": H1, "offset": offset34, "size": (6500,810)},
        "cam2": {"H": H2, "offset": offset34, "size": (6500,810)},
        "cam3": {"H": H3, "offset": offset34, "size": (6500,810)},
        "cam4": {"H": H4, "offset": offset34, "size": (6500,810)},
        "cam5": {"H": H5, "offset": offset34, "size": (6500,810)},
    }
    # Warping
    for cam, params in camera_params.items():
        panorama_gpu, weights_gpu = add_image_gpu(
            panorama_gpu,
            init_imgs[cam],
            params["H"],
            params["offset"],
            weights_gpu,
            (4350,740)
        )
    prev_frames = {cam: None for cam in camera_params} # History frame dict
    print("Main Process Started — Full-resolution Dynamic Update")

    # ------ D. Determine visible cameras and blending ------
    while not stop_flag:
        with lock:
            cur_imgs = {cam: global_images[cam] for cam in camera_params}
            yaw_rad = math.radians(global_yaw)

        if any(img is None for img in cur_imgs.values()):
            time.sleep(0.01)
            continue

        visible = determine_visible_cams(yaw_rad, fov, cam_ranges)

        # Send {visible_cams} to image server
        try:
            msg = ",".join(visible).encode('utf-8')
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(msg, (SEND_BACK_IP, SEND_BACK_PORT))
            sock.close()
        except Exception as e:
            print(f"Visibility feedback error: {e}")
        
        # Blending according visible cams
        for cam in visible:
            img = cur_imgs[cam]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if prev_frames[cam] is None:
                prev_frames[cam] = gray
                continue

            diff = cv2.absdiff(prev_frames[cam], gray)
            _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            ent = shannon_entropy_gpu(mask)
            prev_frames[cam] = gray

            if ent < dynamic_threshold:
                continue

            params = camera_params[cam]
            panorama_gpu, weights_gpu = add_image_gpu(
                panorama_gpu,
                img,
                params["H"],
                params["offset"],
                weights_gpu,
                (4350,740)
            )

        # ------ E. Download Panorama ------
        pano = panorama_gpu.download()
        cv2.imshow('pano', pano)
        ret, buf = cv2.imencode('.jpg', pano, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with image_lock:
                latest_image = buf.tobytes()
        cv2.waitKey(1) & 0xFF

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
                # 根据偏移调整
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
                prefix = struct.pack('!I', len(latest_image))
                try:
                    conn.sendall(prefix + latest_image)
                except:
                    break
        time.sleep(1/30)
    conn.close()
    server.close()

# ---------------- Threading ----------------
if __name__ == '__main__':
    # start threads
    threading.Thread(target=image_receiver, daemon=True).start()
    threading.Thread(target=yaw_server,     daemon=True).start()
    threading.Thread(target=image_server,   daemon=True).start()
    # run main
    main_processing()