import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient
import warnings
import sys
import pickle
import math
import os

# -------- Suppress Warnings / OpenCV logs (best-effort) --------
warnings.filterwarnings("ignore", category=UserWarning)
# Disable MSMF probing to reduce OBSensor noise (sets priority to 0)
os.environ['CV_VIDEOIO_PRIORITY_MSMF'] = '0'
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass
try:
    sys.stderr = DummyFile()
except Exception:
    pass

# -------- Settings: 720p @ 60fps --------
CAP_W, CAP_H, CAP_FPS = 1280, 720, 60
# Display size per camera (we'll show side-by-side)
DISPLAY_W, DISPLAY_H = 640, 360
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS

BLACKLISTED_CAMERAS = ["HTC Multimedia Camera", "Virtual Camera twitch", "OBS Virtual Camera"]

# OSC
VRCHAT_IP = "127.0.0.1"
VRCHAT_PORT = 9000
osc_client = SimpleUDPClient(VRCHAT_IP, VRCHAT_PORT)

# Tracker IDs for hip_center, left_elbow, right_elbow, left_knee, right_knee, left_ankle, right_ankle
tracker_ids = [2, 7, 8, 5, 6, 3, 4]

# Calibration file
CALIB_FILE = "stereo_calib.pkl"

# -------- Calibration pattern settings (tune for your checkerboard) --------
CHESSBOARD = (9, 6)         # internal corners (width, height)
SQUARE_SIZE = 0.024         # meters per square (set to your real square size)
REQUIRED_FRAMES = 15        # number of successful frames to collect per camera
MAX_CAPTURE_ATTEMPTS = 600  # safety limit (not used as blocking - just safety)

# -------- Globals for calibration & rectification (populated when loaded or after calibration) --------
camera_matrix1 = np.eye(3, dtype=np.float64)
dist_coeffs1 = np.zeros((5,1), dtype=np.float64)
camera_matrix2 = np.eye(3, dtype=np.float64)
dist_coeffs2 = np.zeros((5,1), dtype=np.float64)
R = np.eye(3, dtype=np.float64)
T = np.zeros((3,1), dtype=np.float64)
R1 = np.eye(3, dtype=np.float64)
R2 = np.eye(3, dtype=np.float64)
P1 = np.zeros((3,4), dtype=np.float64)
P2 = np.zeros((3,4), dtype=np.float64)
Q = None

# VRChat alignment globals
R_align = np.array([[-1, 0, 0],   # X flip if left/right swapped
                    [0, 1, 0],    # Y up
                    [0, 0, -1]])  # Z flip for forward/back
Z_OFFSET = -0.5                  # Shift back if avatar faces wrong way (tune: try -1.0 to 0.0)
FEET_OFFSET = np.array([0.0, -0.9, 0.0])  # Drop ankles ~0.9m below hip for floor level
hip_reference_height = 1.0       # Target VRChat hip Y (meters)
hip_scale_factor = 1.0           # Dynamic scale

def try_load_calibration(path=CALIB_FILE):
    global camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, R1, R2, P1, P2, Q
    if not os.path.exists(path):
        print("[CALIB] no calibration file found")
        return False
    try:
        with open(path, "rb") as f:
            calib = pickle.load(f)
        camera_matrix1 = calib['camera_matrix1']; dist_coeffs1 = calib['dist_coeffs1']
        camera_matrix2 = calib['camera_matrix2']; dist_coeffs2 = calib['dist_coeffs2']
        R = calib.get('R', R); T = calib.get('T', T)
        R1 = calib.get('R1', R1); R2 = calib.get('R2', R2)
        P1 = calib.get('P1', P1); P2 = calib.get('P2', P2)
        Q = calib.get('Q', Q)
        print(f"[CALIB] Loaded calibration from {path}")
        return True
    except Exception as e:
        print(f"[CALIB] Failed to load calibration: {e}")
        return False

try_load_calibration()

# -------- Utility functions (smoothing, quaternions, OSC send) --------
def smooth_point(new_point, last_point, alpha=0.3):
    if new_point is None:
        return last_point
    new_point = np.asarray(new_point, dtype=float)
    if last_point is None:
        return new_point
    return alpha * new_point + (1 - alpha) * last_point

def quat_from_rotation_matrix(Rm):
    # Rm: 3x3 rotation matrix -> quaternion [w, x, y, z]
    m = Rm
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        S = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        w = (m[2,1] - m[1,2]) / S
        x = 0.25 * S
        y = (m[0,1] + m[1,0]) / S
        z = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]:
        S = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        w = (m[0,2] - m[2,0]) / S
        x = (m[0,1] + m[1,0]) / S
        y = 0.25 * S
        z = (m[1,2] + m[2,1]) / S
    else:
        S = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        w = (m[1,0] - m[0,1]) / S
        x = (m[0,2] + m[2,0]) / S
        y = (m[1,2] + m[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z], dtype=float)

def align_to_vrchat(point_3d, is_feet=False):
    p = np.array(point_3d, dtype=float)
    p = R_align @ p
    if is_feet:
        p += FEET_OFFSET
    p[2] += Z_OFFSET  # Overall depth shift
    return p

def send_tracker(tracker_id, pos, quat=np.array([1.0,0.0,0.0,0.0]), is_feet=False):
    pos = align_to_vrchat(pos, is_feet)
    osc_client.send_message(f"/tracking/trackers/{tracker_id}/position", [float(pos[0]), float(pos[1]), float(pos[2])])
    osc_client.send_message(f"/tracking/trackers/{tracker_id}/rotation", [float(q) for q in quat.tolist()])

def send_head(pos, quat=np.array([1.0,0.0,0.0,0.0])):
    pos = align_to_vrchat(pos)
    osc_client.send_message("/tracking/trackers/head/position", [float(pos[0]), float(pos[1]), float(pos[2])])
    osc_client.send_message("/tracking/trackers/head/rotation", [float(q) for q in quat.tolist()])

# -------- Improved get_video_devices: Test open + read with DSHOW to avoid MSMF/OBSensor issues --------
def get_video_devices(max_devices=8):
    devices = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Force DSHOW early to bypass MSMF
        if not cap.isOpened():
            cap.release()
            continue
        # Set properties to match capture settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret, _ = cap.read()
        backend = cap.getBackendName()
        cap.release()
        if ret:
            name = backend or f"Device {i}"
            if name not in BLACKLISTED_CAMERAS:
                devices.append((i, name))
                print(f"[DEV] Camera {i}: {name} - OK (DSHOW)")
            else:
                print(f"[DEV] Camera {i}: {name} - Blacklisted")
        else:
            print(f"[DEV] Camera {i}: Failed to read frame (even with DSHOW)")
    if len(devices) < 2:
        print(f"[DEV] Warning: Only {len(devices)} valid cameras found. Using mono mode (MediaPipe 3D fallback).")
        if len(devices) == 0:
            print("[DEV] No cameras detected! Check connections/drivers.")
            devices = [(0, "Fallback Device 0")]  # Assume at least 0 for testing
    return devices[:2]  # Max 2

# -------- Camera worker: Force DSHOW to avoid MSMF errors --------
def camera_worker(src, idx, out_queue):
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # Force DSHOW
    if not cap.isOpened():
        print(f"[Cam {idx}] Failed to open {src} with DSHOW")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"[Cam {idx}] Opened {src} - Backend: {cap.getBackendName()}")

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)  # Reduced for FPS

    frame_counter = 0
    last_fps_time = time.perf_counter()
    fps = 0.0

    last_points_2d = None
    last_points_3d = None
    last_world_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[Cam {idx}] Read failed - retrying...")
            time.sleep(0.01)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)

        points_2d = None
        points_3d = None
        world_landmarks = None

        if results.pose_landmarks and results.pose_world_landmarks:
            lm = results.pose_landmarks.landmark
            lm3 = results.pose_world_landmarks.landmark

            # map MediaPipe indices to the 7 trackers we want:
            # hip_center, left_elbow, right_elbow, left_knee, right_knee, left_ankle, right_ankle
            # Build 2D pixel points
            def to2(i): return np.array([lm[i].x * CAP_W, lm[i].y * CAP_H])
            def to3(i): return np.array([lm3[i].x, lm3[i].y, lm3[i].z])

            # Indices: hips 23=right_hip, 24=left_hip; elbows 13=left,14=right; knees 25=left_knee,26=right_knee; ankles 27=right_ankle,28=left_ankle
            left_hip_2d = to2(24); right_hip_2d = to2(23)
            hip_center_2d = (left_hip_2d + right_hip_2d) / 2
            left_elbow_2d = to2(13); right_elbow_2d = to2(14)
            left_knee_2d = to2(25); right_knee_2d = to2(26)
            left_ankle_2d = to2(28); right_ankle_2d = to2(27)

            points_2d = [hip_center_2d, left_elbow_2d, right_elbow_2d,
                         left_knee_2d, right_knee_2d, left_ankle_2d, right_ankle_2d]

            # 3D world landmarks (MediaPipe) for the same used points
            left_hip_3d = to3(24); right_hip_3d = to3(23)
            hip_center_3d = (left_hip_3d + right_hip_3d) / 2
            left_elbow_3d = to3(13); right_elbow_3d = to3(14)
            left_knee_3d = to3(25); right_knee_3d = to3(26)
            left_ankle_3d = to3(28); right_ankle_3d = to3(27)

            points_3d = [hip_center_3d, left_elbow_3d, right_elbow_3d,
                         left_knee_3d, right_knee_3d, left_ankle_3d, right_ankle_3d]

            # also keep the full 33-world-landmark array for more advanced rotation computations
            world_landmarks = np.array([[p.x,p.y,p.z] for p in lm3], dtype=float)

            last_points_2d = points_2d
            last_points_3d = points_3d
            last_world_landmarks = world_landmarks

        frame_counter += 1
        now = time.perf_counter()
        if now - last_fps_time >= 1.0:
            fps = frame_counter / (now - last_fps_time)
            frame_counter = 0
            last_fps_time = now

        # keep queue short
        while out_queue.qsize() > 1:
            try: out_queue.get_nowait()
            except Exception: break
        out_queue.put((idx, frame, fps, last_points_2d, last_points_3d, last_world_landmarks))

# -------- Triangulation helper --------
def triangulate_keypoints(points2d_1, points2d_2):
    """Triangulate corresponding 2D points (lists of (x,y)) -> (N,3) 3D points in camera coord."""
    if points2d_1 is None or points2d_2 is None:
        return None
    pts1 = np.asarray(points2d_1, dtype=np.float32)
    pts2 = np.asarray(points2d_2, dtype=np.float32)
    if pts1.size == 0 or pts2.size == 0 or pts1.shape[0] != pts2.shape[0]:
        return None
    pts1_rect = cv2.undistortPoints(pts1.reshape(-1,1,2), camera_matrix1, dist_coeffs1, R=R1, P=P1)
    pts2_rect = cv2.undistortPoints(pts2.reshape(-1,1,2), camera_matrix2, dist_coeffs2, R=R2, P=P2)
    pts1_for_tri = pts1_rect.reshape(-1,2).T
    pts2_for_tri = pts2_rect.reshape(-1,2).T
    points_4d = cv2.triangulatePoints(P1, P2, pts1_for_tri, pts2_for_tri)
    w = points_4d[3,:]
    valid = np.abs(w) > 1e-8
    points_3d = np.zeros((points_4d.shape[1], 3), dtype=np.float32)
    points_3d[valid,0] = points_4d[0,valid] / w[valid]
    points_3d[valid,1] = points_4d[1,valid] / w[valid]
    points_3d[valid,2] = points_4d[2,valid] / w[valid]
    return points_3d

# -------- Background stereo calibration worker (to avoid blocking UI) --------
def stereo_calibration_worker(imgpts1, imgpts2, objp_template, image_size, result_queue):
    try:
        # prepare object points list repeated
        objpoints = [objp_template.astype(np.float32)] * len(imgpts1)
        # initial single-camera calibrations
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpts1, image_size, None, None)
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpts2, image_size, None, None)
        # stereo calibrate, fixing intrinsics to the single-camera results
        flags = cv2.CALIB_FIX_INTRINSIC
        stereoret, mtx1, dist1, mtx2, dist2, Rmat, Tvec, E, F = cv2.stereoCalibrate(
            objpoints, imgpts1, imgpts2,
            mtx1, dist1, mtx2, dist2,
            image_size,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
            flags=flags
        )
        R1_out, R2_out, P1_out, P2_out, Q_out, _, _ = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, image_size, Rmat, Tvec)
        calib = {
            'camera_matrix1': mtx1, 'dist_coeffs1': dist1,
            'camera_matrix2': mtx2, 'dist_coeffs2': dist2,
            'R': Rmat, 'T': Tvec, 'R1': R1_out, 'R2': R2_out, 'P1': P1_out, 'P2': P2_out, 'Q': Q_out
        }
        result_queue.put(('ok', calib))
    except Exception as e:
        result_queue.put(('err', str(e)))

# -------- Precompute object points template --------
objp_template = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp_template[:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1,2)
objp_template *= SQUARE_SIZE

# -------- Main process --------
if __name__ == "__main__":
    devices = get_video_devices()
    CAMERA_SOURCES = [index for index,name in devices]
    if len(CAMERA_SOURCES) < 2:
        # For mono mode, duplicate the first camera (no stereo triangulation)
        CAMERA_SOURCES += [CAMERA_SOURCES[0]] * (2 - len(CAMERA_SOURCES))
        print(f"[MAIN] Mono mode: Using {CAMERA_SOURCES[0]} for both cams (no stereo)")
    print("Using camera indices:", CAMERA_SOURCES)

    queues = []
    processes = []
    for idx, src in enumerate(CAMERA_SOURCES):
        q = Queue()
        p = Process(target=camera_worker, args=(src, idx, q), daemon=True)
        p.start()
        queues.append(q)
        processes.append(p)
        time.sleep(1)  # Stagger starts to avoid USB conflicts

    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Preview", DISPLAY_W * len(CAMERA_SOURCES), DISPLAY_H)

    frames = [None] * len(CAMERA_SOURCES)
    fps_vals = [0.0] * len(CAMERA_SOURCES)
    points_2d_all = [None] * len(CAMERA_SOURCES)   # 2D detections per cam (list of 7 points)
    points_3d_all = [None] * len(CAMERA_SOURCES)   # MediaPipe world 3D per cam (list of 7 points)
    world_landmarks_all = [None] * len(CAMERA_SOURCES)
    smoothed_points = None
    smoothed_head = None
    reset_smoothed = False  # Flag to indicate if smoothed points should be reset on next detection
    last_detection_time = 0.0
    SMOOTH_TIMEOUT = 2.0

    # Calibration state machine
    calib_state = "idle"      # idle | waiting | collecting | waiting_done | computing | done | failed
    current_cam = 0
    calib_start_time = 0.0
    calib_points_img = {0: [], 1: []}   # list of np.array corners (Nx2) per camera
    calib_objpoints = {0: [], 1: []}
    stereo_process = None
    stereo_result_queue = None

    overlay_text = None
    overlay_until = 0.0
    no_frames_timeout = time.time() + 30  # Exit if no frames in 30s

    try:
        while time.time() < no_frames_timeout:
            # --- pull latest frames from camera queues ---
            frame_count = 0
            for i, q in enumerate(queues):
                # drain and take latest
                data = None
                while True:
                    try:
                        item = q.get_nowait()
                        data = item
                    except Exception:
                        break
                if data:
                    idx, frame, fps, p2d, p3d, world_landmarks = data
                    frames[idx] = frame
                    fps_vals[idx] = fps
                    points_2d_all[idx] = p2d
                    points_3d_all[idx] = p3d
                    world_landmarks_all[idx] = world_landmarks
                    frame_count += 1
                    # Reset timeout on activity
                    no_frames_timeout = time.time() + 30

            if frame_count == 0:
                print("[MAIN] No frames received yet - waiting...")
                time.sleep(1)
                continue

            now = time.time()
            # --- stereo triangulation (if both cameras have 2D detections and calibration exists and stereo mode) ---
            triangulated_points = None
            head_3d = None
            if len(set(CAMERA_SOURCES)) == 2 and points_2d_all[0] is not None and points_2d_all[1] is not None and np.any(P1) and np.any(P2):
                try:
                    triangulated_points = triangulate_keypoints(points_2d_all[0], points_2d_all[1])
                except Exception as e:
                    print(f"[TRIANG] Error: {e}")
                    triangulated_points = None

            # If triangulation isn't available, fallback to first camera's MediaPipe world points
            camera_origin = np.array([0.0, 0.0, 0.0])  # Adjust if cam0 isn’t at (0,0,0)
            if triangulated_points is None:
                triangulated_points = points_3d_all[0] if points_3d_all[0] is not None else points_3d_all[1]
                if triangulated_points is not None:
                    # Transform MediaPipe world coords (origin at cam0) to VRChat space
                    triangulated_points = [np.array(p) + camera_origin for p in triangulated_points]
                head_3d = None
                if world_landmarks_all[0] is not None:
                    # head: use ear average from world_landmarks (indices 7 left_ear, 8 right_ear)
                    try:
                        left_ear = world_landmarks_all[0][7]; right_ear = world_landmarks_all[0][8]
                        head_3d = (left_ear + right_ear) / 2.0 + camera_origin
                    except Exception:
                        head_3d = None

            # --- Send OSC: positions with smoothing; compute rotations from available landmarks ---
            if triangulated_points is not None:
                last_detection_time = now
                if smoothed_points is None or reset_smoothed or (now - last_detection_time > SMOOTH_TIMEOUT):
                    smoothed_points = [np.array(p, dtype=float) for p in triangulated_points]
                    reset_smoothed = False
                else:
                    smoothed_points = [smooth_point(p, sp) for p, sp in zip(triangulated_points, smoothed_points)]

                # Dynamic scaling from detected hip height
                if smoothed_points:
                    hip_pos = smoothed_points[0]  # hip_center
                    detected_hip_y = abs(hip_pos[1])  # MediaPipe Y is up, but could be negative
                    if detected_hip_y > 0.1:  # Valid detection
                        hip_scale_factor = hip_reference_height / detected_hip_y
                    # Apply scale to all points
                    smoothed_points = [np.array(p) * hip_scale_factor for p in smoothed_points]
                    if head_3d is not None:
                        head_3d = np.array(head_3d) * hip_scale_factor

                # Center around hip
                if smoothed_points:
                    hip_pos = np.array(smoothed_points[0])
                    centered_points = [np.array(p) - hip_pos for p in smoothed_points]

                    # Estimate a torso rotation (use world landmarks if available, else infer from triangulated points)
                    quat = np.array([1.0,0.0,0.0,0.0])
                    try:
                        # Prefer full world landmarks for rotation if available (camera 0)
                        wl = world_landmarks_all[0]
                        if wl is not None:
                            # Use shoulders (11 left_shoulder, 12 right_shoulder) and hips (23 right_hip, 24 left_hip)
                            left_sh = wl[11] if wl.shape[0] > 11 else None
                            right_sh = wl[12] if wl.shape[0] > 12 else None
                            left_hp = wl[24] if wl.shape[0] > 24 else None
                            right_hp = wl[23] if wl.shape[0] > 23 else None
                            if left_sh is not None and right_sh is not None and left_hp is not None and right_hp is not None:
                                x_axis = (right_sh - left_sh)
                                if np.linalg.norm(x_axis) < 1e-6:
                                    x_axis = np.array([1.0,0.0,0.0])
                                else:
                                    x_axis /= np.linalg.norm(x_axis)
                                y_axis = ((left_sh + right_sh)/2.0) - ((left_hp + right_hp)/2.0)
                                if np.linalg.norm(y_axis) < 1e-6:
                                    y_axis = np.array([0.0,1.0,0.0])
                                else:
                                    y_axis /= np.linalg.norm(y_axis)
                                z_axis = np.cross(x_axis, y_axis)
                                if np.linalg.norm(z_axis) < 1e-6:
                                    z_axis = np.array([0.0,0.0,1.0])
                                else:
                                    z_axis /= np.linalg.norm(z_axis)
                                # re-orthonormalize y
                                y_axis = np.cross(z_axis, x_axis)
                                y_axis /= np.linalg.norm(y_axis)
                                Rm = np.column_stack((x_axis, y_axis, z_axis))
                                quat = quat_from_rotation_matrix(Rm)
                        else:
                            # fallback: use triangulated hip & elbows to get lateral
                            if len(centered_points) >= 3:
                                hip = np.array(centered_points[0], dtype=float)
                                left_el = np.array(centered_points[1], dtype=float)
                                right_el = np.array(centered_points[2], dtype=float)
                                x_axis = (right_el - left_el)
                                if np.linalg.norm(x_axis) < 1e-6:
                                    x_axis = np.array([1.0,0.0,0.0])
                                else:
                                    x_axis /= np.linalg.norm(x_axis)
                                # For y_axis, since no chest, use a proxy from hip to average elbow height or default
                                avg_elbow = (left_el + right_el) / 2
                                y_axis = avg_elbow - hip
                                if np.linalg.norm(y_axis) < 1e-6:
                                    y_axis = np.array([0.0,1.0,0.0])
                                else:
                                    y_axis /= np.linalg.norm(y_axis)
                                z_axis = np.cross(x_axis, y_axis)
                                if np.linalg.norm(z_axis) < 1e-6:
                                    z_axis = np.array([0.0,0.0,1.0])
                                else:
                                    z_axis /= np.linalg.norm(z_axis)
                                y_axis = np.cross(z_axis, x_axis)
                                y_axis /= np.linalg.norm(y_axis)
                                Rm = np.column_stack((x_axis, y_axis, z_axis))
                                quat = quat_from_rotation_matrix(Rm)
                    except Exception as e:
                        print(f"[ROT] Error computing quat: {e}")
                        quat = np.array([1.0,0.0,0.0,0.0])

                    # Send trackers with per-limb quats where possible (using centered_points)
                    if len(centered_points) >= 7 and wl is not None:
                        # Left knee quat
                        hip = np.array(centered_points[0], dtype=float)
                        left_knee = np.array(centered_points[3], dtype=float)
                        left_ankle = np.array(centered_points[5], dtype=float)
                        quat_left_knee = quat.copy()
                        knee_bone = left_knee - hip
                        calf_bone = left_ankle - left_knee
                        if np.linalg.norm(knee_bone) > 1e-6 and np.linalg.norm(calf_bone) > 1e-6:
                            norm_knee = knee_bone / np.linalg.norm(knee_bone)
                            norm_calf = calf_bone / np.linalg.norm(calf_bone)
                            axis = np.cross(norm_knee, norm_calf)
                            if np.linalg.norm(axis) > 1e-6:
                                axis /= np.linalg.norm(axis)
                                angle = np.arccos(np.clip(np.dot(norm_knee, norm_calf), -1,1))
                                quat_left_knee = np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])
                                quat_left_knee /= np.linalg.norm(quat_left_knee)

                        # Right knee quat (mirror)
                        right_knee = np.array(centered_points[4], dtype=float)
                        right_ankle = np.array(centered_points[6], dtype=float)
                        quat_right_knee = quat.copy()
                        knee_bone_r = right_knee - hip
                        calf_bone_r = right_ankle - right_knee
                        if np.linalg.norm(knee_bone_r) > 1e-6 and np.linalg.norm(calf_bone_r) > 1e-6:
                            norm_knee_r = knee_bone_r / np.linalg.norm(knee_bone_r)
                            norm_calf_r = calf_bone_r / np.linalg.norm(calf_bone_r)
                            axis_r = np.cross(norm_knee_r, norm_calf_r)
                            if np.linalg.norm(axis_r) > 1e-6:
                                axis_r /= np.linalg.norm(axis_r)
                                angle_r = np.arccos(np.clip(np.dot(norm_knee_r, norm_calf_r), -1,1))
                                quat_right_knee = np.array([np.cos(angle_r/2), *(np.sin(angle_r/2) * axis_r)])
                                quat_right_knee /= np.linalg.norm(quat_right_knee)

                        # Send with specific quats
                        send_tracker(tracker_ids[0], centered_points[0], quat)  # hip
                        send_tracker(tracker_ids[1], centered_points[1], quat)  # left_elbow
                        send_tracker(tracker_ids[2], centered_points[2], quat)  # right_elbow
                        send_tracker(tracker_ids[3], centered_points[3], quat_left_knee)  # left_knee
                        send_tracker(tracker_ids[4], centered_points[4], quat_right_knee)  # right_knee
                        send_tracker(tracker_ids[5], centered_points[5], quat, is_feet=True)  # left_ankle
                        send_tracker(tracker_ids[6], centered_points[6], quat, is_feet=True)  # right_ankle
                    else:
                        # Fallback: send all with torso quat
                        for tid, pos in zip(tracker_ids, centered_points):
                            is_foot = tid in [3, 4]  # Left/right ankle
                            send_tracker(tid, pos, quat, is_foot)

                    if head_3d is not None:
                        smoothed_head = smooth_point(head_3d, smoothed_head)
                        send_head(smoothed_head, quat)
            else:
                # No detection: timeout reset
                if now - last_detection_time > SMOOTH_TIMEOUT:
                    for tid in tracker_ids:
                        send_tracker(tid, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
                    if smoothed_head is not None:
                        send_head(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
                    smoothed_points = None
                    smoothed_head = None

            # --- Build preview grid ---
            grid_tiles = []
            for i in range(len(frames)):
                f = frames[i]
                if f is None:
                    tile = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                    cv2.putText(tile, f"Cam {i}: no feed", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    tile = f.copy()
                    # Draw green dots for the 7 tracked points if available
                    if points_2d_all[i] is not None:
                        for pt in points_2d_all[i]:
                            pt = tuple(pt.astype(int))
                            cv2.circle(tile, pt, 5, (0, 255, 0), -1)
                    # Resize for display
                    tile = cv2.resize(tile, (DISPLAY_W, DISPLAY_H))
                    cv2.putText(tile, f"Cam {i}: {fps_vals[i]:.1f} FPS", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                grid_tiles.append(tile)
            grid = np.hstack(grid_tiles)

            # Calibration status overlay
            if not np.any(P1) or not np.any(P2):
                cv2.putText(grid, "UNCALIBRATED - Using MediaPipe Fallback", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if len(set(CAMERA_SOURCES)) < 2:
                cv2.putText(grid, "MONO MODE - No Stereo Triangulation", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # --- Calibration state machine (non-blocking) ---
            if calib_state == "waiting":
                remaining = 10 - int(now - calib_start_time)
                if remaining <= 0:
                    calib_state = "collecting"
                    print(f"[CALIB] Start collecting frames for camera {current_cam}")
                else:
                    cv2.putText(grid, f"Calibration Cam {current_cam+1} in {remaining}s", (30,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

            elif calib_state == "collecting":
                # show progress overlay
                collected = len(calib_points_img[current_cam])
                cv2.putText(grid, f"Collecting Cam {current_cam+1}: {collected}/{REQUIRED_FRAMES}", (30,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

                # attempt detection on the live frame for current_cam
                frame_for_detection = frames[current_cam]
                if frame_for_detection is not None and len(frame_for_detection.shape) == 3:
                    try:
                        gray = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2GRAY)
                        # prefer the more robust SB detector if available
                        found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                        if not found:
                            found, corners = cv2.findChessboardCorners(gray, CHESSBOARD,
                                                                       flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                            if found:
                                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                        if found and corners is not None and corners.shape[0] == CHESSBOARD[0]*CHESSBOARD[1]:
                            # store image points and object points
                            calib_points_img[current_cam].append(corners.reshape(-1,2).astype(np.float32))
                            calib_objpoints[current_cam].append(objp_template.copy())
                            # draw corners onto preview tile
                            vis = frame_for_detection.copy()
                            cv2.drawChessboardCorners(vis, CHESSBOARD, corners, found)
                            grid[0:DISPLAY_H, current_cam*DISPLAY_W:(current_cam+1)*DISPLAY_W] = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
                            print(f"[CALIB] Cam{current_cam+1} collected {len(calib_points_img[current_cam])}/{REQUIRED_FRAMES}")
                            # Optional debounce: comment out for smoother live collection
                            # time.sleep(0.25)
                    except Exception as e:
                        print(f"[CALIB] Detection error for Cam {current_cam+1}: {e}")

                # If enough frames collected for this camera, transition
                if len(calib_points_img[current_cam]) >= REQUIRED_FRAMES:
                    overlay_text = f"Camera {current_cam+1} Calibrated"
                    overlay_until = now + 5.0
                    print(f"[CALIB] Camera {current_cam+1} collection complete")
                    # move state: if this was cam0 -> back to idle and wait for user to press c again for cam1
                    if current_cam == 0:
                        calib_state = "idle"
                        current_cam = 1
                    else:
                        # both cameras now have collections -> start stereo calibration in background
                        # pair up to min length
                        n_pairs = min(len(calib_points_img[0]), len(calib_points_img[1]))
                        if n_pairs >= REQUIRED_FRAMES // 2:  # At least some pairs
                            imgpts1 = calib_points_img[0][:n_pairs]
                            imgpts2 = calib_points_img[1][:n_pairs]
                            objp_used = objp_template
                            stereo_result_queue = Queue()
                            stereo_process = Process(
                                target=stereo_calibration_worker,
                                args=(imgpts1, imgpts2, objp_used, (CAP_W, CAP_H), stereo_result_queue),
                                daemon=True
                            )
                            stereo_process.start()
                            calib_state = "computing"
                            print("[CALIB] Started stereo calibration in background")
                        else:
                            print("[CALIB] Not enough pairs for stereo calibration")
                            calib_state = "idle"

            elif calib_state == "computing":
                cv2.putText(grid, "Computing stereo calibration...", (30,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 3)
                # check for result (non-blocking)
                if stereo_result_queue is not None and not stereo_result_queue.empty():
                    status, payload = stereo_result_queue.get()
                    if status == 'ok':
                        calib = payload
                        # save on disk
                        try:
                            with open(CALIB_FILE, "wb") as f:
                                pickle.dump(calib, f)
                            # update globals
                            camera_matrix1 = calib['camera_matrix1']; dist_coeffs1 = calib['dist_coeffs1']
                            camera_matrix2 = calib['camera_matrix2']; dist_coeffs2 = calib['dist_coeffs2']
                            R = calib['R']; T = calib['T']; R1 = calib['R1']; R2 = calib['R2']; P1 = calib['P1']; P2 = calib['P2']; Q = calib.get('Q', None)
                            overlay_text = "Stereo calibration successful"
                            overlay_until = now + 5.0
                            calib_state = "done"
                            print("[CALIB] Stereo calibration finished successfully and saved.")
                        except Exception as e:
                            overlay_text = f"Save failed: {e}"
                            overlay_until = now + 5.0
                            calib_state = "failed"
                            print(f"[CALIB] Error saving calibration: {e}")
                    else:
                        overlay_text = f"Calibration error: {payload}"
                        overlay_until = now + 6.0
                        calib_state = "failed"
                        print(f"[CALIB] Stereo calibration failed: {payload}")

            elif calib_state in ("done", "failed"):
                if overlay_text:
                    color = (0,255,0) if calib_state=="done" else (0,0,255)
                    cv2.putText(grid, overlay_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                if now > overlay_until:
                    # after showing result, go to idle
                    calib_state = "idle"
                    overlay_text = None

            # overlay any temporary message
            if overlay_text and now <= overlay_until:
                cv2.putText(grid, overlay_text, (30, DISPLAY_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.imshow("Camera Preview", grid)
            key = cv2.waitKey(1) & 0xFF

            # --- keyboard controls ---
            if key == ord('q') or key == 27:
                break
            elif key == ord('g'):
                # reset trackers: send zero positions, but set flag to reinitialize smoothing on next detection
                for tid in tracker_ids:
                    send_tracker(tid, np.zeros(3), np.array([1.0,0.0,0.0,0.0]))
                send_head(np.zeros(3), np.array([1.0,0.0,0.0,0.0]))
                smoothed_points = None  # Reset smoothing to reinitialize on next frame
                smoothed_head = None
                print("[OSC] Trackers reset - will reinitialize on next detection")
            elif key == ord('c'):
                # Initiate calibration flow non-blocking
                if calib_state == "idle":
                    # Clear for current_cam if re-calibrating
                    if len(calib_points_img[0]) >= REQUIRED_FRAMES and len(calib_points_img[1]) >= REQUIRED_FRAMES:
                        print("[CALIB] Both cameras already calibrated; press 'c' again to re-calibrate one.")
                        continue
                    # decide which camera to calibrate next (0 first, then 1)
                    if len(calib_points_img[0]) < REQUIRED_FRAMES:
                        current_cam = 0
                        calib_points_img[0] = []
                        calib_objpoints[0] = []
                    elif len(calib_points_img[1]) < REQUIRED_FRAMES:
                        current_cam = 1
                        calib_points_img[1] = []
                        calib_objpoints[1] = []
                    else:
                        current_cam = 0  # Reset to 0 for re-calib
                        calib_points_img[0] = []
                        calib_objpoints[0] = []

                    calib_state = "waiting"
                    calib_start_time = time.time()
                    overlay_text = f"Calibration for Cam {current_cam+1} starts in 10s"
                    overlay_until = time.time() + 2.0
                    print(f"[CALIB] Prepare checkerboard for camera {current_cam+1} (10s countdown)")

            # keep loop at target framerate
            time.sleep(FRAME_TIME)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("Shutting down processes...")
        for p in processes:
            try:
                p.terminate()
                p.join(timeout=0.5)
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Exit.")