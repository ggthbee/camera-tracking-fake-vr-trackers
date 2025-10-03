import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import threading
import time
import platform
import mediapipe as mp
from pythonosc import udp_client

# -------- Settings --------
CAMERA_SOURCES = [0, 1]
CAP_MODES = [(1920, 1080, 30)]
MODE_INDEX = [0]

CAP_W, CAP_H, CAP_FPS = CAP_MODES[MODE_INDEX[0]]
GRID_ROWS, GRID_COLS = 1, 2
DISPLAY_W, DISPLAY_H = 640, 480  # 480p per camera
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
SHOW_FPS = [True]
FPS_LOG_INTERVAL = 100.0
USE_CONVERT_RGB = True

BACKEND = cv2.CAP_MSMF if platform.system() == "Windows" else 0
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# -------- MediaPipe --------
mp_pose = mp.solutions.pose

# -------- OSC Settings --------
STEAMVR_IP = "127.0.0.1"
STEAMVR_PORT = 9000
osc_client = udp_client.SimpleUDPClient(STEAMVR_IP, STEAMVR_PORT)

# -------- Tracker Names --------
tracker_names = ["LeftFoot", "RightFoot", "LeftKnee", "RightKnee", "HipCenter", "LeftElbow", "RightElbow"]

# -------- Threaded Camera --------
class CameraThread:
    def __init__(self, src, idx):
        self.src = src
        self.idx = idx
        self.lock = threading.Lock()
        self.running = True
        self.frame = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)
        self.ret = False
        self.points = [(0, 0)] * 7
        self.last_fps_time = time.perf_counter()
        self.frame_counter = 0
        self.fps = 0.0
        self.last_log_time = time.perf_counter()
        self.cap = None
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prev_points = None
        self.smooth_alpha = 0.6
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def _open_and_configure(self, width, height, fps):
        cap = cv2.VideoCapture(self.src, BACKEND)
        if not cap.isOpened() and BACKEND != cv2.CAP_DSHOW:
            cap.release()
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[Cam {self.idx}] Failed to open")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        return cap

    def _smooth_points(self, points):
        if self.prev_points is None:
            self.prev_points = points
            return points
        smoothed = []
        for i, (x, y) in enumerate(points):
            px, py = self.prev_points[i]
            sx = self.smooth_alpha * px + (1 - self.smooth_alpha) * x
            sy = self.smooth_alpha * py + (1 - self.smooth_alpha) * y
            smoothed.append((sx, sy))
        self.prev_points = smoothed
        return smoothed

    def run(self):
        self.cap = self._open_and_configure(CAP_W, CAP_H, CAP_FPS)
        if self.cap is None:
            return
        while self.running:
            ret, frame = self.cap.read()
            now = time.perf_counter()
            self.frame_counter += 1
            dt = now - self.last_fps_time
            if dt >= 1.0:
                self.fps = self.frame_counter / dt
                self.frame_counter = 0
                self.last_fps_time = now
            if (now - self.last_log_time) >= FPS_LOG_INTERVAL:
                print(f"[Cam {self.idx}] Measured {self.fps:.1f} FPS")
                self.last_log_time = now
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_detector.process(rgb)
                points = []
                h, w, _ = frame.shape
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    left_ankle = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                                  lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)
                    right_ankle = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                                   lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
                    left_knee = (lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                                 lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h)
                    right_knee = (lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                                  lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
                    left_hip = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                                lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)
                    right_hip = (lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                                 lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                    hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                    left_elbow = (lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                                  lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h)
                    right_elbow = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                                   lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h)
                    points = [left_ankle, right_ankle, left_knee, right_knee, hip_center, left_elbow, right_elbow]
                    points = self._smooth_points(points)
                    # ✅ Draw per-camera points here
                    for (cx, cy) in points:
                        cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 0), -1)
                with self.lock:
                    self.ret = True
                    self.frame = frame
                    self.points = points
            else:
                with self.lock:
                    self.ret = False

    def read(self):
        with self.lock:
            if self.ret and hasattr(self, 'points'):
                return True, self.frame.copy(), self.fps, self.points
            return False, np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8), self.fps, [(0, 0)] * 7

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.5)
        if self.cap:
            self.cap.release()

# -------- OSC Send --------
def send_tracker(name, pos, quat=[0, 0, 0, 1]):
    x, y = pos  # Use 2D points directly
    z = 0.0     # Set z to 0 since we don't have 3D data
    osc_client.send_message(f"/tracker/{name}/pose", [float(x), float(y), float(z)] + quat)

# -------- Grid Display --------
def create_grid(frames, fps_values, rows, cols, scale_w=DISPLAY_W, scale_h=DISPLAY_H):
    tiles = []
    for i, f in enumerate(frames):
        if f is not None and f.size > 0:
            frame = cv2.resize(f, (scale_w, scale_h), interpolation=cv2.INTER_NEAREST)
            txt = f"Cam {i}: {fps_values[i]:.1f} FPS"
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            frame = np.zeros((scale_h, scale_w, 3), dtype=np.uint8)
        tiles.append(frame)
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((scale_h, scale_w, 3), dtype=np.uint8))
    rows_img = []
    for r in range(0, len(tiles), cols):
        rows_img.append(np.hstack(tiles[r:r + cols]))
    return np.vstack(rows_img)

# -------- Init Cameras --------
cameras = [CameraThread(src, idx) for idx, src in enumerate(CAMERA_SOURCES)]
cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Preview", DISPLAY_W * GRID_COLS, DISPLAY_H * GRID_ROWS)

# -------- Main Loop --------
try:
    while True:
        loop_start = time.perf_counter()
        frames, fps_vals, points_all = [], [], []

        for cam in cameras:
            ret, frame, fps, pts = cam.read()
            frames.append(frame)   # ✅ already has dots drawn
            fps_vals.append(fps)
            points_all.append(pts)

        # Send 2D points from first camera (if available)
        if points_all and points_all[0]:
            for name, pos in zip(tracker_names, points_all[0]):
                send_tracker(name, pos)

        grid = create_grid(frames, fps_vals, GRID_ROWS, GRID_COLS)
        cv2.imshow("Camera Preview", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elapsed = time.perf_counter() - loop_start
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    print("Stopping cameras...")
    for cam in cameras:
        cam.stop()
    cv2.destroyAllWindows()
    print("Cleanup complete.")
