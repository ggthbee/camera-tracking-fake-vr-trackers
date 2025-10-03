import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import threading
import time
import platform
import mediapipe as mp

# -------- Settings --------
CAMERA_SOURCES = [0, 1]     # list of camera indices
CAP_MODES = [
    (1920, 1080, 30),        # 1080p @ 30 FPS
]
MODE_INDEX = [0]

CAP_W, CAP_H, CAP_FPS = CAP_MODES[MODE_INDEX[0]]

GRID_ROWS, GRID_COLS = 1, 2
DISPLAY_SCALE = 1
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
SHOW_FPS = [True]

FPS_LOG_INTERVAL = 2.0
USE_CONVERT_RGB = True

# Backend selection
BACKEND = None
if platform.system() == "Windows":
    BACKEND = cv2.CAP_MSMF
else:
    BACKEND = 0

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# -------- MediaPipe --------
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2


# -------- Threaded Camera --------
class CameraThread:
    def __init__(self, src, idx):
        self.src = src
        self.idx = idx

        self.lock = threading.Lock()
        self.running = True

        self.frame = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)
        self.ret = False

        self.last_fps_time = time.perf_counter()
        self.frame_counter = 0
        self.fps = 0.0
        self.last_log_time = time.perf_counter()

        self.cap = None
        self.failed = False

        # Each camera has its own Pose instance
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark smoothing state
        self.prev_landmarks = None
        self.smooth_alpha = 0.7  # smoothing factor

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def _open_and_configure(self, width, height, fps):
        cap = cv2.VideoCapture(self.src, BACKEND)
        if not cap.isOpened() and BACKEND != cv2.CAP_DSHOW:
            cap.release()
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"[Cam {self.idx}] Failed to open")
            self.failed = True
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1 if USE_CONVERT_RGB else 0)
        except Exception:
            pass

        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except Exception:
            pass

        actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps    = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        print(f"[Cam {self.idx}] Requested {width}x{height}@{fps} MJPG -> "
              f"Actual {actual_width}x{actual_height}@{actual_fps:.1f} ({fourcc_str})")

        return cap

    def _smooth_landmarks(self, landmarks):
        """Apply EMA smoothing to landmarks."""
        smoothed = []
        if self.prev_landmarks is None:
            self.prev_landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]

        for i, lm in enumerate(landmarks):
            px, py, pz, pv = self.prev_landmarks[i]
            sx = self.smooth_alpha * px + (1 - self.smooth_alpha) * lm.x
            sy = self.smooth_alpha * py + (1 - self.smooth_alpha) * lm.y
            sz = self.smooth_alpha * pz + (1 - self.smooth_alpha) * lm.z
            sv = self.smooth_alpha * pv + (1 - self.smooth_alpha) * lm.visibility
            smoothed.append((sx, sy, sz, sv))

        self.prev_landmarks = smoothed

        new_landmarks = []
        for (sx, sy, sz, sv) in smoothed:
            landmark = landmark_pb2.NormalizedLandmark()
            landmark.x, landmark.y, landmark.z, landmark.visibility = sx, sy, sz, sv
            new_landmarks.append(landmark)

        return landmark_pb2.NormalizedLandmarkList(landmark=new_landmarks)

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

                if results.pose_landmarks:
                    smoothed_landmarks = self._smooth_landmarks(results.pose_landmarks.landmark)

                    h, w, _ = frame.shape

                    # Grab landmarks
                    left_ankle  = smoothed_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    right_ankle = smoothed_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    left_knee   = smoothed_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    right_knee  = smoothed_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    left_hip    = smoothed_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
                    right_hip   = smoothed_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    left_elbow  = smoothed_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    right_elbow = smoothed_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

                    # Compute hip center
                    hip_x = (left_hip.x + right_hip.x) / 2
                    hip_y = (left_hip.y + right_hip.y) / 2

                    # Draw green blobs
                    points = [
                        (int(left_ankle.x * w), int(left_ankle.y * h)),
                        (int(right_ankle.x * w), int(right_ankle.y * h)),
                        (int(left_knee.x * w), int(left_knee.y * h)),
                        (int(right_knee.x * w), int(right_knee.y * h)),
                        (int(hip_x * w), int(hip_y * h)),
                        (int(left_elbow.x * w), int(left_elbow.y * h)),
                        (int(right_elbow.x * w), int(right_elbow.y * h)),
                    ]

                    for (cx, cy) in points:
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                with self.lock:
                    self.ret = False

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy(), self.fps
            return False, np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8), self.fps

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.5)
        if self.cap:
            self.cap.release()


# -------- Grid display --------
def create_grid(frames, fps_values, rows, cols, show_fps, scale=DISPLAY_SCALE):
    h_resized = max(1, int(CAP_H * scale))
    w_resized = max(1, int(CAP_W * scale))

    tiles = []
    for i, f in enumerate(frames):
        if f is not None and f.size > 0:
            frame = cv2.resize(f, (w_resized, h_resized), interpolation=cv2.INTER_NEAREST)
            if show_fps:
                txt = f"Cam {i}: {fps_values[i]:.1f} FPS"
                cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            frame = np.zeros((h_resized, w_resized, 3), dtype=np.uint8)
            if show_fps:
                cv2.putText(frame, f"Cam {i}: Failed", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        tiles.append(frame)

    while len(tiles) < rows * cols:
        blank = np.zeros((h_resized, w_resized, 3), dtype=np.uint8)
        if show_fps:
            cv2.putText(blank, f"Cam {len(tiles)}: N/A", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        tiles.append(blank)

    rows_img = []
    for r in range(0, len(tiles), cols):
        rows_img.append(np.hstack(tiles[r:r+cols]))
    return np.vstack(rows_img)


# -------- Init cameras --------
cameras = [CameraThread(src, idx) for idx, src in enumerate(CAMERA_SOURCES)]

# Window
cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Preview",
                 int(CAP_W * DISPLAY_SCALE * GRID_COLS),
                 int(CAP_H * DISPLAY_SCALE * GRID_ROWS))

try:
    while True:
        loop_start = time.perf_counter()

        frames, fps_vals = [], []
        for cam in cameras:
            ret, frame, fps = cam.read()
            frames.append(frame if ret else None)
            fps_vals.append(fps)

        grid = create_grid(frames, fps_vals, GRID_ROWS, GRID_COLS, SHOW_FPS[0])
        cv2.imshow("Camera Preview", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in (ord('f'), ord('F')):
            SHOW_FPS[0] = not SHOW_FPS[0]

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
