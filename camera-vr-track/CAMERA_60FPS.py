import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import threading
import time
import platform

# -------- Settings --------
CAMERA_SOURCES = [0, 1, 2, 3]     # Camera indices
CAP_MODES = [
    (1280, 720, 60),              # 720p @ 60 FPS
    (640, 480, 100)               # 480p @ 100 FPS
]
MODE_INDEX = [0]

CAP_W, CAP_H, CAP_FPS = CAP_MODES[MODE_INDEX[0]]

GRID_ROWS, GRID_COLS = 2, 2
DISPLAY_SCALE = 0.5
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS
SHOW_FPS = [True]

FPS_LOG_INTERVAL = 2.0            # shorter log interval for tighter feedback
USE_CONVERT_RGB = True            # set to False to reduce CPU (may affect color/format)

# Try MSMF first on Windows; else default to DSHOW
BACKEND = None
if platform.system() == "Windows":
    BACKEND = cv2.CAP_MSMF  # better 60 fps in many cases
else:
    BACKEND = 0             # default

cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # avoid OpenCV internal thread thrash with our own threading

# -------- Threaded Camera --------
class CameraThread:
    def __init__(self, src, idx):
        self.src = src
        self.idx = idx

        self.lock = threading.Lock()
        self.running = True

        # frame storage
        self.frame = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)
        self.ret = False

        # fps measurement
        self.last_fps_time = time.perf_counter()
        self.frame_counter = 0
        self.fps = 0.0
        self.last_log_time = time.perf_counter()

        self.cap = None
        self.failed = False

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def _open_and_configure(self, width, height, fps):
        # open
        cap = cv2.VideoCapture(self.src, BACKEND)
        if not cap.isOpened() and BACKEND != cv2.CAP_DSHOW:
            # fallback to DSHOW on Windows
            cap.release()
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"[Cam {self.idx}] Failed to open")
            self.failed = True
            return None

        # property order many UVC drivers respect better
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # keep the most recent frame only (reduces lag, helps maintain fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # optionally avoid RGB conversion in the driver (can save CPU)
        try:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1 if USE_CONVERT_RGB else 0)
        except Exception:
            pass

        # try hardware accel hint (no-op if unsupported)
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except Exception:
            pass

        # Read back actuals
        actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps    = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        print(f"[Cam {self.idx}] Requested {width}x{height}@{fps} MJPG -> "
              f"Actual {actual_width}x{actual_height}@{actual_fps:.1f} ({fourcc_str})")

        if actual_fps < fps * 0.9:
            print(f"[Cam {self.idx}] Warning: driver reporting {actual_fps:.1f} FPS (< {fps})")

        return cap

    def run(self):
        self.cap = self._open_and_configure(CAP_W, CAP_H, CAP_FPS)
        if self.cap is None:
            return

        while self.running:
            ret, frame = self.cap.read()
            now = time.perf_counter()

            # FPS calc (per camera, sliding seconds)
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
                # keep lock very short
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                with self.lock:
                    self.ret = False

    def update_settings(self, width, height, fps):
        global CAP_W, CAP_H, CAP_FPS
        CAP_W, CAP_H, CAP_FPS = width, height, fps

        # stop / restart capture with new settings
        if self.cap:
            self.cap.release()
        self.cap = self._open_and_configure(width, height, fps)

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

        frames = []
        fps_vals = []
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
        elif key in (ord('r'), ord('R')):
            MODE_INDEX[0] = (MODE_INDEX[0] + 1) % len(CAP_MODES)
            new_w, new_h, new_fps = CAP_MODES[MODE_INDEX[0]]
            print(f"[ALL] Switching to {new_w}x{new_h}@{new_fps}")
            for cam in cameras:
                cam.update_settings(new_w, new_h, new_fps)

        # cap display loop to ~TARGET_FPS to spare CPU
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
