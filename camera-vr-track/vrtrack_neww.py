import cv2
import numpy as np
import time
from multiprocessing import Process, Queue

# -------- Settings: 720p @ 60fps --------
CAP_W, CAP_H, CAP_FPS = 1280, 720, 60
DISPLAY_W, DISPLAY_H = 1280, 720
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS

BLACKLISTED_CAMERAS = ["HTC Multimedia Camera", "Virtual Camera twitch", "OBS Virtual Camera"]

def get_video_devices(max_devices=8):
    devices = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        frames_read = 0
        for _ in range(5):
            ret, _ = cap.read()
            if ret:
                frames_read += 1
        cap.release()
        if frames_read >= 3:
            devices.append(i)
    if len(devices) < 2:
        devices = [0, 0]
    return devices[:2]

def camera_worker(src, idx, out_queue):
    cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"[Cam {idx}] Failed to open camera {src}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"[Cam {idx}] Opened camera {src}")

    frame_counter = 0
    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Calculate FPS
        frame_counter += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_counter / (now - last_time)
            frame_counter = 0
            last_time = now

        while out_queue.qsize() > 1:
            try: out_queue.get_nowait()
            except Exception: break
        out_queue.put((idx, frame, fps))

if __name__ == "__main__":
    CAMERA_SOURCES = get_video_devices()
    print("Using camera indices:", CAMERA_SOURCES)

    queues = []
    processes = []
    for idx, src in enumerate(CAMERA_SOURCES):
        q = Queue()
        p = Process(target=camera_worker, args=(src, idx, q), daemon=True)
        p.start()
        queues.append(q)
        processes.append(p)
        time.sleep(1)

    num_cameras = max(1, len(CAMERA_SOURCES))
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Preview", DISPLAY_W * num_cameras, DISPLAY_H)

    frames = [None] * len(CAMERA_SOURCES)
    fps_vals = [0.0] * len(CAMERA_SOURCES)

    try:
        while True:
            for i, q in enumerate(queues):
                try:
                    idx, frame, fps = q.get(timeout=0.1)
                    frames[idx] = frame
                    fps_vals[idx] = fps
                except Exception:
                    pass

            grid_tiles = []
            for i, f in enumerate(frames):
                if f is None:
                    tile = 255 * np.ones((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                    cv2.putText(tile, f"Cam {i}: no feed", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    tile = cv2.resize(f, (DISPLAY_W, DISPLAY_H))
                    cv2.putText(tile, f"Cam {i}: {fps_vals[i]:.1f} FPS", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                grid_tiles.append(tile)
            grid = np.hstack(grid_tiles)

            cv2.imshow("Camera Preview", grid)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break
            time.sleep(FRAME_TIME)

    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            try:
                p.terminate()
                p.join(timeout=0.5)
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Exit.")
