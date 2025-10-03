import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from pythonosc import udp_client
import threading
import time

# -------------------- CONFIG --------------------
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
FPS = 30
CAM_IDS = [0, 1, 2, 3]

# OSC config (VRChat)
OSC_IP = "127.0.0.1"
OSC_PORT = 9000

# Blob detector parameters
BLOB_PARAMS = cv2.SimpleBlobDetector_Params()
BLOB_PARAMS.filterByArea = True
BLOB_PARAMS.minArea = 5  # Stricter to reduce noise
BLOB_PARAMS.maxArea = 20
BLOB_PARAMS.filterByColor = True
BLOB_PARAMS.blobColor = 255
BLOB_PARAMS.filterByCircularity = True
BLOB_PARAMS.minCircularity = 0.6
BLOB_PARAMS.filterByConvexity = True
BLOB_PARAMS.minConvexity = 0.6
BLOB_PARAMS.filterByInertia = True
BLOB_PARAMS.minInertiaRatio = 0.4
BLOB_PARAMS.minThreshold = 220  # Higher to reduce noise
BLOB_PARAMS.maxThreshold = 255
BLOB_PARAMS.thresholdStep = 5

# Safe creation for Windows + all OpenCV versions
try:
    detector = cv2.SimpleBlobDetector_create(BLOB_PARAMS)
except AttributeError:
    detector = cv2.SimpleBlobDetector(BLOB_PARAMS)

SMOOTHING_ALPHA = 0.6  # Smoother tracking
MAX_CLUSTER_DIST = 50  # Max distance for blob matching (pixels)

# -------------------- GLOBAL STATE --------------------
osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
cameras = {}  # Store camera objects
calibration_data = [[] for _ in CAM_IDS]  # Store calibrated blobs per camera
rois = [None for _ in CAM_IDS]  # Store ROI per camera (x, y, w, h)
calibrated_cameras = [False for _ in CAM_IDS]  # Calibration status per camera
current_calib_camera = 0  # Current camera being calibrated

# -------------------- ROI SELECTION --------------------
def select_roi(frame, cam_id):
    print(f"[INFO] Select ROI for camera {cam_id} by dragging a rectangle, then press Enter")
    roi = cv2.selectROI(f"Select ROI - Camera {cam_id}", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Select ROI - Camera {cam_id}")
    return roi if roi[2] > 0 and roi[3] > 0 else None

# -------------------- CAMERA SETUP --------------------
def open_camera(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARN] Camera {cam_id} not available")
        return None
    print(f"[INFO] Camera {cam_id} opened")
    cameras[cam_id] = cap
    return cap

# -------------------- FRAME NORMALIZATION --------------------
def normalize_frame(frame):
    return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

# -------------------- IR / HIGH-BIT CONVERSION --------------------
def convert_frame_to_uint8(frame):
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)
    return frame

# -------------------- MARKER DETECTION --------------------
def detect_blobs(gray, roi=None):
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    if roi is not None:
        x, y, w, h = roi
        mask = np.zeros_like(th)
        mask[y:y+h, x:x+w] = 255
        th = cv2.bitwise_and(th, mask)

    keypoints = detector.detect(th)
    filtered_pts = []
    h, w = gray.shape
    for kp in keypoints:
        x, y = kp.pt
        if 10 < x < w - 10 and 10 < y < h - 10:  # Reduced edge margin
            if roi is None or (x >= roi[0] and x <= roi[0] + roi[2] and y >= roi[1] and y <= roi[1] + roi[3]):
                filtered_pts.append((x, y))
    pts = np.array(filtered_pts, dtype=np.float32) if filtered_pts else np.empty((0, 2), dtype=np.float32)
    return pts

# -------------------- CLUSTERING --------------------
def cluster_points(pts, eps=20, min_samples=1):
    if len(pts) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = clustering.labels_
    clusters = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster_pts = pts[labels == lbl]
        cx, cy = np.mean(cluster_pts, axis=0)
        clusters.append((float(cx), float(cy)))
    return clusters

# -------------------- PROCESS FRAME --------------------
def process_frame(gray, roi=None):
    pts = detect_blobs(gray, roi)
    clusters = cluster_points(pts, eps=20)
    return clusters

# -------------------- OSC SENDING --------------------
def send_clusters_to_vr(clusters):
    for i, (x, y) in enumerate(clusters[:8]):  # Limit to 8 trackers
        vx = float(x / TARGET_WIDTH)
        vy = float(y / TARGET_HEIGHT)
        vz = 0.0
        try:
            osc_client.send_message(f"/tracking/trackers/{i+1}/position", [vx, vy, vz])
            osc_client.send_message(f"/tracking/trackers/{i+1}/rotation", [0.0, 0.0, 0.0])
        except Exception as e:
            print(f"[ERROR] OSC send failed for tracker {i+1}: {e}")

# -------------------- CAMERA THREAD --------------------
def camera_thread(cam_id, frame_buffer, clusters_buffer):
    cap = open_camera(cam_id)
    while True:
        if cap is None:
            frame_buffer[cam_id] = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
            clusters_buffer[cam_id] = []
            time.sleep(0.01)
            continue

        for _ in range(2):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            frame_buffer[cam_id] = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
            clusters_buffer[cam_id] = []
            continue

        frame = convert_frame_to_uint8(frame)
        if len(frame.shape) == 2:
            gray_frame = frame
            frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
            gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        frame_bgr = normalize_frame(frame_bgr)
        gray_frame = cv2.resize(gray_frame, (TARGET_WIDTH, TARGET_HEIGHT))

        clusters = process_frame(gray_frame, rois[cam_id] if calibrated_cameras[cam_id] else None)

        for x, y in clusters:
            cv2.circle(frame_bgr, (int(x), int(y)), 5, (0, 0, 255), 2)
        if rois[cam_id] is not None:
            x, y, w, h = rois[cam_id]
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame_buffer[cam_id] = frame_bgr
        clusters_buffer[cam_id] = clusters
        time.sleep(1.0 / FPS)

# -------------------- CALIBRATION AND TRACKING --------------------
def calibrate_camera(cam_id, clusters_buffer, frame_buffer):
    global current_calib_camera, calibrated_cameras
    if clusters_buffer[cam_id]:
        calibration_data[cam_id] = clusters_buffer[cam_id][:]
        rois[cam_id] = select_roi(frame_buffer[cam_id], cam_id)
        calibrated_cameras[cam_id] = True
        print(f"[INFO] Camera {cam_id} calibrated: {len(calibration_data[cam_id])} blobs, ROI: {rois[cam_id]}")
    else:
        print(f"[WARN] No blobs detected for camera {cam_id} during calibration")
    current_calib_camera += 1

def skip_camera(cam_id):
    global current_calib_camera
    calibrated_cameras[cam_id] = False
    calibration_data[cam_id] = []
    rois[cam_id] = None
    print(f"[INFO] Camera {cam_id} skipped")
    current_calib_camera += 1

def compute_relative_clusters(clusters_buffer, calibration_data):
    relative_clusters = []
    for cam_id, (current_clusters, calib_clusters) in enumerate(zip(clusters_buffer, calibration_data)):
        if not calibrated_cameras[cam_id] or not calib_clusters:
            continue
        # Match each current cluster to the closest calibration cluster
        for curr_x, curr_y in current_clusters:
            min_dist = float('inf')
            rel_x, rel_y = 0.0, 0.0
            for calib_x, calib_y in calib_clusters:
                dist = ((curr_x - calib_x) ** 2 + (curr_y - calib_y) ** 2) ** 0.5
                if dist < min_dist and dist < MAX_CLUSTER_DIST:
                    min_dist = dist
                    rel_x = curr_x - calib_x
                    rel_y = curr_y - calib_y
            if min_dist < MAX_CLUSTER_DIST:
                relative_clusters.append((rel_x, rel_y))
    return relative_clusters

# -------------------- MAIN --------------------
def main():
    global current_calib_camera
    frame_buffer = [np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) for _ in CAM_IDS]
    clusters_buffer = [[] for _ in CAM_IDS]
    smoothed_clusters = []

    # Start camera threads
    threads = []
    for cam_id in CAM_IDS:
        t = threading.Thread(target=camera_thread, args=(cam_id, frame_buffer, clusters_buffer), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            tiled = np.hstack(frame_buffer)
            if current_calib_camera < len(CAM_IDS):
                cv2.putText(tiled, f"Calibrate camera {CAM_IDS[current_calib_camera]}: Assume A-pose, press 'c' or 'v' to skip",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(tiled, "Calibration complete, tracking...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Reflective Tracker -> VRChat", tiled)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('c') and current_calib_camera < len(CAM_IDS):
                calibrate_camera(CAM_IDS[current_calib_camera], clusters_buffer, frame_buffer)
            elif key == ord('v') and current_calib_camera < len(CAM_IDS):
                skip_camera(CAM_IDS[current_calib_camera])

            if current_calib_camera >= len(CAM_IDS):
                final_clusters = compute_relative_clusters(clusters_buffer, calibration_data)
                if not final_clusters and any(len(c) > 0 for c in clusters_buffer):
                    print("[DEBUG] No valid relative clusters (check MAX_CLUSTER_DIST or calibration)")

                # Smoothing with persistent tracking
                if smoothed_clusters:
                    new_smoothed = []
                    for new_cluster in final_clusters:
                        min_dist = float('inf')
                        best_old = None
                        for old_cluster in smoothed_clusters:
                            dist = ((new_cluster[0] - old_cluster[0]) ** 2 + (new_cluster[1] - old_cluster[1]) ** 2) ** 0.5
                            if dist < min_dist and dist < MAX_CLUSTER_DIST:
                                min_dist = dist
                                best_old = old_cluster
                        if best_old is not None:
                            x_new = (SMOOTHING_ALPHA * float(new_cluster[0]) + (1 - SMOOTHING_ALPHA) * float(best_old[0]))
                            y_new = (SMOOTHING_ALPHA * float(new_cluster[1]) + (1 - SMOOTHING_ALPHA) * float(best_old[1]))
                            new_smoothed.append((x_new, y_new))
                        else:
                            new_smoothed.append(new_cluster)
                    smoothed_clusters = new_smoothed
                else:
                    smoothed_clusters = final_clusters

                send_clusters_to_vr(smoothed_clusters)
                print(f"[DEBUG] Sent {len(smoothed_clusters)} relative clusters: {smoothed_clusters}")

    finally:
        for cam_id, cap in cameras.items():
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()