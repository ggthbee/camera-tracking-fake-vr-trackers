import cv2
import numpy as np
from pythonosc import udp_client

# Assumed camera intrinsic matrix (adjust based on actual calibration if possible)
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
Dist = np.zeros(5, dtype=np.float32)

# Assumed 3D model points in A-pose (in meters, Unity coordinate system: +X right, +Y up, +Z forward)
# Order: left_foot, right_foot, left_knee, right_knee, left_elbow, right_elbow, hip, chest
body_parts = ['left_foot', 'right_foot', 'left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'hip', 'chest']
tracker_ids = [3, 4, 5, 6, 7, 8, 1, 2]  # VRChat OSC tracker IDs
three_d_model = np.array([
    [-0.15, 0.0, 0.0],   # left_foot
    [0.15, 0.0, 0.0],    # right_foot
    [-0.15, 0.45, 0.0],  # left_knee
    [0.15, 0.45, 0.0],   # right_knee
    [-0.4, 0.6, 0.0],    # left_elbow
    [0.4, 0.6, 0.0],     # right_elbow
    [0.0, 0.9, 0.0],     # hip
    [0.0, 1.4, 0.0]      # chest
], dtype=np.float32)

# OSC client for VRChat (VRChat listens on port 9000)
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)

# Function to detect landmarks from grayscale image
def detect_landmarks(gray):
    try:
        # Use GPU if available
        gpu_gray = cv2.cuda_GpuMat(gray)
        gpu_bin = cv2.cuda.threshold(gpu_gray, 180, 255, cv2.THRESH_BINARY)[1]
        bin_img = gpu_bin.download()
    except:
        # Fallback to CPU
        _, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 100:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids.append([cx, cy])

    if len(centroids) < 10:
        return np.array([])

    centroids_np = np.float32(centroids)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(centroids_np, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Sort centers by decreasing y (assuming higher y = lower height in 3D)
    sort_idx = np.argsort(-centers[:, 1])
    sorted_centers = centers[sort_idx]

    landmarks = np.zeros((8, 2), dtype=np.float32)

    # Assign feet (first 2, sort by x)
    group = sorted_centers[0:2]
    group_sort = np.argsort(group[:, 0])
    landmarks[0] = group[group_sort[0]]  # left
    landmarks[1] = group[group_sort[1]]  # right

    # Assign knees (next 2)
    group = sorted_centers[2:4]
    group_sort = np.argsort(group[:, 0])
    landmarks[2] = group[group_sort[0]]
    landmarks[3] = group[group_sort[1]]

    # Assign elbows (next 2)
    group = sorted_centers[4:6]
    group_sort = np.argsort(group[:, 0])
    landmarks[4] = group[group_sort[0]]
    landmarks[5] = group[group_sort[1]]

    # Assign hip (next 1)
    landmarks[6] = sorted_centers[6]

    # Assign chest (last 1)
    landmarks[7] = sorted_centers[7]

    return landmarks

# Function for multi-view triangulation
def triangulate_multi(proj_mats, points_2d):
    num_views = len(proj_mats)
    A = np.zeros((2 * num_views, 4))
    for i in range(num_views):
        P = proj_mats[i]
        x, y = points_2d[i]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

# Main script
caps = [cv2.VideoCapture(i) for i in range(4)]
brightness = -6
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

projection_matrices = [None] * 4
calibrated = False

while True:
    frames = []
    grays = []
    ret = True
    for cap in caps:
        r, frame = cap.read()
        if not r:
            ret = False
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        # For display, convert back to BGR for drawing
        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        frames.append(display_frame)

    if not ret:
        break

    # Combine into single window (2x2 grid)
    row1 = np.hstack(frames[0:2])
    row2 = np.hstack(frames[2:4])
    big_frame = np.vstack((row1, row2))

    # Process detection and drawing
    image_points_list = []
    for i, gray in enumerate(grays):
        landmarks = detect_landmarks(gray)
        image_points_list.append(landmarks)
        # Draw landmarks on display frame
        for lm in landmarks:
            cv2.circle(frames[i], (int(lm[0]), int(lm[1])), 5, (0, 0, 255), -1)

    # Update combined frame after drawing
    row1 = np.hstack(frames[0:2])
    row2 = np.hstack(frames[2:4])
    big_frame = np.vstack((row1, row2))

    cv2.imshow('Camera Feeds', big_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        brightness += 1
        for cap in caps:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    elif key == ord('-'):
        brightness -= 1
        for cap in caps:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    elif key == ord('c'):
        # Calibration mode
        proj_mats = []
        for i, pts2d in enumerate(image_points_list):
            if len(pts2d) == 8:
                pts2d = pts2d.reshape(-1, 1, 2)
                ret_pnp, rvec, tvec = cv2.solvePnP(three_d_model, pts2d, K, Dist)
                if ret_pnp:
                    R, _ = cv2.Rodrigues(rvec)
                    Rt = np.hstack((R, tvec))
                    P = K @ Rt
                    proj_mats.append(P)
        if len(proj_mats) == 4:
            projection_matrices = proj_mats
            calibrated = True
            print("Calibration successful!")

    if calibrated:
        # Track and send to OSC
        three_d_points = []
        for j in range(8):
            pts2d_per_cam = []
            valid_pmats = []
            for i in range(4):
                if len(image_points_list[i]) == 8:
                    pts2d_per_cam.append(image_points_list[i][j])
                    valid_pmats.append(projection_matrices[i])
            if len(pts2d_per_cam) >= 2:
                pts2d_per_cam = np.array(pts2d_per_cam)
                X = triangulate_multi(valid_pmats, pts2d_per_cam)
                three_d_points.append(X)
            else:
                three_d_points.append(None)

        # Send to OSC
        for idx, X in enumerate(three_d_points):
            if X is not None:
                tid = tracker_ids[idx]
                osc_client.send_message(f"/tracking/trackers/{tid}/position", [float(X[0]), float(X[1]), float(X[2])])
                # Send default rotation (0,0,0)
                osc_client.send_message(f"/tracking/trackers/{tid}/rotation", [0.0, 0.0, 0.0])

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()