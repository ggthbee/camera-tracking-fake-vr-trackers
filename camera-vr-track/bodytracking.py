import cv2
import numpy as np
import mediapipe as mp
import time
import math

# Set up MediaPipe for fallback pose estimation
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to detect available cameras, excluding index 4
def detect_cameras(max_index=5):
    available_cameras = []
    for i in range(max_index):
        if i == 4:  # Skip camera index 4
            print(f"Skipping camera at index {i} as per request")
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera found at index {i} (ELP camera)")
            cap.release()
        else:
            print(f"No camera at index {i}")
    return available_cameras

# Initialize ELP cameras
camera_indices = detect_cameras()
if not camera_indices:
    print("Error: No ELP cameras detected (excluding index 4). Exiting.")
    exit()

cameras = []
pose_instances = []
frame_dimensions = []
for idx in camera_indices:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        cameras.append(cap)
        ret, frame = cap.read()
        if ret:
            print(f"ELP Camera {idx} initialized. Frame dimensions: {frame.shape}")
            frame_dimensions.append((frame.shape[1], frame.shape[0]))
        else:
            print(f"Warning: Could not read initial frame from ELP camera {idx}")
            frame_dimensions.append((640, 480))
        pose_instances.append(mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.7
        ))
    else:
        print(f"Error: Could not open ELP camera at index {idx}")

if not cameras:
    print("Error: No ELP cameras could be initialized. Exiting.")
    exit()

# Set up fullscreen window
cv2.namedWindow("Body Tracking - ELP Cameras", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Body Tracking - ELP Cameras", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize blob detector for IR-reflective markers
def init_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20  # Adjusted for ELP OV2710 IR images
    params.maxArea = 2000
    params.filterByCircularity = True
    params.minCircularity = 0.6  # Relaxed for IR reflections
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.minThreshold = 150  # Lowered for ELP IR sensitivity
    params.maxThreshold = 255
    return cv2.SimpleBlobDetector_create(params)

blob_detectors = [init_blob_detector() for _ in cameras]

# Kalman filter for smoothing marker positions
def init_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    return kf

# Initialize Kalman filters and calibration data
key_landmarks = [11, 12, 13, 14, 15, 16]  # Left/right shoulder, elbow, wrist
kalman_filters = [{i: init_kalman_filter() for i in key_landmarks} for _ in cameras]
smoothed_landmarks = [{i: None for i in key_landmarks} for _ in cameras]
calibration_frames = 30
calibration_data = [{i: [] for i in key_landmarks} for _ in cameras]
calibrated = [False] * len(cameras)
reference_pose = [{} for _ in cameras]

# Preprocess frame for IR and marker detection
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Enhance IR image contrast for ELP cameras
    frame_gray = cv2.equalizeHist(frame_gray)
    # Optional: Apply slight blur to reduce noise
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, frame, frame_gray

# Detect reflective markers and map to landmarks
def detect_markers(frame_gray, detector, frame_shape, results, cam_idx):
    keypoints = detector.detect(frame_gray)
    h, w = frame_shape[:2]
    detected_points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    
    marker_map = {idx: None for idx in key_landmarks}
    if len(detected_points) >= len(key_landmarks):
        # Improved sorting: assume A-pose with left-right symmetry
        # Sort by y (top-to-bottom) and x (left-to-right for left side, right-to-left for right)
        detected_points = sorted(detected_points, key=lambda p: p[1])  # Sort by y
        # Assume top 2 are shoulders, next 2 elbows, last 2 wrists
        if len(detected_points) >= 6:
            left_points = sorted(detected_points[:3], key=lambda p: p[0])  # Left side (smaller x)
            right_points = sorted(detected_points[3:6], key=lambda p: p[0], reverse=True)  # Right side (larger x)
            # Map: 11 (L shoulder), 12 (R shoulder), 13 (L elbow), 14 (R elbow), 15 (L wrist), 16 (R wrist)
            mapping = [11, 13, 15, 12, 14, 16]
            for i, idx in enumerate(mapping):
                marker_map[idx] = detected_points[i]
    elif results.pose_landmarks:
        # Fallback to MediaPipe
        for idx in key_landmarks:
            if results.pose_landmarks.landmark[idx].visibility > 0.5:
                x = results.pose_landmarks.landmark[idx].x * w
                y = results.pose_landmarks.landmark[idx].y * h
                marker_map[idx] = (x, y)
    return marker_map

# Smooth marker positions with Kalman filter
def smooth_and_track_landmarks(marker_map, cam_idx):
    for idx in key_landmarks:
        if marker_map[idx]:
            x, y = marker_map[idx]
            kalman_filters[cam_idx][idx].correct(np.array([[x], [y]], np.float32))
            smoothed_landmarks[cam_idx][idx] = (x, y)
        else:
            predicted = kalman_filters[cam_idx][idx].predict()
            smoothed_landmarks[cam_idx][idx] = (predicted[0, 0], predicted[1, 0])
    return smoothed_landmarks[cam_idx]

# Calibrate A-pose using markers
def calibrate_a_pose(marker_map, cam_idx):
    global calibration_data, calibrated, reference_pose
    if calibrated[cam_idx]:
        return False
    appended = False
    for idx in key_landmarks:
        if marker_map[idx]:
            x, y = marker_map[idx]
            calibration_data[cam_idx][idx].append((x, y))
            appended = True
    return appended

# Force calibration with current frame
def force_calibrate(marker_map, cam_idx):
    global calibration_data, calibrated, reference_pose
    missing = [idx for idx in key_landmarks if not marker_map[idx]]
    if missing:
        print(f"ELP Camera {cam_idx}: Cannot calibrate: Missing markers for landmarks {missing}")
        return False
    calibration_data[cam_idx] = {idx: [marker_map[idx]] * calibration_frames for idx in key_landmarks}
    reference_pose[cam_idx] = {idx: marker_map[idx] for idx in key_landmarks}
    calibrated[cam_idx] = True
    print(f"ELP Camera {cam_idx}: Calibration complete!")
    for idx in key_landmarks:
        x, y = reference_pose[cam_idx][idx]
        kalman_filters[cam_idx][idx].statePost = np.array([[x], [y], [0], [0]], np.float32)
    return True

# Combine frames into a grid
def create_grid(frames, grid_cols):
    if not frames:
        return None
    grid_rows = (len(frames) + grid_cols - 1) // grid_cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((h * grid_rows, w * grid_cols, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        row = i // grid_cols
        col = i % grid_cols
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame
    return grid

# Main loop
frame_count = [0] * len(cameras)
prev_time = time.time()
try:
    while any(cap.isOpened() for cap in cameras):
        frames = []
        for cam_idx, (cap, pose, detector) in enumerate(zip(cameras, pose_instances, blob_detectors)):
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if not ret:
                print(f"ELP Camera {camera_indices[cam_idx]}: Failed to capture frame.")
                continue

            frame_rgb, frame, frame_gray = preprocess_frame(frame)
            results = pose.process(frame_rgb)

            # Detect reflective markers
            marker_map = detect_markers(frame_gray, detector, frame.shape, results, cam_idx)

            debug_text = f"Markers detected: {sum(1 for v in marker_map.values() if v is not None)}"
            if not calibrated[cam_idx]:
                current_count = len(calibration_data[cam_idx][key_landmarks[0]])
                if current_count < calibration_frames and sum(1 for v in marker_map.values() if v is not None) >= len(key_landmarks):
                    cv2.putText(frame, f"Cam {cam_idx}: Hold A-pose ({current_count}/{calibration_frames})", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    calibrate_a_pose(marker_map, cam_idx)
                    if all(len(calibration_data[cam_idx][idx]) >= calibration_frames for idx in key_landmarks):
                        reference_pose[cam_idx] = {idx: np.mean(calibration_data[cam_idx][idx], axis=0) for idx in key_landmarks}
                        calibrated[cam_idx] = True
                        print(f"ELP Camera {cam_idx}: Calibration complete!")
                        for idx in key_landmarks:
                            x, y = reference_pose[cam_idx][idx]
                            kalman_filters[cam_idx][idx].statePost = np.array([[x], [y], [0], [0]], np.float32)
                else:
                    cv2.putText(frame, f"Cam {cam_idx}: Please assume A-pose with markers", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if calibrated[cam_idx]:
                    print(f"ELP Camera {cam_idx}: Recalibrating...")
                    calibrated[cam_idx] = False
                    calibration_data[cam_idx] = {i: [] for i in key_landmarks}
                print(f"ELP Camera {cam_idx}: Manual calibration triggered!")
                force_calibrate(marker_map, cam_idx)

            smoothed_landmarks[cam_idx] = smooth_and_track_landmarks(marker_map, cam_idx)

            # Draw markers and smoothed positions
            for idx in key_landmarks:
                if marker_map[idx]:
                    x, y = marker_map[idx]
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                if smoothed_landmarks[cam_idx][idx]:
                    x, y = smoothed_landmarks[cam_idx][idx]
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)

            # Draw MediaPipe landmarks (fallback)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"Cam {cam_idx} FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if calibrated[cam_idx]:
                cv2.putText(frame, f"Cam {cam_idx}: Calibrated (Press 'c')", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, debug_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frames.append(frame)

        if frames:
            grid_cols = min(len(frames), 2)
            grid = create_grid(frames, grid_cols)
            if grid is not None:
                cv2.imshow("Body Tracking - ELP Cameras", grid)

        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    for pose in pose_instances:
        pose.close()
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()