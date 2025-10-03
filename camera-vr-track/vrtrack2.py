import cv2
import numpy as np
import mediapipe as mp
import time
import math
from pythonosc.udp_client import SimpleUDPClient

# Note: Install python-osc with pip install python-osc

# Set up MediaPipe for body tracking
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.2, min_tracking_confidence=0.5, model_complexity=2)

# OSC client for VRChat
osc_client = SimpleUDPClient("127.0.0.1", 9000)

# User height for scaling (adjust to your real height in meters)
user_height = 1.8
leg_ratio = 0.45  # Approximate ratio of leg length (hip to floor) to full height

# Initialize the camera with index 2
camera = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not camera.isOpened():
    print("Error: Could not open camera at index 2. Check connections and index.")
    exit()

# Print frame dimensions if possible
ret, frame = camera.read()
if ret:
    print(f"Original frame dimensions: {frame.shape}")
else:
    print("Warning: Could not read initial frame, but continuing anyway.")

# Set up fullscreen window
cv2.namedWindow("Body Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Body Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Global variables for click-based calibration
calibration_step = 0  # 0: Left hip, 1: Right hip, 2: Left foot, 3: Right foot
calibration_points = {23: None, 24: None, 27: None, 28: None}  # Store clicked points
calibration_complete = False
last_click_time = 0
click_debounce = 0.5  # Seconds to prevent rapid clicks

# Mouse callback function for clicking landmarks
def mouse_callback(event, x, y, flags, param):
    global calibration_step, calibration_points, calibration_complete, last_click_time
    current_time = time.time()
    if event == cv2.EVENT_LBUTTONDOWN and (current_time - last_click_time) > click_debounce:
        if calibration_step == 0:
            calibration_points[23] = (x, y)
            print(f"Left hip point set at ({x}, {y})")
            calibration_step = 1
        elif calibration_step == 1:
            calibration_points[24] = (x, y)
            print(f"Right hip point set at ({x}, {y})")
            calibration_step = 2
        elif calibration_step == 2:
            calibration_points[27] = (x, y)
            print(f"Left foot point set at ({x}, {y})")
            calibration_step = 3
        elif calibration_step == 3:
            calibration_points[28] = (x, y)
            print(f"Right foot point set at ({x}, {y})")
            calibration_step = 4
            calibration_complete = True
            print("Calibration complete!")
        last_click_time = current_time

cv2.setMouseCallback("Body Tracking", mouse_callback)

# Kalman filter for 2D image coordinates (for display)
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

# Kalman filter for 3D world coordinates (for OSC)
def init_kalman_filter_3d():
    kf = cv2.KalmanFilter(6, 3)
    kf.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1.0
    return kf

# Key landmarks limited to lower body only (hips and ankles)
key_landmarks = [23, 24, 27, 28]  # Left/right hip, left/right ankle

kalman_filters = {i: init_kalman_filter() for i in key_landmarks}
kalman_filters_3d = {i: init_kalman_filter_3d() for i in key_landmarks}
smoothed_landmarks = {i: None for i in key_landmarks}
smoothed_landmarks_world = {i: None for i in key_landmarks}

# Calibration variables
reference_pose = {}
reference_pose_world = {}
scale_factor = 1.0

# Preprocessing with grayscale conversion and CLAHE enhancement for low visibility
def preprocess_frame(frame):
    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_enhanced = clahe.apply(frame_gray)
    frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_GRAY2RGB)
    return frame_rgb, frame

# Smooth landmarks and handle occlusion (2D for display)
def smooth_and_track_landmarks(results, frame_shape):
    global smoothed_landmarks
    h, w = frame_shape[:2]
    for idx in key_landmarks:
        if results.pose_landmarks and results.pose_landmarks.landmark[idx].visibility > 0.5:
            x = results.pose_landmarks.landmark[idx].x * w
            y = results.pose_landmarks.landmark[idx].y * h
            # Check if the detected landmark is close to the clicked point
            if calibration_points[idx] and not calibration_complete:
                cx, cy = calibration_points[idx]
                distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                if distance < 100:  # Threshold in pixels for matching
                    kalman_filters[idx].correct(np.array([[x], [y]], np.float32))
                    smoothed_landmarks[idx] = (x, y)
            elif calibration_complete:
                kalman_filters[idx].correct(np.array([[x], [y]], np.float32))
                smoothed_landmarks[idx] = (x, y)
        else:
            predicted = kalman_filters[idx].predict()
            smoothed_landmarks[idx] = (predicted[0, 0], predicted[1, 0])
    return smoothed_landmarks

# Smooth landmarks and handle occlusion (3D for OSC)
def smooth_and_track_landmarks_world(results, frame_shape):
    global smoothed_landmarks_world
    h, w = frame_shape[:2]
    for idx in key_landmarks:
        if results.pose_landmarks and results.pose_landmarks.landmark[idx].visibility > 0.5:
            x = results.pose_landmarks.landmark[idx].x * w
            y = results.pose_landmarks.landmark[idx].y * h
            # Check if the detected landmark is close to the clicked point
            if calibration_points[idx] and not calibration_complete:
                cx, cy = calibration_points[idx]
                distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                if distance < 100:  # Threshold in pixels for matching
                    wx = results.pose_world_landmarks.landmark[idx].x
                    wy = results.pose_world_landmarks.landmark[idx].y
                    wz = results.pose_world_landmarks.landmark[idx].z
                    kalman_filters_3d[idx].correct(np.array([[wx], [wy], [wz]], np.float32))
                    smoothed_landmarks_world[idx] = (wx, wy, wz)
            elif calibration_complete:
                wx = results.pose_world_landmarks.landmark[idx].x
                wy = results.pose_world_landmarks.landmark[idx].y
                wz = results.pose_world_landmarks.landmark[idx].z
                kalman_filters_3d[idx].correct(np.array([[wx], [wy], [wz]], np.float32))
                smoothed_landmarks_world[idx] = (wx, wy, wz)
        else:
            predicted = kalman_filters_3d[idx].predict()
            smoothed_landmarks_world[idx] = (predicted[0, 0], predicted[1, 0], predicted[2, 0])
    return smoothed_landmarks_world

# Initialize Kalman filters with clicked points
def initialize_kalman_with_clicks(frame_shape):
    global reference_pose, reference_pose_world, scale_factor
    h, w = frame_shape[:2]
    reference_pose = {}
    reference_pose_world = {}
    
    # Map clicked 2D points to 3D estimates
    for idx in key_landmarks:
        if calibration_points[idx]:
            x, y = calibration_points[idx]
            reference_pose[idx] = (x, y)
            # Normalize to [-0.5, 0.5] for 3D approximation
            wx = (x / w) - 0.5
            wy = -((y / h) - 0.5)
            wz = -0.5  # Approximate z
            reference_pose_world[idx] = (wx, wy, wz)
            kalman_filters[idx].statePost = np.array([[x], [y], [0], [0]], np.float32)
            kalman_filters_3d[idx].statePost = np.array([[wx], [wy], [wz], [0], [0], [0]], np.float32)
    
    # Calculate scale_factor from measured leg length
    hip_left_y = reference_pose_world[23][1]
    hip_right_y = reference_pose_world[24][1]
    hip_y_avg = (hip_left_y + hip_right_y) / 2
    foot_left_y = reference_pose_world[27][1]
    foot_right_y = reference_pose_world[28][1]
    foot_y_avg = (foot_left_y + foot_right_y) / 2
    measured_leg = abs(hip_y_avg - foot_y_avg)
    real_leg = user_height * leg_ratio
    scale_factor = real_leg / measured_leg if measured_leg > 0 else 1.0
    print(f"Measured leg length: {measured_leg:.2f}m, Estimated real leg: {real_leg:.2f}m, Scale factor: {scale_factor:.2f}")

# Function to send tracker data via OSC (only hip and feet)
def send_tracker(tracker_id, pos, rot=(0.0, 0.0, 0.0)):
    adjusted_pos = (-pos[0] * scale_factor, pos[1] * scale_factor, -pos[2] * scale_factor)
    if isinstance(tracker_id, int):
        path = f"/tracking/trackers/{tracker_id}"
    else:
        path = f"/tracking/trackers/{tracker_id}"
    osc_client.send_message(f"{path}/position", adjusted_pos)
    osc_client.send_message(f"{path}/rotation", rot)

# Main loop
frame_count = 0
prev_time = time.time()
try:
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame. Check camera connection.")
            break

        frame_rgb, frame = preprocess_frame(frame)
        results = pose.process(frame_rgb)

        debug_text = ""
        if not calibration_complete:
            if calibration_step == 0:
                cv2.putText(frame, "Click left hip", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif calibration_step == 1:
                cv2.putText(frame, "Click right hip", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif calibration_step == 2:
                cv2.putText(frame, "Click left foot", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif calibration_step == 3:
                cv2.putText(frame, "Click right foot", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            debug_text = f"Calibration step: {calibration_step + 1}/4"
        else:
            cv2.putText(frame, "Calibrated (Press 'r' to recalibrate)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Initialize Kalman filters once calibration is complete
            if not reference_pose:
                initialize_kalman_with_clicks(frame.shape)

        # Draw clicked points
        for idx, point in calibration_points.items():
            if point:
                x, y = point
                cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("Recalibrating...")
            calibration_step = 0
            calibration_points = {23: None, 24: None, 27: None, 28: None}
            calibration_complete = False
            reference_pose = {}
            reference_pose_world = {}
            smoothed_landmarks = {i: None for i in key_landmarks}
            smoothed_landmarks_world = {i: None for i in key_landmarks}

        if results.pose_landmarks:
            smoothed_landmarks = smooth_and_track_landmarks(results, frame.shape)
            smoothed_landmarks_world = smooth_and_track_landmarks_world(results, frame.shape)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        if smoothed_landmarks:
            for idx in key_landmarks:
                if smoothed_landmarks[idx]:
                    x, y = smoothed_landmarks[idx]
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)

        if calibration_complete and smoothed_landmarks_world:
            hip = (np.array(smoothed_landmarks_world[23]) + np.array(smoothed_landmarks_world[24])) / 2
            left_foot = np.array(smoothed_landmarks_world[27])
            right_foot = np.array(smoothed_landmarks_world[28])

            send_tracker(1, hip)
            send_tracker(3, left_foot)
            send_tracker(4, right_foot)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if debug_text:
            cv2.putText(frame, debug_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Body Tracking", frame)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    pose.close()
    camera.release()
    cv2.destroyAllWindows()