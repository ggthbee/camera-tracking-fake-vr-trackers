import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
from pythonosc import udp_client

# Set camera index and resolution
CAMERA_INDEX = 2  # Your IR camera
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# VRChat coordinate adjustments for front-facing camera
X_OFFSET = 0.0  # Adjust if left/right off-center (e.g., -0.1 or 0.1)
Y_OFFSET = 0.1  # Adjust if up/down off-center (e.g., -0.1 or 0.2 for lying down)
Z_SCALE = 3.0   # Increase for front-facing depth (e.g., 2.0 to 4.0)

# Set up queues for multithreading
frame_queue = queue.Queue(maxsize=2)
keypoint_queue = queue.Queue(maxsize=2)

# Set up OSC client for VRChat
OSC_IP = "127.0.0.1"
OSC_PORT = 9000
osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# Set up Mediapipe for body tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.4,  # Slightly higher for front-facing stability
    min_tracking_confidence=0.4,
    model_complexity=0  # Lightest model for 60 FPS
)
mp_drawing = mp.solutions.drawing_utils

# Map Mediapipe landmarks to VRChat trackers
TRACKER_MAPPING = {
    0: 1,  # Nose -> Tracker 1 (head)
    11: 2,  # Left shoulder -> Tracker 2
    12: 3,  # Right shoulder -> Tracker 3
    23: 4,  # Left hip -> Tracker 4
    24: 5,  # Right hip -> Tracker 5
    27: 6,  # Left knee -> Tracker 6
    28: 7,  # Right knee -> Tracker 7
    31: 8,  # Left foot -> Tracker 8
}

# Smoothing for keypoints (stronger to reduce jitter)
class KeypointSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha  # Stronger smoothing for VRChat
        self.prev_keypoints = None

    def smooth(self, keypoints):
        if not keypoints:
            return keypoints
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints
            return keypoints
        smoothed = []
        for curr, prev in zip(keypoints, self.prev_keypoints):
            x = self.alpha * curr[0] + (1 - self.alpha) * prev[0]
            y = self.alpha * curr[1] + (1 - self.alpha) * prev[1]
            z = self.alpha * curr[2] + (1 - self.alpha) * prev[2]
            smoothed.append((x, y, z))
        self.prev_keypoints = smoothed
        return smoothed

smoother = KeypointSmoother(alpha=0.2)

# Function to clean up IR image (tuned for front-facing)
def clean_up_image(frame):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))  # Stronger contrast
    frame = clahe.apply(frame)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Stronger blur for stability
    return frame

# Task 1: Capture camera frames
def capture_frames():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}. Check connection/drivers.")
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 60)
    actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Camera {CAMERA_INDEX}: Set to {actual_width}x{actual_height}, FPS: {actual_fps}")
    
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print(f"Warning: Failed to read frame from camera {CAMERA_INDEX}. Retrying...")
            time.sleep(0.05)
            continue
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    camera.release()

# Task 2: Process frames
def process_frames():
    last_time = time.time()
    frame_count = 0
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Clean up frame
        start_time = time.time()
        clean_frame = clean_up_image(frame)
        preprocess_time = (time.time() - start_time) * 1000
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_GRAY2RGB)
        
        # Mediapipe pose estimation
        start_time = time.time()
        results = pose.process(frame_rgb)
        pose_time = (time.time() - start_time) * 1000
        
        # Get and smooth 3D keypoints
        keypoints = []
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                x = landmark.x * IMAGE_WIDTH
                y = landmark.y * IMAGE_HEIGHT
                x = max(0, min(x, IMAGE_WIDTH - 1))
                y = max(0, min(y, IMAGE_HEIGHT - 1))
                z = landmark.z
                keypoints.append((x, y, z))
            # Smooth keypoints
            keypoints = smoother.smooth(keypoints)
            # Send to VRChat
            for i, (x, y, z) in enumerate(keypoints):
                if i in TRACKER_MAPPING:
                    tracker_id = TRACKER_MAPPING[i]
                    x_centered = (x / IMAGE_WIDTH) - 0.5 + X_OFFSET
                    y_centered = (y / IMAGE_HEIGHT) - 0.5 + Y_OFFSET
                    z_scaled = z * Z_SCALE
                    osc_client.send_message(
                        f"/tracking/trackers/{tracker_id}/position",
                        [x_centered, y_centered, z_scaled]
                    )
                    print(f"Tracker {tracker_id}: x={x_centered:.2f}, y={y_centered:.2f}, z={z_scaled:.2f}")
        else:
            print("Warning: No body detected. Adjust camera angle, distance, or move slightly.")
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count / (current_time - last_time)
            print(f"FPS: {fps:.2f}, Preprocess Time: {preprocess_time:.2f}ms, Pose Time: {pose_time:.2f}ms")
            frame_count = 0
            last_time = current_time
        
        # Show frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            cv2.putText(frame, "No Body Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("3D Tracking", frame)
        
        frame_queue.task_done()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Task 3: Send OSC to VRChat (rate-limited)
def send_to_vrchat():
    last_send_time = time.time()
    while True:
        try:
            keypoints = keypoint_queue.get(timeout=0.1)
            current_time = time.time()
            if current_time - last_send_time >= 0.016:  # ~60 Hz
                for i, (x, y, z) in enumerate(keypoints):
                    if i in TRACKER_MAPPING:
                        tracker_id = TRACKER_MAPPING[i]
                        x_centered = (x / IMAGE_WIDTH) - 0.5 + X_OFFSET
                        y_centered = (y / IMAGE_HEIGHT) - 0.5 + Y_OFFSET
                        z_scaled = z * Z_SCALE
                        osc_client.send_message(
                            f"/tracking/trackers/{tracker_id}/position",
                            [x_centered, y_centered, z_scaled]
                        )
                last_send_time = current_time
            keypoint_queue.task_done()
        except queue.Empty:
            pass

# Start threads
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()
threading.Thread(target=send_to_vrchat, daemon=True).start()

# Keep program running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pose.close()
    cv2.destroyAllWindows()