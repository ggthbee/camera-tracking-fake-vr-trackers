import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time

# Set image dimensions for maximum speed (320x240)
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
# Try 640x480 later if 60 FPS is achieved: IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480

# Set up queues for multithreading
frame_queue = queue.Queue(maxsize=2)  # Small queue to minimize latency
keypoint_queue = queue.Queue(maxsize=2)

# Set up Mediapipe for body tracking (optimized for speed)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.3,  # Lower for speed
    min_tracking_confidence=0.3,
    model_complexity=0  # Lightest model for 60 FPS
)
mp_drawing = mp.solutions.drawing_utils

# Function to clean up IR image (ultra-light for speed)
def clean_up_image(frame):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Light contrast adjustment
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))  # Smaller grid for speed
    frame = clahe.apply(frame)
    # Skip blur and normalization for speed
    return frame

# Task 1: Capture camera frames
def capture_frames():
    camera = cv2.VideoCapture(0)  # Try 1 or 2 if needed
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Skip frame
    camera.release()

# Task 2: Process frames
def process_frames():
    last_time = time.time()
    frame_count = 0
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)  # Very fast timeout
        except queue.Empty:
            continue
        
        # Clean up frame
        start_time = time.time()
        clean_frame = clean_up_image(frame)
        preprocess_time = (time.time() - start_time) * 1000  # ms
        
        # Convert to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_GRAY2RGB)
        
        # Mediapipe pose estimation
        start_time = time.time()
        results = pose.process(frame_rgb)
        pose_time = (time.time() - start_time) * 1000  # ms
        
        # Get 3D keypoints (using Mediapipe's z for speed)
        keypoints = []
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * IMAGE_WIDTH)
                y = int(landmark.y * IMAGE_HEIGHT)
                x = max(0, min(x, IMAGE_WIDTH - 1))  # Clip to avoid index errors
                y = max(0, min(y, IMAGE_HEIGHT - 1))
                z = landmark.z  # Mediapipe's z (fast but less accurate)
                keypoints.append((x, y, z))
                print(f"Body part {i}: x={x}, y={y}, z={z:.2f}")
        else:
            print("Warning: No body detected. Try adjusting camera angle, distance, or moving slightly.")
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count / (current_time - last_time)
            print(f"FPS: {fps:.2f}, Preprocess Time: {preprocess_time:.2f}ms, Pose Time: {pose_time:.2f}ms")
            frame_count = 0
            last_time = current_time
        
        # Show frame with FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            cv2.putText(frame, "No Body Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("3D Tracking", frame)
        
        # Add keypoints to queue
        if keypoints:
            try:
                keypoint_queue.put_nowait(keypoints)
            except queue.Full:
                pass
        
        frame_queue.task_done()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start threads
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()

# Keep program running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pose.close()
    cv2.destroyAllWindows()