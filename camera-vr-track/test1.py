import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import socket
import json
import threading

# Camera configuration
CAMERA_IDS = [0, 1]  # Adjust for your camera indices
RESOLUTION = (1280, 720)  # 720p at 60 fps
FPS = 60
UDP_ADDRESS = ('localhost', 12345)
SKELETON_MAP = {
    0: 'head', 1: 'left_shoulder', 2: 'right_shoulder', 3: 'left_elbow', 4: 'right_elbow'
}  # Adjust as needed

# Kalman filter for smoothing and occlusion handling
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 1000
    kf.R = 5
    kf.Q = 0.1
    return kf

# Compute median landmark for a group of markers
def compute_median_landmark(marker_positions):
    if not marker_positions:
        return None
    markers = np.array(marker_positions)
    return (int(np.median(markers[:, 0])), int(np.median(markers[:, 1])))

# Process camera feed
def process_camera(cam_id, camera_params, landmarks_list, lock):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)
    kalman_filters = [create_kalman_filter() for _ in range(len(SKELETON_MAP))]
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        landmarks = []
        for contour in contours:
            if cv2.contourArea(contour) > 30:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    landmarks.append((cx, cy))
        # Group markers (example: 3 markers per band)
        grouped_markers = [landmarks[i:i+3] for i in range(0, len(landmarks), 3)]
        median_landmarks = [compute_median_landmark(group) for group in grouped_markers]
        # Apply Kalman filtering
        with lock:
            for i, landmark in enumerate(median_landmarks[:len(kalman_filters)]):
                if landmark:
                    kalman_filters[i].predict()
                    kalman_filters[i].update(np.array(landmark))
                    median_landmarks[i] = (int(kalman_filters[i].x[0]), int(kalman_filters[i].x[1]))
                else:
                    kalman_filters[i].predict()
                    median_landmarks[i] = (int(kalman_filters[i].x[0]), int(kalman_filters[i].x[1]))
            landmarks_list[cam_id] = median_landmarks
        cv2.imshow(f'Camera {cam_id}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

# Camera calibration (run once per camera)
def calibrate_camera(cam_id):
    checkerboard = (9, 6)
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(cam_id)
    while len(objpoints) < 20:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    cap.release()
    return mtx, dist

# Triangulate 2D landmarks to 3D
def triangulate_landmarks(landmarks_cam1, landmarks_cam2, P1, P2):
    points_4d = cv2.triangulatePoints(P1, P2, np.array(landmarks_cam1).T, np.array(landmarks_cam2).T)
    return (points_4d[:3] / points_4d[3]).T

# Main pipeline
def main():
    # Load or perform camera calibration
    camera_params = [calibrate_camera(i) for i in CAMERA_IDS]
    P1 = np.hstack((camera_params[0][0], np.zeros((3, 1))))  # Example projection matrix
    P2 = np.hstack((camera_params[1][0], np.zeros((3, 1))))  # Adjust with stereo calibration
    landmarks_list = [[] for _ in CAMERA_IDS]
    lock = threading.Lock()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Start camera threads
    threads = [threading.Thread(target=process_camera, args=(i, camera_params[i], landmarks_list, lock))
               for i in range(len(CAMERA_IDS))]
    for t in threads:
        t.start()

    # A-pose calibration (run once)
    input("Assume A-pose and press Enter...")
    with lock:
        a_pose_landmarks = landmarks_list[0]  # Use first camera for reference
        with open('a_pose_calibration.json', 'w') as f:
            json.dump({'landmarks': a_pose_landmarks, 'skeleton_map': SKELETON_MAP}, f)

    # Stream 3D landmarks to Unity
    while True:
        with lock:
            if all(landmarks_list):
                landmarks_3d = triangulate_landmarks(landmarks_list[0], landmarks_list[1], P1, P2)
                sock.sendto(json.dumps(landmarks_3d.tolist()).encode(), UDP_ADDRESS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()