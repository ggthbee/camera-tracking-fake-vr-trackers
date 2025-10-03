import cv2
import numpy as np
import time
from pythonosc.udp_client import SimpleUDPClient
from filterpy.kalman import KalmanFilter

# Helper Functions (unchanged)
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def look_at(eye, target, up):
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    cam_up = np.cross(right, forward)
    R = np.c_[right, cam_up, forward]
    return R

def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z]) * 180 / np.pi

def send_osc(endpoint, vec3):
    try:
        osc_client.send_message(endpoint, [float(v) for v in vec3])
    except Exception as e:
        print(f"OSC send error for {endpoint}: {e}")

def send_calibrate():
    try:
        osc_client.send_message("/tracking/calibrate", 1)
        print("Sent VRChat calibrate button press")
    except Exception as e:
        print(f"OSC calibrate error: {e}")

# Camera Calibration
room_width = 4.0
room_depth = 4.0
camera_height = 3.0
fx = 600.0
fy = 600.0
cx = 320.0
cy = 240.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
cam_positions = [
    np.array([0.0, 0.0, camera_height]),
    np.array([room_width, 0.0, camera_height]),
    np.array([room_width, room_depth, camera_height]),
    np.array([0.0, room_depth, camera_height])
]
target = np.array([room_width / 2, room_depth / 2, 0.0])
up = np.array([0.0, 0.0, 1.0])
projection_matrices = []
for eye in cam_positions:
    R = look_at(eye, target, up)
    t = -np.dot(R, eye)
    ext = np.hstack((R, t.reshape(3, 1)))
    P = np.dot(K, ext)
    projection_matrices.append(P)
playspace_origin = np.array([room_width / 2, room_depth / 2, 0.0])

# Camera Setup (Fixed for broadcasting error)
cameras = []
subtractors = []
valid_cam_ids = []
for cam_id in range(4):
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera {cam_id}: {width}x{height} @ {fps} FPS")
            try:
                subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
                if hasattr(subtractor, 'setLearningRate') and int(cv2.__version__.split('.')[0]) >= 4:
                    subtractor.setLearningRate(0.02)
                subtractors.append(subtractor)
                cameras.append(cap)
                valid_cam_ids.append(cam_id)
            except Exception as e:
                print(f"Error: Failed to create MOG2 for Camera {cam_id}: {e}. Skipping camera.")
                cap.release()
        except Exception as e:
            print(f"Warning: Failed to configure Camera {cam_id}: {e}")
            cap.release()
    else:
        print(f"Warning: Camera {cam_id} failed to open")
if len(cameras) < 2:
    print("Error: At least two cameras required for triangulation")
    exit(1)

# OSC Setup
osc_ip = "127.0.0.1"
osc_port = 9000
osc_client = SimpleUDPClient(osc_ip, osc_port)
OSC_POS_ENDPOINTS = {
    '2': "/tracking/trackers/2/position",  # hip
    '3': "/tracking/trackers/3/position",  # left_elbow
    '4': "/tracking/trackers/4/position",  # right_elbow
    '5': "/tracking/trackers/5/position",  # left_knee
    '6': "/tracking/trackers/6/position",  # right_knee
    '7': "/tracking/trackers/7/position",  # left_foot
    '8': "/tracking/trackers/8/position",  # right_foot
}
OSC_ROT_ENDPOINTS = {
    '2': "/tracking/trackers/2/rotation",
    '3': "/tracking/trackers/3/rotation",
    '4': "/tracking/trackers/4/rotation",
    '5': "/tracking/trackers/5/rotation",
    '6': "/tracking/trackers/6/rotation",
    '7': "/tracking/trackers/7/rotation",
    '8': "/tracking/trackers/8/rotation",
}

# Kalman Filter Setup
trackers = {}
for k in OSC_POS_ENDPOINTS.keys():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = np.array([[1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
    kf.P *= 1000.0
    kf.R = np.eye(3) * 0.2
    kf.Q = np.eye(6) * 0.05
    kf.x[:3] = np.zeros(3)
    trackers[k] = kf

# Belt Detection
def detect_belt_midpoint(frame, fgmask, cam_id):
    try:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, fgmask)
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.putText(frame, f"Cam {cam_id}: No belts", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return []
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 1000:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]
                if mean_intensity > 200:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    angle = 0
                    try:
                        if len(cnt) >= 5:
                            ellipse = cv2.fitEllipse(cnt)
                            angle = ellipse[2]
                        else:
                            rect = cv2.minAreaRect(cnt)
                            angle = rect[2]
                    except:
                        continue
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                    angle_rad = np.deg2rad(angle)
                    end_x = int(cx + 20 * np.cos(angle_rad))
                    end_y = int(cy + 20 * np.sin(angle_rad))
                    cv2.line(frame, (cx, cy), (end_x, end_y), (0, 0, 255), 2)
                    detections.append(((cx, cy), angle))
        if detections:
            cv2.putText(frame, f"Cam {cam_id}: {len(detections)} belts", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Cam {cam_id}: No valid belts", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return detections
    except Exception as e:
        print(f"Error in detect_belt_midpoint (Cam {cam_id}): {e}")
        cv2.putText(frame, f"Cam {cam_id}: Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return []

# Triangulation
def triangulate(det_list):
    try:
        if len(det_list) < 2:
            print(f"Warning: Only {len(det_list)} camera(s) detected belt")
            return None
        A = []
        for cam_id, pt in det_list:
            x, y = pt
            P = projection_matrices[cam_id]
            row1 = x * P[2] - P[0]
            row2 = y * P[2] - P[1]
            A.append(row1)
            A.append(row2)
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        if S[-2] / S[-1] > 1000:
            print("Warning: Unstable triangulation")
            return None
        X = Vt[-1]
        if abs(X[3]) < 1e-6:
            print("Warning: Triangulation division by near-zero")
            return None
        X /= X[3]
        return X[:3]
    except Exception as e:
        print(f"Error in triangulate: {e}")
        return None

# Orientation Estimation
def estimate_3d_orientation(det_list, angles):
    try:
        if len(det_list) < 2:
            return np.eye(3)
        avg_dir = np.zeros(3)
        for (cam_id, pt), angle in zip(det_list, angles):
            R = projection_matrices[cam_id][:3, :3]
            angle_rad = np.deg2rad(angle)
            dir_2d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            dir_3d = R.T @ np.array([dir_2d[0], dir_2d[1], 0])
            avg_dir += normalize(dir_3d)
        if np.linalg.norm(avg_dir) < 1e-6:
            print("Warning: Invalid orientation direction")
            return np.eye(3)
        avg_dir = normalize(avg_dir)
        up = np.array([0, 0, 1])
        right = normalize(np.cross(avg_dir, up))
        new_up = np.cross(right, avg_dir)
        R = np.c_[avg_dir, right, new_up]
        return R
    except Exception as e:
        print(f"Error in estimate_3d_orientation: {e}")
        return np.eye(3)

# Tracker Assignment
def assign_trackers(points_3d, rotations_3d, last_positions):
    assigned = {name: None for name in OSC_POS_ENDPOINTS.keys()}
    assigned_rot = {name: np.eye(3) for name in OSC_POS_ENDPOINTS.keys()}
    points = points_3d.copy()
    rots = rotations_3d.copy()
    for name in OSC_POS_ENDPOINTS.keys():
        if not points:
            break
        distances = [np.linalg.norm(p - last_positions[name]) if np.all(np.isfinite(p)) else np.inf for p in points]
        idx = np.argmin(distances)
        if distances[idx] != np.inf:
            assigned[name] = points.pop(idx)
            assigned_rot[name] = rots.pop(idx)
    return assigned, assigned_rot

# Main Loop
last_positions = {k: np.zeros(3) for k in OSC_POS_ENDPOINTS.keys()}
last_rotations = {k: np.zeros(3) for k in OSC_POS_ENDPOINTS.keys()}
last_update = {k: time.time() for k in OSC_POS_ENDPOINTS.keys()}
calibrating = False

print("Press 'G' to recalibrate with A-pose")

while True:
    try:
        frames = []
        valid_indices = []
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                valid_indices.append(valid_cam_ids[i])
            else:
                print(f"Camera {valid_cam_ids[i]} failed to capture frame")
        if not frames:
            print("No valid frames captured, skipping iteration")
            continue

        all_markers = []
        all_angles = []
        annotated_frames = []
        for i, (frame, cam_id) in enumerate(zip(frames, valid_indices)):
            try:
                fgmask = subtractors[i].apply(frame)
                detections = detect_belt_midpoint(frame, fgmask, cam_id)
                for midpoint, angle in detections:
                    all_markers.append((cam_id, midpoint))
                    all_angles.append(angle)
                annotated_frames.append(frame)
            except Exception as e:
                print(f"Error processing frame for Camera {cam_id}: {e}")
                annotated_frames.append(frame)

        groups = []
        used_cams = set()
        used_points = set()
        for i, (cam_id, pt) in enumerate(all_markers):
            if cam_id in used_cams or i in used_points:
                continue
            group = [(cam_id, pt)]
            group_angles = [all_angles[i]]
            used_cams.add(cam_id)
            used_points.add(i)
            for j, (other_cam_id, other_pt) in enumerate(all_markers):
                if other_cam_id not in used_cams and j not in used_points:
                    dist = np.linalg.norm(np.array(pt) - np.array(other_pt))
                    if dist < 150:
                        group.append((other_cam_id, other_pt))
                        group_angles.append(all_angles[j])
                        used_cams.add(other_cam_id)
                        used_points.add(j)
            groups.append((group, group_angles))

        points_3d = []
        rotations_3d = []
        for group, group_angles in groups:
            pos3d = triangulate(group)
            if pos3d is not None and np.all(np.isfinite(pos3d)):
                points_3d.append(pos3d)
                R = estimate_3d_orientation(group, group_angles)
                rotations_3d.append(R)

        print(f"Detected belts: {len(all_markers)}, Triangulated points: {len(points_3d)}")

        detections, rotations = assign_trackers(points_3d, rotations_3d, last_positions)

        for name in detections.keys():
            try:
                pos3d = detections[name]
                R = rotations[name]
                kf = trackers[name]
                current_time = time.time()
                if pos3d is not None and np.all(np.isfinite(pos3d)):
                    kf.update(pos3d)
                    kf.predict()
                    last_positions[name] = kf.x[:3]
                    last_rotations[name] = rotation_matrix_to_euler(R)
                    last_update[name] = current_time
                else:
                    kf.predict()
                    last_positions[name] = kf.x[:3]
                    if current_time - last_update[name] > 2.0:
                        kf.x = np.zeros(6)
                        kf.P = np.eye(6) * 1000.0
                        last_positions[name] = np.zeros(3)
                        last_rotations[name] = np.zeros(3)
                        print(f"Tracker {name} reset due to timeout")
                pos3d = last_positions[name]
                pos3d = pos3d - playspace_origin
                unity_pos = [pos3d[0], pos3d[2], pos3d[1]]
                send_osc(OSC_POS_ENDPOINTS[name], unity_pos)
                rot_euler = last_rotations[name]
                send_osc(OSC_ROT_ENDPOINTS[name], rot_euler)
            except Exception as e:
                print(f"Error processing tracker {name}: {e}")

        resized = [cv2.resize(f, (960, 540)) for f in annotated_frames]
        if len(resized) >= 2:
            n = len(resized)
            rows = int(np.ceil(n / 2))
            grid = np.zeros((rows * 540, 2 * 960, 3), dtype=np.uint8)
            for i, frame in enumerate(resized):
                row = i // 2
                col = i % 2
                grid[row*540:(row+1)*540, col*960:(col+1)*960] = frame
            grid_resized = cv2.resize(grid, (1920, 1080))
            cv2.imshow("Multi-Camera Belts", grid_resized)
        else:
            single_resized = cv2.resize(annotated_frames[0], (1920, 1080))
            cv2.imshow("Multi-Camera Belts", single_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g') and not calibrating:
            calibrating = True
            print("Recalibration started - hold A-pose for 2 seconds...")
            time.sleep(2)
            print("Recalibration complete! Sending VRChat calibrate")
            send_calibrate()
            calibrating = False

    except Exception as e:
        print(f"Error in main loop: {e}")

for cap in cameras:
    cap.release()
cv2.destroyAllWindows()