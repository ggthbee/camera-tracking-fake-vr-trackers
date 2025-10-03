import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
import threading
import time

# --- CONFIG ---
CAM_IDS = [0, 1, 2, 3]
OSC_IP = "127.0.0.1"
OSC_PORT = 9000
MAX_PIVOT_DIST = 25  # pixels
MARKER_RADIUS = 5

# --- OSC CLIENT ---
client = SimpleUDPClient(OSC_IP, OSC_PORT)

# --- CAMERA HANDLING ---
frames = [None] * len(CAM_IDS)
caps = []
for cam_id in CAM_IDS:
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        caps.append(cap)
    else:
        print(f"Camera {cam_id} failed to initialize.")

def camera_thread(index, cap):
    global frames
    while True:
        ret, frame = cap.read()
        if ret:
            frames[index] = frame
        time.sleep(0.01)

# Start camera threads
for i, cap in enumerate(caps):
    threading.Thread(target=camera_thread, args=(i, cap), daemon=True).start()

# --- HELPER FUNCTIONS ---
def detect_markers(frame, threshold=240):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 2:  # tiny markers
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        markers.append((cx, cy))
    return markers

def group_blobs(blobs, max_dist=MAX_PIVOT_DIST):
    """
    Group nearby blobs into pivot points (3 blobs per pivot)
    """
    groups = []
    used = set()
    for i, b in enumerate(blobs):
        if i in used:
            continue
        group = [b]
        used.add(i)
        for j, other in enumerate(blobs):
            if j in used:
                continue
            if np.linalg.norm(np.array(b) - np.array(other)) < max_dist:
                group.append(other)
                used.add(j)
        if len(group) >= 3:
            group = sorted(group, key=lambda p: (p[1], p[0]))
            groups.append(group[:3])
    return groups

def compute_tracker_rotation(group):
    """
    Compute rotation for 3 points: center, back, left/right
    Returns yaw, pitch, roll in radians
    """
    if len(group) != 3:
        return [0.0, 0.0, 0.0]

    # Lift 2D points to 3D
    m_center = np.array([*group[0], 0], dtype=np.float32)
    m_back = np.array([*group[1], 0], dtype=np.float32)
    m_side = np.array([*group[2], 0], dtype=np.float32)

    x_axis = m_back - m_center
    norm_x = np.linalg.norm(x_axis)
    if norm_x == 0:
        return [0.0, 0.0, 0.0]
    x_axis /= norm_x

    y_axis = m_side - m_center
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    norm_y = np.linalg.norm(y_axis)
    if norm_y == 0:
        return [0.0, 0.0, 0.0]
    y_axis /= norm_y

    z_axis = np.cross(x_axis, y_axis)
    norm_z = np.linalg.norm(z_axis)
    if norm_z == 0:
        return [0.0, 0.0, 0.0]
    z_axis /= norm_z

    yaw = float(np.arctan2(y_axis[1], y_axis[0]))
    pitch = float(np.arcsin(-z_axis[2]))
    roll = float(np.arctan2(-z_axis[1], z_axis[0]))
    return [yaw, pitch, roll]

# --- MAIN LOOP ---
while True:
    combined_frame = None
    pivots_2d = []

    for idx, frame in enumerate(frames):
        if frame is None:
            continue

        markers = detect_markers(frame)
        pivot_groups = group_blobs(markers)

        # Draw pivot points
        for group in pivot_groups:
            colors = [(0,0,255), (0,255,0), (255,0,0)]  # center, back, side
            for j, (x,y) in enumerate(group):
                cv2.circle(frame, (x,y), MARKER_RADIUS, colors[j], -1)

        pivots_2d.append(pivot_groups)

        # Stack frames horizontally for visualization
        if combined_frame is None:
            combined_frame = frame
        else:
            combined_frame = np.hstack((combined_frame, frame))

    if combined_frame is not None:
        cv2.imshow("Pivot Tracker", combined_frame)

    # --- OSC SEND ---
    send_data = []
    for cam_pivots in pivots_2d:
        for group in cam_pivots:
            flat_coords = [float(c) for pt in group for c in pt]
            rot = compute_tracker_rotation(group)
            send_data.append(flat_coords + rot)

    if send_data:
        try:
            client.send_message("/tracking/1/pose", send_data)
        except Exception as e:
            print("OSC Error:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all cameras
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
