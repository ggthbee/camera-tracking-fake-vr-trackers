import time
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

# === VRChat OSC Bridge ===
class VRChatOSCBridge:
    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = SimpleUDPClient(ip, port)
        self.trackers = {
            "hip": "/tracking/trackers/hip",
            "left_elbow": "/tracking/trackers/left_elbow",
            "right_elbow": "/tracking/trackers/right_elbow",
            "left_knee": "/tracking/trackers/left_knee",
            "right_knee": "/tracking/trackers/right_knee",
            "left_ankle": "/tracking/trackers/left_ankle",
            "right_ankle": "/tracking/trackers/right_ankle",
            "head": "/tracking/trackers/head"
        }
        self.last_positions = {k: np.zeros(3) for k in self.trackers}
        self.alpha = 0.5  # smoothing factor

    def smooth(self, key, new_pos):
        """Exponential smoothing filter for positions."""
        last = self.last_positions[key]
        smoothed = self.alpha * np.array(new_pos) + (1 - self.alpha) * last
        self.last_positions[key] = smoothed
        return smoothed

    def send_tracker(self, tracker_id, position, orientation=(0, 0, 0, 1)):
        """Send OSC message for a single tracker."""
        if tracker_id not in self.trackers:
            return
        smoothed = self.smooth(tracker_id, position)

        # Convert from OpenCV/MediaPipe (x right, y down, z forward)
        # to VRChat (x right, y up, z forward)
        vrc_pos = [float(smoothed[0]), float(-smoothed[1]), float(smoothed[2])]

        addr = self.trackers[tracker_id]
        self.client.send_message(addr, list(vrc_pos) + list(orientation))

    def send_all(self, positions):
        """Send multiple trackers at once.
        positions = { "hip": [x,y,z], "head": [x,y,z], ... }
        """
        for k, pos in positions.items():
            self.send_tracker(k, pos)

# === Example Usage ===
if __name__ == "__main__":
    bridge = VRChatOSCBridge()

    # Fake test data: moving hip up and down
    t = 0
    while True:
        hip_y = np.sin(t) * 0.1
        fake_positions = {
            "hip": [0.0, hip_y, 1.0],
            "head": [0.0, hip_y + 0.5, 1.0]
        }
        bridge.send_all(fake_positions)
        t += 0.1
        time.sleep(0.05)
