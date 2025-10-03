import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import time

# -------- Settings --------
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "training-material")
MODEL_PATH = os.path.join(SAVE_DIR, "pose_model.pth")
CAP_W, CAP_H = 1280, 720
DISPLAY_W, DISPLAY_H = 1280, 720
JOINT_NAMES = ["chest","left_hip","left_knee","left_foot","right_foot","right_knee","right_hip"]

# -------- Model (same as training) --------
class PoseNet(nn.Module):
    def __init__(self, num_joints=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_joints*2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------- Load model --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseNet(len(JOINT_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------- Camera --------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CAP_W, CAP_H))
    img_tensor = torch.tensor(img/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(DEVICE)

    # Predict joints
    with torch.no_grad():
        output = model(img_tensor).cpu().numpy().reshape(-1,2)
        output[:,0] *= CAP_W
        output[:,1] *= CAP_H

    # Draw joints
    for i, (x,y) in enumerate(output):
        cv2.circle(frame, (int(x),int(y)), 5, (0,255,0), -1)
        cv2.putText(frame, JOINT_NAMES[i], (int(x)+5,int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # FPS
    curr_time = time.time()
    fps = 0.9*fps + 0.1/(curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
