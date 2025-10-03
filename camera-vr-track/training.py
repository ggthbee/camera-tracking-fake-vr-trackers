import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# -------- Settings --------
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "training-material")
IMAGE_SIZE = (1280, 720)  # Resize images for training
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
JOINT_NAMES = ["chest", "left_hip", "left_knee", "left_foot", "right_foot", "right_knee", "right_hip"]

# -------- Dataset --------
class PoseDataset(Dataset):
    def __init__(self, folder):
        self.samples = []
        for file in os.listdir(folder):
            if file.endswith(".json"):
                json_path = os.path.join(folder, file)
                with open(json_path, "r") as f:
                    data = json.load(f)
                img_path = os.path.join(folder, data["image"])
                if os.path.exists(img_path):
                    joints = []
                    for joint in JOINT_NAMES:
                        x, y = data["joints"][joint]
                        joints.append(x / CAP_W)
                        joints.append(y / CAP_H)
                    self.samples.append((img_path, np.array(joints, dtype=np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, joints = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0  # Normalize
        img = np.transpose(img, (2,0,1))  # CHW
        return torch.tensor(img, dtype=torch.float32), torch.tensor(joints, dtype=torch.float32)

# -------- Model --------
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

# -------- Training --------
def train():
    dataset = PoseDataset(SAVE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PoseNet(num_joints=len(JOINT_NAMES))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, joints in dataloader:
            imgs, joints = imgs.to(device), joints.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, joints)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pose_model.pth"))
    print("Model saved at pose_model.pth")

if __name__ == "__main__":
    CAP_W, CAP_H = 1280, 720  # needed to normalize coordinates
    train()
