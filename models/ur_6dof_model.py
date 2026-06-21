from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pinocchio as pin

class UR6DOFModel(nn.Module):
    def __init__(self):
        super(UR6DOFModel, self).__init__()
        # Define the architecture of your model here
        self.fc1 = nn.Linear(16, 128)  # Input layer for 6D pose (position + orientation)
        self.fc2 = nn.Linear(128, 256) # Hidden layer
        self.fc3 = nn.Linear(256, 6)   # Output layer for joint angles (6 DOF)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class UR6DOFDataset(Dataset):
    def __init__(self, configurations: List[List[float]], poses: List[pin.SE3]):
        self.configurations = configurations
        self.poses = poses

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, idx):
        config = torch.tensor(self.configurations[idx], dtype=torch.float32)
        pose_float = self.poses[idx].homogeneous.flatten()  # Convert SE3 to a flat array (16 elements)
        pose = torch.tensor(pose_float, dtype=torch.float32)
        return config, pose
    