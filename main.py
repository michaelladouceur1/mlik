import torch
import torch.nn as nn
import pinocchio as pin
from torch.utils.data import DataLoader
from models.ur_6dof_model import UR6DOFModel, UR6DOFDataset
from data_generation import generate_random_configurations, calculate_poses_fk

URDF_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/ur10_robot.urdf"
URDF_PACKAGE_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/"

def train_model(robot, model, num_samples=10000, batch_size=32, epochs=30, device='cpu'):
    print(f"Training on device: {device}")

    model.to(device)

    configurations = generate_random_configurations(robot, num_samples)
    tool0_poses = calculate_poses_fk(robot, configurations, "tool0")

    # print configurations and poses size
    print(f"Generated {len(configurations)} configurations and {len(tool0_poses)} poses.")

    dataset = UR6DOFDataset(configurations, tool0_poses)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for configs, poses in dataloader:
            configs, poses = configs.to(device), poses.to(device)
            optimizer.zero_grad()
            outputs = model(poses)
            loss = criterion(outputs, configs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, [URDF_PACKAGE_PATH])
    model = UR6DOFModel()
    train_model(robot, model, device=device)