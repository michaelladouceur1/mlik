import torch
import torch.nn as nn
import pinocchio as pin
import numpy as np
from torch.utils.data import DataLoader
from models.ur_6dof_model import UR6DOFModel, UR6DOFDataset
from data_generation import calculate_configuration_fk, generate_random_configurations, calculate_configurations_fk

URDF_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/ur10_robot.urdf"
URDF_PACKAGE_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/"

def train_model(robot, model, num_samples=10000, batch_size=32, epochs=10, device='cpu'):
    print(f"Training on device: {device}")

    model.to(device)

    configurations = generate_random_configurations(robot, num_samples)
    tool0_poses = calculate_configurations_fk(robot, configurations, "tool0")

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

def evaluate_model(robot, model, device='cpu'):
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_configs = generate_random_configurations(robot, 100)
        test_poses = calculate_configurations_fk(robot, test_configs, "tool0")
        test_dataset = UR6DOFDataset(test_configs, test_poses)
        test_dataloader = DataLoader(test_dataset, batch_size=32)

        total_loss = 0
        criterion = nn.MSELoss()
        for configs, poses in test_dataloader:
            configs, poses = configs.to(device), poses.to(device)
            outputs = model(poses)
            loss = criterion(outputs, configs)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss/len(test_dataloader)}")

        average_distance = 0
        # calculate distance from predicted configurations to true configurations
        for i in range(len(test_configs)):
            pred_config = model(test_dataset[i][1].unsqueeze(0).to(device)).cpu().numpy().flatten()
            true_config = test_dataset[i][0].numpy().flatten()

            pred_tool0_pose = calculate_configuration_fk(robot, pred_config, "tool0")
            true_tool0_pose = calculate_configuration_fk(robot, true_config, "tool0")

            pred_translation = pred_tool0_pose.translation.T
            true_translation = true_tool0_pose.translation.T

            distance = np.linalg.norm(pred_translation - true_translation)
            average_distance += distance

            # print(f"Sample {i}: Predicted Translation: {pred_translation}, True Translation: {true_translation}, Distance: {distance}")

        average_distance /= len(test_configs)
        print(f"Average Distance between Predicted and True Tool0 Translations: {average_distance} meters")

def save_model(model, path="ur6dof_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, [URDF_PACKAGE_PATH])
    model = UR6DOFModel()

    train_model(robot, model, device=device)
    evaluate_model(robot, model, device=device)
    save_model(model)