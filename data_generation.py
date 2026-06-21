from typing import List
import pinocchio as pin
from visualize import initialize_visualizer, display_fk_axes, display_link_axes, display_model_configurations

URDF_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/ur10_robot.urdf"
URDF_PACKAGE_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/"

def generate_random_configurations(robot: pin.RobotWrapper, num_samples: int) -> List[List[float]]:
    configurations = []
    for _ in range(num_samples):
        q = pin.randomConfiguration(robot.model)
        configurations.append(q)
    return configurations

def calculate_poses_fk(robot: pin.RobotWrapper, configurations: List[List[float]], frame: str) -> List[pin.SE3]:
    poses = []
    for q in configurations:
        pose = calculate_pose_fk(robot, q, frame)
        poses.append(pose)
    return poses

def calculate_pose_fk(robot: pin.RobotWrapper, q: List[float], frame: str) -> pin.SE3:
    data = robot.model.createData()
    pin.forwardKinematics(robot.model, data, q)
    pin.updateFramePlacements(robot.model, data)
    tool0_index = robot.model.getFrameId(frame)
    tool0_pose = data.oMf[tool0_index]
    return tool0_pose

def get_robot_intrinsics(robot: pin.RobotWrapper):
    # Placeholder for extracting robot intrinsics (e.g., joint limits, link lengths)
    # This can be expanded based on specific requirements
    return {
        "joint_limits": robot.model.lowerPositionLimit,  # Assuming symmetric limits
        "link_lengths": [robot.model.inertias[i].mass for i in range(robot.model.njoints)],  # Example using mass as a proxy
    }

if __name__ == "__main__":
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, [URDF_PACKAGE_PATH])

    configurations = generate_random_configurations(robot, 1000)
    tool0_poses = calculate_poses_fk(robot, configurations, "tool0")
    # for i, q in enumerate(configurations):
    #     print(f"Configuration {i}: {q.T}")
    #     tool0_pose = calculate_pose_fk(robot, q, "tool0")
    #     print(f"Tool0 pose for configuration {i}: {tool0_pose}\n")

    # visualize_model(robot, configurations)

    initialize_visualizer(robot)
    display_link_axes(robot, ["tool0"])
    display_fk_axes("fk_computed", tool0_poses, axis_length=0.15)
    display_model_configurations(configurations, delay=1.0)