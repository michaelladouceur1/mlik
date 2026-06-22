from typing import List
import pinocchio as pin
import numpy as np
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from visualize import initialize_visualizer, display_fk_axes, display_link_axes, display_model_configurations

URDF_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/ur10_robot.urdf"
URDF_PACKAGE_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/"

def generate_random_configurations(robot: pin.RobotWrapper, num_samples: int) -> List[List[float]]:
    configurations = []
    for _ in range(num_samples):
        q = pin.randomConfiguration(robot.model)
        configurations.append(q)
    return configurations

def generate_random_poses(robot: pin.RobotWrapper, num_samples: int, frame: str) -> List[pin.SE3]:
    poses = []
    for _ in range(num_samples):
        q = pin.randomConfiguration(robot.model)
        pose = calculate_configuration_fk(robot, q, frame)
        poses.append(pose)
    return poses

def calculate_poses_ik(robot: pin.RobotWrapper, poses: List[pin.SE3], frame: str) -> List[List[float]]:
    configurations = []
    for pose in poses:
        q = calculate_pose_ik(robot, pose, frame)
        configurations.append(q)
    return configurations

def calculate_pose_ik(
    robot: pin.RobotWrapper,
    pose: pin.SE3,
    frame: str,
    q_init: np.ndarray = None,
    eps: float = 1e-4,
    max_iter: int = 1000,
    dt: float = 1e-1,
) -> np.ndarray:
    configuration = pink.Configuration(robot.model, robot.data, 
                                       q_init if q_init is not None else pin.neutral(robot.model))

    frame_task = FrameTask(frame, position_cost=1.0, orientation_cost=1.0)
    posture_task = PostureTask(cost=1e-3)  # Regularization towards neutral pose

    frame_task.set_target(pose)
    posture_task.set_target(pin.neutral(robot.model))

    tasks = [frame_task, posture_task]

    for _ in range(max_iter):
        velocity = solve_ik(configuration, tasks, dt, solver="quadprog")
        configuration.integrate_inplace(velocity, dt)

        if np.linalg.norm(frame_task.compute_error(configuration)) < eps:
            return configuration.q

    print("Warning: IK did not converge")
    return configuration.q

def calculate_configurations_fk(robot: pin.RobotWrapper, configurations: List[List[float]], frame: str) -> List[pin.SE3]:
    poses = []
    for q in configurations:
        pose = calculate_configuration_fk(robot, q, frame)
        poses.append(pose)
    return poses

def calculate_configuration_fk(robot: pin.RobotWrapper, q: List[float], frame: str) -> pin.SE3:
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
    tool0_poses = calculate_configurations_fk(robot, configurations, "tool0")
    # for i, q in enumerate(configurations):
    #     print(f"Configuration {i}: {q.T}")
    #     tool0_pose = calculate_pose_fk(robot, q, "tool0")
    #     print(f"Tool0 pose for configuration {i}: {tool0_pose}\n")

    # visualize_model(robot, configurations)

    initialize_visualizer(robot)
    display_link_axes(robot, ["tool0"])
    display_fk_axes("fk_computed", tool0_poses, axis_length=0.15)
    display_model_configurations(configurations, delay=1.0)