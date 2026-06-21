from typing import List
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time

URDF_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/ur10_robot.urdf"
URDF_PACKAGE_PATH = "/home/michael/Documents/mlik/urdf/ur_description/urdf/"

def visualize_model(robot: pin.RobotWrapper, poses: List[List[float]]):
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Enable tool0 frame visualization (RGB axes: X=red, Y=green, Z=blue)
    tool0_id = robot.model.getFrameId("tool0")
    viz.displayFrames(True, frame_ids=[tool0_id], axis_length=0.1, axis_width=4)

    for q in poses:
        viz.display(q)
        time.sleep(0.5)

def generate_random_configurations(robot: pin.RobotWrapper, num_samples: int) -> List[List[float]]:
    configurations = []
    for _ in range(num_samples):
        q = pin.randomConfiguration(robot.model)
        configurations.append(q)
    return configurations

def calculate_tool0_forward_kinematics(robot: pin.RobotWrapper, q: List[float], frame: str) -> pin.SE3:
    data = robot.model.createData()
    pin.forwardKinematics(robot.model, data, q)
    pin.updateFramePlacements(robot.model, data)
    tool0_index = robot.model.getFrameId(frame)
    tool0_pose = data.oMf[tool0_index]
    return tool0_pose

if __name__ == "__main__":
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, [URDF_PACKAGE_PATH])

        # Create data required by the algorithms
    data = robot.model.createData()

    # print names of the joints in the kinematic tree    print("Joint names in the kinematic tree:")
    for name in robot.model.names:
        print(name)
        
    # Perform the forward kinematics over the kinematic tree
    # pin.forwardKinematics(robot.model, data, q)
    
    # Print out the placement of each joint of the kinematic tree
    # for name, oMi in zip(robot.model.names, data.oMi):
    #     print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

    configurations = generate_random_configurations(robot, 10)
    for i, q in enumerate(configurations):
        print(f"Configuration {i}: {q.T}")
        tool0_pose = calculate_tool0_forward_kinematics(robot, q, "tool0")
        print(f"Tool0 pose for configuration {i}: {tool0_pose}\n")

    visualize_model(robot, configurations)