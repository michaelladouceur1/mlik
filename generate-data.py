import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time

def visualize_model():
    robot = pin.RobotWrapper.BuildFromURDF("/home/michael/Documents/mlik/urdf/ur_description/urdf/ur10_robot.urdf", ["/home/michael/Documents/mlik/urdf/ur_description/urdf/"])
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    q = robot.q0  # or pin.neutral(robot.model)
    viz.display(q)

if __name__ == "__main__":
    model = pin.buildModelsFromUrdf("/home/michael/Documents/mlik/urdf/ur_description/urdf/ur10_robot.urdf", ["/home/michael/Documents/mlik/urdf/ur_description/urdf/"])[0]

        # Create data required by the algorithms
    data = model.createData()
    
    np.random.seed(int(time.time() * 1000) % 2**32)

    # Sample a random configuration
    q = pin.randomConfiguration(model)
    print(f"q: {q.T}")
    
    # Perform the forward kinematics over the kinematic tree
    pin.forwardKinematics(model, data, q)
    
    # Print out the placement of each joint of the kinematic tree
    for name, oMi in zip(model.names, data.oMi):
        print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

    visualize_model()