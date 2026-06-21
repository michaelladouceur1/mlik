from typing import List
# from networkx import display
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import numpy as np
import time

viz = None

def initialize_visualizer(robot: pin.RobotWrapper) -> MeshcatVisualizer:
    global viz
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    return viz

def display_fk_axes(name: str, poses: List[pin.SE3], axis_length: float = 0.15):
    for i, pose in enumerate(poses):
        display_fk_axis(f"{name}_{i}", pose, axis_length)

def display_fk_axis(name: str, pose: pin.SE3, axis_length: float = 0.15):
    """Draw the computed SE3 pose as colored XYZ axes in Meshcat.
    Overlaps with displayFrames axes if FK is correct.
    X=red, Y=green, Z=blue
    """
    R = pose.rotation
    t = pose.translation

    for col_idx, (color, label) in enumerate([
        (0x770000, "x"),
        (0x007700, "y"),
        (0x000077, "z"),
    ]):
        axis_end = t + R[:, col_idx] * axis_length
        # PointsGeometry expects a (3, N) float32 array; LineSegments pairs (0,1), (2,3), ...
        vertices = np.column_stack([t, axis_end]).astype(np.float32)
        viz.viewer[f"{name}/{label}"].set_object(
            mg.LineSegments(
                mg.PointsGeometry(vertices),
                mg.LineBasicMaterial(color=color, linewidth=3),
            )
        )

def display_link_axes(robot: pin.RobotWrapper, frames: List[str]):
    frame_ids = [robot.model.getFrameId(frame) for frame in frames]
    viz.displayFrames(True, frame_ids=frame_ids, axis_length=0.1, axis_width=10)

def display_model_configurations(configurations: List[List[float]], delay: float = 1.0):
    for q in configurations:
        viz.display(q)
        time.sleep(delay)