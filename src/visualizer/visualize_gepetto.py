import argparse
import os
import time

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
DEFAULT_URDF = os.path.join(_PROJECT_ROOT, "assets", "orcahand", "orcahand_scaled.urdf")
DEFAULT_ACTIONS = os.path.join(_PROJECT_ROOT, "data", "actions", "retargeted_angles.npy")
DEFAULT_TARGETS = os.path.join(_PROJECT_ROOT, "data", "sam3d_outputs", "dexsuite_joints1.npy")
DEFAULT_FPS = 30


def _refresh_visualizer() -> None:
    """Redraw GUI without advancing dynamics (see playback loop)."""
    if hasattr(p, "COV_ENABLE_SINGLE_STEP_RENDERING"):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    else:
        p.getCameraImage(1, 1, renderer=p.ER_TINY_RENDERER)


def main():
    ap = argparse.ArgumentParser(description="Play back joint angles on the OrcaHand URDF in PyBullet.")
    ap.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    ap.add_argument(
        "--actions",
        type=str,
        default=DEFAULT_ACTIONS,
        help="Path to retargeted_angles.npy.",
    )
    ap.add_argument(
        "--targets", 
        type=str, 
        default=DEFAULT_TARGETS,
        help="Path to the SAM3D target points dataset."
    )
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = ap.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    actions_path = os.path.abspath(args.actions)
    targets_path = os.path.abspath(args.targets)
    fps = float(args.fps)

    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not os.path.isfile(actions_path):
        raise FileNotFoundError(f"Actions not found: {actions_path}. Run kinematics first.")
    if not os.path.isfile(targets_path):
        raise FileNotFoundError(f"Targets not found: {targets_path}.")

    print("Loading Pinocchio model for joint / q layout...")
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    pin_joint_to_qidx: dict[str, int] = {}
    
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        if j.nq == 0:
            continue
        name = model.names[jid]
        if name == "root_joint" and j.nq == 7:
            continue
        if j.nq != 1:
            raise RuntimeError(f"Joint {name} has nq={j.nq}; this script expects only 1-DoF joints.")
        pin_joint_to_qidx[name] = j.idx_q

    print("Loading actions and target data...")
    actions = np.load(actions_path)
    target_data = np.load(targets_path)
    right_hand_targets = target_data[:, 1, :, :]
    
    num_frames = actions.shape[0]
    if actions.shape[1] != model.nq:
        raise ValueError(f"Actions have shape {actions.shape}, but model.nq={model.nq}")

    print("Starting PyBullet...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, 0)

    # Note: We remove the start_orientation offset so it matches Pinocchio's coordinate math perfectly
    hand_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
    )

    pb_joints: dict[str, int] = {}
    for i in range(p.getNumJoints(hand_id)):
        info = p.getJointInfo(hand_id, i)
        joint_name = info[1].decode("utf-8")
        if info[2] != p.JOINT_FIXED:
            pb_joints[joint_name] = i
            p.setJointMotorControl2(hand_id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    # --- NEW: Spawn 21 Red Spheres for Target Visualization ---
    print("Spawning target markers...")
    sphere_radius = 0.006 
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 0, 0, 0.8])
    marker_ids = [p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id) for _ in range(21)]
    # ----------------------------------------------------------

    missing = [jn for jn in pin_joint_to_qidx if jn not in pb_joints]
    if missing:
        print("Warning: Pinocchio joints not found in PyBullet:", missing)
        
    print(f"Playback: kinematic-only. {num_frames} frames at {fps} FPS. Close the window to exit.")

    try:
        while p.isConnected():
            for frame_idx in range(num_frames):
                current_q = actions[frame_idx]
                
                # 1. Update Robot Base (FreeFlyer)
                p.resetBasePositionAndOrientation(hand_id, current_q[0:3], current_q[3:7])

                # 2. Update Robot Fingers
                for joint_name, q_idx in pin_joint_to_qidx.items():
                    if joint_name not in pb_joints:
                        continue
                    pb_idx = pb_joints[joint_name]
                    p.resetJointState(hand_id, pb_idx, float(current_q[q_idx]))
                
                # 3. Update Target Spheres
                # Loop the targets if the actions array is longer than the targets array
                target_frame = right_hand_targets[frame_idx % right_hand_targets.shape[0]].copy()
                
                # Apply normalization and axis flips exactly as done in kinematics.py
                target_frame = target_frame - target_frame[20].copy()
                target_frame[:, 2] = -target_frame[:, 2] # Flips Z axis
                
                for pt_idx in range(21):
                    p.resetBasePositionAndOrientation(
                        marker_ids[pt_idx], 
                        target_frame[pt_idx], 
                        [0, 0, 0, 1]
                    )

                _refresh_visualizer()
                time.sleep(1.0 / fps)
                
    except p.error:
        print("Visualizer closed.")
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()