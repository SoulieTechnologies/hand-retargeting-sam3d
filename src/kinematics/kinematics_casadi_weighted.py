import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi
from urdf_scaler import scale_and_shift_urdf

hand_data = np.load("/home/theo/code/hand-retargeting-sam3d/data/sam3d_outputs/dexsuite_joints1.npy")
right_hand_points = hand_data[:, 1, :, :]

base_dir = "/home/theo/code/hand-retargeting-sam3d/assets/sharpa_hand"
base_urdf = f"{base_dir}/wave_01/right_sharpa_wave/right_sharpa_wave.urdf"
scaled_urdf = f"{base_dir}/wave_01/right_sharpa_wave/right_sharpa_wave_scaled.urdf"
z_correction = -0.04
global_scale = 0.85
print(f"Scaling factor: {global_scale} and z-shifting: {z_correction}")
scale_and_shift_urdf(base_urdf, scaled_urdf, global_scale, z_correction)

model = pin.buildModelFromUrdf(
    scaled_urdf, 
    pin.JointModelFreeFlyer()
)
data = model.createData()

cmodel = cpin.Model(model)
cdata = cmodel.createData()

# Sharpa hand
joint_mapping = {
    # --- THUMB ---
    0: "right_thumb_fingertip",  1: "right_thumb_DP",   2: "right_thumb_PP",   3: "right_thumb_MC",
    # --- INDEX ---
    4: "right_index_fingertip",  5: "right_index_DP",   6: "right_index_MP",   7: "right_index_PP",
    # --- MIDDLE ---
    8: "right_middle_fingertip", 9: "right_middle_DP",  10: "right_middle_MP", 11: "right_middle_PP",
    # --- RING ---
    12: "right_ring_fingertip",  13: "right_ring_DP",   14: "right_ring_MP",   15: "right_ring_PP",
    # --- PINKY ---
    16: "right_pinky_fingertip", 17: "right_pinky_DP",  18: "right_pinky_MP",  19: "right_pinky_PP"
}

frame_ids = {
    point_idx: model.getFrameId(frame_name)
    for point_idx, frame_name in joint_mapping.items()
}

cq = casadi.SX.sym("q", model.nq)
ctargets = casadi.SX.sym("targets", len(frame_ids), 3)
cpin.forwardKinematics(cmodel, cdata, cq)
cpin.updateFramePlacements(cmodel, cdata)

# --- NOUVEAU : SYSTÈME DE POIDS (WEIGHTS) ---
point_weights = {
    # Bouts des doigts (Priorité Absolue : 50x plus importants)
    0: 50.0, 4: 50.0, 8: 50.0, 12: 50.0, 16: 50.0,
    
    # Phalanges intermédiaires PIP/DIP (Priorité Moyenne : Garde la forme du doigt)
    1: 2.0, 2: 2.0,   
    5: 2.0, 6: 2.0,   
    9: 2.0, 10: 2.0,  
    13: 2.0, 14: 2.0, 
    17: 2.0, 18: 2.0, 
    
    # Jointures de base MCP (Priorité Faible : On les laisse dériver pour absorber l'erreur anatomique)
    3: 0.1, 7: 0.1, 11: 0.1, 15: 0.1, 19: 0.1
}

total_error = 0

for i, (point_idx, frame_id) in enumerate(frame_ids.items()):
    current_pos = cdata.oMf[frame_id].translation
    target_pos = ctargets[i, :].T
    
    w = point_weights.get(point_idx, 1.0)
    
    total_error += w * casadi.sumsqr(current_pos - target_pos)
# -------------------------------------------

nlp = {'x': cq, 'f': total_error, 'p': ctargets}

opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
solver = casadi.nlpsol('solver', 'ipopt', nlp, opts)

q = pin.neutral(model)
joint_angles_trajectory = []

for frame_idx in range(right_hand_points.shape[0]):
    target_points = right_hand_points[frame_idx].copy()

    wrist_target = target_points[20].copy()
    target_points = target_points - wrist_target
    target_points[:, 2] = -target_points[:, 2]

    active_targets = np.array([target_points[idx] for idx in frame_ids.keys()])

    solution = solver(
        x0=q,
        lbx=model.lowerPositionLimit,
        ubx=model.upperPositionLimit,
        p=active_targets
    )

    q = solution['x'].full().flatten()
    
    pin.framesForwardKinematics(model, data, q)
    tip_error_cm = np.linalg.norm(data.oMf[frame_ids[4]].translation - target_points[4]) * 100
    
    q[3:7] /= np.linalg.norm(q[3:7])

    joint_angles_trajectory.append(q.copy())

    print(f"Frame {frame_idx:03d}/{right_hand_points.shape[0]} | Index Tip Error: {tip_error_cm:.2f} cm")
    
np.save("data/actions/retargeted_angles_casadi_weighted.npy", np.array(joint_angles_trajectory))
print("Retargeting complete! Saved to retargeted_angles_casadi_weighted.npy")