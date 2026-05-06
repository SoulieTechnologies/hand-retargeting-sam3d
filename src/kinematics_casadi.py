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
z_correction = -0.15
global_scale = 0.8  
print(f"Scaling factor: {global_scale} and z-shifting: {z_correction}")
scale_and_shift_urdf(base_urdf, scaled_urdf, global_scale, z_correction)

model = pin.buildModelFromUrdf(
    scaled_urdf, 
    pin.JointModelFreeFlyer()
)
data = model.createData()

cmodel = cpin.Model(model)
cdata = cmodel.createData()

#Orca_hand
# # joint_mapping = {
#     0: "T-DP_b7429e50",                # Thumb Tip[cite: 1]
#     1: "T-PP_68395e98",                # Thumb IP[cite: 1]
#     2: "R-T-AP_a9723101",              # Thumb MCP[cite: 1]
#     3: "T-TP-R_1c2b802d",              # Thumb Base/CMC[cite: 1]
#     4: "I-FingerTipAssembly_ec49c16c", # Index Tip[cite: 1]
#     6: "I-PP_bacbd481",                # Index PIP[cite: 1]
#     7: "I-AP-R_d95d02d1",              # Index Base/MCP[cite: 1]
#     8: "M-FingerTipAssembly_424a8e75", # Middle Tip[cite: 1]
#     10: "M-PP_8660a1eb",               # Middle PIP[cite: 1]
#     11: "M-AP_6ec59111",               # Middle Base/MCP[cite: 1]
#     12: "M-FingerTipAssembly_34afb748",# Ring Tip[cite: 1]
#     14: "M-PP_08efa608",               # Ring PIP[cite: 1]
#     15: "M-AP_e04a96f2",               # Ring Base/MCP[cite: 1]
#     16: "P-FingerTipAssembly_cd219176",# Pinky Tip[cite: 1]
#     18: "P-PP_1d411b9b",               # Pinky PIP[cite: 1]
#     19: "P-AP_f5e42b61",               # Pinky Base/MCP[cite: 1]
#     20: "R-Carpals_8d1f1041"           # Wrist[cite: 1]
# }

# Sharpa hand
joint_mapping = {
    # --- THUMB ---
    0: "right_thumb_fingertip",  # Tip[cite: 2]
    1: "right_thumb_DP",         # Distal/IP[cite: 2]
    2: "right_thumb_PP",         # Proximal/MCP[cite: 2]
    3: "right_thumb_MC",         # Base/CMC[cite: 2]

    # --- INDEX ---
    4: "right_index_fingertip",  # Tip[cite: 2]
    5: "right_index_DP",         # DIP[cite: 2]
    6: "right_index_MP",         # PIP[cite: 2]
    7: "right_index_PP",         # MCP[cite: 2]

    # --- MIDDLE ---
    8: "right_middle_fingertip", # Tip[cite: 2]
    9: "right_middle_DP",        # DIP[cite: 2]
    10: "right_middle_MP",       # PIP[cite: 2]
    11: "right_middle_PP",       # MCP[cite: 2]

    # --- RING ---
    12: "right_ring_fingertip",  # Tip[cite: 2]
    13: "right_ring_DP",         # DIP[cite: 2]
    14: "right_ring_MP",         # PIP[cite: 2]
    15: "right_ring_PP",         # MCP[cite: 2]

    # --- PINKY ---
    16: "right_pinky_fingertip", # Tip[cite: 2]
    17: "right_pinky_DP",        # DIP[cite: 2]
    18: "right_pinky_MP",        # PIP[cite: 2]
    19: "right_pinky_PP",        # MCP[cite: 2]

    # --- WRIST ---
    #20: "right_hand_C_MC"        # Palm/Wrist Base[cite: 2]
}

frame_ids = {
    point_idx: model.getFrameId(frame_name)
    for point_idx, frame_name in joint_mapping.items()
}

cq = casadi.SX.sym("q", model.nq)

ctargets = casadi.SX.sym("targets", len(frame_ids), 3)
cpin.forwardKinematics(cmodel, cdata, cq)
cpin.updateFramePlacements(cmodel, cdata)

total_error = 0

for i, (point_idx, frame_id) in enumerate(frame_ids.items()):
    current_pos = cdata.oMf[frame_id].translation
    target_pos = ctargets[i, :].T
    total_error += casadi.sumsqr(current_pos - target_pos)

nlp = {'x': cq, 'f': total_error, 'p': ctargets}

opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-4}
solver = casadi.nlpsol('solver', 'ipopt', nlp, opts)

q = pin.neutral(model)
joint_angles_trajectory = []

for frame_idx in range(right_hand_points.shape[0]):
    target_points = right_hand_points[frame_idx].copy()

    wrist_target = target_points[11].copy()
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
    q[3:7] /= np.linalg.norm(q[3:7])

    joint_angles_trajectory.append(q.copy())

    if frame_idx % 10 == 0:
        print(f"Processed {frame_idx}/{right_hand_points.shape[0]} frames...")

np.save("data/actions/retargeted_angles_casadi.npy", np.array(joint_angles_trajectory))
print("Retargeting complete! Saved to retargeted_angles.npy")