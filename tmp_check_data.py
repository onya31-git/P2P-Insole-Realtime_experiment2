import numpy as np
import plotly.graph_objects as go

PARENTS = [
    -1,  0,  0,  0, 
     1,  2,  3, 
     4,  5,  6, 
     7,  8,  9, 
     9,  9, 12, 
    13, 14, 
    16, 17, 
    18, 19, 
    20, 21
]

OFFSETS_DUMMY = np.array([
    [ 0.0,   0.0,   0.0  ],  # 0: Pelvis
    [-0.09, -0.05,  0.0  ],  # 1: L_Hip
    [ 0.09, -0.05,  0.0  ],  # 2: R_Hip
    [ 0.0,   0.10,  0.0  ],  # 3: Spine1
    [ 0.0,  -0.45,  0.0  ],  # 4: L_Knee
    [ 0.0,  -0.45,  0.0  ],  # 5: R_Knee
    [ 0.0,   0.12,  0.0  ],  # 6: Spine2
    [ 0.0,  -0.42,  0.0  ],  # 7: L_Ankle
    [ 0.0,  -0.42,  0.0  ],  # 8: R_Ankle
    [ 0.0,   0.12,  0.0  ],  # 9: Spine3
    [ 0.0,  -0.05,  0.15 ],  # 10: L_Foot
    [ 0.0,  -0.05,  0.15 ],  # 11: R_Foot
    [ 0.0,   0.15,  0.0  ],  # 12: Neck
    [-0.05,  0.15,  0.0  ],  # 13: L_Collar
    [ 0.05,  0.15,  0.0  ],  # 14: R_Collar
    [ 0.0,   0.15,  0.0  ],  # 15: Head
    [-0.15,  0.0,   0.0  ],  # 16: L_Shoulder
    [ 0.15,  0.0,   0.0  ],  # 17: R_Shoulder
    [ 0.0,  -0.28,  0.0  ],  # 18: L_Elbow
    [ 0.0,  -0.28,  0.0  ],  # 19: R_Elbow
    [ 0.0,  -0.25,  0.0  ],  # 20: L_Wrist
    [ 0.0,  -0.25,  0.0  ],  # 21: R_Wrist
    [ 0.0,  -0.10,  0.0  ],  # 22: L_Hand
    [ 0.0,  -0.10,  0.0  ]   # 23: R_Hand
], dtype=np.float32)

def calculate_fk(quaternions):
    positions = np.zeros((24, 3))
    global_rots = [np.eye(3) for _ in range(24)]
    
    for i in range(24):
        q = quaternions[i]
        try:
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat(q)
            local_rot = r.as_matrix()
        except:
            local_rot = np.eye(3)
        
        parent = PARENTS[i]
        if parent == -1:
            global_rots[i] = local_rot
            positions[i] = OFFSETS_DUMMY[i]
        else:
            global_rots[i] = global_rots[parent] @ local_rot
            positions[i] = positions[parent] + (global_rots[parent] @ OFFSETS_DUMMY[i])
            
    return positions

identity_quats = [[0,0,0,1] for _ in range(24)]
pos = calculate_fk(identity_quats)
output = ""
for i, p in enumerate(pos):
    output += f"{i}: {p}\n"
with open('output_fk_test.txt', 'w') as f:
    f.write(output)
