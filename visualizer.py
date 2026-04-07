import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import socket
import json
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R

# ====================================================
# UDP Receiver Setup
# ====================================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5006

# Thread-safe global variable for the latest pose
latest_pose_lock = threading.Lock()
latest_pose_data = None

def udp_listener():
    global latest_pose_data
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP Listener waiting on {UDP_IP}:{UDP_PORT}...")
    
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            payload = json.loads(data.decode('utf-8'))
            with latest_pose_lock:
                latest_pose_data = payload
        except Exception as e:
            print(f"Error receiving UDP data: {e}")

# Start the UDP Listener in a background thread
listener_thread = threading.Thread(target=udp_listener, daemon=True)
listener_thread.start()

# ====================================================
# Forward Kinematics (FK) Dummy Setup
# Assuming 24-joints SMPL-like hierarchy
# ====================================================
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

# 適当なダミーベースオフセットを設定 (単位: メートル等)
# 実際の物理モデルに合わせて後で調整する必要があります。
OFFSETS_DUMMY = np.array([
    [ 0.0,  0.0,  0.0],  # 0: Pelvis
    [-0.5, -0.2,  0.0],  # 1: L_Hip
    [ 0.5, -0.2,  0.0],  # 2: R_Hip
    [ 0.0,  0.6,  0.0],  # 3: Spine1
    [ 0.0, -1.0,  0.0],  # 4: L_Knee
    [ 0.0, -1.0,  0.0],  # 5: R_Knee
    [ 0.0,  0.6,  0.0],  # 6: Spine2
    [ 0.0, -0.8,  0.0],  # 7: L_Ankle
    [ 0.0, -0.8,  0.0],  # 8: R_Ankle
    [ 0.0,  0.6,  0.0],  # 9: Spine3
    [ 0.0, -0.2,  0.4],  # 10: L_Foot
    [ 0.0, -0.2,  0.4],  # 11: R_Foot
    [ 0.0,  0.5,  0.0],  # 12: Neck
    [-0.6,  0.5,  0.0],  # 13: L_Collar
    [ 0.6,  0.5,  0.0],  # 14: R_Collar
    [ 0.0,  0.5,  0.0],  # 15: Head
    [-0.4,  0.0,  0.0],  # 16: L_Shoulder
    [ 0.4,  0.0,  0.0],  # 17: R_Shoulder
    [ 0.0, -0.8,  0.0],  # 18: L_Elbow
    [ 0.0, -0.8,  0.0],  # 19: R_Elbow
    [ 0.0, -0.6,  0.0],  # 20: L_Wrist
    [ 0.0, -0.6,  0.0],  # 21: R_Wrist
    [ 0.0, -0.2,  0.0],  # 22: L_Hand
    [ 0.0, -0.2,  0.0]   # 23: R_Hand
], dtype=np.float32)

def calculate_fk(quaternions):
    """
    24x4のクォータニオン配列を受け取り、
    グローバル座標(XYZ)をForward Kinematicsで計算します。
    ※モデル出力のquaternionが [x, y, z, w] 前提 (scipy.Rotation対応)
    ※もし [w, x, y, z] であれば、別途並び替えが必要。
    """
    if not quaternions or len(quaternions) < 24:
        return np.zeros((24, 3))
    
    positions = np.zeros((24, 3))
    global_rots = [np.eye(3) for _ in range(24)]
    
    for i in range(24):
        q = quaternions[i]
        # Try to convert to rotation matrix
        try:
            # Assuming models output valid quaternion [x, y, z, w]
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

# ====================================================
# Dash Application for Plotly Real-Time Rendering
# ====================================================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time 3D Skeleton Visualizer"),
    html.Div(id='status-text'),
    dcc.Graph(id='3d-scatter', style={'height': '80vh'}),
    dcc.Interval(
        id='interval-component',
        interval=100, # ミリ秒単位での更新 (10 FPS相当)
        n_intervals=0
    )
])

@app.callback(
    [Output('3d-scatter', 'figure'), Output('status-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    global latest_pose_data
    
    with latest_pose_lock:
        if latest_pose_data is None:
            # データがない場合のダミー描画
            positions = np.zeros((24, 3))
            status = "Waiting for UDP data on port 5006..."
            latency_text = ""
        else:
            quats = latest_pose_data.get("pose_quaternion", [])
            latency = latest_pose_data.get("latency_ms", 0)
            positions = calculate_fk(quats)
            status = "Receiving UDP data! "
            latency_text = f" | Latency: {latency} ms"

    # ボーン（線）の描画用データ作成
    bones_x, bones_y, bones_z = [], [], []
    for i, parent in enumerate(PARENTS):
        if parent != -1:
            bones_x.extend([positions[parent, 0], positions[i, 0], None])
            bones_y.extend([positions[parent, 1], positions[i, 1], None])
            bones_z.extend([positions[parent, 2], positions[i, 2], None])

    # Plotly Figureの構築
    fig = go.Figure()

    # ジョイントへの散布図プロット
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Joints'
    ))

    # 骨格（線）へのプロット
    fig.add_trace(go.Scatter3d(
        x=bones_x,
        y=bones_y,
        z=bones_z,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Bones'
    ))

    # 3D空間の見た目調整（等尺性、軸範囲の固定などによるガタつき防止）
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(range=[-2, 2], title='X'),
            yaxis=dict(range=[-2, 2], title='Y'),
            zaxis=dict(range=[-3, 3], title='Z'),
            aspectmode='cube'
        ),
        showlegend=False
    )
    
    return fig, html.H3(f"{status}{latency_text}")

if __name__ == '__main__':
    print("Starting visualization server... Open http://127.0.0.1:8050 to see the skeleton.")
    app.run_server(debug=True, use_reloader=False)
