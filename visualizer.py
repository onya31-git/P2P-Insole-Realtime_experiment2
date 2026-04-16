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

# モデル出力順序(CSVのPositions出現順)から、可視化用(PARENTS)のインデックスへのマッピング
CSV_INDEX_TO_VIS_INDEX = {
    0: 20, # LWrist
    1: 18, # LElbow
    2: 16, # LShoulder
    3: 21, # RWrist
    4: 19, # RElbow
    5: 17, # RShoulder
    6: 10, # LToe
    7: 7,  # LAnkle
    8: 4,  # LKnee
    9: 1,  # LHip
    10: 11, # RToe
    11: 8,  # RAnkle
    12: 5,  # RKnee
    13: 2,  # RHip
    15: 13, # LClavicle
    16: 22, # LHandEnd
    18: 14, # RClavicle
    19: 23, # RHandEnd
    21: 0,  # Spine -> Pelvis
    22: 3,  # Spine1
    23: 6,  # Spine2
}

def reorder_positions(raw_positions):
    """
    受信した24x3の座標データを、可視化用(PARENTS)のインデックス順に並べ替えます。
    マッピングされていない関節（Spine3, Neck, Head）は隣接関節から補間推定します。
    """
    remapped = np.zeros((24, 3))
    if not raw_positions or len(raw_positions) < 24:
        return remapped

    for model_idx, vis_idx in CSV_INDEX_TO_VIS_INDEX.items():
        if model_idx < len(raw_positions):
            remapped[vis_idx] = raw_positions[model_idx]

    # -------------------------------------------------------
    # 未マッピング関節の補間推定（再学習不要）
    # vis 9  = Spine3 : Spine2（vis 6）と肩中点の間
    # vis 12 = Neck   : 肩中点より少し上
    # vis 15 = Head   : Neck より頭一個分上
    # -------------------------------------------------------
    l_shoulder = remapped[16]  # vis 16: LShoulder
    r_shoulder = remapped[17]  # vis 17: RShoulder
    spine2     = remapped[6]   # vis 6:  Spine2

    if np.any(l_shoulder != 0) and np.any(r_shoulder != 0):
        shoulder_mid = (l_shoulder + r_shoulder) / 2.0

        # Spine3: Spine2と肩中点の中間（やや肩側）
        remapped[9]  = spine2 * 0.4 + shoulder_mid * 0.6

        # Neck: 肩中点をわずかに上方向（Yが高さ）へ
        remapped[12] = shoulder_mid + np.array([0.0, 0.08, 0.0])

        # Head: Neckのさらに上
        remapped[15] = remapped[12] + np.array([0.0, 0.20, 0.0])

    return remapped

# ====================================================
# Dash Application for Plotly Real-Time Rendering
# ====================================================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time 3D Skeleton Visualizer (Direct Position)"),
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
            raw_positions = np.zeros((24, 3))
            status = "Waiting for UDP data on port 5006..."
            latency_text = ""
        else:
            raw_positions = latest_pose_data.get("pose_positions", [])
            latency = latest_pose_data.get("latency_ms", 0)
            status = "Receiving UDP data! "
            latency_text = f" | Latency: {latency} ms"

    # モデルからの生出力を階層の順序に再配置
    positions = reorder_positions(raw_positions)

    # ======================================================
    # 座標系変換: データはY=高さ、PlotlyはZ=上
    # Plotly用マッピング: data_X -> plot_X, data_Y -> plot_Z(上), data_Z -> plot_Y(奥行)
    # ======================================================
    px = positions[:, 0]   # X: 左右
    py = positions[:, 2]   # Y(奥行): データのZ
    pz = positions[:, 1]   # Z(高さ): データのY

    # ボーン（線）の描画用データ作成
    bones_px, bones_py, bones_pz = [], [], []
    for i, parent in enumerate(PARENTS):
        if parent != -1:
            bones_px.extend([positions[parent, 0], positions[i, 0], None])
            bones_py.extend([positions[parent, 2], positions[i, 2], None])  # data Z -> plot Y
            bones_pz.extend([positions[parent, 1], positions[i, 1], None])  # data Y -> plot Z

    # Plotly Figureの構築
    fig = go.Figure()

    # ジョイントへの散布図プロット
    fig.add_trace(go.Scatter3d(
        x=px,
        y=py,
        z=pz,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Joints'
    ))

    # 骨格（線）へのプロット
    fig.add_trace(go.Scatter3d(
        x=bones_px,
        y=bones_py,
        z=bones_pz,
        mode='lines',
        line=dict(color='royalblue', width=4),
        name='Bones'
    ))

    # 3D空間の見た目調整
    # Z=高さ(0〜2m), X=左右(-1〜1m), Y=奥行(-1〜1m)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(range=[-1, 1], title='X (左右)'),
            yaxis=dict(range=[-1, 1], title='Y (奥行)'),
            zaxis=dict(range=[-1.2, 1.2], title='Z (高さ)'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),  # 正面やや斜め上から見る
                up=dict(x=0, y=0, z=1)             # Z軸を上方向に固定
            )
        ),
        showlegend=False
    )

    return fig, html.H3(f"{status}{latency_text}")

if __name__ == '__main__':
    print("Starting visualization server... Open http://127.0.0.1:8050 to see the skeleton.")
    app.run(debug=True, use_reloader=False)
