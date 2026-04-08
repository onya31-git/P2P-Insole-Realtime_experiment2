import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from models.model import KinematicFusionModel
from processor.filter import OneEuroFilter

# ==========================================
# 5. 損失関数と学習ループ
# ==========================================
class KinematicLoss(nn.Module):
    def __init__(self, lambda_bone=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_bone = lambda_bone

    def forward(self, pred_quat, target_quat):
        # クォータニオンに対するMSE誤差 
        # (※ 厳密には q と -q は同じ回転を表すが、連続した時系列予測ならMSEでも一定の学習は可能。
        # 正確には 1 - |q1*q2| などを利用する場合もあるが、要件に合わせてMSEを採用)
        loss_quat = self.mse_loss(pred_quat, target_quat)
        
        # 骨の長さを比較する制約ペナルティ（モック。実際はFKで3D座標を求めて距離を計算）
        loss_bone = torch.tensor(0.0, device=pred_quat.device)
        
        loss = loss_quat + self.lambda_bone * loss_bone
        return loss

def train_dummy():
    """ダミーデータを用いたトレーニングループ"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # パラメータ設定
    BATCH_SIZE = 8
    SEQ_LEN = 20
    H, W = 32, 32
    NUM_SENSORS = 5
    NUM_JOINTS = 24
    EPOCHS = 3
    
    model = KinematicFusionModel(num_joints=NUM_JOINTS, imu_sensors=NUM_SENSORS).to(device)
    model.set_stateful(False) # 学習時はBatch・Sequence全体を一括処理
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = KinematicLoss()
    
    print("--- Training Started ---")
    model.train()
    for epoch in range(EPOCHS):
        # ダミーデータの生成
        # foot_pressure: (B, Seq, Channels, H, W) ※入力要件に合わせてチャンネルの次元を変更
        dummy_foot = torch.rand((BATCH_SIZE, SEQ_LEN, 2, H, W)).to(device)
        # imu_data: (B, Seq, N_sensors, 6)
        dummy_imu = torch.rand((BATCH_SIZE, SEQ_LEN, NUM_SENSORS, 6)).to(device)
        
        # Target Quaternion: 正規化済みのダミーデータ
        target_quat = torch.rand((BATCH_SIZE, SEQ_LEN, NUM_JOINTS, 4)).to(device)
        target_quat = F.normalize(target_quat, p=2, dim=-1)
        
        optimizer.zero_grad()
        
        # 順伝播
        outputs = model(dummy_foot, dummy_imu)
        
        # 損失計算
        loss = criterion(outputs, target_quat)
        
        # 逆伝播・パラメータ更新
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
        
    # 重みの保存処理
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"kinematic_model_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete ---")
    print(f"Model weights saved to {save_path}")
    return save_path

def inference_realtime_dummy(weight_path=None):
    """1 Euro Filterを統合したリアルタイム推論のシミュレーション"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    H, W = 32, 32
    NUM_SENSORS = 5
    NUM_JOINTS = 24
    
    model = KinematicFusionModel(num_joints=NUM_JOINTS, imu_sensors=NUM_SENSORS).to(device)
    
    # --- スクリプト内でロードする重みファイルを指定 ---
    # `weight_path` が渡されない場合は、ここに直接ファイル名を手動で指定します
    TARGET_WEIGHT_PATH = weight_path if weight_path else "weights/kinematic_model_YYYYMMDD_HHMMSS.pth"
    
    if os.path.exists(TARGET_WEIGHT_PATH):
        try:
            model.load_state_dict(torch.load(TARGET_WEIGHT_PATH, map_location=device))
            print(f"Loaded weights from {TARGET_WEIGHT_PATH}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print(f"Warning: Weight file '{TARGET_WEIGHT_PATH}' not found, using initialized weights.")

    model.eval()
    model.set_stateful(True) # ステートフルに設定（前フレームの隠れ状態を保持）
    
    euro_filter = OneEuroFilter(mincutoff=1.0, beta=0.01, dcutoff=1.0)
    
    print("\n--- Realtime Inference Simulation Started ---")
    with torch.no_grad():
        for i in range(10): # 10フレームシミュレーション
            start_time = time.time()
            
            # 1フレームごとのストリームデータ（Batch=1, Seq=1）
            # 実際の環境では補間 (Interpolation) 等で同期済のデータが入る想定
            st_foot = torch.rand((1, 1, 2, H, W)).to(device)
            st_imu = torch.rand((1, 1, NUM_SENSORS, 6)).to(device)
            
            out_quat = model(st_foot, st_imu)
            out_quat_filtered = euro_filter(start_time, out_quat)
            
            elapsed = (time.time() - start_time) * 1000 # ms単位
            print(f"Frame {i+1}: Latency = {elapsed:.2f} ms | Output Shape: {out_quat_filtered.shape}")

if __name__ == "__main__":
    saved_model_path = train_dummy()
    inference_realtime_dummy(saved_model_path)
