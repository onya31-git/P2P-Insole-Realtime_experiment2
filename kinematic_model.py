import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ==========================================
# 1. 足裏圧力分布 (2D-CNN) Encoder
# ==========================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

class FootPressureEncoder(nn.Module):
    def __init__(self, in_channels=2, out_features=128):
        super().__init__()
        # PyTorchのConv2dの入力は (Batch, Channels, H, W)
        self.net = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv2d(64, out_features, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )
        self.out_features = out_features

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.net(x)
        x = x.view(x.size(0), -1) # Flatten -> (B, out_features)
        return x

# ==========================================
# 2. IMU Encoder (1D-CNN with Causal Padding)
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (Batch, Channels, SeqLens)
        x = F.pad(x, (self.pad, 0)) # 未来のパディングは0にして因果性を維持
        x = self.conv(x)
        return self.relu(x)

class IMUEncoder(nn.Module):
    def __init__(self, in_channels=6, num_sensors=5, out_features=128):
        super().__init__()
        self.in_channels = in_channels * num_sensors
        self.net = nn.Sequential(
            CausalConv1d(self.in_channels, 64, kernel_size=3),
            CausalConv1d(64, 128, kernel_size=3, dilation=2),
            CausalConv1d(128, out_features, kernel_size=3, dilation=4)
        )
        self.out_features = out_features

    def forward(self, x):
        # x: (B, in_channels, SeqLens)
        x = self.net(x)
        return x # (B, out_features, SeqLens)

# ==========================================
# 3. 時系列統合層 (Lightweight LSTM)
# ==========================================
class StatefulLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = None
        self.is_stateful = False

    def set_stateful(self, stateful: bool):
        self.is_stateful = stateful
        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def forward(self, x):
        # x: (B, Seq, input_size)
        if self.is_stateful:
            out, self.hidden_state = self.lstm(x, self.hidden_state)
            # 次のイテレーションで計算グラフを切断してメモリリークを防ぐ
            self.hidden_state = (self.hidden_state[0].detach(), self.hidden_state[1].detach())
        else:
            out, _ = self.lstm(x)
        return out

# ==========================================
# ハイブリッドモデル本体
# ==========================================
class KinematicFusionModel(nn.Module):
    def __init__(self, foot_channels=2, imu_sensors=5, imu_channels=6, 
                 foot_out=128, imu_out=128, lstm_hidden=256, num_joints=24, lstm_layers=1):
        super().__init__()
        self.foot_encoder = FootPressureEncoder(in_channels=foot_channels, out_features=foot_out)
        self.imu_encoder = IMUEncoder(in_channels=imu_channels, num_sensors=imu_sensors, out_features=imu_out)
        
        lstm_input_size = foot_out + imu_out
        self.fusion_lstm = StatefulLSTM(input_size=lstm_input_size, hidden_size=lstm_hidden, num_layers=lstm_layers)
        
        self.fc_out = nn.Linear(lstm_hidden, num_joints * 4)
        self.num_joints = num_joints

    def set_stateful(self, stateful: bool):
        """リアルタイム推論モード切替"""
        self.fusion_lstm.set_stateful(stateful)

    def forward(self, foot_pressure, imu_data):
        """
        foot_pressure: (B, Seq, 2, H, W)
        imu_data: (B, Seq, N, 6)
        """
        B, Seq, C, H, W = foot_pressure.size()
        
        # 足圧データのエンコード (時間軸もまとめて処理し、後で戻す)
        f_in = foot_pressure.view(B * Seq, C, H, W)
        foot_feat = self.foot_encoder(f_in) 
        foot_feat = foot_feat.view(B, Seq, -1) # (B, Seq, foot_out)
        
        # IMUデータのエンコード (Conv1Dに入力するため、[B, C, Seq]の形に変形)
        _, _, N, D = imu_data.size()
        i_in = imu_data.view(B, Seq, N * D).transpose(1, 2) # (B, N*D, Seq)
        imu_feat = self.imu_encoder(i_in) # (B, imu_out, Seq)
        imu_feat = imu_feat.transpose(1, 2) # (B, Seq, imu_out)
        
        # 特徴量の結合
        fusion_feat = torch.cat((foot_feat, imu_feat), dim=-1) # (B, Seq, foot_out+imu_out)
        
        # LSTMによる時系列統合
        lstm_out = self.fusion_lstm(fusion_feat) # (B, Seq, lstm_hidden)
        
        # ポーズ出力への変換と正規化 (クォータニオンとして)
        out = self.fc_out(lstm_out) 
        out = out.view(B, Seq, self.num_joints, 4)
        out = F.normalize(out, p=2, dim=-1) 
        return out

# ==========================================
# 4. ジッタ抑制 (1 Euro Filter)
# ==========================================
class OneEuroFilter:
    """1 Euro Filter for PyTorch Tensors. Applies low-pass filter to smooth signal."""
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = torch.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        t_e_tensor = torch.tensor(t_e, device=x.device, dtype=x.dtype)
        
        if t_e <= 0:
            return x
            
        a_d = self.smoothing_factor(t_e_tensor, self.dcutoff)
        dx = (x - self.x_prev) / t_e_tensor
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 動的なカットオフ周波数
        cutoff = self.mincutoff + self.beta * torch.abs(dx_hat)
        a = self.smoothing_factor(t_e_tensor, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

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

def inference_realtime_dummy():
    """1 Euro Filterを統合したリアルタイム推論のシミュレーション"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    H, W = 32, 32
    NUM_SENSORS = 5
    NUM_JOINTS = 24
    
    model = KinematicFusionModel(num_joints=NUM_JOINTS, imu_sensors=NUM_SENSORS).to(device)
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
    train_dummy()
    inference_realtime_dummy()
