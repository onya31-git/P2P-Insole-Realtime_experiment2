import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 足裏圧力分布 (2D-CNN) Encoder
# ==========================================


class FootPressureEncoder(nn.Module):
    def __init__(self, in_features=70, out_features=128):
        super().__init__()
        # 左右の足それぞれ35個の圧力点 -> 計70個などの1Dベクトル入力を想定
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )
        self.out_features = out_features

    def forward(self, x):
        # x: (B*Seq, in_features)
        x = self.net(x)
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
