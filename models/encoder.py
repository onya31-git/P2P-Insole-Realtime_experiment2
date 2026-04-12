import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 足裏圧力 Encoder（大型化）
# ==========================================
class FootPressureEncoder(nn.Module):
    def __init__(self, in_features=70, out_features=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )
        self.out_features = out_features

    def forward(self, x):
        # x: (B*Seq, in_features)
        return self.net(x)


# ==========================================
# 2. IMU Encoder（大型化・Causal CNN）
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act  = nn.GELU()

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class IMUEncoder(nn.Module):
    def __init__(self, in_channels=9, num_sensors=2, out_features=256):
        super().__init__()
        self.in_channels = in_channels * num_sensors
        self.net = nn.Sequential(
            CausalConv1d(self.in_channels, 128, kernel_size=3),
            CausalConv1d(128, 256, kernel_size=3, dilation=2),
            CausalConv1d(256, out_features, kernel_size=3, dilation=4),
        )
        self.out_features = out_features

    def forward(self, x):
        # x: (B, in_channels, SeqLen)
        return self.net(x)  # (B, out_features, SeqLen)


# ==========================================
# 3. ステートフル LSTM（2層・大型化）
# ==========================================
class StatefulLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0.0)
        self.hidden_state = None
        self.is_stateful  = False

    def set_stateful(self, stateful: bool):
        self.is_stateful  = stateful
        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def forward(self, x):
        if self.is_stateful:
            out, self.hidden_state = self.lstm(x, self.hidden_state)
            self.hidden_state = (
                self.hidden_state[0].detach(),
                self.hidden_state[1].detach(),
            )
        else:
            out, _ = self.lstm(x)
        return out
