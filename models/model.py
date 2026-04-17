import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FootPressureEncoder, IMUEncoder, StatefulLSTM

class KinematicFusionModel(nn.Module):
    def __init__(self, foot_features=70, imu_sensors=2, imu_channels=6,
                 foot_out=256, imu_out=256, lstm_hidden=512, num_joints=24, lstm_layers=2):
        super().__init__()
        self.foot_encoder = FootPressureEncoder(in_features=foot_features, out_features=foot_out)
        self.imu_encoder = IMUEncoder(in_channels=imu_channels, num_sensors=imu_sensors, out_features=imu_out)
        
        lstm_input_size = foot_out + imu_out
        self.fusion_lstm = StatefulLSTM(input_size=lstm_input_size, hidden_size=lstm_hidden, num_layers=lstm_layers)
        
        self.fc_out = nn.Linear(lstm_hidden, num_joints * 3)
        self.num_joints = num_joints

    def set_stateful(self, stateful: bool):
        """リアルタイム推論モード切替"""
        self.fusion_lstm.set_stateful(stateful)

    def forward(self, foot_pressure, imu_data):
        # foot_pressure: (B, Seq, F)
        B, Seq, F_dim = foot_pressure.size()
        
        f_in = foot_pressure.view(B * Seq, F_dim)
        foot_feat = self.foot_encoder(f_in) 
        foot_feat = foot_feat.view(B, Seq, -1)
        
        _, _, N, D = imu_data.size()
        i_in = imu_data.view(B, Seq, N * D).transpose(1, 2)
        imu_feat = self.imu_encoder(i_in)
        imu_feat = imu_feat.transpose(1, 2)
        
        fusion_feat = torch.cat((foot_feat, imu_feat), dim=-1)
        lstm_out = self.fusion_lstm(fusion_feat)
        
        out = self.fc_out(lstm_out) 
        out = out.view(B, Seq, self.num_joints, 3)
        return out


class HierarchicalKinematicFusionModel(nn.Module):
    """
    下半身の関節位置を先に推定し、その特徴を用いて上半身の関節位置を推定する階層型モデル。
    """
    def __init__(self, foot_features=70, imu_sensors=2, imu_channels=6,
                 foot_out=256, imu_out=256, lstm_hidden=512, num_joints=24, lstm_layers=2):
        super().__init__()
        self.foot_encoder = FootPressureEncoder(in_features=foot_features, out_features=foot_out)
        self.imu_encoder = IMUEncoder(in_channels=imu_channels, num_sensors=imu_sensors, out_features=imu_out)
        
        lstm_input_size = foot_out + imu_out
        self.fusion_lstm = StatefulLSTM(input_size=lstm_input_size, hidden_size=lstm_hidden, num_layers=lstm_layers)
        
        # 下半身：Toe, Ankle, Knee, Hip, ToesEnd, Spine (11箇所)
        self.lower_indices = [6, 7, 8, 9, 10, 11, 12, 13, 17, 20, 21]
        # 上半身：残りの13箇所
        self.upper_indices = [i for i in range(num_joints) if i not in self.lower_indices]
        
        self.num_lower = len(self.lower_indices)
        self.num_upper = len(self.upper_indices)
        
        # 下半身の推定
        self.fc_lower = nn.Linear(lstm_hidden, self.num_lower * 3)
        
        # 上半身の推定（LSTM特徴量 + 下半身の推定結果を利用）
        self.fc_upper = nn.Linear(lstm_hidden + self.num_lower * 3, self.num_upper * 3)
        
        self.num_joints = num_joints

    def set_stateful(self, stateful: bool):
        """リアルタイム推論モード切替"""
        self.fusion_lstm.set_stateful(stateful)

    def forward(self, foot_pressure, imu_data):
        # foot_pressure: (B, Seq, F)
        B, Seq, F_dim = foot_pressure.size()
        
        f_in = foot_pressure.view(B * Seq, F_dim)
        foot_feat = self.foot_encoder(f_in) 
        foot_feat = foot_feat.view(B, Seq, -1)
        
        _, _, N, D = imu_data.size()
        i_in = imu_data.view(B, Seq, N * D).transpose(1, 2)
        imu_feat = self.imu_encoder(i_in)
        imu_feat = imu_feat.transpose(1, 2)
        
        fusion_feat = torch.cat((foot_feat, imu_feat), dim=-1)
        lstm_out = self.fusion_lstm(fusion_feat)
        
        # 1. 下半身の推定
        out_lower = self.fc_lower(lstm_out) # (B, Seq, num_lower * 3)
        
        # 2. 下半身の推定結果を加味して上半身の推定
        upper_input = torch.cat([lstm_out, out_lower], dim=-1)
        out_upper = self.fc_upper(upper_input) # (B, Seq, num_upper * 3)
        
        # 3. 24関節の出力フォーマット (B, Seq, 24, 3) に結合
        out = torch.zeros(B, Seq, self.num_joints, 3, device=lstm_out.device)
        
        out_lower_view = out_lower.view(B, Seq, self.num_lower, 3)
        out_upper_view = out_upper.view(B, Seq, self.num_upper, 3)
        
        out[:, :, self.lower_indices, :] = out_lower_view
        out[:, :, self.upper_indices, :] = out_upper_view
        
        return out
