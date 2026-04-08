import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FootPressureEncoder, IMUEncoder, StatefulLSTM

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
        B, Seq, C, H, W = foot_pressure.size()
        
        f_in = foot_pressure.view(B * Seq, C, H, W)
        foot_feat = self.foot_encoder(f_in) 
        foot_feat = foot_feat.view(B, Seq, -1)
        
        _, _, N, D = imu_data.size()
        i_in = imu_data.view(B, Seq, N * D).transpose(1, 2)
        imu_feat = self.imu_encoder(i_in)
        imu_feat = imu_feat.transpose(1, 2)
        
        fusion_feat = torch.cat((foot_feat, imu_feat), dim=-1)
        lstm_out = self.fusion_lstm(fusion_feat)
        
        out = self.fc_out(lstm_out) 
        out = out.view(B, Seq, self.num_joints, 4)
        out = F.normalize(out, p=2, dim=-1) 
        return out
