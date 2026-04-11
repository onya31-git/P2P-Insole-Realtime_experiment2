import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class KinematicDataset(Dataset):
    def __init__(self, insole_dir='data/insole', skeleton_dir='data/skeleton', seq_len=20, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints

        # 1. データの読み込み
        insole_l_path = os.path.join(insole_dir, 'Insole_l.csv')
        insole_r_path = os.path.join(insole_dir, 'Insole_r.csv')
        skeleton_path = os.path.join(skeleton_dir, 'skeleton.csv')

        # 足圧・IMUデータの読み込み
        df_l = pd.read_csv(insole_l_path, skiprows=1)
        df_r = pd.read_csv(insole_r_path, skiprows=1)

        # 骨格データの読み込み (6行目から実際の数値データ)
        df_skel = pd.read_csv(skeleton_path, skiprows=5, header=None, low_memory=False)

        # 2. 足圧とIMUのパース
        # columns: Timestamp, P1..P35, Mag(3), Gyro(3), Acc(3)
        foot_l = df_l.iloc[:, 1:36].values.astype(np.float32)
        foot_r = df_r.iloc[:, 1:36].values.astype(np.float32)
        imu_l = df_l.iloc[:, 36:45].values.astype(np.float32)
        imu_r = df_r.iloc[:, 36:45].values.astype(np.float32)

        # 2a. IMUデータの正規化
        # Mag(地磁気) [-100, 100] -> [-1, 1]
        # Gyro(角速度) [-500, 500] deg/s -> [-1, 1]
        # Acc(加速度) [-8, 8] g -> [-1, 1]
        for data in [imu_l, imu_r]:
            data[:, 0:3] /= 100.0   # Mag正規化
            data[:, 3:6] /= 500.0   # Gyro正規化
            data[:, 6:9] /= 8.0     # Acc正規化

        # 2b. 足圧データの正規化（最大値2000を目安に0~1に）
        foot_l = foot_l / 2000.0
        foot_r = foot_r / 2000.0

        # 3. 骨格データのPositions列を正確に抽出
        # ★修正: 'Positions'(s付き)と'Position'(s無し)の両方が存在する
        # 調査結果に基づく24関節の列開始インデックス（各関節はX,Y,Zの3列）:
        # [LWrist, LElbow, LShoulder, RWrist, RElbow, RShoulder,
        #  LToe, LAnkle, LKnee, LHip, RToe, RAnkle, RKnee, RHip,
        #  RShoulder2, LClavicle, LHandEnd, LToesEnd, RClavicle, RHandEnd, RToesEnd,
        #  Spine, Spine1, Spine2]
        TARGET_POS_COL_STARTS = [
            6,    # LWristPositions
            12,   # LElbowPositions
            18,   # LShoulderPositions
            24,   # RWristPositions
            30,   # RElbowPositions
            36,   # RShoulderPositions
            42,   # LToePositions
            48,   # LAnklePositions
            54,   # LKneePositions
            60,   # LHipPositions
            66,   # RToePositions
            72,   # RAnklePositions
            78,   # RKneePositions
            84,   # RHipPositions
            90,   # RShoulderPosition  (s無し)
            96,   # LClaviclePosition  (s無し)
            102,  # LHandEndPosition   (s無し)
            108,  # LToesEndPosition   (s無し)
            114,  # RClaviclePosition  (s無し)
            120,  # RHandEndPosition   (s無し)
            126,  # RToesEndPosition   (s無し)
            132,  # SpinePosition      (s無し)
            138,  # Spine1Position     (s無し)
            144,  # Spine2Position     (s無し)
        ]
        assert len(TARGET_POS_COL_STARTS) == self.num_joints, \
            f"Column count {len(TARGET_POS_COL_STARTS)} != num_joints {self.num_joints}"

        target_cols = []
        for s in TARGET_POS_COL_STARTS:
            target_cols.extend([s, s + 1, s + 2])

        skel_positions = df_skel.iloc[:, target_cols].values.astype(np.float32)

        # 4. 同期処理
        # インソール(100Hz)とスケルトン(100fps)は同レートなので行番号で対応
        min_length = min(len(foot_l), len(foot_r), len(skel_positions))

        foot_l = foot_l[:min_length]
        foot_r = foot_r[:min_length]
        imu_l = imu_l[:min_length]
        imu_r = imu_r[:min_length]
        skel_positions = skel_positions[:min_length]

        # 5. 結合・テンソル化
        self.foot_data = np.concatenate([foot_l, foot_r], axis=-1)  # (N, 70)
        self.imu_data = np.stack([imu_l, imu_r], axis=1)            # (N, 2, 9)

        # 骨格座標: mm -> m 変換、shape: (N, num_joints, 3)
        self.pos_data = skel_positions.reshape(min_length, self.num_joints, 3) / 1000.0

        self.foot_data = torch.tensor(self.foot_data, dtype=torch.float32)
        self.imu_data = torch.tensor(self.imu_data, dtype=torch.float32)
        self.pos_data = torch.tensor(self.pos_data, dtype=torch.float32)

        print(f"[Dataset] {min_length} frames loaded. "
              f"foot={self.foot_data.shape}, imu={self.imu_data.shape}, pos={self.pos_data.shape}")

    def __len__(self):
        return max(0, len(self.foot_data) - self.seq_len + 1)

    def __getitem__(self, idx):
        foot = self.foot_data[idx: idx + self.seq_len]   # (Seq, 70)
        imu = self.imu_data[idx: idx + self.seq_len]     # (Seq, 2, 9)
        pos = self.pos_data[idx: idx + self.seq_len]     # (Seq, 24, 3)
        return foot, imu, pos
