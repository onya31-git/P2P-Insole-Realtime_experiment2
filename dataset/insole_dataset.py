import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

class KinematicDataset(Dataset):
    def __init__(self, insole_dir='data/insole', skeleton_dir='data/skeleton', seq_len=20, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints
        
        # 1. データの読み込み
        insole_l_path = os.path.join(insole_dir, 'Insole_l.csv')
        insole_r_path = os.path.join(insole_dir, 'Insole_r.csv')
        skeleton_path = os.path.join(skeleton_dir, 'skeleton.csv')
        
        # 足圧・IMUデータの読み込み (先頭の `// DN:` 等をスキップ)
        df_l = pd.read_csv(insole_l_path, skiprows=1)
        df_r = pd.read_csv(insole_r_path, skiprows=1)
        
        # 骨格データの読み込み (6行目から実際の数値データ)
        df_skel = pd.read_csv(skeleton_path, skiprows=5, header=None)
        
        # 2. 足圧とIMUのパース
        # columns: Timestamp, P1..P35, Mag(3), Gyro(3), Acc(3)
        # P1..P35: index 1 to 35
        # IMU: index 36 to 44
        foot_l = df_l.iloc[:, 1:36].values
        foot_r = df_r.iloc[:, 1:36].values
        
        imu_l = df_l.iloc[:, 36:45].values
        imu_r = df_r.iloc[:, 36:45].values
        
        # 3. 骨格データ(角度)の抽出
        # skeleton.csv は多くの列を持つが、今回は "Angle" 列 (Euler角 X,Y,Z) を抽出
        # ヘッダー構成（2行目にラベル、3行目にX,Y,Z）をハードコードせず、特定の規則に従って抽出するか、
        # 簡易的に全てのX,Y,Zのセットを取得して先頭24関節分を利用する。
        # 今回はダミーから実データへの移行として、数値データ行の全列から3列単位(Euler角)で取り得るものをQuaternionに変換
        # 実際の skeleton はフォーマットが複雑なため、ここでは Angles とみられる列を手動マッピングする簡略化を行う
        # ※ 実運用ではジョイント名の明確なマッピングを推奨
        
        # csvの2行目から "Angles" を含むセルの列インデックスを特定する処理（通常はX,Y,Zの3列が対応）
        skel_header_joints = pd.read_csv(skeleton_path, skiprows=2, nrows=1, header=None).values[0]
        skel_header_xyz = pd.read_csv(skeleton_path, skiprows=3, nrows=1, header=None).values[0]
        
        angle_cols_indices = []
        for i, val in enumerate(skel_header_joints):
            if isinstance(val, str) and 'Angles' in val:
                # i 列目が X, i+1 が Y, i+2 が Z と仮定
                angle_cols_indices.extend([i, i+1, i+2])
                
        # 必要な24ジョイント分 (24 * 3 = 72 columns) を確保
        target_cols = angle_cols_indices[:self.num_joints * 3]
        
        # 不足分がある場合はパディング
        while len(target_cols) < self.num_joints * 3:
            target_cols.append(target_cols[-1] if target_cols else 0)
            
        skel_angles_euler = df_skel.iloc[:, target_cols].values
        
        # 4. 同期とパディング/トリミング処理
        # それぞれのレートが異なる（または長さが違う）場合、一番短い長さに切り詰める (簡易同期)
        min_length = min(len(foot_l), len(foot_r), len(skel_angles_euler))
        
        foot_l = foot_l[:min_length]
        foot_r = foot_r[:min_length]
        imu_l = imu_l[:min_length]
        imu_r = imu_r[:min_length]
        skel_angles_euler = skel_angles_euler[:min_length]
        
        # 結合
        self.foot_data = np.concatenate([foot_l, foot_r], axis=-1)  # shape: (N, 70)
        # imu_data: (N, 2, 9) = 2 sensors, 9 channels each
        self.imu_data = np.stack([imu_l, imu_r], axis=1) # shape: (N, 2, 9)
        
        # 骨格（Euler -> Quaternion）
        # shape: (N, num_joints, 3) 
        skel_angles_euler = skel_angles_euler.reshape(min_length, self.num_joints, 3)
        quat_data = np.zeros((min_length, self.num_joints, 4))
        for j in range(self.num_joints):
            # Euler角をQuaternionに変換 (Scipy Rotation)
            # skeleton.csvの回転順序は一般的に XYZ または ZYX など。ここでは XYZ とする
            r = R.from_euler('xyz', skel_angles_euler[:, j, :], degrees=True)
            quat_data[:, j, :] = r.as_quat() # x, y, z, w
        self.quat_data = quat_data
        
        # テンソル化
        self.foot_data = torch.tensor(self.foot_data, dtype=torch.float32)
        self.imu_data = torch.tensor(self.imu_data, dtype=torch.float32)
        self.quat_data = torch.tensor(self.quat_data, dtype=torch.float32)

    def __len__(self):
        # seq_len分の窓をとるため
        return max(0, len(self.foot_data) - self.seq_len + 1)

    def __getitem__(self, idx):
        foot = self.foot_data[idx: idx + self.seq_len]      # (Seq, 70)
        imu = self.imu_data[idx: idx + self.seq_len]        # (Seq, 2, 9)
        quat = self.quat_data[idx: idx + self.seq_len]      # (Seq, 24, 4)
        return foot, imu, quat
