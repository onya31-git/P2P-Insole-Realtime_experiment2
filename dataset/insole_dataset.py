import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class KinematicDataset(Dataset):
    def __init__(self, insole_dir='data/insole', skeleton_dir='data/skeleton', seq_len=50, num_joints=24):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints

        # 1. IDを見つける (.csvファイルの中から `_skeleton.csv` の前の部分を抽出)
        skeleton_files = glob.glob(os.path.join(skeleton_dir, '*_skeleton.csv'))
        
        # もし `*_skeleton.csv` が見つからない場合（レガシーな `skeleton.csv` など）へのフォールバック対応
        if len(skeleton_files) == 0:
            legacy_skel = os.path.join(skeleton_dir, 'skeleton.csv')
            if os.path.exists(legacy_skel):
                dataset_ids = [None] # レガシーID
            else:
                dataset_ids = []
        else:
            dataset_ids = [os.path.basename(f).replace('_skeleton.csv', '') for f in skeleton_files]
            dataset_ids.sort() # D1, D2 と順に並ぶようにする
        
        all_foot = []
        all_imu = []
        all_pos = []
        self.valid_indices = []
        
        current_offset = 0 # 結合時の現在のフレーム先頭インデックス
        
        for did in dataset_ids:
            if did is None:
                in_l = os.path.join(insole_dir, 'Insole_l.csv')
                in_r = os.path.join(insole_dir, 'Insole_r.csv')
                skel = os.path.join(skeleton_dir, 'skeleton.csv')
            else:
                in_l = os.path.join(insole_dir, f'{did}_Insole_l.csv')
                in_r = os.path.join(insole_dir, f'{did}_Insole_r.csv')
                skel = os.path.join(skeleton_dir, f'{did}_skeleton.csv')
            
            if not (os.path.exists(in_l) and os.path.exists(in_r) and os.path.exists(skel)):
                print(f"Warning: Data files for ID '{did}' are incomplete. Skipping.")
                continue
                
            # 足圧・IMUと骨格データの読み込み
            df_l = pd.read_csv(in_l, skiprows=1)
            df_r = pd.read_csv(in_r, skiprows=1)
            df_skel = pd.read_csv(skel, skiprows=5, header=None, low_memory=False)

            # 2. 足圧とIMUのパース
            foot_l = df_l.iloc[:, 1:36].values.astype(np.float32)
            foot_r = df_r.iloc[:, 1:36].values.astype(np.float32)
            
            # Mag (36:39) を除外し、Gyro (39:42) と Acc (42:45) のみを抽出
            imu_l = df_l.iloc[:, 39:45].values.astype(np.float32)
            imu_r = df_r.iloc[:, 39:45].values.astype(np.float32)

            for data in [imu_l, imu_r]:
                data[:, 0:3] /= 500.0   # Gyro
                data[:, 3:6] /= 8.0     # Acc

            foot_l = foot_l / 2000.0
            foot_r = foot_r / 2000.0

            TARGET_POS_COL_STARTS = [
                6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 
                90, 96, 102, 108, 114, 120, 126, 132, 138, 144
            ]
            
            target_cols = []
            for s in TARGET_POS_COL_STARTS:
                target_cols.extend([s, s + 1, s + 2])

            skel_positions = df_skel.iloc[:, target_cols].values.astype(np.float32)

            min_length = min(len(foot_l), len(foot_r), len(skel_positions))
            
            # --- 個別データセットのチャンクの作成 ---
            c_foot_l = foot_l[:min_length]
            c_foot_r = foot_r[:min_length]
            c_imu_l = imu_l[:min_length]
            c_imu_r = imu_r[:min_length]
            c_skel_pos = skel_positions[:min_length]
            
            c_foot_data = np.concatenate([c_foot_l, c_foot_r], axis=-1)
            c_imu_data = np.stack([c_imu_l, c_imu_r], axis=1)
            c_pos_data = c_skel_pos.reshape(min_length, self.num_joints, 3) / 1000.0
            
            # Root-Relative 化 (Spine, idx=21)
            spine_pos = c_pos_data[:, 21:22, :].copy()
            c_pos_data = c_pos_data - spine_pos
            
            all_foot.append(c_foot_data)
            all_imu.append(c_imu_data)
            all_pos.append(c_pos_data)
            
            # 3. データの境界をまたがない「有効な開始インデックス」を計算
            # min_length=100, seq_len=50 ならば、チャンク内の valid な start は 0~50。
            num_valid = max(0, min_length - self.seq_len + 1)
            for i in range(num_valid):
                self.valid_indices.append(current_offset + i)
                
            current_offset += min_length
            _disp_id = did if did is not None else "legacy"
            print(f"[Dataset] ID: {_disp_id} -> {min_length} frames loaded.")
            
        if len(all_foot) == 0:
            raise ValueError("No valid data found in the specified directories.")
            
        # 4. 全てのデータを結合
        self.foot_data = torch.tensor(np.concatenate(all_foot, axis=0), dtype=torch.float32)
        self.imu_data = torch.tensor(np.concatenate(all_imu, axis=0), dtype=torch.float32)
        self.pos_data = torch.tensor(np.concatenate(all_pos, axis=0), dtype=torch.float32)
        
        print(f"[Dataset] Combined -> foot: {self.foot_data.shape}, imu: {self.imu_data.shape}, pos: {self.pos_data.shape}")
        print(f"[Dataset] Valid sequences (ignoring boundaries) -> {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        foot = self.foot_data[real_idx : real_idx + self.seq_len]
        imu = self.imu_data[real_idx : real_idx + self.seq_len]
        pos = self.pos_data[real_idx : real_idx + self.seq_len]
        return foot, imu, pos

if __name__ == "__main__":
    # 単体動作テスト
    dataset = KinematicDataset(seq_len=50)
    if len(dataset) > 0:
        f, i, p = dataset[0]
        print(f"Sample 0 -> foot: {f.shape}, imu: {i.shape}, pos: {p.shape}")
