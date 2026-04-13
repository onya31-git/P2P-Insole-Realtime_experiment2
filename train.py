import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random

from models.model import KinematicFusionModel
from processor.filter import OneEuroFilter

# ==========================================
# 骨の接続ペア定義（CSVのPositions列順序）
# ==========================================
# 各タプルは (関節A, 関節B) — モデル出力の24関節インデックス
BONE_PAIRS = [
    (0,  1),   # LWrist  - LElbow
    (1,  2),   # LElbow  - LShoulder
    (3,  4),   # RWrist  - RElbow
    (4,  5),   # RElbow  - RShoulder
    (6,  7),   # LToe    - LAnkle
    (7,  8),   # LAnkle  - LKnee
    (8,  9),   # LKnee   - LHip
    (10, 11),  # RToe    - RAnkle
    (11, 12),  # RAnkle  - RKnee
    (12, 13),  # RKnee   - RHip
    (9,  21),  # LHip    - Spine
    (13, 21),  # RHip    - Spine
    (21, 22),  # Spine   - Spine1
    (22, 23),  # Spine1  - Spine2
    (2,  15),  # LShoulder - LClavicle
    (5,  18),  # RShoulder - RClavicle
    (0,  16),  # LWrist  - LHandEnd
    (3,  19),  # RWrist  - RHandEnd
]


# ==========================================
# 損失関数 (Weighted SmoothL1 + 骨長制約 + 速度 + 加速度)
# ==========================================
class KinematicLoss(nn.Module):
    def __init__(self, lambda_bone=0.3, lambda_vel=0.1, lambda_acc=0.5):
        super().__init__()
        self.lambda_bone = lambda_bone
        self.lambda_vel  = lambda_vel
        self.lambda_acc  = lambda_acc

        # 関節ごとの重み (計24関節)
        # 6,10,17,20 (足先) を 2.5倍、7,8,11,12 (足首/膝) を 1.5倍に設定
        weights = torch.ones(24)
        weights[[6, 10, 17, 20]] = 2.5  # Toes
        weights[[7, 8, 11, 12]]  = 1.5  # Ankle / Knee
        self.register_buffer('joint_weights', weights.view(1, 1, 24, 1))

        # 安定させたい中心軸の関節 (Hip, Spine)
        self.stable_joints = [9, 13, 21, 22, 23]

    def forward(self, pred_pos, target_pos):
        """
        pred_pos, target_pos: (B, Seq, 24, 3)
        """
        # 1. 重み付き位置ロス (SmoothL1)
        # beta=0.05: 5cm以下の誤差を二乗で、それ以上を線形で扱う
        loss_pos = F.smooth_l1_loss(pred_pos, target_pos, reduction='none', beta=0.05)
        loss_pos = (loss_pos * self.joint_weights).mean()

        # 2. 骨の長さ一定制約 (MAE)
        bone_loss = 0.0
        for i, j in BONE_PAIRS:
            pred_len   = torch.norm(pred_pos[..., i, :] - pred_pos[..., j, :], dim=-1)
            target_len = torch.norm(target_pos[..., i, :] - target_pos[..., j, :], dim=-1)
            bone_loss += F.l1_loss(pred_len, target_len)
        bone_loss /= len(BONE_PAIRS)

        # 3. 速度一貫性ロス (MAE)
        if pred_pos.size(1) > 1:
            pred_vel   = pred_pos[:, 1:, :, :] - pred_pos[:, :-1, :, :]
            target_vel = target_pos[:, 1:, :, :] - target_pos[:, :-1, :, :]
            vel_loss   = F.l1_loss(pred_vel, target_vel)
        else:
            vel_loss = torch.tensor(0.0, device=pred_pos.device)

        # 4. 加速度（安定性）ペナルティ
        # 特定の関節 (Spine/Hip) の急激な変化を抑制
        if pred_pos.size(1) > 2:
            # 加速度 = (x_{t+1} - x_t) - (x_t - x_{t-1}) = x_{t+1} - 2x_t + x_{t-1}
            acc = pred_pos[:, 2:, self.stable_joints, :] - 2*pred_pos[:, 1:-1, self.stable_joints, :] + pred_pos[:, :-2, self.stable_joints, :]
            acc_loss = torch.norm(acc, dim=-1).mean()
        else:
            acc_loss = torch.tensor(0.0, device=pred_pos.device)

        return loss_pos + self.lambda_bone * bone_loss + self.lambda_vel * vel_loss + self.lambda_acc * acc_loss


from dataset.insole_dataset import KinematicDataset
from torch.utils.data import DataLoader


def train():
    """実データを用いたトレーニングループ（GPU対応・大型モデル・骨長制約・SEQ_LEN=50）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ==========================================
    # パラメータ設定
    # ==========================================
    BATCH_SIZE = 16
    SEQ_LEN    = 50
    NUM_JOINTS = 24
    EPOCHS     = 50

    # モデル: lstmを256に戻して過学習を防止（foot/imuエンコーダは大きいまま）
    model = KinematicFusionModel(
        foot_features=70, imu_sensors=2, imu_channels=9, num_joints=NUM_JOINTS,
        foot_out=256, imu_out=256, lstm_hidden=512, lstm_layers=2
    ).to(device)
    model.set_stateful(False)

    # 最適化とスケジューラー（コサインアニーリング）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    criterion = KinematicLoss(lambda_bone=0.3, lambda_vel=0.1, lambda_acc=0.5).to(device)

    print("--- Loading Dataset ---")
    dataset    = KinematicDataset(
        insole_dir='data/insole', skeleton_dir='data/skeleton',
        seq_len=SEQ_LEN, num_joints=NUM_JOINTS
    )
    use_pin_memory = (device.type == 'cuda')
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=use_pin_memory
    )

    print(f"Dataset Size: {len(dataset)} sequences")
    print(f"Batch Size: {BATCH_SIZE}  |  Seq Len: {SEQ_LEN}  |  Epochs: {EPOCHS}")
    print("--- Training Started ---")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for foot_pressure, imu_data, target_pos in dataloader:
            foot_pressure = foot_pressure.to(device, non_blocking=True)
            imu_data      = imu_data.to(device, non_blocking=True)
            target_pos    = target_pos.to(device, non_blocking=True)

            # ★ データ拡張: 軽微なガウスノイズ
            # （両足データが揃うためDrop-Foot Augmentationは使用しない）
            if random.random() < 0.5:
                foot_pressure = foot_pressure + torch.randn_like(foot_pressure) * 0.01
                imu_data = imu_data + torch.randn_like(imu_data) * 0.005

            optimizer.zero_grad()
            outputs = model(foot_pressure, imu_data)
            loss    = criterion(outputs, target_pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 勾配クリッピング
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}], Loss: {avg_loss:.5f}, LR: {current_lr:.6f}")
        scheduler.step()

    # 重みの保存
    save_dir  = "weights"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"kinematic_model_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete ---")
    print(f"Model weights saved to {save_path}")
    return save_path


def inference_realtime_dummy(weight_path=None):
    """動作確認用リアルタイム推論シミュレーション"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_JOINTS = 24
    model = KinematicFusionModel(num_joints=NUM_JOINTS).to(device)

    import glob
    weight_files    = sorted(glob.glob("weights/*.pth"))
    target_path     = weight_path if weight_path else (weight_files[-1] if weight_files else "")
    if target_path and os.path.exists(target_path):
        model.load_state_dict(torch.load(target_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {target_path}")

    model.eval()
    model.set_stateful(True)

    # 施策D: OneEuroFilter パラメータ
    # mincutoff=3.0: 動きをより通過させる（0.5は過剰スムーシング）
    # beta=0.05: 速い動きにも素早く追従
    euro_filter = OneEuroFilter(mincutoff=3.0, beta=0.05, dcutoff=1.0)

    print("\n--- Realtime Inference Simulation ---")
    with torch.no_grad():
        for i in range(10):
            start_time = time.time()
            st_foot = torch.rand((1, 1, 70)).to(device)
            st_imu  = torch.rand((1, 1, 2, 9)).to(device)
            out     = model(st_foot, st_imu)
            out_f   = euro_filter(start_time, out)
            elapsed = (time.time() - start_time) * 1000
            print(f"Frame {i+1}: Latency = {elapsed:.2f} ms | Shape: {out_f.shape}")


if __name__ == "__main__":
    saved_path = train()
    inference_realtime_dummy(saved_path)
