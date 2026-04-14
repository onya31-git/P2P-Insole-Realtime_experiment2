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
# 損失関数
# ==========================================
class KinematicLoss(nn.Module):
    def __init__(self, lambda_bone=0.5):
        super().__init__()
        self.mse_loss    = nn.MSELoss()
        self.lambda_bone = lambda_bone

    def forward(self, pred_pos, target_pos):
        """
        pred_pos, target_pos: (B, Seq, 24, 3)
        """
        # 位置座標MSE
        loss_pos = self.mse_loss(pred_pos, target_pos)

        # 骨の長さ一定制約 Loss
        # 予測と正解の骨長が一致するように学習する
        bone_loss = 0.0
        for i, j in BONE_PAIRS:
            pred_len   = torch.norm(pred_pos[..., i, :]   - pred_pos[..., j, :],   dim=-1)  # (B, Seq)
            target_len = torch.norm(target_pos[..., i, :] - target_pos[..., j, :], dim=-1)  # (B, Seq)
            bone_loss += F.mse_loss(pred_len, target_len)
        bone_loss /= len(BONE_PAIRS)

        return loss_pos + self.lambda_bone * bone_loss


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
    BATCH_SIZE = 16   # GPUを活用するためバッチサイズを増加
    SEQ_LEN    = 50   # 施策B: シーケンス長を50フレームに拡大
    NUM_JOINTS = 24
    EPOCHS     = 50   # より多くのエポックで安定した収束を狙う

    # モデル（デフォルトで foot_out=256, imu_out=256, lstm_hidden=512, lstm_layers=2）
    model = KinematicFusionModel(
        foot_features=70, imu_sensors=2, imu_channels=9, num_joints=NUM_JOINTS
    ).to(device)
    model.set_stateful(False)

    # 最適化とスケジューラー（コサインアニーリング）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    criterion = KinematicLoss(lambda_bone=0.5)

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

    # 施策D: OneEuroFilter パラメータ最適化
    # mincutoff=0.5にすることでスムーシングを強化、beta=0.003で速い動きへの追従を保つ
    euro_filter = OneEuroFilter(mincutoff=0.5, beta=0.003, dcutoff=1.0)

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
