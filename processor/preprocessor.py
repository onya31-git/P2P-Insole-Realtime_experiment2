import json
import torch

# 学習時と同じ正規化スケールを定数として定義
FOOT_PRESSURE_SCALE = 2000.0   # 学習時: foot / 2000.0
IMU_MAG_SCALE = 100.0           # Mag: [-100, 100] -> [-1, 1]
IMU_GYRO_SCALE = 500.0          # Gyro: [-500, 500] -> [-1, 1]
IMU_ACC_SCALE = 8.0             # Acc: [-8, 8] -> [-1, 1]


def preprocess_foot_pressure(p_list, device):
    """
    35点の圧力データを (1, 1, 70) のテンソルに変換します。
    学習と同じスケールに正規化します。
    現状のストリームは片足（左足と仮定）のみのため、右足分は0でパディングします。
    """
    tensor_l = torch.tensor(p_list, dtype=torch.float32, device=device) / FOOT_PRESSURE_SCALE
    tensor_r = torch.zeros_like(tensor_l)  # 右足はゼロ

    # 左右それぞれ35点ずつとして結合 (70次元)
    out_tensor = torch.cat([tensor_l, tensor_r], dim=0).unsqueeze(0).unsqueeze(0)
    return out_tensor


def preprocess_imu(acc, gyro, device):
    """
    加速度(3)とジャイロ(3)から(1, 1, 2, 9)のIMUテンソルを作成します。
    学習時と同じ正規化を適用します。
    片足（左足）にデータを入力し、右足は全てゼロとします。
    """
    mag = [0.0, 0.0, 0.0]  # Magデータはストリームに含まれないので0

    # ★正規化: 学習時と同スケール
    mag_norm = [v / IMU_MAG_SCALE for v in mag]
    gyro_norm = [v / IMU_GYRO_SCALE for v in gyro]
    acc_norm = [v / IMU_ACC_SCALE for v in acc]

    combined_l = torch.tensor(mag_norm + gyro_norm + acc_norm, dtype=torch.float32, device=device)
    combined_r = torch.zeros_like(combined_l)

    # 2つのセンサー(左右)として結合 -> Shape: (2, 9)
    combined = torch.stack([combined_l, combined_r], dim=0)

    out_tensor = combined.unsqueeze(0).unsqueeze(0)
    return out_tensor


def parse_sse_payload(payload_str):
    """
    \"data: {...}\" 形式の文字列から JSON をパースして辞書で返します。
    """
    payload_str = payload_str.strip()
    if payload_str.startswith("data:"):
        payload_str = payload_str[5:].strip()

    if not payload_str:
        return None

    try:
        return json.loads(payload_str)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
