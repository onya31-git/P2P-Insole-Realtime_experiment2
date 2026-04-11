import json
import torch
import numpy as np

# 学習時と同じ正規化スケール定数
FOOT_PRESSURE_SCALE = 2000.0  # foot / 2000.0
IMU_MAG_SCALE       = 100.0   # Mag:  [-100, 100] -> [-1, 1]
IMU_GYRO_SCALE      = 500.0   # Gyro: [-500, 500] -> [-1, 1]
IMU_ACC_SCALE       = 8.0     # Acc:  [-8,  8]   -> [-1, 1]


def preprocess_both_feet(p_l, acc_l, gyro_l, mag_l,
                         p_r, acc_r, gyro_r, mag_r,
                         device):
    """
    左足・右足それぞれの圧力・IMUデータを受け取り、
    モデル入力テンソル (foot, imu) を返します。

    Returns:
        foot_tensor: (1, 1, 70)   — 左足35点 + 右足35点
        imu_tensor:  (1, 1, 2, 9) — [左センサー, 右センサー] x [mag3, gyro3, acc3]
    """
    # --- 足圧 ---
    fl = np.array(p_l, dtype=np.float32) / FOOT_PRESSURE_SCALE
    fr = np.array(p_r, dtype=np.float32) / FOOT_PRESSURE_SCALE
    foot = np.concatenate([fl, fr])  # (70,)
    foot_tensor = torch.tensor(foot, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # --- IMU (左足) ---
    ml = np.array(mag_l,  dtype=np.float32) / IMU_MAG_SCALE
    gl = np.array(gyro_l, dtype=np.float32) / IMU_GYRO_SCALE
    al = np.array(acc_l,  dtype=np.float32) / IMU_ACC_SCALE
    imu_l = np.concatenate([ml, gl, al])  # (9,)

    # --- IMU (右足) ---
    mr = np.array(mag_r,  dtype=np.float32) / IMU_MAG_SCALE
    gr = np.array(gyro_r, dtype=np.float32) / IMU_GYRO_SCALE
    ar = np.array(acc_r,  dtype=np.float32) / IMU_ACC_SCALE
    imu_r = np.concatenate([mr, gr, ar])  # (9,)

    imu = np.stack([imu_l, imu_r], axis=0)  # (2, 9)
    imu_tensor = torch.tensor(imu, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    return foot_tensor, imu_tensor


def preprocess_foot_pressure(p_list, device):
    """
    後方互換用: 片足35点 -> (1,1,70)。右足はゼロパディング。
    ※ 両足データが揃う場合は preprocess_both_feet を使用してください。
    """
    tensor_l = torch.tensor(p_list, dtype=torch.float32, device=device) / FOOT_PRESSURE_SCALE
    tensor_r = torch.zeros_like(tensor_l)
    out_tensor = torch.cat([tensor_l, tensor_r], dim=0).unsqueeze(0).unsqueeze(0)
    return out_tensor


def preprocess_imu(acc, gyro, device, mag=None):
    """
    後方互換用: 片足IMU -> (1,1,2,9)。右センサーはゼロパディング。
    ※ 両足データが揃う場合は preprocess_both_feet を使用してください。
    """
    if mag is None:
        mag = [0.0, 0.0, 0.0]
    mag_norm  = np.array(mag,  dtype=np.float32) / IMU_MAG_SCALE
    gyro_norm = np.array(gyro, dtype=np.float32) / IMU_GYRO_SCALE
    acc_norm  = np.array(acc,  dtype=np.float32) / IMU_ACC_SCALE
    combined_l = np.concatenate([mag_norm, gyro_norm, acc_norm])
    combined_r = np.zeros(9, dtype=np.float32)
    combined = np.stack([combined_l, combined_r], axis=0)
    out_tensor = torch.tensor(combined, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return out_tensor


def parse_sse_payload(payload_str):
    """
    SSEストリームの1行をパースして辞書で返します。
    "data: {...}" 形式または "event: xxx" 形式に対応。
    """
    payload_str = payload_str.strip()

    if payload_str.startswith("event:"):
        return None  # イベント行は無視

    if payload_str.startswith("data:"):
        payload_str = payload_str[5:].strip()

    if not payload_str:
        return None

    try:
        return json.loads(payload_str)
    except json.JSONDecodeError:
        return None
