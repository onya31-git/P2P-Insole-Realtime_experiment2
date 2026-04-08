import json
import torch

def preprocess_foot_pressure(p_list, device):
    """
    35点の圧力データを (1, 1, 2, 32, 32) のテンソルに変換します。
    ※現状はダミーのマッピング（単一の特徴として複製およびパディング）を行っています。
    ※両足分(Channel=2)として、同じデータを複製しています。
    """
    tensor_p = torch.tensor(p_list, dtype=torch.float32, device=device)
    
    grid = torch.zeros((32 * 32,), dtype=torch.float32, device=device)
    grid[:35] = tensor_p
    grid = grid.view(32, 32)
    
    out_tensor = torch.stack([grid, grid], dim=0).unsqueeze(0).unsqueeze(0)
    return out_tensor

def preprocess_imu(acc, gyro, device):
    """
    加速度(3)とジャイロ(3)を結合し、(1, 1, 5, 6) のIMUテンソルに変換します。
    ※現状は取得した1つのセンサーデータを5つの関節センサ（Hip, Knee, Ankle等）に複製しています。
    """
    combined = torch.tensor(acc + gyro, dtype=torch.float32, device=device)
    
    repeated = combined.unsqueeze(0).repeat(5, 1)
    
    out_tensor = repeated.unsqueeze(0).unsqueeze(0)
    return out_tensor

def parse_sse_payload(payload_str):
    """
    "data: {...}" 形式の文字列から JSON をパースして辞書で返します。
    """
    payload_str = payload_str.strip()
    if payload_str.startswith("data:"):
        payload_str = payload_str[5:].strip()
    
    try:
        return json.loads(payload_str)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
