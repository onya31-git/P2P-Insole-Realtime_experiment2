import json
import torch

def preprocess_foot_pressure(p_list, device):
    """
    35点の圧力データを (1, 1, 70) のテンソルに変換します。
    ※現状のストリームは片足分しか送られてこないと仮定し、左・右の両方として同じデータを複製して結合します。
    """
    tensor_p = torch.tensor(p_list, dtype=torch.float32, device=device)
    
    # 左右それぞれ35点ずつとして結合 (70次元)
    out_tensor = torch.cat([tensor_p, tensor_p], dim=0).unsqueeze(0).unsqueeze(0)
    return out_tensor

def preprocess_imu(acc, gyro, device):
    """
    加速度(3)とジャイロ(3)から、(1, 1, 2, 9) のIMUテンソルに変換します。
    ※データにMag(地磁気)が含まれていないため、0で補完します。
    ※現状は取得した1つのセンサーデータを左右の2つの足用（センサー数=2）に複製しています。
    """
    mag = [0.0, 0.0, 0.0]
    # データの順序がCSVに合わせて (Mag_x,y,z, Gyro_x,y,z, Acc_x,y,z) 等であると仮定
    # CSVでは Mag, Gyro, Acc の順
    combined = torch.tensor(mag + gyro + acc, dtype=torch.float32, device=device)
    
    # 2つのセンサー(左右)として複製 -> Shape: (2, 9)
    repeated = combined.unsqueeze(0).repeat(2, 1)
    
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
