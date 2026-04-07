import socket
import json
import torch
import time
import argparse
import urllib.request
import urllib.error
from kinematic_model import KinematicFusionModel, OneEuroFilter

def preprocess_foot_pressure(p_list, device):
    """
    35点の圧力データを (1, 1, 2, 32, 32) のテンソルに変換します。
    ※現状はダミーのマッピング（単一の特徴として複製およびパディング）を行っています。
    ※両足分(Channel=2)として、同じデータを複製しています。
    """
    # p_list: 35 elements
    tensor_p = torch.tensor(p_list, dtype=torch.float32, device=device)
    
    # 32x32 = 1024点の画像データに対して、35点を適当にマッピングします。
    # 実際には物理的なセンサー位置に基づいたグリッド補間（Grid interpolation）等が必要です。
    grid = torch.zeros((32 * 32,), dtype=torch.float32, device=device)
    grid[:35] = tensor_p
    grid = grid.view(32, 32)
    
    # [B=1, Seq=1, C=2, H=32, W=32]
    # C=2 (Left/Right) として同じデータを両方のチャネルに入れています
    out_tensor = torch.stack([grid, grid], dim=0).unsqueeze(0).unsqueeze(0)
    return out_tensor

def preprocess_imu(acc, gyro, device):
    """
    加速度(3)とジャイロ(3)を結合し、(1, 1, 5, 6) のIMUテンソルに変換します。
    ※現状は取得した1つのセンサーデータを5つの関節センサ（Hip, Knee, Ankle等）に複製しています。
    """
    combined = torch.tensor(acc + gyro, dtype=torch.float32, device=device) # (6,)
    
    # 5センサー分に複製する [5, 6]
    repeated = combined.unsqueeze(0).repeat(5, 1)
    
    # [B=1, Seq=1, N_sensors=5, D=6]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_url", type=str, default="http://163.143.136.103:5001/stream", help="HTTP Stream URL")
    parser.add_argument("--send_ip", type=str, default="127.0.0.1", help="UDP Sending IP")
    parser.add_argument("--send_port", type=int, default=5006, help="UDP Sending Port")
    parser.add_argument("--weights", type=str, default="", help="Path to model weights (.pth)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==============================
    # 1. モデルとフィルタの初期化
    # ==============================
    NUM_SENSORS = 5
    NUM_JOINTS = 24
    model = KinematicFusionModel(num_joints=NUM_JOINTS, imu_sensors=NUM_SENSORS).to(device)
    
    if args.weights:
        try:
            model.load_state_dict(torch.load(args.weights, map_location=device))
            print(f"Loaded weights from {args.weights}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            
    model.eval()
    model.set_stateful(True) # リアルタイム推論モード（前フレーム状態保持）

    euro_filter = OneEuroFilter(mincutoff=1.0, beta=0.01, dcutoff=1.0)

    # ==============================
    # 2. 通信の初期化
    # ==============================
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Connecting to HTTP Stream at {args.stream_url}")
    print(f"Computed Output will be sent to {args.send_ip}:{args.send_port}")
    print("Waiting for data...")

    # ==============================
    # 3. リアルタイム推論ループ
    # ==============================
    try:
        req = urllib.request.Request(args.stream_url)
        with urllib.request.urlopen(req) as response:
            for line in response:
                raw_str = line.decode('utf-8').strip()
                if not raw_str:
                    continue
                
                parsed = parse_sse_payload(raw_str)
                if not parsed or "payload" not in parsed:
                    continue
                    
                payload_data = parsed["payload"]
                
                p_data = payload_data.get("p", [])
                acc = payload_data.get("acc", [])
                gyro = payload_data.get("gyro", [])
                
                if len(p_data) != 35 or len(acc) != 3 or len(gyro) != 3:
                    # 不正な形状の場合はスキップ
                    continue
                
                # --- 前処理 ---
                foot_tensor = preprocess_foot_pressure(p_data, device)
                imu_tensor = preprocess_imu(acc, gyro, device)
                
                # --- 推論 ---
                start_time = time.time()
                with torch.no_grad():
                    out_quat = model(foot_tensor, imu_tensor)
                    # ジッタ抑制フィルタの適用
                    out_quat_filtered = euro_filter(start_time, out_quat)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # --- 後処理とデータ送信 ---
                # (Batch=1, Seq=1, Joints=24, 4) -> (24, 4) のリストへ変換
                quat_list = out_quat_filtered.squeeze().cpu().numpy().tolist()
                
                output_msg = {
                    "ts": time.time(),
                    "latency_ms": round(latency_ms, 2),
                    "pose_quaternion": quat_list
                }
                
                # ストリーム出力として指定ポートへ送信
                sock_send.sendto(json.dumps(output_msg).encode('utf-8'), (args.send_ip, args.send_port))
                
                print(f"Processed frame. Latency: {latency_ms:.2f} ms | Output Joints: {len(quat_list)}")

    except KeyboardInterrupt:
        print("\nStopped real-time inference.")
    except urllib.error.URLError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        sock_send.close()

if __name__ == "__main__":
    main()
