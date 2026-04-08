import socket
import json
import torch
import time
import argparse
import os
import urllib.request
import urllib.error
from models.model import KinematicFusionModel
from processor.filter import OneEuroFilter
from processor.preprocessor import preprocess_foot_pressure, preprocess_imu, parse_sse_payload

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
    
    # --- スクリプト内でロードする重みファイルを指定 ---
    # 例: "weights/kinematic_model_20260408_154441.pth" （空文字の場合は引数 --weights が優先されます）
    TARGET_WEIGHT_PATH = ""
    
    weight_to_load = TARGET_WEIGHT_PATH if TARGET_WEIGHT_PATH else args.weights

    if weight_to_load:
        if os.path.exists(weight_to_load):
            try:
                model.load_state_dict(torch.load(weight_to_load, map_location=device))
                print(f"Loaded weights from {weight_to_load}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
        else:
            print(f"Warning: Weight file '{weight_to_load}' not found. Using initialized weights.")
            
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
