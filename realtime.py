import socket
import json
import torch
import time
import argparse
import os
import glob
import urllib.request
import urllib.error
from collections import deque
from models.model import HierarchicalKinematicFusionModel
from models.model import KinematicFusionModel
from processor.filter import OneEuroFilter
from processor.preprocessor import preprocess_both_feet, parse_sse_payload

# ==============================
# デバイスID（左足・右足）
# ==============================
LEFT_FOOT_DN  = "3030F9284F54"
RIGHT_FOOT_DN = "3030F92685D4"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_url", type=str, default="http://163.143.136.103:5001/stream", help="HTTP Stream URL")
    parser.add_argument("--send_ip",   type=str, default="127.0.0.1", help="UDP Sending IP")
    parser.add_argument("--send_port", type=int, default=5006,         help="UDP Sending Port")
    parser.add_argument("--weights",   type=str, default="",           help="Path to model weights (.pth)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==============================
    # 1. モデルとフィルタの初期化
    # ==============================
    NUM_JOINTS = 24
    # model = KinematicFusionModel(
    model = HierarchicalKinematicFusionModel(
        foot_features=70, imu_sensors=2, imu_channels=9, num_joints=NUM_JOINTS
    ).to(device)

    # 最新の重みを自動選択
    weight_files = sorted(glob.glob("weights/*.pth"))
    TARGET_WEIGHT_PATH = weight_files[-1] if weight_files else ""
    weight_to_load = args.weights if args.weights else TARGET_WEIGHT_PATH

    if weight_to_load:
        if os.path.exists(weight_to_load):
            try:
                model.load_state_dict(torch.load(weight_to_load, map_location=device, weights_only=True))
                print(f"Loaded weights from {weight_to_load}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
                raise RuntimeError("モデルと重みのアーキテクチャが一致しません。") from e
        else:
            print(f"Warning: Weight file '{weight_to_load}' not found.")

    model.eval()
    model.set_stateful(False)  # リアルタイム推論モード（学習時と同等のステートレス・スライディングウィンドウ方式）

    # 施策D: OneEuroFilter パラメータ最適化
    euro_filter = OneEuroFilter(mincutoff=0.5, beta=0.003, dcutoff=1.0)

    # ==============================
    # 2. 通信の初期化
    # ==============================
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Connecting to HTTP Stream at {args.stream_url}")
    print(f"Output will be sent to {args.send_ip}:{args.send_port}")
    print(f"Left foot DN : {LEFT_FOOT_DN}")
    print(f"Right foot DN: {RIGHT_FOOT_DN}")
    print("Waiting for data from both feet...")

    # ==============================
    # 3. バッファ（スライディングウィンドウ用）
    # ==============================
    buffer_left  = None  # {"p": [...], "acc": [...], "gyro": [...], "mag": [...]}
    buffer_right = None
    
    SEQ_LEN = 50
    sliding_foot = deque(maxlen=SEQ_LEN)
    sliding_imu = deque(maxlen=SEQ_LEN)

    # ==============================
    # 4. リアルタイム推論ループ
    # ==============================
    try:
        req = urllib.request.Request(args.stream_url)
        with urllib.request.urlopen(req) as response:
            for line in response:
                raw_str = line.decode('utf-8').strip()
                if not raw_str:
                    continue

                parsed = parse_sse_payload(raw_str)
                if not parsed:
                    continue

                # ストリームのルートに直接 dn と payload がある場合と
                # payload 内に dn がある場合の両方に対応
                dn = parsed.get("dn", "")
                payload_data = parsed.get("payload", {})
                if not dn and payload_data:
                    dn = payload_data.get("dn", "")

                p_data = payload_data.get("p",    [])
                acc    = payload_data.get("acc",   [])
                gyro   = payload_data.get("gyro",  [])
                mag    = payload_data.get("mag",   [0.0, 0.0, 0.0])

                # データの形状チェック
                if len(p_data) != 35 or len(acc) != 3 or len(gyro) != 3:
                    continue

                frame_data = {"p": p_data, "acc": acc, "gyro": gyro, "mag": mag}

                # 左右どちらのデバイスか判定してバッファに保存
                if dn == LEFT_FOOT_DN:
                    buffer_left = frame_data
                elif dn == RIGHT_FOOT_DN:
                    buffer_right = frame_data
                else:
                    # 未知のデバイスはスキップ
                    continue

                # 両足のデータが揃っていなければ次のフレームへ
                if buffer_left is None or buffer_right is None:
                    foot_side = "左足" if dn == LEFT_FOOT_DN else "右足"
                    print(f"[{foot_side}] データ受信中. もう片足を待機...")
                    continue

                # ==============================
                # 5. 前処理（両足データ使用）
                # ==============================
                foot_tensor, imu_tensor = preprocess_both_feet(
                    p_l   = buffer_left["p"],
                    acc_l = buffer_left["acc"],
                    gyro_l= buffer_left["gyro"],
                    mag_l = buffer_left["mag"],
                    p_r   = buffer_right["p"],
                    acc_r = buffer_right["acc"],
                    gyro_r= buffer_right["gyro"],
                    mag_r = buffer_right["mag"],
                    device=device
                )

                # ==============================
                # 6. スライディングウィンドウへ追加・推論
                # ==============================
                # (1, 1, F) -> (F,) にしてdequeへ
                sliding_foot.append(foot_tensor.squeeze(0).squeeze(0))
                sliding_imu.append(imu_tensor.squeeze(0).squeeze(0))
                
                # 50フレーム溜まるまでは待機 (最初の0.5秒間)
                if len(sliding_foot) < SEQ_LEN:
                    print(f"Buffering window... ({len(sliding_foot)}/{SEQ_LEN})")
                    continue

                start_time = time.time()
                with torch.no_grad():
                    # (SEQ_LEN, F) のリストを (1, SEQ_LEN, F) テンソルにスタックして推論
                    foot_seq = torch.stack(list(sliding_foot)).unsqueeze(0)
                    imu_seq = torch.stack(list(sliding_imu)).unsqueeze(0)
                    
                    out_pos = model(foot_seq, imu_seq) # (1, SEQ_LEN, 24, 3)
                    out_pos_last = out_pos[:, -1:, :, :] # 最後のフレームのみ抽出 (1, 1, 24, 3)
                    
                    out_pos_filtered = euro_filter(start_time, out_pos_last)

                latency_ms = (time.time() - start_time) * 1000

                # ==============================
                # 7. 後処理とデータ送信
                # ==============================
                # (1, 1, 24, 3) -> (24, 3) のリスト
                pos_list = out_pos_filtered.squeeze().cpu().numpy().tolist()

                output_msg = {
                    "ts": time.time(),
                    "latency_ms": round(latency_ms, 2),
                    "pose_positions": pos_list
                }

                sock_send.sendto(
                    json.dumps(output_msg).encode('utf-8'),
                    (args.send_ip, args.send_port)
                )

                print(f"Processed frame (both feet). Latency: {latency_ms:.2f} ms")

    except KeyboardInterrupt:
        print("\nStopped real-time inference.")
    except urllib.error.URLError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during processing: {e}")
    finally:
        sock_send.close()

if __name__ == "__main__":
    main()
