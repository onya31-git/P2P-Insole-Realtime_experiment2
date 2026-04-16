import urllib.request
import urllib.error
from processor.preprocessor import parse_sse_payload

# デバイスID（左足・右足）
LEFT_FOOT_DN  = "3030F9284F54"
RIGHT_FOOT_DN = "3030F92685D4"
STREAM_URL = "http://163.143.136.103:5001/stream"

def check_stream():
    print(f"Connecting to {STREAM_URL} for data verification...")
    try:
        req = urllib.request.Request(STREAM_URL)
        with urllib.request.urlopen(req) as response:
            left_data = None
            right_data = None

            for line in response:
                raw_str = line.decode('utf-8').strip()
                if not raw_str:
                    continue

                parsed = parse_sse_payload(raw_str)
                if not parsed:
                    continue

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

                if dn == LEFT_FOOT_DN and left_data is None:
                    left_data = {"p": p_data, "acc": acc, "gyro": gyro, "mag": mag}
                    print("\n--- 左足データ取得成功 ---")
                    print(f"圧力 (Min/Max/Mean): {min(p_data):.1f} / {max(p_data):.1f} / {(sum(p_data)/len(p_data)):.1f}")
                    print(f"IMU Acc: {acc}")
                    print(f"IMU Gyro: {gyro}")
                    print(f"IMU Mag: {mag}")

                elif dn == RIGHT_FOOT_DN and right_data is None:
                    right_data = {"p": p_data, "acc": acc, "gyro": gyro, "mag": mag}
                    print("\n--- 右足データ取得成功 ---")
                    print(f"圧力 (Min/Max/Mean): {min(p_data):.1f} / {max(p_data):.1f} / {(sum(p_data)/len(p_data)):.1f}")
                    print(f"IMU Acc: {acc}")
                    print(f"IMU Gyro: {gyro}")
                    print(f"IMU Mag: {mag}")

                if left_data is not None and right_data is not None:
                    print("\n両足のデータ正常に取得完了しました。スクリプトを終了します。")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_stream()
