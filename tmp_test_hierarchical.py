import torch
from models.model import HierarchicalKinematicFusionModel

def test_model():
    print("Testing HierarchicalKinematicFusionModel...")
    model = HierarchicalKinematicFusionModel()
    
    # バッチサイズ 2、シーケンス長 10
    B = 2
    Seq = 10
    
    foot_pressure = torch.randn(B, Seq, 70)
    imu_data = torch.randn(B, Seq, 2, 9)
    
    # フォワードパス
    out = model(foot_pressure, imu_data)
    
    # 出力シェイプの検証
    expected_shape = (B, Seq, 24, 3)
    assert out.shape == expected_shape, f"Expected {expected_shape}, but got {out.shape}"
    
    print(f"Output shape validated successfully: {out.shape}")
    
    # リアルタイム（Stateful）モードの検証
    model.set_stateful(True)
    foot_pressure_rt = torch.randn(1, 1, 70)
    imu_data_rt = torch.randn(1, 1, 2, 9)
    out_rt = model(foot_pressure_rt, imu_data_rt)
    
    expected_shape_rt = (1, 1, 24, 3)
    assert out_rt.shape == expected_shape_rt, f"Expected {expected_shape_rt}, but got {out_rt.shape}"
    
    print(f"Realtime output shape validated successfully: {out_rt.shape}")
    print("All tests passed!")

if __name__ == "__main__":
    test_model()
