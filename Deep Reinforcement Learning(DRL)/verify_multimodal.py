#!/usr/bin/env python3
"""Verification script for multimodal architecture upgrade."""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import MultimodalFusionNet, create_model
from data_collector import SensorDataCollector
from config import MODEL_CONFIG

def test_model_creation():
    """Test that the multimodal model can be created."""
    print("Testing model creation...")
    
    try:
        model = create_model(device='cpu')
        print(f"✓ Model created successfully: {type(model).__name__}")
        
        # Print model summary
        print(f"  Model type: {MODEL_CONFIG.get('model_type', 'unknown')}")
        print(f"  CNN features: {MODEL_CONFIG.get('cnn_features', 'N/A')}")
        print(f"  Lidar features: {MODEL_CONFIG.get('lidar_features', 'N/A')}")
        print(f"  Sensor features: {MODEL_CONFIG.get('sensor_features', 'N/A')}")
        print(f"  Output size: {MODEL_CONFIG.get('output_size', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

def test_forward_pass():
    """Test that data flows through the model correctly."""
    print("\nTesting forward pass...")
    
    try:
        model = create_model(device='cpu')
        model.eval()
        
        # Create dummy inputs matching the expected shapes
        batch_size = 2
        camera = torch.randn(batch_size, 3, 64, 64)  # (batch, channels, height, width)
        lidar = torch.randn(batch_size, 64)           # (batch, lidar_points)
        sensors = torch.randn(batch_size, 3)          # (batch, gps_x, gps_y, gyro_z)
        
        # Forward pass
        with torch.no_grad():
            output = model(camera, lidar, sensors)
        
        assert output.shape == (batch_size, 2), f"Expected output shape (2, 2), got {output.shape}"
        print(f"✓ Forward pass successful")
        print(f"  Input shapes: camera{camera.shape}, lidar{lidar.shape}, sensors{sensors.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values: {output.cpu().numpy()}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sensor_collector():
    """Test that sensor data collector processes data correctly."""
    print("\nTesting sensor data collector...")
    
    try:
        collector = SensorDataCollector()
        
        # Create dummy sensor data
        camera_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        lidar_ranges = np.array([1.0, 1.5, 2.0, 2.5, 3.0] * 13, dtype=np.float32)[:64]
        gps_data = np.array([10.0, 20.0, 0.0])  # [x, y, z]
        gyro_data = np.array([0.1, 0.2, 0.5])   # [rx, ry, rz]
        
        # Process sensor data
        sensor_dict = collector.collect_sensor_data(camera_image, lidar_ranges, gps_data, gyro_data)
        
        # Verify output structure
        assert 'camera' in sensor_dict, "Missing 'camera' in output"
        assert 'lidar' in sensor_dict, "Missing 'lidar' in output"
        assert 'gps' in sensor_dict, "Missing 'gps' in output"
        assert 'gyro' in sensor_dict, "Missing 'gyro' in output"
        
        # Verify shapes
        assert sensor_dict['camera'].shape == (64, 64, 3), f"Camera shape mismatch: {sensor_dict['camera'].shape}"
        assert sensor_dict['lidar'].shape == (64,), f"Lidar shape mismatch: {sensor_dict['lidar'].shape}"
        assert sensor_dict['gps'].shape == (2,), f"GPS shape mismatch: {sensor_dict['gps'].shape}"
        assert sensor_dict['gyro'].shape == (1,), f"Gyro shape mismatch: {sensor_dict['gyro'].shape}"
        
        # Verify value ranges
        assert np.all(sensor_dict['camera'] >= 0) and np.all(sensor_dict['camera'] <= 1), "Camera values out of range"
        assert np.all(sensor_dict['lidar'] >= 0) and np.all(sensor_dict['lidar'] <= 30), "Lidar values out of range"
        
        print("✓ Sensor data collection successful")
        print(f"  Camera shape: {sensor_dict['camera'].shape}")
        print(f"  Lidar shape: {sensor_dict['lidar'].shape}")
        print(f"  GPS shape: {sensor_dict['gps'].shape}")
        print(f"  Gyro shape: {sensor_dict['gyro'].shape}")
        print(f"  Camera value range: [{sensor_dict['camera'].min():.3f}, {sensor_dict['camera'].max():.3f}]")
        print(f"  Lidar value range: [{sensor_dict['lidar'].min():.3f}, {sensor_dict['lidar'].max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Sensor data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_saving():
    """Test that batches can be saved and loaded in multimodal format."""
    print("\nTesting batch save/load...")
    
    try:
        collector = SensorDataCollector(max_buffer_size=10)
        
        # Generate sample data
        for i in range(5):
            camera_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            lidar_ranges = np.linspace(0.5, 3.0, 64).astype(np.float32)
            gps_data = np.array([10.0 + i, 20.0 + i, 0.0])
            gyro_data = np.array([0.1, 0.2, 0.5])
            
            sensor_dict = collector.collect_sensor_data(camera_image, lidar_ranges, gps_data, gyro_data)
            collector.add_sample(sensor_dict, steering_angle=0.1, speed_control=0.5)
        
        # Save batch
        collector.save_batch()
        
        # Check if batch file was created (with absolute path)
        batch_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "training_data", 
            "batch_0.npz"
        )
        assert os.path.exists(batch_file), f"Batch file not created: {batch_file}"
        
        # Load and verify
        data = np.load(batch_file)
        assert 'camera' in data, "Missing 'camera' in saved batch"
        assert 'lidar' in data, "Missing 'lidar' in saved batch"
        assert 'gps' in data, "Missing 'gps' in saved batch"
        assert 'gyro' in data, "Missing 'gyro' in saved batch"
        assert 'labels' in data, "Missing 'labels' in saved batch"
        
        assert data['camera'].shape == (5, 64, 64, 3), f"Camera batch shape: {data['camera'].shape}"
        assert data['lidar'].shape == (5, 64), f"Lidar batch shape: {data['lidar'].shape}"
        assert data['gps'].shape == (5, 2), f"GPS batch shape: {data['gps'].shape}"
        assert data['gyro'].shape == (5, 1), f"Gyro batch shape: {data['gyro'].shape}"
        assert data['labels'].shape == (5, 2), f"Labels batch shape: {data['labels'].shape}"
        
        print("✓ Batch save/load successful")
        print(f"  Saved batch file: {batch_file}")
        print(f"  Camera batch shape: {data['camera'].shape}")
        print(f"  Lidar batch shape: {data['lidar'].shape}")
        print(f"  GPS batch shape: {data['gps'].shape}")
        print(f"  Gyro batch shape: {data['gyro'].shape}")
        print(f"  Labels batch shape: {data['labels'].shape}")
        
        # Close the file
        data.close()
        
        # Clean up
        import time
        time.sleep(0.5)  # Give system time to release file
        try:
            os.remove(batch_file)
        except Exception as e:
            print(f"  Warning: Could not delete test batch file: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Batch save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Multimodal Architecture Verification Tests")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_sensor_collector,
        test_batch_saving,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The multimodal architecture is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
