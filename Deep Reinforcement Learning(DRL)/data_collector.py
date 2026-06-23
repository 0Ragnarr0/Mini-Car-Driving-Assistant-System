"""Data collection utilities for autonomous driving training."""

import os
import sys
import numpy as np
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Sensors.camera import process_3dCamera_data, process_fisheye_data
from config import SENSOR_CONFIG, DATA_CONFIG

try:
    from dataset import DataManager
except Exception:
    DataManager = None


class SensorDataCollector:
    """Collects sensor data from Webots simulation."""
    
    def __init__(self, max_buffer_size=1000):
        """
        Initialize the data collector.
        
        Args:
            max_buffer_size: Maximum number of samples to buffer before saving
        """
        self.sensor_buffer = deque(maxlen=max_buffer_size)
        self.label_buffer = deque(maxlen=max_buffer_size)
        self.data_manager = DataManager() if DataManager is not None else None
        self.batch_count = 0

    def _save_batch_fallback(self, sensor_data, labels):
        """Save training data without DataManager dependencies."""
        data_dir = DATA_CONFIG.get('data_dir', 'training_data')
        os.makedirs(data_dir, exist_ok=True)
        batch_file = os.path.join(data_dir, f'batch_{self.batch_count}.npz')
        
        # sensor_data is list of dicts, convert to structured arrays
        camera_list = np.array([s['camera'] for s in sensor_data], dtype=np.float32)
        lidar_list = np.array([s['lidar'] for s in sensor_data], dtype=np.float32)
        gps_list = np.array([s['gps'] for s in sensor_data], dtype=np.float32)
        gyro_list = np.array([s['gyro'] for s in sensor_data], dtype=np.float32)
        
        np.savez_compressed(
            batch_file,
            camera=camera_list,
            lidar=lidar_list,
            gps=gps_list,
            gyro=gyro_list,
            labels=labels
        )
        print(f"Saved training batch {self.batch_count} to {batch_file}")
    
    def collect_sensor_data(self, camera_image, lidar_ranges, gps_data, gyro_data):
        """
        Collect and process multimodal sensor data.
        
        Args:
            camera_image: RGB camera image (H, W, 3) or (H, W, 4)
            lidar_ranges: Lidar range measurements (N,)
            gps_data: GPS position [x, y]
            gyro_data: Gyroscope rotation [rx, ry, rz] - using rz (yaw)
        
        Returns:
            Dictionary with processed sensor inputs ready for model
        """
        import cv2
        
        # Process camera: resize to 64x64 RGB
        if camera_image is not None and len(camera_image.shape) > 0:
            if camera_image.shape[2] == 4:  # RGBA
                camera_image = camera_image[:, :, :3]  # Drop alpha
            camera_rgb = cv2.resize(camera_image, (64, 64), interpolation=cv2.INTER_LINEAR)
            camera_rgb = camera_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
        else:
            camera_rgb = np.zeros((64, 64, 3), dtype=np.float32)
        
        # Process lidar: resample to 64 points
        if lidar_ranges is not None and len(lidar_ranges) > 0:
            lidar_array = np.array(lidar_ranges, dtype=np.float32)
            lidar_array = np.clip(lidar_array, 0.0, 30.0)  # Clip to max distance
            # Resample to 64 points
            if len(lidar_array) != 64:
                lidar_array = np.interp(
                    np.linspace(0, len(lidar_array)-1, 64),
                    np.arange(len(lidar_array)),
                    lidar_array
                )
        else:
            lidar_array = np.zeros(64, dtype=np.float32)
        
        # Process GPS: ensure 2D position
        if gps_data is not None and len(gps_data) >= 2:
            gps_array = np.array(gps_data[:2], dtype=np.float32)
        else:
            gps_array = np.zeros(2, dtype=np.float32)
        
        # Process Gyro: use yaw rate (z-axis rotation)
        if gyro_data is not None and len(gyro_data) >= 3:
            gyro_array = np.array([gyro_data[2]], dtype=np.float32)  # yaw rate
        else:
            gyro_array = np.zeros(1, dtype=np.float32)
        
        return {
            'camera': camera_rgb,  # (64, 64, 3)
            'lidar': lidar_array,  # (64,)
            'gps': gps_array,  # (2,)
            'gyro': gyro_array,  # (1,)
        }
    
    def add_sample(self, sensor_input_dict, steering_angle, speed_control):
        """
        Add a sample to the buffer.
        
        Args:
            sensor_input_dict: Dictionary with keys 'camera', 'lidar', 'gps', 'gyro'
            steering_angle: Ground truth steering angle
            speed_control: Ground truth speed control value
        """
        self.sensor_buffer.append(sensor_input_dict)
        self.label_buffer.append(np.array([steering_angle, speed_control], dtype=np.float32))
    
    def save_batch(self):
        """Save current buffer as a training batch (multimodal format)."""
        if len(self.sensor_buffer) == 0:
            print("No data to save")
            return

        # Convert buffer of dicts to separate arrays
        camera_list = np.array([s['camera'] for s in self.sensor_buffer], dtype=np.float32)
        lidar_list = np.array([s['lidar'] for s in self.sensor_buffer], dtype=np.float32)
        gps_list = np.array([s['gps'] for s in self.sensor_buffer], dtype=np.float32)
        gyro_list = np.array([s['gyro'] for s in self.sensor_buffer], dtype=np.float32)
        labels = np.array(list(self.label_buffer), dtype=np.float32)

        # Save as multimodal format
        data_dir = DATA_CONFIG.get('data_dir', 'training_data')
        os.makedirs(data_dir, exist_ok=True)
        batch_file = os.path.join(data_dir, f'batch_{self.batch_count}.npz')
        
        np.savez_compressed(
            batch_file,
            camera=camera_list,
            lidar=lidar_list,
            gps=gps_list,
            gyro=gyro_list,
            labels=labels
        )
        print(f"Saved training batch {self.batch_count} to {batch_file}")

        self.batch_count += 1
        
        # Clear buffers
        self.sensor_buffer.clear()
        self.label_buffer.clear()
    
    def get_buffer_status(self):
        """Get current buffer status."""
        return {
            'current_size': len(self.sensor_buffer),
            'max_size': self.sensor_buffer.maxlen,
            'batches_saved': self.batch_count,
            'total_samples': self.batch_count * len(self.sensor_buffer) + len(self.sensor_buffer)
        }


class DataAugmentation:
    """Data augmentation techniques for training."""
    
    @staticmethod
    def add_gaussian_noise(sensor_data, noise_level=0.01):
        """Add Gaussian noise to sensor data."""
        noise = np.random.normal(0, noise_level, sensor_data.shape)
        return sensor_data + noise
    
    @staticmethod
    def scale_sensor_values(sensor_data, scale_factor=1.0):
        """Scale sensor values."""
        return sensor_data * scale_factor
    
    @staticmethod
    def augment_batch(sensor_data, labels, num_augmentations=2):
        """
        Augment a batch of data.
        
        Args:
            sensor_data: Sensor input batch
            labels: Labels batch
            num_augmentations: Number of augmentations per sample
        
        Returns:
            Augmented sensor data and labels
        """
        augmented_data = [sensor_data]
        augmented_labels = [labels]
        
        for _ in range(num_augmentations):
            # Add noise
            noisy_data = DataAugmentation.add_gaussian_noise(sensor_data, noise_level=0.02)
            augmented_data.append(noisy_data)
            augmented_labels.append(labels)
        
        return np.vstack(augmented_data), np.vstack(augmented_labels)
