"""Dataset loader and preparation for training."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TRAINING_CONFIG, DATA_CONFIG


class AutonomousDrivingDataset(Dataset):
    """Custom dataset for autonomous driving training data (multimodal)."""
    
    def __init__(self, camera_data, lidar_data, sensor_data, labels):
        """
        Args:
            camera_data: Numpy array of camera images (N, 64, 64, 3)
            lidar_data: Numpy array of lidar ranges (N, 64)
            sensor_data: Numpy array of GPS+Gyro (N, 3)
            labels: Numpy array of labels (N, 2)
        """
        self.camera = torch.from_numpy(camera_data).float()
        self.lidar = torch.from_numpy(lidar_data).float()
        self.sensors = torch.from_numpy(sensor_data).float()
        self.labels = torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        camera = self.camera[idx]
        lidar = self.lidar[idx]
        sensors = self.sensors[idx]
        label = self.labels[idx]
        
        # Permute camera to (C, H, W) for CNN
        camera = camera.permute(2, 0, 1)  # (3, 64, 64)
        
        return (camera, lidar, sensors), label


class DataManager:
    """Manages data collection, loading, and preprocessing."""
    
    def __init__(self, data_dir=DATA_CONFIG['data_dir']):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(DATA_CONFIG['models_dir'], exist_ok=True)
        os.makedirs(DATA_CONFIG['logs_dir'], exist_ok=True)
    
    def save_training_batch(self, sensor_data, labels, batch_num):
        """Save a batch of training data."""
        batch_file = os.path.join(self.data_dir, f'batch_{batch_num}.npz')
        np.savez_compressed(batch_file, sensor_data=sensor_data, labels=labels)
        print(f"Saved training batch {batch_num} to {batch_file}")
        return batch_file
    
    def load_training_batch(self, batch_num):
        """Load a batch of training data."""
        batch_file = os.path.join(self.data_dir, f'batch_{batch_num}.npz')
        data = np.load(batch_file)
        return data['sensor_data'], data['labels']
    
    def get_all_training_data(self):
        """Load all available training data in multimodal format."""
        camera_list = []
        lidar_list = []
        gps_list = []
        gyro_list = []
        labels_list = []
        
        batch_num = 0
        while True:
            batch_file = os.path.join(self.data_dir, f'batch_{batch_num}.npz')
            if not os.path.exists(batch_file):
                break
            
            try:
                data = np.load(batch_file)
                
                # Try new multimodal format first
                if 'camera' in data:
                    camera_list.append(data['camera'])
                    lidar_list.append(data['lidar'])
                    gps_list.append(data['gps'])
                    gyro_list.append(data['gyro'])
                    labels_list.append(data['labels'])
                else:
                    # Fallback: old flat format - skip or convert
                    print(f"Warning: {batch_file} is in old format, skipping")
                    batch_num += 1
                    continue
            except Exception as e:
                print(f"Error loading {batch_file}: {e}")
                batch_num += 1
                continue
            
            batch_num += 1
        
        if not camera_list:
            return None, None, None, None
        
        all_camera = np.vstack(camera_list)
        all_lidar = np.vstack(lidar_list)
        all_gps = np.vstack(gps_list)
        all_gyro = np.vstack(gyro_list)
        all_labels = np.vstack(labels_list)
        
        # Combine GPS + Gyro
        all_sensors = np.hstack([all_gps, all_gyro])
        
        return all_camera, all_lidar, all_sensors, all_labels
    
    def create_data_loaders(self, batch_size=TRAINING_CONFIG['batch_size']):
        """Create train and validation data loaders (multimodal format)."""
        camera_data, lidar_data, sensor_data, labels = self.get_all_training_data()
        
        if camera_data is None:
            raise ValueError("No training data found. Please collect data first.")
        
        # Split data
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=TRAINING_CONFIG['validation_split'],
            random_state=42
        )
        
        # Split into train/val sets
        camera_train = camera_data[train_idx]
        camera_val = camera_data[val_idx]
        
        lidar_train = lidar_data[train_idx]
        lidar_val = lidar_data[val_idx]
        
        sensor_train = sensor_data[train_idx]
        sensor_val = sensor_data[val_idx]
        
        labels_train = labels[train_idx]
        labels_val = labels[val_idx]
        
        # Create datasets
        train_dataset = AutonomousDrivingDataset(
            camera_train, lidar_train, sensor_train, labels_train
        )
        val_dataset = AutonomousDrivingDataset(
            camera_val, lidar_val, sensor_val, labels_val
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def save_scaler(self, scaler_path):
        """Save the fitted scaler for inference."""
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, scaler_path):
        """Load the fitted scaler for inference."""
        import pickle
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
