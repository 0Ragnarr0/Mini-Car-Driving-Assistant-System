"""Inference module for real-time predictions in Webots."""

import os
import pickle
import numpy as np
import torch
from config import DATA_CONFIG, SENSOR_CONFIG, MODEL_CONFIG
from model import load_model
from data_collector import SensorDataCollector


class AutonomousDrivingInference:
    """Inference engine for autonomous driving."""
    
    def __init__(self, model_path=None, scaler_path=None, device='cpu'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model weights
            scaler_path: Path to fitted scaler for normalization
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.scaler = None
        self.data_collector = SensorDataCollector()
        
        # Load model and scaler
        if model_path is None:
            model_path = os.path.join(DATA_CONFIG['models_dir'], 'autonomous_driving_model_final.pth')
        if scaler_path is None:
            scaler_path = os.path.join(DATA_CONFIG['models_dir'], 'scaler.pkl')
        
        self.load_model(model_path)
        self.load_scaler(scaler_path)
    
    def load_model(self, model_path):
        """Load the trained model."""
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            print("Using untrained model for inference")
            from model import create_model
            self.model = create_model(device=self.device)
        else:
            try:
                self.model = load_model(model_path, device=self.device)
                print(f"Loaded model from {model_path}")
            except RuntimeError as e:
                print(f"Warning: Failed to load model - architecture mismatch or corrupted checkpoint")
                print(f"  Error: {str(e)[:100]}...")
                print("Using untrained multimodal model for inference")
                print("Please train a new model with multimodal data using: python trainer.py")
                from model import create_model
                self.model = create_model(device=self.device)
    
    def load_scaler(self, scaler_path):
        """Load the fitted scaler."""
        if not os.path.exists(scaler_path):
            print(f"Warning: Scaler not found at {scaler_path}")
            print("Inference may not work correctly")
            self.scaler = None
        else:
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Loaded scaler from {scaler_path}")
            except Exception as exc:
                print(f"Warning: Failed to load scaler from {scaler_path}: {exc}")
                print("Continuing inference without scaler normalization")
                self.scaler = None
    
    def predict(self, camera_image=None, lidar_ranges=None, gps_data=None, gyro_data=None):
        """
        Make prediction from multimodal sensor inputs.
        
        Args:
            camera_image: RGB camera image (H, W, 3)
            lidar_ranges: Lidar range measurements
            gps_data: GPS position [x, y]
            gyro_data: Gyroscope data [rx, ry, rz]
        
        Returns:
            Tuple of (steering_angle, speed_control) in normalized range [-1, 1]
        """
        # Collect and process sensor data
        sensor_dict = self.data_collector.collect_sensor_data(
            camera_image, lidar_ranges, gps_data, gyro_data
        )
        
        # Convert to tensors
        camera_tensor = torch.from_numpy(sensor_dict['camera']).float().unsqueeze(0)  # (1, 64, 64, 3)
        camera_tensor = camera_tensor.permute(0, 3, 1, 2)  # (1, 3, 64, 64) for CNN
        
        lidar_tensor = torch.from_numpy(sensor_dict['lidar']).float().unsqueeze(0)  # (1, 64)
        
        gps_tensor = torch.from_numpy(sensor_dict['gps']).float().unsqueeze(0)  # (1, 2)
        gyro_tensor = torch.from_numpy(sensor_dict['gyro']).float().unsqueeze(0)  # (1, 1)
        
        # Combine GPS + Gyro
        sensor_tensor = torch.cat([gps_tensor, gyro_tensor], dim=1).to(self.device)  # (1, 3)
        camera_tensor = camera_tensor.to(self.device)
        lidar_tensor = lidar_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(camera_tensor, lidar_tensor, sensor_tensor)
        
        # Convert to numpy
        prediction = output.cpu().numpy()[0]
        steering_angle = float(prediction[0])
        speed_control = float(prediction[1])
        
        # Denormalize outputs to real values
        steering_angle = steering_angle * SENSOR_CONFIG['max_steering_angle']
        speed_control = (speed_control + 1) / 2 * SENSOR_CONFIG['max_speed']  # Convert [-1,1] to [0, max_speed]
        
        return steering_angle, speed_control
    
    def predict_batch(self, sensor_inputs):
        """
        Make batch predictions.
        
        Args:
            sensor_inputs: Batch of sensor inputs (N, input_size)
        
        Returns:
            Batch of predictions (N, 2)
        """
        # Normalize if scaler available
        if self.scaler is not None:
            sensor_inputs = self.scaler.transform(sensor_inputs)
        
        # Convert to tensor
        sensor_tensor = torch.from_numpy(sensor_inputs).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(sensor_tensor)
        
        # Convert to numpy
        predictions = outputs.cpu().numpy()
        
        # Denormalize
        predictions[:, 0] *= SENSOR_CONFIG['max_steering_angle']
        predictions[:, 1] = (predictions[:, 1] + 1) / 2 * SENSOR_CONFIG['max_speed']
        
        return predictions
