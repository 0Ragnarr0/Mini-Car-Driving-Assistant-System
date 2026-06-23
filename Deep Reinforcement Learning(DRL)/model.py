"""Neural network model for autonomous driving."""

import torch
import torch.nn as nn
from config import MODEL_CONFIG


class CNNEncoder(nn.Module):
    """CNN encoder for camera images."""
    
    def __init__(self, in_channels=3, out_features=128):
        super(CNNEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_features)
    
    def forward(self, x):
        """x shape: (batch, 3, 64, 64)"""
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultimodalFusionNet(nn.Module):
    """Multimodal fusion network combining camera, lidar, GPS, gyro."""
    
    def __init__(self, config=None):
        super(MultimodalFusionNet, self).__init__()
        if config is None:
            config = MODEL_CONFIG
        
        self.config = config
        
        # Camera encoder
        self.camera_encoder = CNNEncoder(
            in_channels=config.get('camera_channels', 3),
            out_features=config.get('cnn_features', 128)
        )
        
        # Lidar encoder
        lidar_points = config.get('lidar_points', 64)
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_points, 128),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.3)),
            nn.Linear(128, config.get('lidar_features', 32))
        )
        
        # Sensor encoder (GPS + Gyro)
        sensor_input = config.get('gps_dims', 2) + config.get('gyro_dims', 1)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_input, 32),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.3)),
            nn.Linear(32, config.get('sensor_features', 16))
        )
        
        # Fusion layers
        fusion_input = (
            config.get('cnn_features', 128) +
            config.get('lidar_features', 32) +
            config.get('sensor_features', 16)
        )
        
        fusion_hidden = config.get('fusion_hidden', [256, 128])
        layers = []
        prev_size = fusion_input
        
        for hidden_size in fusion_hidden:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.get('dropout_rate', 0.3)))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, config.get('output_size', 2)))
        layers.append(nn.Tanh())
        
        self.fusion_network = nn.Sequential(*layers)
    
    def forward(self, camera, lidar, sensors):
        """
        Forward pass.
        
        Args:
            camera: (batch, 3, 64, 64) - RGB camera image
            lidar: (batch, 64) - Lidar ranges
            sensors: (batch, 3) - [gps_x, gps_y, gyro_yaw]
        
        Returns:
            (batch, 2) - [steering, speed]
        """
        camera_feat = self.camera_encoder(camera)
        lidar_feat = self.lidar_encoder(lidar)
        sensor_feat = self.sensor_encoder(sensors)
        
        # Concatenate features
        fused = torch.cat([camera_feat, lidar_feat, sensor_feat], dim=1)
        
        # Fusion network
        output = self.fusion_network(fused)
        
        return output


class AutonomousDrivingNet(nn.Module):
    """Deep neural network for autonomous driving control (legacy flat MLP)."""
    
    def __init__(self, input_size=None, hidden_layers=None, output_size=None, dropout_rate=0.3):
        super(AutonomousDrivingNet, self).__init__()
        
        if input_size is None:
            input_size = 4106  # Old default
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        if output_size is None:
            output_size = 2
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def predict(self, sensor_input):
        """
        Make a prediction from sensor input.
        
        Args:
            sensor_input: Tensor of shape (1, input_size) or (batch_size, input_size)
        
        Returns:
            Tensor of predictions (steering_angle, speed_control)
        """
        with torch.no_grad():
            return self.forward(sensor_input)

def create_model(device='cpu', model_type=None):
    """Create and return the model."""
    if model_type is None:
        model_type = MODEL_CONFIG.get('model_type', 'cnn_multimodal')
    
    if model_type == 'cnn_multimodal':
        model = MultimodalFusionNet(MODEL_CONFIG)
    else:
        # Legacy flat MLP
        model = AutonomousDrivingNet(**MODEL_CONFIG)
    
    model.to(device)
    return model


def save_model(model, filepath):
    """Save model weights to file."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device='cpu'):
    """Load model from file."""
    model = create_model(device=device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model
