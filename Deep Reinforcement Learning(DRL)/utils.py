"""Utilities for autonomous driving system."""

import os
import json
import numpy as np
from datetime import datetime


class Logger:
    """Simple logging utility."""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir,
            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
    
    def log(self, message):
        """Log a message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')


class MetricsTracker:
    """Track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name, value):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_mean(self, name):
        """Get mean of a metric."""
        if name not in self.metrics:
            return None
        return np.mean(self.metrics[name])
    
    def get_stats(self, name):
        """Get statistics for a metric."""
        if name not in self.metrics:
            return None
        values = np.array(self.metrics[name])
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def save_metrics(self, filepath):
        """Save metrics to file."""
        stats = {}
        for name in self.metrics:
            stats[name] = self.get_stats(name)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


class ValidationUtils:
    """Validation utilities for training."""
    
    @staticmethod
    def validate_sensor_input(sensor_input, expected_size=4106):
        """Validate sensor input shape and values."""
        if len(sensor_input) != expected_size:
            raise ValueError(f"Expected sensor input size {expected_size}, got {len(sensor_input)}")
        
        if np.any(np.isnan(sensor_input)):
            raise ValueError("Sensor input contains NaN values")
        
        if np.any(np.isinf(sensor_input)):
            raise ValueError("Sensor input contains infinite values")
        
        return True
    
    @staticmethod
    def validate_label(label, steering_range=1.57, speed_range=5.0):
        """Validate label (ground truth) values."""
        steering, speed = label
        
        if abs(steering) > steering_range:
            print(f"Warning: Steering angle {steering} exceeds range")
        
        if abs(speed) > speed_range:
            print(f"Warning: Speed {speed} exceeds range")
        
        return True


class SimulationMonitor:
    """Monitor simulation state and performance."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.step_count = 0
        self.collisions = 0
        self.distance_traveled = 0.0
        self.avg_speed = 0.0
    
    def update_step(self):
        """Update step counter."""
        self.step_count += 1
    
    def record_collision(self):
        """Record a collision."""
        self.collisions += 1
    
    def update_position(self, position_delta):
        """Update distance traveled."""
        self.distance_traveled += position_delta
    
    def get_elapsed_time(self):
        """Get elapsed simulation time."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_stats(self):
        """Get simulation statistics."""
        elapsed = self.get_elapsed_time()
        return {
            'steps': self.step_count,
            'elapsed_time_seconds': elapsed,
            'collisions': self.collisions,
            'distance_traveled': self.distance_traveled,
            'average_speed': self.distance_traveled / max(elapsed, 0.1)
        }


def create_demo_training_data(num_samples=1000):
    """
    Create demo training data for testing.
    
    Args:
        num_samples: Number of training samples to generate
    
    Returns:
        Tuple of (sensor_data, labels)
    """
    input_size = 4106  # 4096 fisheye + 10 LIDAR points
    
    # Generate random sensor data
    sensor_data = np.random.rand(num_samples, input_size).astype(np.float32)
    
    # Generate corresponding labels (steering, speed)
    labels = np.random.uniform(-1, 1, (num_samples, 2)).astype(np.float32)
    
    return sensor_data, labels
