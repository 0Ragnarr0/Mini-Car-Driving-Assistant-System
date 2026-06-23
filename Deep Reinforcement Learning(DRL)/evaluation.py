"""Evaluation script for model validation."""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from Deep_Reinforcement_Learning_DRL_.model import load_model
from Deep_Reinforcement_Learning_DRL_.dataset import DataManager
from Deep_Reinforcement_Learning_DRL_.config import DATA_CONFIG
from Deep_Reinforcement_Learning_DRL_.utils import MetricsTracker


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self, model_path, device='cpu'):
        """Initialize evaluator."""
        self.device = device
        self.model = load_model(model_path, device=device)
        self.data_manager = DataManager()
        self.metrics = MetricsTracker()
    
    def evaluate_on_test_set(self):
        """Evaluate model on test set."""
        print("Evaluating model on test set...")
        
        # Load all data
        sensor_data, labels = self.data_manager.get_all_training_data()
        
        if sensor_data is None:
            print("No training data found!")
            return
        
        # Normalize data
        sensor_data = self.data_manager.scaler.transform(sensor_data)
        
        # Convert to tensors
        sensor_tensor = torch.from_numpy(sensor_data).float().to(self.device)
        labels_tensor = torch.from_numpy(labels).float().to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(sensor_tensor)
        
        # Calculate metrics
        predictions_np = predictions.cpu().numpy()
        labels_np = labels
        
        mse = mean_squared_error(labels_np, predictions_np)
        mae = mean_absolute_error(labels_np, predictions_np)
        rmse = np.sqrt(mse)
        
        # Per-output metrics
        steering_mae = mean_absolute_error(labels_np[:, 0], predictions_np[:, 0])
        speed_mae = mean_absolute_error(labels_np[:, 1], predictions_np[:, 1])
        
        print("\n" + "="*50)
        print("Model Evaluation Results")
        print("="*50)
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"\nPer-output metrics:")
        print(f"  Steering MAE: {steering_mae:.6f} radians")
        print(f"  Speed MAE: {speed_mae:.6f} m/s")
        print("="*50)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'steering_mae': steering_mae,
            'speed_mae': speed_mae
        }
    
    def evaluate_predictions(self, num_samples=100):
        """Evaluate specific predictions."""
        print(f"\nEvaluating {num_samples} predictions...")
        
        # Generate random sensor data
        sensor_data = np.random.rand(num_samples, 4106).astype(np.float32)
        
        # Normalize
        sensor_data = self.data_manager.scaler.transform(sensor_data)
        
        # Make predictions
        sensor_tensor = torch.from_numpy(sensor_data).float().to(self.device)
        
        with torch.no_grad():
            predictions = self.model(sensor_tensor)
        
        predictions_np = predictions.cpu().numpy()
        
        print(f"\nPrediction Statistics (n={num_samples}):")
        print(f"  Steering angle - Mean: {predictions_np[:, 0].mean():.4f}, "
              f"Std: {predictions_np[:, 0].std():.4f}")
        print(f"  Speed control - Mean: {predictions_np[:, 1].mean():.4f}, "
              f"Std: {predictions_np[:, 1].std():.4f}")
        
        return predictions_np


def benchmark_model(model_path, device='cpu'):
    """Benchmark model performance."""
    print("Starting model benchmark...")
    
    model = load_model(model_path, device=device)
    
    # Benchmark inference speed
    import time
    
    num_runs = 100
    batch_size = 32
    input_size = 4106
    
    sensor_data = torch.randn(batch_size, input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sensor_data)
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sensor_data)
    elapsed = time.time() - start
    
    samples_per_second = (num_runs * batch_size) / elapsed
    
    print("\n" + "="*50)
    print("Model Benchmark Results")
    print("="*50)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {samples_per_second:.1f} samples/second")
    print(f"Latency per sample: {(elapsed / (num_runs * batch_size)) * 1000:.2f}ms")
    print("="*50)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = args.model
    if model_path is None:
        model_path = os.path.join(DATA_CONFIG['models_dir'], 'autonomous_driving_model_final.pth')
    
    # Run evaluation
    if args.benchmark:
        benchmark_model(model_path, device=args.device)
    else:
        evaluator = ModelEvaluator(model_path, device=args.device)
        evaluator.evaluate_on_test_set()
        evaluator.evaluate_predictions(num_samples=100)
