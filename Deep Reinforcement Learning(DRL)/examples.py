"""Example scripts for using the autonomous driving system."""

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from Deep_Reinforcement_Learning_DRL_.trainer import ModelTrainer
from Deep_Reinforcement_Learning_DRL_.inference import AutonomousDrivingInference
from Deep_Reinforcement_Learning_DRL_.utils import create_demo_training_data, Logger
from Deep_Reinforcement_Learning_DRL_.config import DATA_CONFIG
import torch


def example_training():
    """Example: Train the autonomous driving model."""
    print("\n" + "="*50)
    print("Example 1: Training the Model")
    print("="*50 + "\n")
    
    # Create demo training data
    print("Creating demo training data...")
    sensor_data, labels = create_demo_training_data(num_samples=5000)
    
    # Save demo data
    os.makedirs(DATA_CONFIG['data_dir'], exist_ok=True)
    np.savez_compressed(
        os.path.join(DATA_CONFIG['data_dir'], 'batch_0.npz'),
        sensor_data=sensor_data,
        labels=labels
    )
    print(f"Saved demo data with {len(sensor_data)} samples")
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    trainer = ModelTrainer(device=device)
    trainer.train(epochs=10)  # Use fewer epochs for demo
    
    print("\nTraining completed!")


def example_inference():
    """Example: Use the trained model for inference."""
    print("\n" + "="*50)
    print("Example 2: Running Inference")
    print("="*50 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize inference engine
    inference = AutonomousDrivingInference(device=device)
    
    # Create synthetic sensor data
    print("Creating synthetic sensor data...")
    depth_data = np.random.randint(0, 3000, (480, 640), dtype=np.uint16)
    fisheye_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Make prediction
    print("Making prediction...")
    steering_angle, speed_control = inference.predict(depth_data, fisheye_data)
    
    print(f"Prediction Results:")
    print(f"  Steering Angle: {steering_angle:.4f} radians ({np.degrees(steering_angle):.2f} degrees)")
    print(f"  Speed Control: {speed_control:.4f} m/s")


def example_batch_inference():
    """Example: Batch inference for multiple samples."""
    print("\n" + "="*50)
    print("Example 3: Batch Inference")
    print("="*50 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference = AutonomousDrivingInference(device=device)
    
    # Create batch of sensor data
    num_samples = 10
    input_size = 4106
    sensor_batch = np.random.rand(num_samples, input_size).astype(np.float32)
    
    print(f"Running batch inference on {num_samples} samples...")
    
    # Make batch predictions
    predictions = inference.predict_batch(sensor_batch)
    
    print("Batch Prediction Results:")
    print(f"  Shape: {predictions.shape}")
    print(f"  Mean Steering: {predictions[:, 0].mean():.4f} radians")
    print(f"  Mean Speed: {predictions[:, 1].mean():.4f} m/s")
    print(f"\nFirst 3 predictions:")
    for i in range(min(3, len(predictions))):
        print(f"  Sample {i+1}: Steering={predictions[i, 0]:.4f}, Speed={predictions[i, 1]:.4f}")


def example_data_collection():
    """Example: Data collection workflow."""
    print("\n" + "="*50)
    print("Example 4: Data Collection Workflow")
    print("="*50 + "\n")
    
    from Deep_Reinforcement_Learning_DRL_.data_collector import SensorDataCollector, DataAugmentation
    
    collector = SensorDataCollector(max_buffer_size=100)
    
    print("Simulating sensor data collection...")
    for i in range(50):
        # Simulate sensor data
        depth = np.random.randint(0, 3000, (480, 640), dtype=np.uint16)
        fisheye = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Collect processed data
        sensor_input = collector.collect_sensor_data(depth, fisheye)
        
        # Generate dummy labels
        steering = np.random.uniform(-1, 1)
        speed = np.random.uniform(0, 1)
        
        collector.add_sample(sensor_input, steering, speed)
    
    print(f"Buffer Status: {collector.get_buffer_status()}")
    
    # Save batch
    collector.save_batch()
    print("Batch saved successfully!")
    
    # Data augmentation example
    print("\nDemonstrating data augmentation...")
    sample_data = np.random.rand(10, 4106)
    sample_labels = np.random.rand(10, 2)
    
    augmented_data, augmented_labels = DataAugmentation.augment_batch(
        sample_data, sample_labels, num_augmentations=2
    )
    print(f"Original shape: {sample_data.shape}")
    print(f"Augmented shape: {augmented_data.shape}")


def example_system_workflow():
    """Example: Complete system workflow."""
    print("\n" + "="*50)
    print("Example 5: Complete System Workflow")
    print("="*50 + "\n")
    
    logger = Logger()
    
    logger.log("Starting complete autonomous driving workflow...")
    
    # Step 1: Data collection (simulated)
    logger.log("Step 1: Data Collection")
    sensor_data, labels = create_demo_training_data(num_samples=1000)
    logger.log(f"  Collected {len(sensor_data)} training samples")
    
    # Step 2: Save data
    logger.log("Step 2: Saving Training Data")
    os.makedirs(DATA_CONFIG['data_dir'], exist_ok=True)
    np.savez_compressed(
        os.path.join(DATA_CONFIG['data_dir'], 'batch_0.npz'),
        sensor_data=sensor_data,
        labels=labels
    )
    logger.log("  Training data saved")
    
    # Step 3: Train model
    logger.log("Step 3: Training Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(device=device)
    logger.log(f"  Using device: {device}")
    # (Model training would happen here)
    
    logger.log("Workflow completed successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Driving Examples')
    parser.add_argument('example', type=int, choices=[1, 2, 3, 4, 5],
                       help='Example number to run')
    
    args = parser.parse_args()
    
    examples = {
        1: example_training,
        2: example_inference,
        3: example_batch_inference,
        4: example_data_collection,
        5: example_system_workflow,
    }
    
    examples[args.example]()
