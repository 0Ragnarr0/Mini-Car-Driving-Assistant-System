"""Quick start script for autonomous driving system."""

import os
import sys
import argparse

# Add project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def setup_environment():
    """Check and setup environment."""
    print("Setting up environment...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ required")
        return False
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("✓ GPU not available (will use CPU)")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Check other dependencies
    try:
        import numpy
        import cv2
        import sklearn
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    
    # Create directories
    os.makedirs('training_data', exist_ok=True)
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    print("✓ Directories created")
    
    return True


def show_menu():
    """Show main menu."""
    print("\n" + "="*50)
    print("Autonomous Driving AI System")
    print("="*50)
    print("1. Check Environment")
    print("2. Create Demo Training Data")
    print("3. Train Model")
    print("4. Test Inference")
    print("5. Evaluate Model")
    print("6. Run Examples")
    print("7. Exit")
    print("="*50)
    return input("Select option: ")


def create_demo_data():
    """Create demo training data."""
    print("\nCreating demo training data...")
    from Deep_Reinforcement_Learning_DRL_.utils import create_demo_training_data
    from Deep_Reinforcement_Learning_DRL_.config import DATA_CONFIG
    import numpy as np
    
    num_samples = int(input("Number of samples (default 1000): ") or "1000")
    
    sensor_data, labels = create_demo_training_data(num_samples=num_samples)
    
    os.makedirs(DATA_CONFIG['data_dir'], exist_ok=True)
    np.savez_compressed(
        os.path.join(DATA_CONFIG['data_dir'], 'batch_0.npz'),
        sensor_data=sensor_data,
        labels=labels
    )
    print(f"✓ Created {len(sensor_data)} training samples")


def train_model():
    """Train the model."""
    print("\nTraining model...")
    import torch
    from Deep_Reinforcement_Learning_DRL_.trainer import ModelTrainer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = int(input("Number of epochs (default 100): ") or "100")
    
    trainer = ModelTrainer(device=device)
    trainer.train(epochs=epochs)
    print("✓ Training completed")


def test_inference():
    """Test inference."""
    print("\nTesting inference...")
    import torch
    import numpy as np
    from Deep_Reinforcement_Learning_DRL_.inference import AutonomousDrivingInference
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inference = AutonomousDrivingInference(device=device)
    
    # Create synthetic data
    depth = np.random.randint(0, 3000, (480, 640), dtype=np.uint16)
    fisheye = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Make prediction
    steering, speed = inference.predict(depth, fisheye)
    
    print(f"✓ Prediction made")
    print(f"  Steering: {steering:.4f} radians")
    print(f"  Speed: {speed:.4f} m/s")


def evaluate_model():
    """Evaluate model."""
    print("\nEvaluating model...")
    import torch
    from Deep_Reinforcement_Learning_DRL_.evaluation import ModelEvaluator
    from Deep_Reinforcement_Learning_DRL_.config import DATA_CONFIG
    import os
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(DATA_CONFIG['models_dir'], 'autonomous_driving_model_final.pth')
    
    if not os.path.exists(model_path):
        print("✗ Model not found. Please train first.")
        return
    
    evaluator = ModelEvaluator(model_path, device=device)
    evaluator.evaluate_on_test_set()


def run_examples():
    """Run example scripts."""
    print("\nRunning examples...")
    print("1. Training (with demo data)")
    print("2. Inference")
    print("3. Batch Inference")
    print("4. Data Collection")
    print("5. Complete Workflow")
    
    choice = input("Select example: ")
    
    os.system(f"cd Deep_Reinforcement_Learning_DRL_ && python examples.py {choice}")


def main():
    """Main menu loop."""
    parser = argparse.ArgumentParser(description='Autonomous Driving AI Quick Start')
    parser.add_argument('--no-menu', action='store_true', help='Skip menu')
    parser.add_argument('--setup-only', action='store_true', help='Only setup and exit')
    parser.add_argument('--demo', action='store_true', help='Run quick demo')
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_environment():
        print("\nEnvironment setup failed!")
        return
    
    if args.setup_only:
        print("\nSetup complete!")
        return
    
    if args.demo:
        print("\nRunning demo...")
        create_demo_data()
        train_model()
        test_inference()
        print("\nDemo completed!")
        return
    
    if args.no_menu:
        return
    
    # Main menu loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            if setup_environment():
                print("✓ Environment OK")
        elif choice == '2':
            create_demo_data()
        elif choice == '3':
            train_model()
        elif choice == '4':
            test_inference()
        elif choice == '5':
            evaluate_model()
        elif choice == '6':
            run_examples()
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid option")


if __name__ == '__main__':
    main()
