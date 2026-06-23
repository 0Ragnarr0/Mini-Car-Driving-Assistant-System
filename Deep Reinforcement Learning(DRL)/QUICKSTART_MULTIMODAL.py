#!/usr/bin/env python3
"""
Quick-start guide for multimodal autonomous driving system.

This script provides step-by-step instructions and functions to:
1. Collect training data with multimodal sensors
2. Train the new CNN + fusion model
3. Test inference
"""

import os
import sys

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_step(number, title, description=""):
    """Print a formatted step."""
    print(f"\n[Step {number}] {title}")
    if description:
        print(f"  {description}")

def quickstart_phase1_data_collection():
    """Guide for Phase 1: Manual data collection."""
    print_section("PHASE 1: MULTIMODAL DATA COLLECTION")
    
    print_step(1, "Start Webots", 
        "Open Webots and load the world 'worlds/Auto-Driving System.wbt'")
    
    print_step(2, "Verify Controller",
        "The controller 'nawnaw_robot' should be assigned to the BmwX5 robot")
    
    print_step(3, "Enable Data Collection",
        "Edit 'controllers/nawnaw_robot/nawnaw_robot.py' and set:\n"
        "  use_model = False    # Use manual control\n"
        "  collect_data = True  # Collect multimodal sensor data")
    
    print_step(4, "Drive the Robot",
        "Use keyboard or joystick to manually control the robot:\n"
        "  - Drive through lanes\n"
        "  - Navigate intersections\n"
        "  - Make turns and adjust speed\n"
        "  - Drive at various speeds and angles")
    
    print_step(5, "Monitor Collection",
        "Watch the controller output for:\n"
        "  [Data Collection] Sample N collected\n"
        "  [Batch] Saved batch X samples")
    
    print_step(6, "Save Training Data",
        "Collected batches are automatically saved to 'training_data/batch_*.npz'\n"
        "Each batch contains multimodal data:\n"
        "  - camera (64×64 RGB images)\n"
        "  - lidar (64-point range measurements)\n"
        "  - gps (x, y positions)\n"
        "  - gyro (yaw rotation rates)")
    
    print("\n📊 Data Collection Tips:")
    print("  • Collect 500+ samples for meaningful training")
    print("  • Cover diverse scenarios: straight roads, curves, speed changes")
    print("  • Include edge cases: slow speeds, sharp turns, obstacles")
    print("  • Drive smoothly to generate clean steering/speed labels")

def quickstart_phase2_training():
    """Guide for Phase 2: Model training."""
    print_section("PHASE 2: TRAIN MULTIMODAL MODEL")
    
    print_step(1, "Prepare Environment",
        "Ensure you're in the project directory:\n"
        "  cd 'Deep Reinforcement Learning(DRL)'")
    
    print_step(2, "Verify Training Data",
        "Check that batch files exist:\n"
        "  ls ../training_data/batch_*.npz")
    
    print_step(3, "Start Training",
        "Run the trainer:\n"
        "  python trainer.py")
    
    print_step(4, "Monitor Training",
        "Watch for output:\n"
        "  --- Epoch 1/100 ---\n"
        "  Batch 1/N, Loss: X.XXXXXX\n"
        "  Train Loss: X.XXXXXX, Val Loss: X.XXXXXX")
    
    print_step(5, "Training Completion",
        "Model saves checkpoints during training:\n"
        "  trained_models/autonomous_driving_model_epoch_*.pth\n"
        "Best model kept based on validation loss")
    
    print("\n📈 Training Tips:")
    print("  • First training may take 5-10 minutes on CPU")
    print("  • Loss should decrease over epochs (convergence)")
    print("  • If loss increases, check learning rate (config.py)")
    print("  • Early stopping prevents overfitting")

def quickstart_phase3_inference():
    """Guide for Phase 3: Real-time inference."""
    print_section("PHASE 3: REAL-TIME INFERENCE")
    
    print_step(1, "Enable Model Inference",
        "Edit 'controllers/nawnaw_robot/nawnaw_robot.py' and set:\n"
        "  use_model = True     # Use AI model\n"
        "  collect_data = True  # (optional) collect more data for refinement")
    
    print_step(2, "Run Simulation",
        "Start Webots with the same world\n"
        "The robot should now drive autonomously using the trained model")
    
    print_step(3, "Monitor Performance",
        "Watch for:\n"
        "  [Controller] Model loaded successfully\n"
        "  [Inference] Steering: X.XX rad, Speed: X.XX m/s\n"
        "  Robot follows lanes and navigates intersections")
    
    print_step(4, "Evaluate Results",
        "Check if model successfully:\n"
        "  ✓ Keeps car in lanes\n"
        "  ✓ Navigates intersections\n"
        "  ✓ Maintains consistent speed\n"
        "  ✓ Responds to obstacles")
    
    print("\n🎯 Inference Tips:")
    print("  • Model uses multimodal inputs: camera + lidar + GPS + gyro")
    print("  • CNN branch handles visual lane detection")
    print("  • Lidar branch ensures collision avoidance")
    print("  • GPS branch provides position context")
    print("  • Gyro branch enables steering smoothness")

def quickstart_advanced_options():
    """Guide for advanced users."""
    print_section("ADVANCED OPTIONS")
    
    print_step(1, "Hyperparameter Tuning",
        "Edit 'Deep Reinforcement Learning(DRL)/config.py':\n"
        "  - Learning rate: default 0.001\n"
        "  - Batch size: default 32\n"
        "  - Epochs: default 100\n"
        "  - Dropout: default 0.3\n"
        "  - Fusion layers: [256, 128]")
    
    print_step(2, "Model Architecture",
        "Adjust multimodal network in 'model.py':\n"
        "  - CNN features: increase for better visual processing\n"
        "  - Lidar features: increase for finer distance sensing\n"
        "  - Sensor features: adjust GPS/gyro importance\n"
        "  - Fusion hidden: deeper network for complex tasks")
    
    print_step(3, "Data Augmentation",
        "Modify 'data_collector.py' to add:\n"
        "  - Gaussian noise to camera\n"
        "  - Range scaling to lidar\n"
        "  - Position jitter to GPS")
    
    print_step(4, "Inference Optimization",
        "For production deployment:\n"
        "  - Quantize model weights (reduce size)\n"
        "  - Export to ONNX format\n"
        "  - Test on target hardware")

def quickstart_troubleshooting():
    """Troubleshooting guide."""
    print_section("TROUBLESHOOTING")
    
    issues = [
        ("No training data collected", [
            "1. Check that collect_data=True in controller",
            "2. Verify Webots simulation is running",
            "3. Check sensor names match BmwX5 configuration",
            "4. Look for error messages in controller output"
        ]),
        ("Training loss not decreasing", [
            "1. Increase learning rate (config.py)",
            "2. Collect more diverse training data",
            "3. Check data normalization (camera/lidar ranges)",
            "4. Verify batch sizes aren't too large"
        ]),
        ("Model inference crashes", [
            "1. Ensure model is trained (not random weights)",
            "2. Check sensor data types match expected shapes",
            "3. Verify GPU/CPU device selection",
            "4. Look for shape mismatches in forward pass"
        ]),
        ("Robot drives erratically", [
            "1. Model may need more training data",
            "2. Collect data on similar road types",
            "3. Increase fusion network capacity",
            "4. Fine-tune learning rate during training"
        ]),
    ]
    
    for issue, solutions in issues:
        print(f"\n❌ Issue: {issue}")
        for solution in solutions:
            print(f"   {solution}")

def main():
    """Print the complete quick-start guide."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MULTIMODAL AUTONOMOUS DRIVING SYSTEM" + " " * 16 + "║")
    print("║" + " " * 22 + "Quick-Start Guide" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")
    
    quickstart_phase1_data_collection()
    quickstart_phase2_training()
    quickstart_phase3_inference()
    quickstart_advanced_options()
    quickstart_troubleshooting()
    
    print("\n" + "=" * 70)
    print("  System Ready - Begin with Phase 1: Data Collection")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
