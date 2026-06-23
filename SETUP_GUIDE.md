"""Setup and installation guide."""

# Setup Guide for Autonomous Driving AI System

## Prerequisites

- Python 3.8+
- Webots simulator installed
- NVIDIA GPU (optional but recommended)

## Installation Steps

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd c:\Users\user\u22-car(own)

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Check torch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}')"

# Check other dependencies
python -c "import numpy, cv2, sklearn; print('All dependencies installed!')"
```

### Step 3: Webots Configuration

1. Open Webots
2. Load the world: `worlds/Auto-Driving System.wbt`
3. Verify robot has:
   - Left and right motors
   - Depth camera (for 3D depth data)
   - Fisheye camera (omnidirectional RGB)
   - Distance sensors (for obstacle detection)
   - Optional: GPS and inertial unit

### Step 4: Data Collection Phase

For new projects without training data:

```bash
# Configure controller for data collection
# Edit controllers/auto-driving/auto-driving.py and set:
# controller = AutonomousDrivingController(use_model=False, collect_data=True)

# Run in Webots
# The controller will collect sensor data and save batches
```

Data will be saved to: `training_data/batch_*.npz`

### Step 5: Train the Model

```bash
cd Deep\ Reinforcement\ Learning\(DRL\)

# Train with default parameters
python trainer.py

# Training will:
# - Load all collected data from training_data/
# - Normalize and split into train/validation
# - Train for up to 100 epochs
# - Save best model and checkpoints
# - Save fitted scaler for inference
```

Trained models saved to: `trained_models/`

### Step 6: Deploy for Inference

```bash
# Update controller for inference mode
# Edit controllers/auto-driving/auto-driving.py and set:
# controller = AutonomousDrivingController(use_model=True, collect_data=False)

# Run in Webots
# The controller will use trained model for real-time control
```

## File Structure After Setup

```
trained_models/
├── autonomous_driving_model_final.pth
├── autonomous_driving_model_epoch_*.pth
└── scaler.pkl

training_data/
├── batch_0.npz
├── batch_1.npz
└── ...

logs/
└── log_*.txt
```

## Quick Test

### Test 1: Check Model Creation

```bash
python -c "
import sys
sys.path.insert(0, '.')
from Deep_Reinforcement_Learning_DRL_.model import create_model
model = create_model()
print(f'Model created: {type(model)}')
print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
"
```

### Test 2: Create Demo Data and Train

```bash
cd Deep\ Reinforcement\ Learning\(DRL\)
python examples.py 1
```

### Test 3: Run Inference on Demo Data

```bash
cd Deep\ Reinforcement\ Learning\(DRL\)
python examples.py 2
```

## Troubleshooting

### Issue: ImportError for PyTorch

**Solution:** 
```bash
# Install with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Webots Controller Not Starting

**Solution:**
1. Check Webots robot definitions match device names
2. Verify camera resolution matches expected dimensions
3. Check controller Python path is correct

### Issue: Out of Memory (OOM)

**Solution:**
- Reduce batch size in `config.py`: `'batch_size': 16`
- Reduce model size: smaller hidden layers
- Use CPU for training

### Issue: Poor Model Performance

**Solution:**
- Collect more diverse training data
- Increase training epochs
- Check data normalization
- Verify sensor data quality

## Performance Tuning

### For Faster Training

```python
# Edit config.py
TRAINING_CONFIG = {
    'learning_rate': 0.01,      # Increase learning rate
    'batch_size': 64,           # Increase batch size
    'epochs': 50,               # Reduce epochs initially
}
```

### For Better Accuracy

```python
# Edit config.py
MODEL_CONFIG = {
    'hidden_layers': [1024, 512, 256],  # Add layers
    'dropout_rate': 0.2,                # Reduce dropout
}
```

### For Real-time Performance

```python
# Edit config.py
MODEL_CONFIG = {
    'hidden_layers': [256, 128],        # Smaller model
    'dropout_rate': 0.5,                # More dropout
}
```

## Next Steps

1. Collect training data in realistic scenarios
2. Train the model
3. Evaluate performance
4. Iterate and improve

See [README.md](README.md) for detailed documentation.
