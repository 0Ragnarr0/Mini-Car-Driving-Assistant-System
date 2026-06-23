"""Documentation and README for the autonomous driving AI system."""

# Autonomous Driving AI System

## Overview
This is a complete AI-based autonomous driving system that integrates with Webots simulation. The system uses deep neural networks trained on sensor data to control a simulated vehicle.

## Project Structure

```
├── Deep Reinforcement Learning(DRL)/
│   ├── config.py              # Configuration parameters
│   ├── model.py               # Neural network model definition
│   ├── dataset.py             # Dataset and data loader utilities
│   ├── trainer.py             # Training script
│   ├── inference.py           # Inference engine
│   ├── data_collector.py      # Data collection from Webots
│   ├── utils.py               # Utility functions
│   └── examples.py            # Example scripts
├── controllers/auto-driving/
│   └── auto-driving.py        # Webots controller
├── Sensors/
│   └── camera.py              # Camera data processing
├── requirements.txt           # Python dependencies
└── training_data/             # Training data storage
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. System Components

#### Configuration (`config.py`)
- Training hyperparameters
- Model architecture parameters
- Sensor specifications
- Data and storage paths

#### Model Architecture (`model.py`)
- Deep neural network with:
  - Input: Combined sensor data (fisheye camera 4096 + LIDAR 10 points = 4106)
  - Hidden layers: [512, 256, 128] with ReLU and dropout
  - Output: [steering_angle, speed_control]
  - Activation: Tanh for normalized output

#### Training Pipeline

1. **Data Collection** (`data_collector.py`)
   - Processes depth camera data into virtual LIDAR (10 points)
   - Converts fisheye to 64x64 grayscale
   - Normalizes and combines sensor inputs

2. **Dataset Handling** (`dataset.py`)
   - Loads and prepares training data
   - Splits into train/validation
   - Provides PyTorch DataLoaders
   - Applies data normalization

3. **Training** (`trainer.py`)
   ```bash
   python trainer.py
   ```
   - Trains the model using MSE loss
   - Implements early stopping
   - Saves checkpoints and best model
   - Supports GPU acceleration

#### Inference Engine (`inference.py`)
- Loads trained model
- Processes real-time sensor data
- Makes predictions for control commands
- Denormalizes outputs to real-world units

### 3. Data Collection Workflow

For collecting training data in Webots:

```python
controller = AutonomousDrivingController(use_model=False, collect_data=True)
controller.run()
```

This will:
- Record sensor data and control commands
- Save batches every 500 samples
- Store data in `training_data/batch_*.npz`

### 4. Training the Model

```bash
python Deep\ Reinforcement\ Learning\(DRL\)/trainer.py
```

### 5. Running Inference in Webots

Update the controller in `controllers/auto-driving/auto-driving.py`:

```python
controller = AutonomousDrivingController(use_model=True, collect_data=False)
controller.run()
```

### 5.5 True Deep Reinforcement Learning (Online)

Use the controller in RL mode to train directly from interaction (no behavior labels).

PowerShell:
```powershell
$env:NAWNAW_MODE="rl"
```

Optional runtime controls:
```powershell
# Resume from existing checkpoint prefix (without _actor.pth / _critic.pth)
$env:NAWNAW_RL_CKPT_PREFIX="trained_models/ddpg_autonomous_driving_ep_50"

# Reward profile: balanced | aggressive_avoidance | smooth_drive
$env:NAWNAW_RL_REWARD_PROFILE="aggressive_avoidance"
```

Then run the `nawnaw_robot` controller in Webots. This mode uses an online DDPG agent with:
- continuous action space: `[steering, speed]`
- replay buffer + target networks
- reward from forward progress, obstacle risk, smooth steering, and collision penalty

Checkpoints are saved into `trained_models/` as:
- `ddpg_autonomous_driving_ep_<N>_actor.pth`
- `ddpg_autonomous_driving_ep_<N>_critic.pth`

For deterministic policy evaluation (no exploration, no updates):
```powershell
$env:NAWNAW_MODE="rl_eval"
$env:NAWNAW_RL_CKPT_PREFIX="trained_models/ddpg_autonomous_driving_ep_50"
```

Evaluation episode count is controlled by `RL_CONFIG['eval_episodes']` in `config.py`.

### 6. Example Scripts

Run example workflows:

```bash
# Example 1: Training
python Deep\ Reinforcement\ Learning\(DRL\)/examples.py 1

# Example 2: Inference
python Deep\ Reinforcement\ Learning\(DRL\)/examples.py 2

# Example 3: Batch inference
python Deep\ Reinforcement\ Learning\(DRL\)/examples.py 3

# Example 4: Data collection
python Deep\ Reinforcement\ Learning\(DRL\)/examples.py 4

# Example 5: Complete workflow
python Deep\ Reinforcement\ Learning\(DRL\)/examples.py 5
```

## Model I/O

### Input Sensors
- **Fisheye Camera**: 480x640 RGB → 64x64 grayscale (4096 values)
- **Depth Camera**: 480x640 depth map → 10 virtual LIDAR points
- **Total Input**: 4106 features

### Output Control
- **Steering Angle**: [-1.57, 1.57] radians (-90° to +90°)
- **Speed Control**: [0, 5.0] m/s

## Training Workflow

1. **Data Collection Phase**
   - Run Webots with `collect_data=True`
   - Collect diverse driving scenarios
   - Save training batches

2. **Data Preparation**
   - Load all batches
   - Normalize with StandardScaler
   - Split into train/validation

3. **Model Training**
   - Use GPU if available
   - Monitor train/validation loss
   - Early stopping after 10 epochs without improvement
   - Learning rate reduction on plateau

4. **Model Deployment**
   - Load trained weights
   - Load scaler for normalization
   - Run inference in Webots controller

## Key Features

✓ Real-time sensor processing
✓ GPU acceleration support
✓ Data augmentation with noise
✓ Batch training with DataLoaders
✓ Early stopping and model checkpointing
✓ Logging and metrics tracking
✓ Webots integration
✓ Manual fallback control logic

## Performance Considerations

### Hardware Requirements
- GPU recommended (NVIDIA CUDA)
- Minimum: 4GB RAM
- For Webots: 8GB+ RAM recommended

### Optimization Tips
1. Use GPU for training
2. Batch size: 32 (adjust based on GPU memory)
3. Collect diverse training data (different scenarios)
4. Validate on different track configurations
5. Use data augmentation for robustness

## Troubleshooting

### No GPU Available
- Falls back to CPU automatically
- Training will be slower

### Model Not Found
- Make sure to train first or download pre-trained weights
- Check `trained_models/` directory

### Poor Inference Results
- Collect more training data
- Check sensor data normalization
- Validate training data quality
- Increase model capacity if needed

## Configuration Tuning

Edit `config.py` to adjust:
- Learning rate and batch size
- Model architecture (hidden layers)
- Data split ratios
- Sensor processing parameters

## Future Enhancements

- [ ] Convolutional layers for image processing
- [ ] LSTM for temporal dependencies
- [ ] Reinforcement learning with reward shaping
- [ ] Multi-agent training
- [ ] ROS integration
- [ ] Real-world deployment

## License

Internal project - auto-driving system development

## Contact

For issues or questions, check the system documentation or project proposals.
