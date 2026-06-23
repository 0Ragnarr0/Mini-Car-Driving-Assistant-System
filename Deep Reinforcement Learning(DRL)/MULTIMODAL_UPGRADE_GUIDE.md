# Multimodal Sensor Architecture Upgrade

## Overview
The autonomous driving system has been upgraded to use multiple sensor modalities (camera, lidar, GPS, gyro) with a CNN + fusion architecture for improved driving behavior on complex scenarios (intersections, lane-keeping, traffic lights).

## Architecture Changes

### 1. Model Architecture (model.py)
**Old**: Flat MLP (4106 inputs → hidden layers → 2 outputs)
**New**: Multimodal Fusion Network with three specialized branches:

- **Camera Branch**: CNN encoder (64×64 RGB → 128 features)
  - 2 conv layers (32 → 64 → 128 channels)
  - Max pooling for spatial reduction
  - Fully connected output (128 features)

- **Lidar Branch**: MLP encoder (64 lidar ranges → 32 features)
  - Linear layer with ReLU + Dropout
  - Output features for fusion

- **Sensor Branch**: MLP encoder (GPS x,y + Gyro yaw → 16 features)
  - Linear layer with ReLU + Dropout
  - Combines positional and rotational data

- **Fusion Network**: Combines all branches
  - Concatenates features from all branches
  - 2-layer MLP with ReLU + Dropout + BatchNorm
  - Tanh output for [-1, 1] range

### 2. Data Format Changes (data_collector.py)
**Old Format**: Flat concatenated array (4096 camera + 10 lidar = 4106)
**New Format**: Structured multimodal dictionary per sample:
```
{
  'camera': (64, 64, 3) RGB image normalized [0, 1]
  'lidar': (64,) range measurements in meters
  'gps': (2,) position [x, y]
  'gyro': (1,) yaw rotation rate
}
```

File format for saved batches (.npz):
- `camera`: (N, 64, 64, 3) - N samples of RGB images
- `lidar`: (N, 64) - N samples of 64-point lidar
- `gps`: (N, 2) - N samples of GPS positions
- `gyro`: (N, 1) - N samples of yaw rates
- `labels`: (N, 2) - N samples of [steering, speed]

### 3. Controller Integration (nawnaw_robot.py)
- Sensor data collection now includes:
  - Raw lidar ranges from Sick LMS 291
  - GPS position from GPS device
  - Gyro rotation rates from InertialUnit
- Passes all sensor data to inference engine

### 4. Inference Pipeline (inference.py)
- New signature: `predict(camera_image, lidar_ranges, gps_data, gyro_data)`
- Internally:
  1. Processes camera to (64, 64, 3) RGB
  2. Resamples lidar to 64 points
  3. Extracts GPS [x, y] and gyro [z]
  4. Runs multimodal model
  5. Returns steering angle and speed control

### 5. Training Pipeline (trainer.py, dataset.py)
- Dataset loads multimodal inputs as tuples: `(camera, lidar, sensors), label`
- Train/validate loops unpack tuples and pass to model separately
- No more global scaler needed (CNN handles normalization)
- Batch loading respects new .npz format with separate camera/lidar/gps/gyro arrays

## Configuration (config.py)

**Updated MODEL_CONFIG**:
```python
MODEL_CONFIG = {
    'model_type': 'cnn_multimodal',
    'cnn_features': 128,           # Camera CNN output size
    'lidar_features': 32,          # Lidar encoder output size
    'sensor_features': 16,         # GPS+Gyro encoder output size
    'fusion_hidden': [256, 128],   # Fusion network hidden layers
    'output_size': 2,              # Steering + Speed
    'dropout_rate': 0.3,
    'camera_height': 64,
    'camera_width': 64,
    'camera_channels': 3,
    'lidar_points': 64,
    'gps_dims': 2,
    'gyro_dims': 1,
}
```

## Migration Steps

### For New Data Collection:
1. Start Webots with world "Auto-Driving System.wbt"
2. Controller nawnaw_robot will:
   - Set `use_model=False` (for manual control mode)
   - Set `collect_data=True` (to capture multimodal data)
   - Use manual control logic to drive the car
   - Collect camera, lidar, GPS, gyro samples
3. Samples automatically saved as new multimodal format in `training_data/batch_*.npz`

### For Training:
```bash
cd "Deep Reinforcement Learning(DRL)"
python trainer.py
```

The trainer will:
1. Load all batch_*.npz files from training_data/
2. Extract camera, lidar, gps, gyro arrays
3. Create train/val split
4. Train multimodal fusion model
5. Save best model as `autonomous_driving_model_epoch_*_loss_*.pth`

### For Inference:
```python
from inference import AutonomousDrivingInference

inference = AutonomousDrivingInference(device='cpu')
steering, speed = inference.predict(
    camera_image=fisheye_rgb,      # (H, W, 3)
    lidar_ranges=lidar_array,      # (N,)
    gps_data=[x, y],               # [2]
    gyro_data=[rx, ry, rz]         # [3]
)
```

## Expected Performance Improvements

1. **Lane Detection**: CNN branch processes spatial patterns in camera images
2. **Collision Avoidance**: Lidar branch provides precise distance measurements
3. **Navigation**: GPS branch provides absolute position context
4. **Steering Smoothness**: Gyro branch enables gyroscopic feedback control
5. **Better Generalization**: Fusion approach reduces overfitting to single sensor

## Backward Compatibility

- Old model checkpoints incompatible (different architecture)
- Old .npz batches (with `sensor_data` key) will be skipped during training
- To use old data: manually convert to new format or recollect

## Troubleshooting

**"Model loading error"**: Ensure model is trained with new architecture
**"Data shape mismatch"**: Verify batch files use new multimodal format
**"Inference producing zeros"**: Check camera/lidar data are being read correctly
**"Training not converging"**: May need more multimodal data samples for fusion learning

## Files Modified

- `model.py` - Added CNNEncoder, MultimodalFusionNet classes
- `data_collector.py` - Updated collect_sensor_data, add_sample, save_batch methods
- `inference.py` - Updated predict signature for multimodal inputs
- `trainer.py` - Updated train_epoch, validate for multimodal batches
- `dataset.py` - Updated AutonomousDrivingDataset, DataManager for multimodal format
- `controllers/nawnaw_robot.py` - Updated read_sensor_data, control_with_ai methods
- `config.py` - Updated MODEL_CONFIG with multimodal specs

## Next Steps

1. Collect multimodal training data (Phase 1: manual control)
2. Train new model with multimodal architecture
3. Validate inference on test scenarios
4. Fine-tune hyperparameters (fusion_hidden, dropout_rate, etc.)
5. Consider data augmentation for edge cases
