"""Project Structure and File Organization Guide."""

# Complete Autonomous Driving AI System - File Organization

## Directory Structure

```
c:\Users\user\u22-car(own)\
│
├── README.md                              # Project overview
├── requirements.txt                       # Python dependencies
├── SETUP_GUIDE.md                         # Installation and setup instructions
├── quickstart.py                          # Interactive quick start script
│
├── Deep Reinforcement Learning(DRL)/      # AI/ML Core System
│   ├── __init__.py                        # Package initialization
│   ├── config.py                          # Configuration and hyperparameters
│   ├── model.py                           # Neural network model definition
│   ├── dataset.py                         # Data loading and preprocessing
│   ├── trainer.py                         # Training loop and model training
│   ├── inference.py                       # Real-time inference engine
│   ├── data_collector.py                  # Sensor data collection utilities
│   ├── evaluation.py                      # Model evaluation and benchmarking
│   ├── utils.py                           # Helper utilities
│   ├── examples.py                        # Example scripts
│   └── README.md                          # DRL system documentation
│
├── controllers/
│   └── auto-driving/
│       └── auto-driving.py                # Webots controller with AI integration
│
├── Sensors/
│   └── camera.py                          # Camera sensor processing (existing)
│
├── training_data/                         # Training data storage (auto-created)
│   └── batch_*.npz                        # Training batches
│
├── trained_models/                        # Model storage (auto-created)
│   ├── autonomous_driving_model_final.pth
│   ├── autonomous_driving_model_epoch_*.pth
│   └── scaler.pkl
│
├── logs/                                  # Training logs (auto-created)
│   └── log_*.txt
│
├── worlds/
│   └── Auto-Driving System.wbt            # Webots simulation world
│
└── [Other existing folders...]
    ├── plugins/
    ├── Perception/
    ├── System_Test/
    └── ...
```

## Core Files Description

### 1. Configuration Layer

**File:** `Deep Reinforcement Learning(DRL)/config.py`
- **Purpose:** Centralized configuration management
- **Contents:**
  - Model architecture (layers, sizes, dropout)
  - Training parameters (learning rate, batch size, epochs)
  - Sensor specifications (resolution, ranges)
  - Data and storage paths
  - Webots simulation parameters

### 2. Model Layer

**File:** `Deep Reinforcement Learning(DRL)/model.py`
- **Purpose:** Neural network definition
- **Architecture:**
  - Input: 4106 features (4096 fisheye + 10 LIDAR)
  - Hidden: [512, 256, 128] with ReLU + Dropout + BatchNorm
  - Output: 2 values (steering, speed)
  - Activation: Tanh (normalized output)
- **Functions:**
  - `AutonomousDrivingNet`: Main model class
  - `create_model()`: Factory function
  - `save_model()`: Save weights
  - `load_model()`: Load pretrained weights

### 3. Data Layer

**File:** `Deep Reinforcement Learning(DRL)/dataset.py`
- **Purpose:** Data handling and preprocessing
- **Classes:**
  - `AutonomousDrivingDataset`: PyTorch Dataset
  - `DataManager`: Load, save, normalize data
- **Features:**
  - Train/validation split
  - StandardScaler normalization
  - PyTorch DataLoader integration
  - Batch storage (compressed .npz)

**File:** `Deep Reinforcement Learning(DRL)/data_collector.py`
- **Purpose:** Sensor data collection from simulation
- **Classes:**
  - `SensorDataCollector`: Collect and buffer sensor data
  - `DataAugmentation`: Augmentation techniques
- **Features:**
  - Depth camera processing (LIDAR extraction)
  - Fisheye camera processing (resize, normalize)
  - Noise augmentation
  - Batch management

### 4. Training Layer

**File:** `Deep Reinforcement Learning(DRL)/trainer.py`
- **Purpose:** Model training orchestration
- **Class:** `ModelTrainer`
- **Features:**
  - GPU support
  - Early stopping
  - Learning rate scheduling
  - Checkpoint saving
  - Loss tracking
- **Usage:**
  ```bash
  python trainer.py
  ```

### 5. Inference Layer

**File:** `Deep Reinforcement Learning(DRL)/inference.py`
- **Purpose:** Real-time predictions for control
- **Class:** `AutonomousDrivingInference`
- **Methods:**
  - `predict()`: Single sample prediction
  - `predict_batch()`: Multiple samples
  - `load_model()`: Load trained weights
  - `load_scaler()`: Load normalization scaler
- **Workflow:**
  1. Process sensor inputs
  2. Normalize with scaler
  3. Forward through model
  4. Denormalize outputs to real units

### 6. Webots Controller

**File:** `controllers/auto-driving/auto-driving.py`
- **Purpose:** Main Webots controller script
- **Class:** `AutonomousDrivingController`
- **Modes:**
  - Training: `use_model=False, collect_data=True`
  - Inference: `use_model=True, collect_data=False`
- **Functionality:**
  - Motor control (velocity-based)
  - Sensor reading and processing
  - AI-based decision making
  - Data collection for training
  - Fallback manual control

### 7. Evaluation Layer

**File:** `Deep Reinforcement Learning(DRL)/evaluation.py`
- **Purpose:** Model validation and benchmarking
- **Class:** `ModelEvaluator`
- **Metrics:**
  - MSE, RMSE, MAE
  - Per-output accuracy
  - Inference throughput
- **Usage:**
  ```bash
  python evaluation.py --model trained_models/autonomous_driving_model_final.pth
  python evaluation.py --benchmark
  ```

### 8. Utilities

**File:** `Deep Reinforcement Learning(DRL)/utils.py`
- **Purpose:** Helper functions and utilities
- **Classes:**
  - `Logger`: Logging with timestamps
  - `MetricsTracker`: Track training metrics
  - `ValidationUtils`: Data validation
  - `SimulationMonitor`: Simulation tracking
- **Functions:**
  - `create_demo_training_data()`: Generate synthetic data

**File:** `Deep Reinforcement Learning(DRL)/examples.py`
- **Purpose:** Example scripts and workflows
- **Examples:**
  1. Training workflow
  2. Inference demonstration
  3. Batch inference
  4. Data collection
  5. Complete system workflow
- **Usage:**
  ```bash
  python examples.py 1  # Train
  python examples.py 2  # Infer
  ```

### 9. Quick Start

**File:** `quickstart.py`
- **Purpose:** Interactive setup and demo
- **Features:**
  - Environment checking
  - Interactive menu
  - Demo mode
  - Example running
- **Usage:**
  ```bash
  python quickstart.py
  python quickstart.py --demo
  ```

## Data Flow Diagram

```
TRAINING PHASE:
┌─────────────────────────────────────────────────┐
│           Webots Simulation                     │
│  - Depth Camera → Virtual LIDAR (10 points)    │
│  - Fisheye Camera → 64x64 Grayscale            │
│  - Ground Truth: Steering, Speed               │
└──────────────┬──────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────┐
│      Data Collection (data_collector.py)        │
│  - Process sensors                             │
│  - Buffer samples                              │
│  - Save batches (training_data/batch_*.npz)    │
└──────────────┬──────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────┐
│      Data Preparation (dataset.py)             │
│  - Load batches                                │
│  - Normalize (StandardScaler)                  │
│  - Train/val split                             │
└──────────────┬──────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────┐
│    Model Training (trainer.py)                  │
│  - Define loss (MSE)                           │
│  - Optimize (Adam)                             │
│  - Early stopping                              │
│  - Save model (trained_models/)                │
└──────────────┬──────────────────────────────────┘
               │
               ↓
         TRAINED MODEL
         
INFERENCE PHASE:
┌─────────────────────────────────────────────────┐
│           Webots Simulation                     │
│  - Real-time sensor stream                     │
└──────────────┬──────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────┐
│   Data Collection (collect_sensor_data)        │
│  - Process depth → LIDAR                       │
│  - Process fisheye → image features            │
│  - Combine (4106 features)                     │
└──────────────┬──────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────┐
│      Inference (inference.py)                   │
│  - Normalize with scaler                       │
│  - Forward through model                       │
│  - Denormalize outputs                         │
│  - Return [steering, speed]                    │
└──────────────┬──────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────┐
│      Controller Application                     │
│  - Apply steering servo                        │
│  - Set motor velocities                        │
└─────────────────────────────────────────────────┘
```

## Workflow Stages

### Stage 1: Environment Setup
1. Install Python dependencies (`pip install -r requirements.txt`)
2. Verify PyTorch installation
3. Create necessary directories
4. Run `quickstart.py --setup-only`

### Stage 2: Data Collection
1. Configure Webots robot with required sensors
2. Set controller: `use_model=False, collect_data=True`
3. Run simulation
4. Collect diverse driving scenarios
5. Data saved to `training_data/batch_*.npz`

### Stage 3: Model Training
1. Verify training data exists
2. Run `python trainer.py`
3. Monitor loss metrics
4. Model saved to `trained_models/`
5. Scaler saved to `trained_models/scaler.pkl`

### Stage 4: Model Evaluation
1. Run `python evaluation.py`
2. Review performance metrics
3. Adjust model if needed
4. Benchmark throughput

### Stage 5: Deployment
1. Load trained model in controller
2. Set controller: `use_model=True, collect_data=False`
3. Run simulation with AI control
4. Monitor performance

## File Dependencies

```
quickstart.py
    ├── config.py
    ├── model.py
    ├── trainer.py
    ├── inference.py
    └── utils.py

auto-driving.py
    ├── config.py
    ├── inference.py
    ├── data_collector.py
    └── Sensors/camera.py

trainer.py
    ├── config.py
    ├── model.py
    └── dataset.py

data_collector.py
    ├── config.py
    ├── Sensors/camera.py
    └── dataset.py

inference.py
    ├── config.py
    ├── model.py
    └── data_collector.py
```

## Configuration Guide

See `config.py` for detailed configuration options:

- **TRAINING_CONFIG**: Learning rate, batch size, epochs
- **MODEL_CONFIG**: Architecture (layers, dropout)
- **SENSOR_CONFIG**: Resolution, ranges, normalization
- **DATA_CONFIG**: Storage paths
- **WEBOTS_CONFIG**: Simulation parameters

## Performance Metrics

**Training:**
- Loss: MSE between predictions and labels
- Metrics: MAE, RMSE per output
- Hardware: GPU acceleration (~10x faster)

**Inference:**
- Latency: ~5-10ms per sample on GPU
- Throughput: ~100+ samples/second
- Real-time capability: Yes for 32ms timestep

## Next Steps

1. Review SETUP_GUIDE.md for detailed setup
2. Run quickstart.py for interactive demo
3. Check examples.py for usage patterns
4. Collect training data in Webots
5. Train the model
6. Deploy for autonomous driving

## Support Resources

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Installation guide
- [Deep Reinforcement Learning(DRL)/README.md](Deep\ Reinforcement\ Learning(DRL)/README.md) - System documentation
- [examples.py](Deep\ Reinforcement\ Learning(DRL)/examples.py) - Usage examples
- [config.py](Deep\ Reinforcement\ Learning(DRL)/config.py) - Configuration options
