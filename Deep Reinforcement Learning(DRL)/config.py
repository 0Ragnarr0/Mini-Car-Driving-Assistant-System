"""Configuration parameters for autonomous driving AI model."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Training parameters
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'train_test_split': 0.8,
    'validation_split': 0.2,
}

# Model architecture parameters
MODEL_CONFIG = {
    'model_type': 'cnn_multimodal',  # CNN for images + MLP for numeric inputs
    'cnn_features': 128,  # Output features from CNN
    'lidar_features': 32,  # Lidar encoding size
    'sensor_features': 16,  # GPS + Gyro encoding size
    'fusion_hidden': [256, 128],  # Fusion network hidden layers
    'output_size': 2,  # Steering angle + Speed control
    'dropout_rate': 0.3,
    # Input specs
    'camera_height': 64,
    'camera_width': 64,
    'camera_channels': 3,  # RGB
    'lidar_points': 64,
    'gps_dims': 2,  # x, y position
    'gyro_dims': 1,  # yaw rate
}

# Sensor parameters
SENSOR_CONFIG = {
    'fisheye_resolution': (64, 64),
    'lidar_points': 10,
    'max_distance': 3.0,  # meters
    'max_speed': 5.0,  # m/s
    'max_steering_angle': 1.57,  # radians (~90 degrees)
}

# Data collection parameters
DATA_CONFIG = {
    'data_dir': os.path.join(PROJECT_ROOT, 'training_data'),
    'models_dir': os.path.join(PROJECT_ROOT, 'trained_models'),
    'logs_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'batch_collection_size': 1000,  # Collect 1000 samples per session
}

# Webots simulation parameters
WEBOTS_CONFIG = {
    'timestep': 32,  # milliseconds
    'physics_engine': 'ode',
    'max_simulation_steps': 50000,
}

# Reinforcement learning parameters (online training in Webots)
RL_CONFIG = {
    'gamma': 0.99,
    'tau': 0.005,
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    'batch_size': 128,
    'replay_size': 100000,
    'warmup_steps': 2000,
    'action_noise': 0.4,  # Increased from 0.15 for better exploration
    'hidden_dim': 256,
    'max_episode_steps': 1200,
    'save_every_episodes': 10,
    'eval_episodes': 5,
    'lidar_feature_bins': 12,
    'collision_distance_m': 0.25,
    'default_reward_profile': 'balanced',
    # Optional checkpoint prefix (without _actor.pth / _critic.pth suffix)
    # Example: trained_models/ddpg_autonomous_driving_ep_50
    'resume_checkpoint_prefix': '',
    'reward_profiles': {
        'balanced': {
            'base_reward': 0.10,
            'forward_weight': 0.60,
            'steering_penalty_weight': 0.08,
            'risk_penalty_weight': 0.50,
            'avoidance_bonus_weight': 0.25,
            'collision_penalty': 3.00,
            'lane_keeping_bonus_weight': 0.15,  # Reward for staying in lane
            'lane_violation_penalty': 0.50,  # Penalty for going off-road
        },
        'aggressive_avoidance': {
            'base_reward': 0.05,
            'forward_weight': 0.45,
            'steering_penalty_weight': 0.04,
            'risk_penalty_weight': 0.90,
            'avoidance_bonus_weight': 0.40,
            'collision_penalty': 4.00,
            'lane_keeping_bonus_weight': 0.10,
            'lane_violation_penalty': 0.40,
        },
        'smooth_drive': {
            'base_reward': 0.12,
            'forward_weight': 0.70,
            'steering_penalty_weight': 0.14,
            'risk_penalty_weight': 0.45,
            'avoidance_bonus_weight': 0.16,
            'collision_penalty': 3.20,
            'lane_keeping_bonus_weight': 0.20,  # Prioritize lane-keeping
            'lane_violation_penalty': 0.60,
        },
    },
}
