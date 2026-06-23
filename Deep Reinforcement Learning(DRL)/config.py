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
    'batch_size': 64,  # Increased from 32 for more stable learning early
    'replay_size': 100000,
    'warmup_steps': 500,  # Reduced from 2000 - start learning early to prevent collision loops
    'action_noise': 0.4,  # Increased from 0.15 for better exploration
    'hidden_dim': 256,
    'max_episode_steps': 1200,
    'save_every_episodes': 1,  # Save after EVERY episode to persist progress across runs
    'eval_episodes': 5,
    'lidar_feature_bins': 12,
    'collision_distance_m': 1.5,  # Soft collision threshold (can recover) - warning zone at 1.5m
    'collision_distance_hard_m': 0.8,  # Hard collision threshold (end episode) - physical contact at 0.8m
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

# Safety constraints for RL training (hard limits, not soft rewards)
SAFETY_CONFIG = {
    'enable_safety_checks': True,
    'max_speed_limit': 5.0,  # m/s hard limit (cannot exceed)
    'min_collision_distance': 0.80,  # meters (hard emergency brake) - matches hard collision threshold
    'max_steering_rate': 1.0,  # radians/step (prevent jerky steering)
    'max_episode_reward': 1000.0,  # Flag if episode reward > this (likely reward hacking)
    'min_episode_reward': -100.0,  # Flag if episode reward < this (trapped/crashing)
    'max_speed_for_turning': 2.0,  # m/s (reduce speed when steering >0.5)
    'emergency_brake_lidar_threshold': 0.25,  # Force brake if lidar < this
    'allow_unsafe_actions': False,  # Disable actions that violate constraints
}

# Reward validation thresholds (detect anomalies)
REWARD_VALIDATION = {
    'enable_validation': True,
    'max_single_step_reward': 10.0,  # Flag if single step reward > this
    'episode_reward_ema_window': 10,  # Exponential moving average window
    'collision_rate_threshold': 0.5,  # Flag if collision_count/episode_count > 50%
    'episode_efficiency_min': 0.1,  # Min reward per step (detect low-quality episodes)
}
