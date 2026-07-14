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
    'camera_height': 64,  # Processing resolution (downsampled from 256×256)
    'camera_width': 64,  # Processing resolution (downsampled from 256×256)
    'camera_channels': 3,  # RGB
    'lidar_points': 64,
    'gps_dims': 2,  # x, y position
    'gyro_dims': 1,  # yaw rate
}

# Sensor parameters
SENSOR_CONFIG = {
    'fisheye_resolution': (256, 256),  # Raw capture resolution (downsampled to 64×64 for processing)
    'fisheye_fov': 1.5708,  # Radians (90 degrees, wide automotive angle)
    'lidar_points': 10,
    'lidar_fov': 4.71,  # ~270 degrees (Sick LMS 291 standard)
    'lidar_max_range': 30.0,  # meters
    'distance_sensor_aperture': 1.2,  # Radians (wide side detection)
    'distance_sensor_range': 1.5,  # meters
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
    # TD3-specific (Twin Delayed DDPG) hyperparameters
    'policy_delay': 2,      # Update actor + target nets every N critic updates
    'target_noise': 0.2,    # Std of clipped noise added to target action (smoothing)
    'noise_clip': 0.5,      # Clip range for target-policy smoothing noise
    'warmup_steps': 200,  # Start learning before early collisions around step ~400
    'action_noise': 0.25,  # Base exploration noise (used as fallback)
    'action_noise_initial': 0.25,
    'action_noise_final': 0.05,
    'action_noise_decay_steps': 30000,
    'action_noise_steer_scale': 0.45,  # Steering exploration smaller than speed exploration
    'action_noise_speed_scale': 0.80,
    'max_steer_delta_per_step': 0.12,  # Normalized steering action change cap per step
    'max_speed_delta_per_step': 0.20,  # Normalized speed action change cap per step
    'early_episode_steer_limit': 0.35,  # Normalized action limit for first N steps in each episode
    'early_episode_speed_limit': 0.55,  # Normalized action limit for first N steps in each episode
    'early_episode_limit_steps': 250,
    'lane_assist_enabled': True,  # Confidence-gated steering assist on top of RL action
    'lane_assist_conf_threshold': 0.12,
    'lane_assist_kp': 0.85,
    'lane_assist_kd': 0.30,
    'lane_assist_blend_max': 0.45,  # Max blending factor with RL steering action
    'lane_assist_max_steer': 0.40,  # Normalized steering cap for assist output
    'lane_assist_speed_cap': 0.45,  # Cap speed action when lane offset is high
    'lane_bias_correction_enabled': True,  # Adaptive correction for persistent lane-offset bias
    'lane_bias_alpha': 0.02,  # EMA update rate for learned lane-center bias
    'lane_bias_max_abs': 0.35,  # Clamp bias correction magnitude
    'lane_bias_conf_threshold': 0.65,  # Update bias only on confident lane detections
    'lane_bias_update_max_steer': 0.12,  # Update bias only when steering is near straight
    'lane_bias_min_samples': 30,  # Minimum confident samples before applying correction
    'lane_seg_model_enabled': True,  # Optional TorchScript lane segmentation helper
    'lane_seg_model_path': '',  # Optional absolute/relative path to lane model (.ts/.pt)
    'lane_seg_threshold': 0.55,  # Probability threshold for lane mask binarization
    'hidden_dim': 256,
    'max_episode_steps': 1200,
    'save_every_episodes': 1,  # Save after EVERY episode to persist progress across runs
    'eval_episodes': 5,
    'lidar_feature_bins': 12,
    'collision_distance_m': 0.90,  # Soft collision threshold (can recover)
    'collision_distance_hard_m': 0.35,  # Hard collision threshold (end episode) - close physical contact
    'hard_collision_soft_streak_steps': 5,  # Escalate to hard collision faster for repeated scraping
    'default_reward_profile': 'balanced',
    # Optional checkpoint prefix (without _actor.pth / _critic.pth suffix)
    # Example: trained_models/ddpg_autonomous_driving_ep_50
    'resume_checkpoint_prefix': '',
    'reward_profiles': {
        'balanced': {
            'base_reward': 0.10,
            'forward_weight': 0.60,
            'steering_penalty_weight': 0.08,
            'risk_penalty_weight': 0.70,
            'avoidance_bonus_weight': 0.25,
            'collision_penalty': 5.00,
            'lane_keeping_bonus_weight': 0.15,  # Reward for staying in lane
            'lane_violation_penalty': 1.30,  # Strong penalty for poor lane-keeping
            'side_collision_penalty_weight': 0.60,  # Penalty for left/right side drift toward barriers
            'lane_conf_threshold': 0.15,
            'lane_center_tolerance': 0.20,
            'lane_edge_threshold': 0.45,
            'lane_violation_curve': 1.7,
            'no_lane_visible_penalty': 0.25,
            'hard_collision_penalty_multiplier': 3.0,
            'soft_collision_penalty_multiplier': 1.0,
            'repeat_collision_step_penalty': 0.25,
            'touch_collision_extra_penalty': 1.0,
        },
        'aggressive_avoidance': {
            'base_reward': 0.05,
            'forward_weight': 0.45,
            'steering_penalty_weight': 0.04,
            'risk_penalty_weight': 1.00,
            'avoidance_bonus_weight': 0.40,
            'collision_penalty': 6.00,
            'lane_keeping_bonus_weight': 0.10,
            'lane_violation_penalty': 1.40,
            'side_collision_penalty_weight': 0.50,  # Stronger side avoidance
            'lane_conf_threshold': 0.15,
            'lane_center_tolerance': 0.20,
            'lane_edge_threshold': 0.45,
            'lane_violation_curve': 1.8,
            'no_lane_visible_penalty': 0.30,
            'hard_collision_penalty_multiplier': 3.0,
            'soft_collision_penalty_multiplier': 1.2,
            'repeat_collision_step_penalty': 0.30,
            'touch_collision_extra_penalty': 1.2,
        },
        'smooth_drive': {
            'base_reward': 0.12,
            'forward_weight': 0.70,
            'steering_penalty_weight': 0.14,
            'risk_penalty_weight': 0.60,
            'avoidance_bonus_weight': 0.16,
            'collision_penalty': 4.80,
            'lane_keeping_bonus_weight': 0.20,  # Prioritize lane-keeping
            'lane_violation_penalty': 1.20,
            'side_collision_penalty_weight': 0.55,
            'lane_conf_threshold': 0.15,
            'lane_center_tolerance': 0.22,
            'lane_edge_threshold': 0.48,
            'lane_violation_curve': 1.7,
            'no_lane_visible_penalty': 0.25,
            'hard_collision_penalty_multiplier': 2.8,
            'soft_collision_penalty_multiplier': 0.9,
            'repeat_collision_step_penalty': 0.22,
            'touch_collision_extra_penalty': 1.0,
        },
    },
}

# Safety constraints for RL training (hard limits, not soft rewards)
SAFETY_CONFIG = {
    'enable_safety_checks': True,
    'max_speed_limit': 5.0,  # m/s hard limit (cannot exceed)
    'min_collision_distance': 0.35,  # meters (hard emergency brake) - matches hard collision threshold
    'max_steering_rate': 1.0,  # radians/step (prevent jerky steering)
    'max_episode_reward': 1000.0,  # Flag if episode reward > this (likely reward hacking)
    'min_episode_reward': -100.0,  # Flag if episode reward < this (trapped/crashing)
    'max_speed_for_turning': 2.0,  # m/s (reduce speed when steering >0.5)
    'emergency_brake_lidar_threshold': 0.35,  # Force brake if lidar < this
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
