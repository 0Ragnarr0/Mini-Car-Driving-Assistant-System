"""Webots controller for autonomous driving with AI model integration."""

import os
import sys
import importlib
import traceback
import faulthandler
import numpy as np

faulthandler.enable(all_threads=True)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
drl_path = os.path.join(project_root, 'Deep Reinforcement Learning(DRL)')
sys.path.insert(0, project_root)
sys.path.insert(0, drl_path)

controller_module = importlib.import_module('controller')
Robot = controller_module.Robot
Camera = controller_module.Camera
Motor = controller_module.Motor
DistanceSensor = controller_module.DistanceSensor

try:
    vehicle_module = importlib.import_module('vehicle')
    Driver = vehicle_module.Driver
except Exception:
    Driver = None

inference_module = importlib.import_module('inference')
data_collector_module = importlib.import_module('data_collector')
config_module = importlib.import_module('config')
rl_agent_module = importlib.import_module('rl_agent')

AutonomousDrivingInference = inference_module.AutonomousDrivingInference
SensorDataCollector = data_collector_module.SensorDataCollector
WEBOTS_CONFIG = config_module.WEBOTS_CONFIG
SENSOR_CONFIG = config_module.SENSOR_CONFIG
DATA_CONFIG = config_module.DATA_CONFIG
RL_CONFIG = getattr(config_module, 'RL_CONFIG', {})
DDPGAgent = rl_agent_module.DDPGAgent


class AutonomousDrivingController:
    """Main controller for autonomous driving in Webots."""
    
    def __init__(self, use_model=True, collect_data=False, rl_train=False, rl_eval=False):
        """
        Initialize the controller.
        
        Args:
            use_model: Whether to use trained AI model for control
            collect_data: Whether to collect data for training
            rl_train: Whether to run online deep reinforcement learning
            rl_eval: Whether to run deterministic RL policy evaluation
        """
        self.driver = Driver() if Driver is not None else None
        self.robot = self.driver if self.driver is not None else Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.use_model = use_model
        self.collect_data = collect_data
        self.rl_train = rl_train
        self.rl_eval = rl_eval
        self.step_count = 0
        self._device_map = self._discover_devices()
        
        # Initialize components
        self._setup_devices()
        self._setup_sensors()
        print(f"[Controller] Distance sensors found: {len(self.distance_sensors)} "
              f"({', '.join(ds.getName() for ds in self.distance_sensors) if self.distance_sensors else 'none'})")
        
        if self.use_model and not self.rl_train and not self.rl_eval:
            self._setup_ai_model()
        if self.rl_train or self.rl_eval:
            self._setup_rl_agent()
        
        if self.collect_data:
            self.data_collector = SensorDataCollector()

    def _discover_devices(self):
        """Discover available device names once to avoid noisy failed lookups."""
        device_map = {}
        if not hasattr(self.robot, 'getNumberOfDevices') or not hasattr(self.robot, 'getDeviceByIndex'):
            return device_map

        try:
            count = self.robot.getNumberOfDevices()
        except Exception:
            return device_map

        for i in range(count):
            try:
                device = self.robot.getDeviceByIndex(i)
                name = device.getName()
                if name:
                    device_map[name.lower()] = name
            except Exception:
                pass

        if device_map:
            print(f"[Controller] Discovered devices: {', '.join(sorted(device_map.values()))}")
        return device_map

    def _get_device_if_exists(self, candidates):
        """Return the first available device among candidate names."""
        for candidate in candidates:
            actual_name = self._device_map.get(candidate.lower())
            if actual_name is not None:
                try:
                    return self.robot.getDevice(actual_name)
                except Exception:
                    pass
        return None
    
    def _find_latest_checkpoint(self, model_prefix):
        """Find the latest checkpoint matching model_prefix pattern."""
        try:
            model_dir = os.path.dirname(model_prefix)
            if not os.path.isdir(model_dir):
                return ""
            
            model_base = os.path.basename(model_prefix)
            # Look for files like ddpg_autonomous_driving_ep_10_actor.pth
            import glob
            actor_files = glob.glob(f"{model_prefix}_ep_*_actor.pth")
            if not actor_files:
                return ""
            
            # Extract episode numbers and find the highest
            episodes = []
            for f in actor_files:
                try:
                    # Extract "_ep_NUMBER_" from filename
                    import re
                    match = re.search(r'_ep_(\d+)_', f)
                    if match:
                        episodes.append((int(match.group(1)), f))
                except:
                    pass
            
            if episodes:
                latest_ep, _ = max(episodes, key=lambda x: x[0])
                latest_prefix = f"{model_prefix}_ep_{latest_ep}"
                return latest_prefix
        except Exception as e:
            pass
        return ""
    
    def _setup_devices(self):
        """Setup robot motors and devices."""
        if self.driver is not None:
            self.left_motor = None
            self.right_motor = None
            self.steering_servo = None
            self.brake = None
            return

        # Get left and right motors
        try:
            self.left_motor = self.robot.getDevice('left motor')
            self.right_motor = self.robot.getDevice('right motor')
        except Exception:
            self.left_motor = None
            self.right_motor = None
        
        # Set motors to velocity mode
        if self.left_motor is not None and self.right_motor is not None:
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        
        # Optional: Get additional actuators
        try:
            self.steering_servo = self.robot.getDevice('steering')
        except:
            self.steering_servo = None
        
        try:
            self.brake = self.robot.getDevice('brake')
        except:
            self.brake = None
    
    def _setup_sensors(self):
        """Setup and enable sensors."""
        self.depth_camera = None
        self.fisheye_camera = None
        self.lidar = None

        # Try standard names first, then fall back to world device names.
        self.depth_camera = self._get_device_if_exists(['depth camera', 'Depth camera', 'range-finder'])
        if self.depth_camera is not None:
            self.depth_camera.enable(self.timestep)

        self.fisheye_camera = self._get_device_if_exists(['fisheye camera', 'camera', 'Camera'])
        if self.fisheye_camera is not None:
            self.fisheye_camera.enable(self.timestep)

        self.lidar = self._get_device_if_exists(['Sick LMS 291', 'sick lms 291', 'lidar', 'LDS-01'])
        if self.lidar is not None:
            self.lidar.enable(self.timestep)
        
        # Optional: Distance sensors
        self.distance_sensors = []
        for actual_name in self._device_map.values():
            if 'distance sensor' in actual_name.lower():
                try:
                    ds = self.robot.getDevice(actual_name)
                    ds.enable(self.timestep)
                    self.distance_sensors.append(ds)
                except Exception:
                    pass
        
        # Optional: GPS/IMU
        self.gps = self._get_device_if_exists(['gps', 'GPS'])
        if self.gps is not None:
            self.gps.enable(self.timestep)

        # Prefer InertialUnit when available, otherwise fall back to Gyro.
        self.inertial_unit = self._get_device_if_exists(['inertial unit', 'InertialUnit'])
        self.gyro_sensor = None
        if self.inertial_unit is not None:
            self.inertial_unit.enable(self.timestep)
        else:
            self.gyro_sensor = self._get_device_if_exists(['gyro', 'Gyro'])
            if self.gyro_sensor is not None:
                self.gyro_sensor.enable(self.timestep)
    
    def _setup_ai_model(self):
        """Setup the AI inference engine."""
        print("[Controller] Initializing AI model...")
        device = 'cuda' if self._has_gpu() else 'cpu'
        self.inference = AutonomousDrivingInference(device=device)
        print(f"[Controller] AI model loaded on device: {device}")

    def _setup_rl_agent(self):
        """Setup DDPG agent and RL episode state."""
        print("[Controller] Initializing RL agent (DDPG)...")
        device = 'cuda' if self._has_gpu() else 'cpu'

        lidar_bins = int(RL_CONFIG.get('lidar_feature_bins', 12))
        self.rl_state_dim = lidar_bins + 6  # lidar bins + distance stats + speed + yaw + previous steering
        self.rl_agent = DDPGAgent(
            state_dim=self.rl_state_dim,
            action_dim=2,
            config=RL_CONFIG,
            device=device,
        )

        self.rl_episode_idx = 0
        self.rl_episode_step = 0
        self.rl_episode_reward = 0.0
        self.rl_episode_collision_count = 0  # Track collisions per episode
        self.rl_max_episode_steps = int(RL_CONFIG.get('max_episode_steps', 1200))
        self.rl_collision_distance = float(RL_CONFIG.get('collision_distance_m', 0.25))
        self.rl_collision_distance_hard = float(RL_CONFIG.get('collision_distance_hard_m', 0.10))  # Hard collision threshold
        self.rl_last_steering = 0.0
        self.rl_last_losses = None
        self.rl_save_every = int(RL_CONFIG.get('save_every_episodes', 10))
        self.rl_eval_episodes = int(RL_CONFIG.get('eval_episodes', 5))
        self.rl_model_prefix = os.path.join(DATA_CONFIG['models_dir'], 'ddpg_autonomous_driving')

        # Reward profile selection can be overridden at runtime.
        profile_name = os.environ.get(
            'NAWNAW_RL_REWARD_PROFILE',
            RL_CONFIG.get('default_reward_profile', 'balanced')
        ).strip().lower()
        reward_profiles = RL_CONFIG.get('reward_profiles', {})
        if profile_name not in reward_profiles:
            print(f"[Controller] Unknown reward profile '{profile_name}', using 'balanced'")
            profile_name = 'balanced'
        self.rl_reward_profile_name = profile_name
        self.rl_reward_profile = reward_profiles.get(profile_name, {})

        # Optional checkpoint resume for training or required checkpoint for evaluation.
        ckpt_prefix = os.environ.get('NAWNAW_RL_CKPT_PREFIX', '').strip()
        if not ckpt_prefix:
            ckpt_prefix = str(RL_CONFIG.get('resume_checkpoint_prefix', '')).strip()
        
        # Auto-find latest checkpoint if none specified
        if not ckpt_prefix and self.rl_train:
            ckpt_prefix = self._find_latest_checkpoint(self.rl_model_prefix)
        
        self.rl_resume_prefix = ckpt_prefix

        if ckpt_prefix:
            actor_path = f"{ckpt_prefix}_actor.pth"
            critic_path = f"{ckpt_prefix}_critic.pth"
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                try:
                    self.rl_agent.load(ckpt_prefix)
                    print(f"[Controller] ✓ RESUMED RL checkpoint: {ckpt_prefix}")
                except Exception as exc:
                    print(f"[Controller] Failed to load checkpoint '{ckpt_prefix}': {exc}")
            else:
                print(f"[Controller] RL checkpoint files not found for prefix: {ckpt_prefix}")
        else:
            if self.rl_train:
                print("[Controller] Starting fresh (no checkpoint found)")
            else:
                print("[Controller] RL evaluation mode has no checkpoint prefix; policy will be untrained")

        print(f"[Controller] RL agent ready on device: {device} (state_dim={self.rl_state_dim})")
        print(f"[Controller] RL reward profile: {self.rl_reward_profile_name}")
    
    def _has_gpu(self):
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def read_sensor_data(self):
        """Read all sensor data from robot."""
        # Depth source (preferred: depth camera; fallback: lidar -> pseudo depth map)
        depth_data = None
        if self.depth_camera is not None:
            try:
                depth_image = self.depth_camera.getImage()
                width = self.depth_camera.getWidth()
                height = self.depth_camera.getHeight()
                depth_data = np.frombuffer(depth_image, np.uint16).reshape((height, width))
            except Exception:
                depth_data = None
        
        if depth_data is None and self.lidar is not None and hasattr(self.lidar, 'getRangeImage'):
            try:
                ranges = np.array(self.lidar.getRangeImage(), dtype=np.float32)
                ranges = np.clip(ranges, 0.0, 30.0)
                # Convert meters to millimeters and tile as pseudo 2D depth map.
                row_mm = (ranges * 1000.0).astype(np.uint16)
                depth_data = np.tile(row_mm, (32, 1))
            except Exception:
                depth_data = None

        if depth_data is None:
            depth_data = np.zeros((32, 64), dtype=np.uint16)

        # RGB source (preferred: fisheye; fallback: regular camera)
        fisheye_data = None
        if self.fisheye_camera is not None:
            try:
                fisheye_image = self.fisheye_camera.getImage()
                f_width = self.fisheye_camera.getWidth()
                f_height = self.fisheye_camera.getHeight()
                fisheye_data = np.frombuffer(fisheye_image, np.uint8).reshape((f_height, f_width, 4))[:, :, :3]
            except Exception:
                fisheye_data = None

        if fisheye_data is None:
            fisheye_data = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Get distance sensor readings
        distances = []
        for ds in self.distance_sensors:
            distances.append(ds.getValue())
        
        # Get lidar ranges
        lidar_ranges = None
        if self.lidar is not None and hasattr(self.lidar, 'getRangeImage'):
            try:
                lidar_ranges = np.array(self.lidar.getRangeImage(), dtype=np.float32)
            except Exception:
                lidar_ranges = None
        
        # Get GPS position
        gps_data = None
        if self.gps is not None:
            try:
                gps_data = self.gps.getValues()  # [x, y, z]
            except Exception:
                gps_data = None
        
        # Get gyro/IMU data (rotation rates)
        gyro_data = None
        if self.inertial_unit is not None:
            try:
                gyro_data = self.inertial_unit.getValues()  # [roll, pitch, yaw]
            except Exception:
                gyro_data = None
        elif self.gyro_sensor is not None:
            try:
                gyro_data = self.gyro_sensor.getValues()  # [roll, pitch, yaw]
            except Exception:
                gyro_data = None

        imu_data = None
        if self.inertial_unit is not None:
            try:
                imu_data = self.inertial_unit.getValues()
            except Exception:
                imu_data = None
        
        return {
            'depth': depth_data,
            'fisheye': fisheye_data,
            'distances': distances,
            'gps': gps_data,
            'imu': imu_data,
            'lidar': lidar_ranges,
            'gyro': gyro_data,
        }
    
    def control_with_ai(self, sensor_data):
        """
        Control robot using AI model with multimodal inputs.
        
        Args:
            sensor_data: Dictionary containing sensor readings
        
        Returns:
            Tuple of (steering_angle, speed_control)
        """
        steering_angle, speed_control = self.inference.predict(
            camera_image=sensor_data['fisheye'],
            lidar_ranges=sensor_data.get('lidar'),
            gps_data=sensor_data.get('gps'),
            gyro_data=sensor_data.get('gyro')
        )
        return steering_angle, speed_control
    
    def control_with_manual_logic(self, sensor_data):
        """
        Basic manual control logic (fallback).
        
        Args:
            sensor_data: Dictionary containing sensor readings
        
        Returns:
            Tuple of (steering_angle, speed_control)
        """
        # Simple obstacle avoidance
        distances = sensor_data['distances']
        if not distances:
            return 0.0, 1.0
        
        # Check distances
        min_distance = min(distances) if distances else 10.0
        
        if min_distance < 0.3:
            # Obstacle detected
            return np.sign(distances[-1] - distances[0]) * 0.5, -0.5
        else:
            # Go straight
            return 0.0, 1.0
    
    def apply_control(self, steering_angle, speed_control):
        """
        Apply control commands to robot.
        
        Args:
            steering_angle: Target steering angle in radians
            speed_control: Target speed in m/s (normalized -1 to 1)
        """
        # Driver API path (BmwX5 / car models)
        if self.driver is not None:
            steering = float(np.clip(steering_angle, -0.5, 0.5))
            # Convert normalized speed to m/s then to km/h for Driver API.
            speed_ms = float((speed_control + 1) / 2 * 8.0)
            speed_kmh = max(0.0, speed_ms * 3.6)
            self.driver.setSteeringAngle(steering)
            self.driver.setCruisingSpeed(speed_kmh)
            return

        # Apply steering
        if self.steering_servo:
            self.steering_servo.setPosition(steering_angle)
        
        # Apply speed control to motors
        # Convert speed_control [-1, 1] to motor velocities
        base_velocity = (speed_control + 1) / 2 * 5.0  # Scale to [0, 5] m/s
        
        # Apply differential drive for steering
        turn_radius = max(0.1, abs(steering_angle))
        left_speed = base_velocity * (1 - abs(steering_angle) * 0.5)
        right_speed = base_velocity * (1 - abs(steering_angle) * 0.5)
        
        if steering_angle > 0:  # Turn right
            right_speed *= (1 - abs(steering_angle))
        elif steering_angle < 0:  # Turn left
            left_speed *= (1 - abs(steering_angle))
        
        if self.left_motor is not None and self.right_motor is not None:
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
    
    def collect_training_data(self, sensor_data, steering_angle, speed_control):
        """
        Collect data for training.
        
        Args:
            sensor_data: Current sensor readings
            steering_angle: Ground truth steering angle
            speed_control: Ground truth speed control
        """
        if not self.collect_data:
            return
        
        sensor_input = self.data_collector.collect_sensor_data(
            camera_image=sensor_data['fisheye'],
            lidar_ranges=sensor_data.get('lidar'),
            gps_data=sensor_data.get('gps'),
            gyro_data=sensor_data.get('gyro')
        )
        
        self.data_collector.add_sample(
            sensor_input,
            steering_angle,
            speed_control
        )
        
        # Save batch periodically
        if len(self.data_collector.sensor_buffer) >= 500:
            self.data_collector.save_batch()
            print(f"[Controller] Data batch saved. Status: {self.data_collector.get_buffer_status()}")

    def _build_rl_state(self, sensor_data):
        """Build compact RL state vector from multimodal sensors."""
        lidar_bins = int(RL_CONFIG.get('lidar_feature_bins', 12))
        lidar_max_range = max(5.0, float(SENSOR_CONFIG.get('max_distance', 30.0)))

        if sensor_data.get('lidar') is not None and len(sensor_data['lidar']) > 0:
            lidar = np.asarray(sensor_data['lidar'], dtype=np.float32)
            lidar = np.clip(lidar, 0.0, lidar_max_range)
            sample_idx = np.linspace(0, len(lidar) - 1, lidar_bins).astype(np.int32)
            lidar_feat = lidar[sample_idx] / lidar_max_range
        else:
            lidar_feat = np.ones(lidar_bins, dtype=np.float32)

        distances = sensor_data.get('distances', [])
        if distances:
            d = np.asarray(distances, dtype=np.float32)
            # Saturate unknown sensor units into a stable [0, 1] range.
            d_norm = np.tanh(np.clip(d, 0.0, 2000.0) / 500.0)
            d_stats = np.array([d_norm.min(), d_norm.mean(), d_norm.max()], dtype=np.float32)
        else:
            d_stats = np.zeros(3, dtype=np.float32)

        speed_norm = 0.0
        if self.driver is not None and hasattr(self.driver, 'getCurrentSpeed'):
            try:
                speed_kmh = float(self.driver.getCurrentSpeed())
                speed_ms = max(0.0, speed_kmh / 3.6)
                speed_norm = np.clip(speed_ms / max(0.1, float(SENSOR_CONFIG.get('max_speed', 5.0))), 0.0, 1.0)
            except Exception:
                speed_norm = 0.0

        yaw_norm = 0.0
        gyro = sensor_data.get('gyro')
        if gyro is not None and len(gyro) >= 3:
            yaw_norm = float(np.clip(np.asarray(gyro, dtype=np.float32)[2] / 5.0, -1.0, 1.0))

        state = np.concatenate(
            [
                lidar_feat.astype(np.float32),
                d_stats,
                np.array([speed_norm, yaw_norm, self.rl_last_steering], dtype=np.float32),
            ],
            axis=0,
        )
        return state.astype(np.float32)

    def _detect_lane_from_camera(self, camera_image):
        """
        Detect lane markings from camera image (white/yellow pixels).
        Returns lateral offset from lane center (-1 to +1).
        """
        try:
            if camera_image is None or camera_image.size == 0:
                return 0.0  # No detection = assume centered
            
            img = np.asarray(camera_image, dtype=np.float32)
            if img.shape != (64, 64, 3):
                return 0.0
            
            # Look for white/yellow lane markings in lower half of image (forward view)
            # Webots fisheye points forward-down, so use bottom rows
            roi_start = 32  # Bottom half
            roi = img[roi_start:, :, :]  # Shape: (32, 64, 3)
            
            # White: R>200, G>200, B>200
            # Yellow: R>200, G>200, B<100
            white_mask = (roi[:, :, 0] > 180) & (roi[:, :, 1] > 180) & (roi[:, :, 2] > 180)
            yellow_mask = (roi[:, :, 0] > 180) & (roi[:, :, 1] > 180) & (roi[:, :, 2] < 100)
            lane_mask = white_mask | yellow_mask
            
            if np.sum(lane_mask) < 10:  # Too few pixels detected
                return 0.0
            
            # Find weighted center column of lane markings
            cols = np.arange(64)
            weights = np.sum(lane_mask, axis=0)  # Sum per column
            weighted_col = np.sum(cols * weights) / np.sum(weights)
            
            # Normalize to [-1, +1]: center (32) = 0, left (0) = -1, right (63) = +1
            lateral_offset = (weighted_col - 32.0) / 32.0
            return float(np.clip(lateral_offset, -1.0, 1.0))
        except Exception:
            return 0.0

    def _compute_rl_reward_done(self, sensor_data, action):
        """Compute reward and terminal condition from current observation and action."""
        steer_cmd = float(action[0])
        speed_cmd = float(action[1])

        lidar_min = 30.0
        if sensor_data.get('lidar') is not None and len(sensor_data['lidar']) > 0:
            lidar_min = float(np.min(np.asarray(sensor_data['lidar'], dtype=np.float32)))

        dist_min = 10.0
        distances = sensor_data.get('distances', [])
        if distances:
            dist_min = float(np.min(np.asarray(distances, dtype=np.float32)))

        cfg = self.rl_reward_profile
        base_reward = float(cfg.get('base_reward', 0.10))
        forward_weight = float(cfg.get('forward_weight', 0.60))
        steering_penalty_weight = float(cfg.get('steering_penalty_weight', 0.08))
        risk_penalty_weight = float(cfg.get('risk_penalty_weight', 0.50))
        avoidance_bonus_weight = float(cfg.get('avoidance_bonus_weight', 0.25))
        collision_penalty = float(cfg.get('collision_penalty', 3.00))
        lane_keeping_bonus_weight = float(cfg.get('lane_keeping_bonus_weight', 0.15))
        lane_violation_penalty = float(cfg.get('lane_violation_penalty', 0.50))

        forward_reward = max(0.0, (speed_cmd + 1.0) * 0.5)
        steering_penalty = steering_penalty_weight * abs(steer_cmd)

        obstacle_risk = max(0.0, 1.0 - (lidar_min / 1.5))
        risk_penalty = risk_penalty_weight * obstacle_risk * max(0.0, (speed_cmd + 1.0) * 0.5)
        avoidance_bonus = avoidance_bonus_weight * obstacle_risk * abs(steer_cmd)

        # Vision-based lane detection: bonus for staying within lane markings
        # Detect lane from camera image (white/yellow road markings)
        lane_keeping_bonus = 0.0
        lane_violation = 0.0
        
        camera_image = sensor_data.get('fisheye')
        lateral_offset = self._detect_lane_from_camera(camera_image)
        
        # Lateral offset from lane center: [-1, +1]
        # Reward for small lateral offset (near center): abs(offset) < 0.3
        # Penalty for large offset (near lane edge): abs(offset) > 0.6
        abs_offset = abs(lateral_offset)
        if abs_offset < 0.3:
            # Near lane center - good!
            lane_keeping_bonus = lane_keeping_bonus_weight * (1.0 - abs_offset / 0.3)
        elif abs_offset > 0.6:
            # Near lane edge - warning!
            lane_violation = lane_violation_penalty * (abs_offset - 0.6) / 0.4
        
        # Fallback: Also check distance sensors (left/right) in case lane detection fails
        if distances and len(distances) >= 2:
            side_distances = np.asarray(distances, dtype=np.float32)
            side_min = float(np.min(side_distances))
            # Hard penalty for going off-road completely (< 0.2m to boundary)
            if side_min < 0.2:
                lane_violation += lane_violation_penalty * 0.5

        reward = (
            base_reward 
            + forward_weight * forward_reward 
            + avoidance_bonus 
            + lane_keeping_bonus
            - steering_penalty 
            - risk_penalty 
            - lane_violation
        )

        # Two-tier collision detection: soft (recoverable) vs hard (end episode)
        soft_collision = (lidar_min < self.rl_collision_distance) and (lidar_min >= self.rl_collision_distance_hard)
        hard_collision = (lidar_min < self.rl_collision_distance_hard) or (dist_min < 0.05)
        
        collision_type = 'none'
        if hard_collision:
            collision_type = 'hard'
            reward -= collision_penalty * 1.5  # Severe penalty for hard collision
            self.rl_episode_collision_count += 1
        elif soft_collision:
            collision_type = 'soft'
            reward -= collision_penalty * 0.5  # Mild penalty for soft collision (can recover)
            self.rl_episode_collision_count += 1
        
        # Episode penalty: penalize episodes with multiple collisions to prevent crash-recovery learning
        # Each additional collision in episode reduces final reward
        if self.rl_episode_collision_count > 0:
            reward -= 0.1 * (self.rl_episode_collision_count - 1)  # Increasing penalty for repeat collisions
        
        # End episode on hard collision or step limit (soft collisions allow recovery)
        done = hard_collision or (self.rl_episode_step >= self.rl_max_episode_steps)
        info = {
            'collision': collision_type,
            'collision_count': self.rl_episode_collision_count,
            'lidar_min': lidar_min,
            'dist_min': dist_min,
        }
        return float(reward), done, info

    def _reset_rl_episode(self):
        """Best-effort episode reset in non-supervisor mode."""
        self.apply_control(0.0, -1.0)

        # Do not call simulationResetPhysics from a Robot controller.
        # Webots allows this only from Supervisor and prints an error otherwise.

        self.rl_episode_idx += 1
        self.rl_episode_step = 0
        self.rl_episode_reward = 0.0
        self.rl_episode_collision_count = 0  # Reset collision counter
        self.rl_last_steering = 0.0

    def run_rl(self):
        """Main online RL loop using DDPG and Webots interaction."""
        mode_label = 'evaluation' if self.rl_eval else 'training'
        print(f"[Controller] Starting true deep reinforcement learning loop ({mode_label})...")
        print(f"[RL] Warmup phase: first {self.rl_agent.warmup_steps} steps collect experience without learning")
        print(f"[RL] After warmup, actor/critic networks will update and losses will be printed")

        episode_rewards = []  # Track rewards for progress
        start_time = None

        # Advance one simulation step before first sensor read.
        # Some Webots devices are not safe to read before an initial step.
        initial_step = self.driver.step() if self.driver is not None else self.robot.step(self.timestep)
        if initial_step == -1:
            return

        sensor_data = self.read_sensor_data()
        state = self._build_rl_state(sensor_data)

        while True:
            # Exploration is enabled only while training.
            action = self.rl_agent.select_action(state, explore=self.rl_train)
            steering_cmd = float(action[0]) * float(SENSOR_CONFIG.get('max_steering_angle', 1.57))
            speed_cmd = float(action[1])

            self.apply_control(steering_cmd, speed_cmd)
            self.rl_last_steering = float(action[0])
            
            # Debug: print action every 200 steps during warmup phase
            if self.step_count < 2500 and self.step_count % 200 == 0:
                learning_status = "WARMUP (no learning yet)" if len(self.rl_agent.replay) < self.rl_agent.warmup_steps else "LEARNING (networks updating)"
                print(
                    f"[RL] step={self.step_count}, {learning_status}, "
                    f"action=[{action[0]:+.3f}, {action[1]:+.3f}], "
                    f"steering_cmd={steering_cmd:+.3f}rad, speed_cmd={speed_cmd:+.3f}, "
                    f"buffer={len(self.rl_agent.replay)}/{self.rl_agent.replay_size}"
                )

            # Advance simulation and observe transition
            step_result = self.driver.step() if self.driver is not None else self.robot.step(self.timestep)
            if step_result == -1:
                break

            if start_time is None:
                import time
                start_time = time.time()

            next_sensor_data = self.read_sensor_data()
            next_state = self._build_rl_state(next_sensor_data)
            reward, done, info = self._compute_rl_reward_done(next_sensor_data, action)

            if self.rl_train:
                self.rl_agent.store_transition(state, action, reward, next_state, done)
                self.rl_last_losses = self.rl_agent.update()

            self.step_count += 1
            self.rl_episode_step += 1
            self.rl_episode_reward += reward

            if self.step_count % 100 == 0:
                loss_msg = ""
                if self.rl_train and self.rl_last_losses is not None:
                    loss_msg = (
                        f", actor_loss={self.rl_last_losses['actor_loss']:.6f}"
                        f", critic_loss={self.rl_last_losses['critic_loss']:.6f}"
                        f", updates={self.rl_last_losses['updates']}"
                    )
                learning_active = "✓LEARNING" if len(self.rl_agent.replay) >= self.rl_agent.warmup_steps else "warming up"
                print(
                    f"[RL] [{learning_active}] step={self.step_count}, episode={self.rl_episode_idx}, "
                    f"ep_step={self.rl_episode_step}, ep_reward={self.rl_episode_reward:.2f}, "
                    f"lidar_min={info['lidar_min']:.2f}m{loss_msg}"
                )

            if done:
                episode_rewards.append(self.rl_episode_reward)
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                collision_info = f"collisions={self.rl_episode_collision_count}" if self.rl_episode_collision_count > 0 else "collision=none"
                print(
                    f"[RL] Episode {self.rl_episode_idx} finished: "
                    f"steps={self.rl_episode_step}, reward={self.rl_episode_reward:.2f}, "
                    f"avg_10ep={avg_reward:.2f}, {collision_info}"
                )

                if self.rl_train and (self.rl_episode_idx + 1) % max(1, self.rl_save_every) == 0:
                    ckpt_prefix = f"{self.rl_model_prefix}_ep_{self.rl_episode_idx + 1}"
                    self.rl_agent.save(ckpt_prefix)
                    print(f"[RL] ✓ SAVED checkpoint: {ckpt_prefix}_actor.pth / _critic.pth")

                if self.rl_eval and (self.rl_episode_idx + 1) >= max(1, self.rl_eval_episodes):
                    print(f"[RL] Evaluation finished after {self.rl_eval_episodes} episodes")
                    break

                self._reset_rl_episode()
                step_result = self.driver.step() if self.driver is not None else self.robot.step(self.timestep)
                if step_result == -1:
                    break
                sensor_data = self.read_sensor_data()
                state = self._build_rl_state(sensor_data)
                continue

            # Continue rollout from the observed next state.
            sensor_data = next_sensor_data
            state = next_state
    
    def run(self):
        """Main control loop."""
        if self.rl_train or self.rl_eval:
            self.run_rl()
            return

        print("[Controller] Starting autonomous driving controller...")

        while (self.driver.step() if self.driver is not None else self.robot.step(self.timestep)) != -1:
            # Read sensors
            sensor_data = self.read_sensor_data()
            
            # Get control commands
            if self.use_model:
                try:
                    steering_angle, speed_control = self.control_with_ai(sensor_data)
                except Exception as e:
                    print(f"[Controller] AI control error: {e}")
                    steering_angle, speed_control = self.control_with_manual_logic(sensor_data)
            else:
                steering_angle, speed_control = self.control_with_manual_logic(sensor_data)
            
            # Apply control
            self.apply_control(steering_angle, speed_control)
            
            # Collect training data if enabled
            if self.collect_data:
                self.collect_training_data(sensor_data, steering_angle, speed_control)
            
            self.step_count += 1
            
            # Print status periodically
            if self.step_count % 100 == 0:
                print(f"[Controller] Step: {self.step_count}, Steering: {steering_angle:.3f}, Speed: {speed_control:.3f}")


def main():
    """Main entry point."""
    # Choose mode with environment variable to keep supervised workflow explicit.
    # Supported values for NAWNAW_MODE:
    #   collect -> supervised data collection (manual/rule labels)
    #   infer   -> run trained model only
    #   hybrid  -> run trained model and keep collecting new labels
    #   rl      -> true online deep reinforcement learning (DDPG)
    #   rl_eval -> deterministic policy rollout using an RL checkpoint
    mode = os.environ.get('NAWNAW_MODE', '').strip().lower()
    mode_source = 'env'

    # Webots can pass startup args through `controllerArgs` in the world file.
    if not mode and len(sys.argv) > 1:
        arg_mode = sys.argv[1].strip().lower()
        if arg_mode.startswith('--mode='):
            mode = arg_mode.split('=', 1)[1]
            mode_source = 'controllerArgs(--mode=...)'
        else:
            mode = arg_mode
            mode_source = 'controllerArgs'

    if not mode:
        mode = 'collect'
        mode_source = 'default'

    if mode == 'collect':
        use_model = False
        collect_data = True
        rl_train = False
        rl_eval = False
    elif mode == 'infer':
        use_model = True
        collect_data = False
        rl_train = False
        rl_eval = False
    elif mode == 'hybrid':
        use_model = True
        collect_data = True
        rl_train = False
        rl_eval = False
    elif mode == 'rl':
        use_model = False
        collect_data = False
        rl_train = True
        rl_eval = False
    elif mode == 'rl_eval':
        use_model = False
        collect_data = False
        rl_train = False
        rl_eval = True
    else:
        print(f"[Controller] Unknown NAWNAW_MODE='{mode}', defaulting to 'collect'")
        use_model = False
        collect_data = True
        rl_train = False
        rl_eval = False

    print(
        f"[Controller] mode='{mode}' (source={mode_source}) | "
        f"use_model={use_model}, collect_data={collect_data}, "
        f"rl_train={rl_train}, rl_eval={rl_eval}"
    )
    controller = AutonomousDrivingController(
        use_model=use_model,
        collect_data=collect_data,
        rl_train=rl_train,
        rl_eval=rl_eval,
    )
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("[Controller] Shutting down...")
    except Exception as exc:
        print(f"[Controller] Fatal error: {exc}")
        traceback.print_exc()
    finally:
        if controller.collect_data:
            controller.data_collector.save_batch()
            print("[Controller] Saved final batch")


if __name__ == '__main__':
    main()
