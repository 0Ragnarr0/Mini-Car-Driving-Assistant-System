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
TouchSensor = controller_module.TouchSensor

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
SAFETY_CONFIG = getattr(config_module, 'SAFETY_CONFIG', {})
REWARD_VALIDATION = getattr(config_module, 'REWARD_VALIDATION', {})
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
        print(f"[Controller] Touch sensors found: {len(self.touch_sensors)} "
              f"({', '.join(ts.getName() for ts in self.touch_sensors) if self.touch_sensors else 'none'})")
        
        if self.use_model:
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
    
    def _find_latest_checkpoint(self, model_prefix, strategy='mtime'):
        """Find a checkpoint matching model_prefix using selection strategy.

        strategy:
            - 'mtime'   : newest modified actor/critic checkpoint pair (default)
            - 'episode' : highest episode number
        """
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
            
            # Build candidate pairs with episode and modification time.
            # Only consider checkpoints that have both actor and critic files.
            candidates = []
            for f in actor_files:
                try:
                    # Extract "_ep_NUMBER_" from filename
                    import re
                    match = re.search(r'_ep_(\d+)_', f)
                    if match:
                        ep = int(match.group(1))
                        prefix = f.rsplit('_actor.pth', 1)[0]
                        critic_file = f"{prefix}_critic.pth"
                        if not os.path.exists(critic_file):
                            continue

                        actor_mtime = os.path.getmtime(f)
                        critic_mtime = os.path.getmtime(critic_file)
                        pair_mtime = max(actor_mtime, critic_mtime)
                        candidates.append((ep, pair_mtime, prefix))
                except:
                    pass
            
            if candidates:
                select = str(strategy).strip().lower()
                if select == 'episode':
                    # Highest episode number; tie-break by newer mtime
                    ep, _, prefix = max(candidates, key=lambda x: (x[0], x[1]))
                    return prefix

                # Default: most recently modified checkpoint pair; tie-break by episode number
                _, _, prefix = max(candidates, key=lambda x: (x[1], x[0]))
                return prefix
        except Exception:
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
        self.sensor_caps = {
            'camera_enabled': False,
            'camera_width': 0,
            'camera_height': 0,
            'camera_fov': 0.0,
            'lidar_enabled': False,
            'lidar_width': 0,
            'lidar_max_range': 30.0,
            'lidar_fov': 0.0,
            'gps_enabled': False,
            'imu_enabled': False,
            'gyro_enabled': False,
            'touch_enabled': False,
            'touch_count': 0,
        }

        self.depth_camera = None
        self.fisheye_camera = None
        self.lidar = None

        # Try standard names first, then fall back to world device names.
        self.depth_camera = self._get_device_if_exists(['depth camera', 'Depth camera', 'range-finder'])
        if self.depth_camera is not None:
            try:
                self.depth_camera.enable(self.timestep)
            except Exception:
                self.depth_camera = None

        self.fisheye_camera = self._get_device_if_exists(['fisheye camera', 'camera', 'Camera'])
        if self.fisheye_camera is not None:
            try:
                self.fisheye_camera.enable(self.timestep)
                self.sensor_caps['camera_enabled'] = True
                if hasattr(self.fisheye_camera, 'getWidth'):
                    self.sensor_caps['camera_width'] = int(self.fisheye_camera.getWidth())
                if hasattr(self.fisheye_camera, 'getHeight'):
                    self.sensor_caps['camera_height'] = int(self.fisheye_camera.getHeight())
                if hasattr(self.fisheye_camera, 'getFov'):
                    self.sensor_caps['camera_fov'] = float(self.fisheye_camera.getFov())
            except Exception:
                self.fisheye_camera = None

        self.lidar = self._get_device_if_exists(['Sick LMS 291', 'sick lms 291', 'lidar', 'LDS-01'])
        if self.lidar is not None:
            try:
                self.lidar.enable(self.timestep)
                self.sensor_caps['lidar_enabled'] = True
                if hasattr(self.lidar, 'getHorizontalResolution'):
                    self.sensor_caps['lidar_width'] = int(self.lidar.getHorizontalResolution())
                if hasattr(self.lidar, 'getMaxRange'):
                    self.sensor_caps['lidar_max_range'] = float(self.lidar.getMaxRange())
                if hasattr(self.lidar, 'getFov'):
                    self.sensor_caps['lidar_fov'] = float(self.lidar.getFov())
            except Exception:
                self.lidar = None
        
        # Optional: Distance sensors (left/right for lane keeping)
        self.distance_sensors = []
        # Try specific names first
        for name_pattern in ['left_distance', 'right_distance', 'distance_sensor']:
            ds = self._get_device_if_exists([name_pattern, name_pattern.lower(), name_pattern.replace('_', ' ')])
            if ds is not None:
                try:
                    ds.enable(self.timestep)
                    self.distance_sensors.append(ds)
                except Exception:
                    pass
        
        # Fallback: generic search for any remaining distance sensors
        if len(self.distance_sensors) < 2:
            for actual_name in self._device_map.values():
                if 'distance' in actual_name.lower() and actual_name.lower() not in [d.getName().lower() for d in self.distance_sensors]:
                    try:
                        ds = self.robot.getDevice(actual_name)
                        ds.enable(self.timestep)
                        self.distance_sensors.append(ds)
                    except Exception:
                        pass
        
        # Optional: GPS/IMU
        self.gps = self._get_device_if_exists(['gps', 'GPS'])
        if self.gps is not None:
            try:
                self.gps.enable(self.timestep)
                self.sensor_caps['gps_enabled'] = True
            except Exception:
                self.gps = None

        # Prefer InertialUnit when available, otherwise fall back to Gyro.
        self.inertial_unit = self._get_device_if_exists(['inertial unit', 'InertialUnit'])
        self.gyro_sensor = None
        if self.inertial_unit is not None:
            try:
                self.inertial_unit.enable(self.timestep)
                self.sensor_caps['imu_enabled'] = True
            except Exception:
                self.inertial_unit = None
        else:
            self.gyro_sensor = self._get_device_if_exists(['gyro', 'Gyro'])
            if self.gyro_sensor is not None:
                try:
                    self.gyro_sensor.enable(self.timestep)
                    self.sensor_caps['gyro_enabled'] = True
                except Exception:
                    self.gyro_sensor = None

        # Optional: touch sensors (bumper-style contact detection)
        self.touch_sensors = []
        for name_pattern in [
            'touch_front_left', 'touch_front_right', 'touch_rear_left', 'touch_rear_right',
            'touch', 'bumper'
        ]:
            ts = self._get_device_if_exists([name_pattern, name_pattern.lower(), name_pattern.replace('_', ' ')])
            if ts is not None:
                try:
                    ts.enable(self.timestep)
                    if ts.getName().lower() not in [t.getName().lower() for t in self.touch_sensors]:
                        self.touch_sensors.append(ts)
                except Exception:
                    pass

        # Fallback: discover any touch sensors by name
        for actual_name in self._device_map.values():
            low_name = actual_name.lower()
            if ('touch' in low_name or 'bumper' in low_name) and low_name not in [t.getName().lower() for t in self.touch_sensors]:
                try:
                    ts = self.robot.getDevice(actual_name)
                    ts.enable(self.timestep)
                    self.touch_sensors.append(ts)
                except Exception:
                    pass

        self.sensor_caps['touch_enabled'] = len(self.touch_sensors) > 0
        self.sensor_caps['touch_count'] = len(self.touch_sensors)

        camera_status = (
            f"{self.sensor_caps['camera_width']}x{self.sensor_caps['camera_height']}"
            if self.sensor_caps['camera_enabled'] else "disabled"
        )
        lidar_status = (
            f"res={self.sensor_caps['lidar_width']}, range={self.sensor_caps['lidar_max_range']:.1f}m"
            if self.sensor_caps['lidar_enabled'] else "disabled"
        )
        gyro_mode = "imu" if self.sensor_caps['imu_enabled'] else ("gyro" if self.sensor_caps['gyro_enabled'] else "disabled")
        print(
            f"[Controller] Sensor profile: camera={camera_status}, lidar={lidar_status}, "
            f"gps={'enabled' if self.sensor_caps['gps_enabled'] else 'disabled'}, "
            f"gyro={gyro_mode}, touch={self.sensor_caps['touch_count']}"
        )
    
    def _setup_ai_model(self):
        """Setup the AI inference engine."""
        print("[Controller] Initializing AI model...")
        device = 'cuda' if self._has_gpu() else 'cpu'
        self.inference = AutonomousDrivingInference(device=device)
        self._setup_lane_segmentation_model(device=device)
        print(f"[Controller] AI model loaded on device: {device}")

    def _setup_lane_segmentation_model(self, device='cpu'):
        """Optionally load a tiny lane-segmentation model (TorchScript)."""
        self.lane_seg_model = None
        self.lane_seg_enabled = False
        self.lane_seg_threshold = float(RL_CONFIG.get('lane_seg_threshold', 0.55))

        enabled = bool(RL_CONFIG.get('lane_seg_model_enabled', False))
        if not enabled:
            return

        model_path = os.environ.get('NAWNAW_LANE_SEG_MODEL', '').strip()
        if not model_path:
            model_path = str(RL_CONFIG.get('lane_seg_model_path', '')).strip()
        if not model_path:
            model_path = os.path.join(DATA_CONFIG['models_dir'], 'lane_segmentation_tiny.ts')

        if not os.path.exists(model_path):
            print(f"[LaneSeg] Model not found at {model_path}; using geometry detector fallback")
            return

        try:
            import torch
            self.lane_seg_model = torch.jit.load(model_path, map_location=device)
            self.lane_seg_model.eval()
            self.lane_seg_enabled = True
            print(f"[LaneSeg] Loaded TorchScript lane model: {model_path}")
        except Exception as exc:
            self.lane_seg_model = None
            self.lane_seg_enabled = False
            print(f"[LaneSeg] Failed to load model ({exc}); using geometry detector fallback")

    def _predict_lane_mask_from_model(self, rgb_image):
        """Predict a binary lane mask and confidence from the optional lane model."""
        if not self.lane_seg_enabled or self.lane_seg_model is None:
            return None, 0.0

        try:
            import torch

            img = np.asarray(rgb_image, dtype=np.float32)
            if img.shape != (64, 64, 3):
                return None, 0.0

            x = torch.from_numpy(img / 255.0).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                y = self.lane_seg_model(x)

            if isinstance(y, (tuple, list)):
                y = y[0]

            if y.ndim != 4:
                return None, 0.0

            y_np = y.detach().cpu().numpy()
            if y_np.shape[1] == 1:
                prob = 1.0 / (1.0 + np.exp(-y_np[0, 0]))
            else:
                logits = y_np[0]
                logits = logits - np.max(logits, axis=0, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.clip(np.sum(exp_logits, axis=0, keepdims=True), 1e-6, None)
                prob = probs[min(1, probs.shape[0] - 1)]

            # Resize to 64x64 if needed via nearest indexing.
            if prob.shape != (64, 64):
                h, w = prob.shape
                ys = np.clip((np.linspace(0, h - 1, 64)).astype(np.int32), 0, h - 1)
                xs = np.clip((np.linspace(0, w - 1, 64)).astype(np.int32), 0, w - 1)
                prob = prob[np.ix_(ys, xs)]

            mask = prob > self.lane_seg_threshold
            coverage = float(np.mean(mask))
            if coverage < 0.005:
                return None, 0.0

            conf = float(np.clip(np.mean(prob[mask]) if np.any(mask) else 0.0, 0.0, 1.0))
            return mask.astype(bool), conf
        except Exception:
            return None, 0.0

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
        self.rl_episode_lane_detected_steps = 0
        self.rl_episode_lane_centered_steps = 0
        self.rl_episode_lane_abs_offset_sum = 0.0
        self.rl_lane_offset_history = []
        self.rl_soft_collision_streak = 0
        self.rl_hard_collision_soft_streak_steps = int(RL_CONFIG.get('hard_collision_soft_streak_steps', 8))
        self.rl_max_episode_steps = int(RL_CONFIG.get('max_episode_steps', 1200))
        self.rl_collision_distance = float(RL_CONFIG.get('collision_distance_m', 0.25))
        self.rl_collision_distance_hard = float(RL_CONFIG.get('collision_distance_hard_m', 0.10))  # Hard collision threshold
        self.rl_action_noise_initial = float(RL_CONFIG.get('action_noise_initial', RL_CONFIG.get('action_noise', 0.25)))
        self.rl_action_noise_final = float(RL_CONFIG.get('action_noise_final', 0.05))
        self.rl_action_noise_decay_steps = max(1, int(RL_CONFIG.get('action_noise_decay_steps', 30000)))
        self.rl_action_noise_steer_scale = float(RL_CONFIG.get('action_noise_steer_scale', 0.45))
        self.rl_action_noise_speed_scale = float(RL_CONFIG.get('action_noise_speed_scale', 0.80))
        self.rl_max_steer_delta_per_step = float(RL_CONFIG.get('max_steer_delta_per_step', 0.12))
        self.rl_max_speed_delta_per_step = float(RL_CONFIG.get('max_speed_delta_per_step', 0.20))
        self.rl_early_episode_steer_limit = float(RL_CONFIG.get('early_episode_steer_limit', 0.35))
        self.rl_early_episode_speed_limit = float(RL_CONFIG.get('early_episode_speed_limit', 0.55))
        self.rl_early_episode_limit_steps = max(0, int(RL_CONFIG.get('early_episode_limit_steps', 250)))
        self.rl_lane_assist_enabled = bool(RL_CONFIG.get('lane_assist_enabled', True))
        self.rl_lane_assist_conf_threshold = float(RL_CONFIG.get('lane_assist_conf_threshold', 0.12))
        self.rl_lane_assist_kp = float(RL_CONFIG.get('lane_assist_kp', 0.85))
        self.rl_lane_assist_kd = float(RL_CONFIG.get('lane_assist_kd', 0.30))
        self.rl_lane_assist_blend_max = float(RL_CONFIG.get('lane_assist_blend_max', 0.45))
        self.rl_lane_assist_max_steer = float(RL_CONFIG.get('lane_assist_max_steer', 0.40))
        self.rl_lane_assist_speed_cap = float(RL_CONFIG.get('lane_assist_speed_cap', 0.45))
        self.rl_lane_assist_prev_offset = 0.0
        self.rl_lane_bias_correction_enabled = bool(RL_CONFIG.get('lane_bias_correction_enabled', True))
        self.rl_lane_bias_alpha = float(RL_CONFIG.get('lane_bias_alpha', 0.02))
        self.rl_lane_bias_max_abs = float(RL_CONFIG.get('lane_bias_max_abs', 0.35))
        self.rl_lane_bias_conf_threshold = float(RL_CONFIG.get('lane_bias_conf_threshold', 0.65))
        self.rl_lane_bias_update_max_steer = float(RL_CONFIG.get('lane_bias_update_max_steer', 0.12))
        self.rl_lane_bias_min_samples = max(5, int(RL_CONFIG.get('lane_bias_min_samples', 30)))
        self.rl_lane_bias_ema = 0.0
        self.rl_lane_bias_samples = 0
        self.rl_lane_bias_last_log_step = -100000
        self.rl_lane_offcenter_streak = 0
        self.rl_use_model_prior = bool(RL_CONFIG.get('use_model_prior', True)) and self.use_model and self.rl_train
        self.rl_model_prior_blend_initial = float(RL_CONFIG.get('model_prior_blend_initial', 0.65))
        self.rl_model_prior_blend_final = float(RL_CONFIG.get('model_prior_blend_final', 0.10))
        self.rl_model_prior_decay_steps = max(1, int(RL_CONFIG.get('model_prior_decay_steps', 10000)))

        # Optional runtime override for model-prior blending during RL training.
        env_prior = os.environ.get('NAWNAW_RL_USE_MODEL_PRIOR', '').strip().lower()
        if env_prior in ('0', 'false', 'no', 'off'):
            self.rl_use_model_prior = False
        elif env_prior in ('1', 'true', 'yes', 'on'):
            self.rl_use_model_prior = self.use_model and self.rl_train

        if self.rl_use_model_prior and not hasattr(self, 'inference'):
            print("[Controller] Model prior requested but inference model not available; disabling model prior")
            self.rl_use_model_prior = False

        self.rl_last_steering = 0.0
        self.rl_last_losses = None
        self.rl_save_every = int(RL_CONFIG.get('save_every_episodes', 10))
        self.rl_eval_episodes = int(RL_CONFIG.get('eval_episodes', 5))
        self.rl_model_prefix = os.path.join(DATA_CONFIG['models_dir'], 'ddpg_autonomous_driving')
        
        # Log collision thresholds
        print(f"[Controller] Collision thresholds: soft={self.rl_collision_distance}m, hard={self.rl_collision_distance_hard}m")
        print(
            f"[Controller] Exploration noise schedule: {self.rl_action_noise_initial:.3f} -> "
            f"{self.rl_action_noise_final:.3f} over {self.rl_action_noise_decay_steps} steps"
        )
        print(
            f"[Controller] Exploration safety: steer_noise_scale={self.rl_action_noise_steer_scale:.2f}, "
            f"speed_noise_scale={self.rl_action_noise_speed_scale:.2f}, "
            f"max_deltas=[steer:{self.rl_max_steer_delta_per_step:.2f}, speed:{self.rl_max_speed_delta_per_step:.2f}]"
        )
        print(
            f"[Controller] Lane assist: {'enabled' if self.rl_lane_assist_enabled else 'disabled'}, "
            f"conf>={self.rl_lane_assist_conf_threshold:.2f}, blend_max={self.rl_lane_assist_blend_max:.2f}"
        )
        print(
            f"[Controller] Lane bias calibration: {'enabled' if self.rl_lane_bias_correction_enabled else 'disabled'}, "
            f"alpha={self.rl_lane_bias_alpha:.3f}, max_bias={self.rl_lane_bias_max_abs:.2f}, "
            f"conf>={self.rl_lane_bias_conf_threshold:.2f}"
        )
        print(
            f"[Controller] RL model prior: {'enabled' if self.rl_use_model_prior else 'disabled'}, "
            f"blend {self.rl_model_prior_blend_initial:.2f}->{self.rl_model_prior_blend_final:.2f} "
            f"over {self.rl_model_prior_decay_steps} steps"
        )

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
            ckpt_select = os.environ.get('NAWNAW_RL_CKPT_SELECT', 'mtime').strip().lower()
            if ckpt_select not in ('mtime', 'episode'):
                ckpt_select = 'mtime'
            ckpt_prefix = self._find_latest_checkpoint(self.rl_model_prefix, strategy=ckpt_select)
            if ckpt_prefix:
                print(f"[Controller] Auto-selected RL checkpoint ({ckpt_select}): {ckpt_prefix}")
        
        self.rl_resume_prefix = ckpt_prefix

        if ckpt_prefix:
            actor_path = f"{ckpt_prefix}_actor.pth"
            critic_path = f"{ckpt_prefix}_critic.pth"
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                try:
                    self.rl_agent.load(ckpt_prefix)
                    # Continue episode numbering from resumed checkpoint suffix (_ep_N).
                    try:
                        import re
                        match = re.search(r'_ep_(\d+)$', ckpt_prefix)
                        if match:
                            resumed_ep = int(match.group(1))
                            self.rl_episode_idx = resumed_ep
                            print(f"[Controller] Resume episode index set to {self.rl_episode_idx} (next save: ep_{self.rl_episode_idx + 1})")
                    except Exception:
                        pass
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
                if depth_image is not None and width > 0 and height > 0:
                    depth_data = np.frombuffer(depth_image, np.uint16).reshape((height, width))
            except Exception:
                depth_data = None
        
        if depth_data is None and self.sensor_caps.get('lidar_enabled', False) and self.lidar is not None and hasattr(self.lidar, 'getRangeImage'):
            try:
                ranges = np.array(self.lidar.getRangeImage(), dtype=np.float32)
                max_range = float(self.sensor_caps.get('lidar_max_range', 30.0))
                ranges = np.clip(ranges, 0.0, max_range)
                # Convert meters to millimeters and tile as pseudo 2D depth map.
                row_mm = (ranges * 1000.0).astype(np.uint16)
                depth_data = np.tile(row_mm, (32, 1))
            except Exception:
                depth_data = None

        if depth_data is None:
            depth_data = np.zeros((32, 64), dtype=np.uint16)

        # RGB source (preferred: fisheye; fallback: regular camera)
        fisheye_data = None
        camera_ok = False
        if self.sensor_caps.get('camera_enabled', False) and self.fisheye_camera is not None:
            try:
                fisheye_image = self.fisheye_camera.getImage()
                f_width = self.fisheye_camera.getWidth()
                f_height = self.fisheye_camera.getHeight()
                # Webots camera buffer is BGRA; convert to RGB for color-based perception.
                if fisheye_image is not None and f_width > 0 and f_height > 0:
                    fisheye_bgra = np.frombuffer(fisheye_image, np.uint8).reshape((f_height, f_width, 4))
                    fisheye_data = fisheye_bgra[:, :, [2, 1, 0]]
                    
                    # Resize to 64x64 for consistent processing (captures high-res, processes efficiently)
                    if fisheye_data.shape != (64, 64, 3):
                        # Simple nearest-neighbor downsampling
                        step = max(f_height // 64, 1)
                        fisheye_data = fisheye_data[::step, ::step, :]
                        if fisheye_data.shape[0] > 64:
                            fisheye_data = fisheye_data[:64, :, :]
                        if fisheye_data.shape[1] > 64:
                            fisheye_data = fisheye_data[:, :64, :]
                        # Pad if smaller
                        if fisheye_data.shape[0] < 64 or fisheye_data.shape[1] < 64:
                            padded = np.zeros((64, 64, 3), dtype=np.uint8)
                            h, w = fisheye_data.shape[:2]
                            padded[:h, :w, :] = fisheye_data
                            fisheye_data = padded
                    
                    camera_ok = True
                    # Log camera status once on first frame
                    if not hasattr(self, '_camera_diagnostic_logged'):
                        self._camera_diagnostic_logged = True
                        print(f"[Sensor] Camera OK: {f_width}×{f_height} BGRA→RGB (→64×64), FOV={self.sensor_caps.get('camera_fov', 'N/A')} rad")
            except Exception as e:
                if not hasattr(self, '_camera_error_logged'):
                    self._camera_error_logged = True
                    print(f"[Sensor] Camera read error: {e}")
                fisheye_data = None

        if fisheye_data is None:
            fisheye_data = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Get distance sensor readings
        distances = []
        for ds in self.distance_sensors:
            distances.append(ds.getValue())

        # Get touch sensor readings (contact = value > 0)
        touch_values = []
        for ts in getattr(self, 'touch_sensors', []):
            try:
                touch_values.append(float(ts.getValue()))
            except Exception:
                touch_values.append(0.0)
        touch_contact = any(v > 0.0 for v in touch_values)
        
        # Get lidar ranges
        lidar_ranges = None
        if self.sensor_caps.get('lidar_enabled', False) and self.lidar is not None and hasattr(self.lidar, 'getRangeImage'):
            try:
                lidar_ranges = np.array(self.lidar.getRangeImage(), dtype=np.float32)
                max_range = float(self.sensor_caps.get('lidar_max_range', 30.0))
                lidar_ranges = np.clip(lidar_ranges, 0.0, max_range)
            except Exception:
                lidar_ranges = None
        
        # Get GPS position
        gps_data = None
        if self.sensor_caps.get('gps_enabled', False) and self.gps is not None:
            try:
                gps_data = self.gps.getValues()  # [x, y, z]
            except Exception:
                gps_data = None
        
        # Get gyro/IMU data (rotation rates)
        gyro_data = None
        if self.sensor_caps.get('imu_enabled', False) and self.inertial_unit is not None:
            try:
                gyro_data = self.inertial_unit.getValues()  # [roll, pitch, yaw]
            except Exception:
                gyro_data = None
        elif self.sensor_caps.get('gyro_enabled', False) and self.gyro_sensor is not None:
            try:
                gyro_data = self.gyro_sensor.getValues()  # [roll, pitch, yaw]
            except Exception:
                gyro_data = None

        imu_data = None
        if self.sensor_caps.get('imu_enabled', False) and self.inertial_unit is not None:
            try:
                imu_data = self.inertial_unit.getValues()
            except Exception:
                imu_data = None
        
        return {
            'depth': depth_data,
            'fisheye': fisheye_data,
            'distances': distances,
            'touch': touch_values,
            'touch_contact': touch_contact,
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
        Apply control commands to robot with safety constraints.
        
        Args:
            steering_angle: Target steering angle in radians
            speed_control: Target speed in m/s (normalized -1 to 1)
        """
        # Safety constraint: Apply hard limits if enabled
        safety_enabled = bool(SAFETY_CONFIG.get('enable_safety_checks', True))
        if safety_enabled:
            max_steering = float(SENSOR_CONFIG.get('max_steering_angle', 1.57))
            steering_angle = float(np.clip(steering_angle, -max_steering, max_steering))
            
            # Limit steering rate (smooth steering, not jerky)
            max_steering_rate = float(SAFETY_CONFIG.get('max_steering_rate', 1.0))
            if hasattr(self, 'rl_last_steering'):
                steering_delta = abs(steering_angle - self.rl_last_steering)
                if steering_delta > max_steering_rate:
                    direction = 1.0 if steering_angle > self.rl_last_steering else -1.0
                    steering_angle = self.rl_last_steering + direction * max_steering_rate
            
            # Reduce speed when steering heavily (safer cornering)
            max_steering_for_full_speed = 0.3
            if abs(steering_angle) > max_steering_for_full_speed:
                speed_reduction = (abs(steering_angle) - max_steering_for_full_speed) / (max_steering - max_steering_for_full_speed)
                speed_control = speed_control * (1.0 - speed_reduction * 0.5)  # Up to 50% speed reduction
        
        # Driver API path (BmwX5 / car models)
        if self.driver is not None:
            steering = float(np.clip(steering_angle, -0.5, 0.5))
            # Convert normalized speed to m/s then to km/h for Driver API.
            speed_ms = float((speed_control + 1) / 2 * 8.0)
            max_speed = float(SAFETY_CONFIG.get('max_speed_limit', 5.0))
            speed_ms = float(np.clip(speed_ms, 0.0, max_speed))
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
            # Support both meter-valued sensors and legacy large-unit sensors.
            if float(np.max(d)) <= 10.0:
                d_norm = np.clip(d / 2.0, 0.0, 1.0)
            else:
                d_norm = np.tanh(np.clip(d, 0.0, 2000.0) / 500.0)
            dist_min_norm = float(np.min(d_norm))
        else:
            dist_min_norm = 1.0

        lane_offset, lane_confidence = self._detect_lane_from_camera(sensor_data.get('fisheye'))

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
                np.array(
                    [
                        dist_min_norm,
                        lane_offset,
                        lane_confidence,
                        speed_norm,
                        yaw_norm,
                        self.rl_last_steering,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        return state.astype(np.float32)

    def _detect_lane_from_camera(self, camera_image):
        """
        Detect lane markings from camera image (white/yellow pixels).
        Returns (lateral_offset, confidence):
            - lateral_offset in [-1, +1]
            - confidence in [0, 1], where 0 means lane not reliably detected
        """
        try:
            if camera_image is None or camera_image.size == 0:
                return 0.0, 0.0
            
            img = np.asarray(camera_image, dtype=np.float32)
            if img.shape != (64, 64, 3):
                return 0.0, 0.0
            
            # Look for white/yellow lane markings in lower part of image (forward view).
            # Use a wider lower ROI to increase chance of lane detection.
            roi_start = 24
            roi = img[roi_start:, :, :]  # Shape: (32, 64, 3)

            # Optional learned segmentation mask (if a lane model is available).
            seg_mask_full, seg_conf = self._predict_lane_mask_from_model(img)
            seg_mask = None
            if seg_mask_full is not None:
                seg_mask = seg_mask_full[roi_start:, :]
            
            # Adaptive thresholds for varied lighting.
            # White lanes: bright and low color variance.
            max_ch = np.max(roi, axis=2)
            min_ch = np.min(roi, axis=2)
            white_mask = (max_ch > 150) & ((max_ch - min_ch) < 35)

            # Yellow lanes in RGB: high R/G, low B.
            yellow_mask = (roi[:, :, 0] > 150) & (roi[:, :, 1] > 120) & (roi[:, :, 2] < 130)

            # Webots autonomous_vehicle reference color (BGR 95,187,203 -> RGB 203,187,95).
            # A color-distance mask is robust when absolute thresholds fail due exposure.
            ref_yellow = np.array([203.0, 187.0, 95.0], dtype=np.float32)
            color_diff = np.sum(np.abs(roi - ref_yellow), axis=2)
            yellow_ref_mask = color_diff < 95.0

            # Gate color masks with learned segmentation when available.
            if seg_mask is not None:
                white_mask = white_mask & seg_mask
                yellow_mask = yellow_mask & seg_mask
                yellow_ref_mask = yellow_ref_mask & seg_mask

            yellow_combined = yellow_mask | yellow_ref_mask
            lane_mask = white_mask | yellow_combined

            lane_pixels = int(np.sum(lane_mask))
            if lane_pixels < 3:  # Too few pixels detected
                # Log first failure to help diagnose camera content
                if not hasattr(self, '_lane_detection_failure_logged'):
                    self._lane_detection_failure_logged = True
                    print(f"[Lane] No markings found in ROI (white:{int(np.sum(white_mask))} + yellow:{int(np.sum(yellow_mask))} + ref_yellow:{int(np.sum(yellow_ref_mask))})")
                return 0.0, 0.0

            cols = np.arange(64, dtype=np.float32)
            lane_weights = np.sum(lane_mask, axis=0).astype(np.float32)
            yellow_weights = np.sum(yellow_combined, axis=0).astype(np.float32)
            white_weights = np.sum(white_mask, axis=0).astype(np.float32)

            yellow_pixels = int(np.sum(yellow_combined))
            white_pixels = int(np.sum(white_mask))

            yellow_col = None
            white_col = None
            if yellow_pixels >= 3 and float(np.sum(yellow_weights)) > 0.0:
                yellow_col = float(np.sum(cols * yellow_weights) / np.sum(yellow_weights))
            if white_pixels >= 3 and float(np.sum(white_weights)) > 0.0:
                white_col = float(np.sum(cols * white_weights) / np.sum(white_weights))

            lane_center_col = None
            confidence = 0.0
            expected_half_lane_width_px = 14.0

            if yellow_col is not None and white_col is not None:
                left_col = float(min(yellow_col, white_col))
                right_col = float(max(yellow_col, white_col))
                lane_span = right_col - left_col
                lane_center_col = 0.5 * (left_col + right_col)

                # Geometric confidence: both boundaries present with plausible separation.
                span_score = float(np.clip((lane_span - 8.0) / 20.0, 0.0, 1.0))
                if lane_span > 32.0:
                    span_score *= float(np.clip((48.0 - lane_span) / 16.0, 0.0, 1.0))

                pair_balance = 1.0 - abs(yellow_pixels - white_pixels) / max(1.0, float(yellow_pixels + white_pixels))
                pair_balance = float(np.clip(pair_balance, 0.0, 1.0))

                col_coverage = float(np.count_nonzero(lane_weights > 0) / 64.0)
                conf_pixels = float(np.clip(lane_pixels / 120.0, 0.0, 1.0))
                conf_spread = float(np.clip(col_coverage / 0.20, 0.0, 1.0))
                confidence = 0.50 * span_score + 0.20 * pair_balance + 0.20 * conf_pixels + 0.10 * conf_spread
            else:
                # One boundary only: infer approximate lane center with lower confidence.
                if yellow_col is not None:
                    lane_center_col = yellow_col + expected_half_lane_width_px
                elif white_col is not None:
                    lane_center_col = white_col - expected_half_lane_width_px

                if lane_center_col is not None:
                    lane_center_col = float(np.clip(lane_center_col, 0.0, 63.0))
                    col_coverage = float(np.count_nonzero(lane_weights > 0) / 64.0)
                    conf_pixels = float(np.clip(lane_pixels / 120.0, 0.0, 1.0))
                    conf_spread = float(np.clip(col_coverage / 0.20, 0.0, 1.0))
                    confidence = 0.30 * conf_pixels + 0.20 * conf_spread
                    confidence = min(confidence, 0.45)

            if lane_center_col is None:
                return 0.0, 0.0

            # Normalize to [-1, +1]: center (32) = 0, left (0) = -1, right (63) = +1
            lateral_offset_raw = float(np.clip((lane_center_col - 32.0) / 32.0, -1.0, 1.0))

            # Suppress unrealistic confidence for edge-biased centers.
            if lane_center_col < 6.0 or lane_center_col > 58.0:
                confidence *= 0.5

            # Blend in learned segmentation confidence if available.
            if seg_mask is not None:
                confidence = 0.7 * confidence + 0.3 * float(np.clip(seg_conf, 0.0, 1.0))

            # Optional low-rate diagnostics for lane-boundary geometry.
            if self.step_count % 600 == 0:
                yc = f"{yellow_col:.1f}" if yellow_col is not None else "None"
                wc = f"{white_col:.1f}" if white_col is not None else "None"
                print(f"[Lane] boundaries: yellow={yc}, white={wc}, center={lane_center_col:.1f}, conf={confidence:.2f}")

            # Adaptive offset-bias calibration.
            # Update bias only in high-confidence and low-steer segments,
            # so sustained camera mounting bias can be removed without absorbing corner geometry.
            lateral_offset = lateral_offset_raw
            if self.rl_lane_bias_correction_enabled:
                can_update_bias = (
                    confidence >= self.rl_lane_bias_conf_threshold
                    and abs(self.rl_last_steering) <= self.rl_lane_bias_update_max_steer
                    and abs(lateral_offset_raw) < 0.95
                )
                if can_update_bias:
                    alpha = float(np.clip(self.rl_lane_bias_alpha, 0.001, 0.2))
                    self.rl_lane_bias_ema = (1.0 - alpha) * self.rl_lane_bias_ema + alpha * lateral_offset_raw
                    self.rl_lane_bias_samples += 1

                if self.rl_lane_bias_samples >= self.rl_lane_bias_min_samples:
                    bias = float(np.clip(self.rl_lane_bias_ema, -self.rl_lane_bias_max_abs, self.rl_lane_bias_max_abs))
                    lateral_offset = float(np.clip(lateral_offset_raw - bias, -1.0, 1.0))

                    if (self.step_count - self.rl_lane_bias_last_log_step) >= 800:
                        self.rl_lane_bias_last_log_step = int(self.step_count)
                        print(
                            f"[Lane] Bias calibration active: raw={lateral_offset_raw:+.2f}, "
                            f"bias={bias:+.2f}, corrected={lateral_offset:+.2f}, conf={confidence:.2f}"
                        )

            # Short temporal smoothing (similar to Webots autonomous_vehicle filter).
            self.rl_lane_offset_history.append(lateral_offset)
            if len(self.rl_lane_offset_history) > 3:
                self.rl_lane_offset_history.pop(0)
            lateral_offset = float(np.mean(self.rl_lane_offset_history))
            return lateral_offset, float(np.clip(confidence, 0.0, 1.0))
        except Exception:
            return 0.0, 0.0

    def _recover_from_collision(self):
        """Try to move away from obstacle to avoid repeated 1-step episodes after reset."""
        max_recovery_steps = 20
        clear_threshold = self.rl_collision_distance + 0.2

        for i in range(max_recovery_steps):
            sensor_data = self.read_sensor_data()
            lidar_min = 30.0
            if sensor_data.get('lidar') is not None and len(sensor_data['lidar']) > 0:
                lidar_min = float(np.min(np.asarray(sensor_data['lidar'], dtype=np.float32)))

            if lidar_min >= clear_threshold:
                return

            # Reverse with a small steering oscillation to escape contacts.
            escape_steer = -0.25 if (i % 2 == 0) else 0.25
            self.apply_control(escape_steer, -0.8)
            step_result = self.driver.step() if self.driver is not None else self.robot.step(self.timestep)
            if step_result == -1:
                return

    def _apply_lane_assist_action(self, action, sensor_data):
        """Blend confidence-gated lane assist with RL action."""
        lane_offset, lane_confidence = self._detect_lane_from_camera(sensor_data.get('fisheye'))

        if not self.rl_lane_assist_enabled:
            return action, lane_offset, lane_confidence

        if lane_confidence < self.rl_lane_assist_conf_threshold:
            self.rl_lane_assist_prev_offset = lane_offset
            return action, lane_offset, lane_confidence

        d_err = lane_offset - self.rl_lane_assist_prev_offset
        self.rl_lane_assist_prev_offset = lane_offset

        lane_steer = -(
            self.rl_lane_assist_kp * lane_offset
            + self.rl_lane_assist_kd * d_err
        )
        lane_steer = float(np.clip(lane_steer, -self.rl_lane_assist_max_steer, self.rl_lane_assist_max_steer))

        blend = float(np.clip(lane_confidence * self.rl_lane_assist_blend_max, 0.0, self.rl_lane_assist_blend_max))
        action[0] = float((1.0 - blend) * action[0] + blend * lane_steer)

        if abs(lane_offset) > 0.35:
            action[1] = float(min(action[1], self.rl_lane_assist_speed_cap))

        return action, lane_offset, lane_confidence

    def _detect_traffic_light_from_camera(self, camera_image):
        """
        Detect traffic light state from camera image (red/yellow/green lights).
        Returns traffic light state: 'red', 'yellow', 'green', or 'none'.
        """
        try:
            if camera_image is None or camera_image.size == 0:
                return 'none'
            
            img = np.asarray(camera_image, dtype=np.float32)
            if img.shape != (64, 64, 3):
                return 'none'
            
            # Look for traffic lights in upper half of image (forward view, above horizon)
            # Webots fisheye points forward-down, traffic lights appear in upper portion
            roi_end = 40  # Upper/middle portion (0-40)
            roi = img[:roi_end, :, :]  # Shape: (40, 64, 3)
            
            # Red light: R>180, G<100, B<100
            red_mask = (roi[:, :, 0] > 180) & (roi[:, :, 1] < 100) & (roi[:, :, 2] < 100)
            red_pixels = np.sum(red_mask)
            
            # Green light: R<100, G>180, B<100
            green_mask = (roi[:, :, 0] < 100) & (roi[:, :, 1] > 180) & (roi[:, :, 2] < 100)
            green_pixels = np.sum(green_mask)
            
            # Yellow light: R>180, G>180, B<100
            yellow_mask = (roi[:, :, 0] > 180) & (roi[:, :, 1] > 180) & (roi[:, :, 2] < 100)
            yellow_pixels = np.sum(yellow_mask)
            
            # Threshold: need at least 15 pixels to consider detected
            threshold = 15
            if red_pixels > threshold and red_pixels > green_pixels and red_pixels > yellow_pixels:
                return 'red'
            elif green_pixels > threshold and green_pixels > red_pixels and green_pixels > yellow_pixels:
                return 'green'
            elif yellow_pixels > threshold and yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                return 'yellow'
            else:
                return 'none'
        except Exception:
            return 'none'

    def _compute_rl_reward_done(self, sensor_data, action):
        """Compute reward and terminal condition from current observation and action."""
        steer_cmd = float(action[0])
        speed_cmd = float(action[1])

        lidar_min = 30.0
        lidar_left_min = 30.0
        lidar_right_min = 30.0
        if sensor_data.get('lidar') is not None and len(sensor_data['lidar']) > 0:
            lidar_ranges = np.asarray(sensor_data['lidar'], dtype=np.float32)
            lidar_min = float(np.min(lidar_ranges))
            # Multi-sector lidar: left/right side detection (prevent side bumping)
            # Lidar typically scans 270 degrees: assume left = indices 0-40%, right = 60-100%
            n_bins = len(lidar_ranges)
            left_sector_end = int(n_bins * 0.4)
            right_sector_start = int(n_bins * 0.6)
            if n_bins > 0:
                lidar_left_min = float(np.min(lidar_ranges[:left_sector_end])) if left_sector_end > 0 else 30.0
                lidar_right_min = float(np.min(lidar_ranges[right_sector_start:])) if right_sector_start < n_bins else 30.0

        dist_min = 10.0
        distances = sensor_data.get('distances', [])
        if distances:
            dist_min = float(np.min(np.asarray(distances, dtype=np.float32)))
        touch_contact = bool(sensor_data.get('touch_contact', False))

        cfg = self.rl_reward_profile
        base_reward = float(cfg.get('base_reward', 0.10))
        forward_weight = float(cfg.get('forward_weight', 0.60))
        steering_penalty_weight = float(cfg.get('steering_penalty_weight', 0.08))
        risk_penalty_weight = float(cfg.get('risk_penalty_weight', 0.50))
        avoidance_bonus_weight = float(cfg.get('avoidance_bonus_weight', 0.25))
        collision_penalty = float(cfg.get('collision_penalty', 3.00))
        lane_keeping_bonus_weight = float(cfg.get('lane_keeping_bonus_weight', 0.15))
        lane_violation_penalty = float(cfg.get('lane_violation_penalty', 0.50))
        side_collision_penalty_weight = float(cfg.get('side_collision_penalty_weight', 0.30))  # New: left/right side avoidance
        lane_conf_threshold = float(cfg.get('lane_conf_threshold', 0.15))
        lane_center_tolerance = float(cfg.get('lane_center_tolerance', 0.20))
        lane_edge_threshold = float(cfg.get('lane_edge_threshold', 0.45))
        lane_violation_curve = float(cfg.get('lane_violation_curve', 1.7))
        no_lane_visible_penalty = float(cfg.get('no_lane_visible_penalty', 0.25))
        hard_collision_penalty_multiplier = float(cfg.get('hard_collision_penalty_multiplier', 3.0))
        soft_collision_penalty_multiplier = float(cfg.get('soft_collision_penalty_multiplier', 1.0))
        repeat_collision_step_penalty = float(cfg.get('repeat_collision_step_penalty', 0.25))
        touch_collision_extra_penalty = float(cfg.get('touch_collision_extra_penalty', 1.0))

        forward_reward = max(0.0, (speed_cmd + 1.0) * 0.5)
        steering_penalty = steering_penalty_weight * abs(steer_cmd)

        obstacle_risk = max(0.0, 1.0 - (lidar_min / 1.5))
        risk_penalty = risk_penalty_weight * obstacle_risk * max(0.0, (speed_cmd + 1.0) * 0.5)
        avoidance_bonus = avoidance_bonus_weight * obstacle_risk * abs(steer_cmd)
        
        # Side-collision risk: penalize left/right side drift towards barriers
        side_risk = max(0.0, max(1.0 - (lidar_left_min / 1.2), 1.0 - (lidar_right_min / 1.2)))
        side_penalty = side_collision_penalty_weight * side_risk * max(0.0, (speed_cmd + 1.0) * 0.5)

        # Vision-based lane detection: bonus for staying within lane markings
        # Detect lane from camera image (white/yellow road markings)
        lane_keeping_bonus = 0.0
        lane_violation = 0.0
        
        camera_image = sensor_data.get('fisheye')
        lateral_offset, lane_confidence = self._detect_lane_from_camera(camera_image)
        
        # Lateral offset from lane center: [-1, +1]
        # Strongly penalize sustained off-center driving to enforce lane discipline.
        abs_offset = abs(lateral_offset)
        if lane_confidence >= lane_conf_threshold:
            if abs_offset < lane_center_tolerance:
                # Near lane center - good!
                lane_keeping_bonus = lane_keeping_bonus_weight * (1.0 - abs_offset / max(lane_center_tolerance, 1e-6)) * lane_confidence

            lane_violation += lane_violation_penalty * (abs_offset ** lane_violation_curve) * lane_confidence
            if abs_offset > lane_edge_threshold:
                lane_violation += lane_violation_penalty * ((abs_offset - lane_edge_threshold) / max(1e-6, 1.0 - lane_edge_threshold)) * lane_confidence
        else:
            # If lane is not visible, avoid rewarding a fake "centered" signal.
            lane_violation += no_lane_visible_penalty * lane_violation_penalty
        
        # Fallback: Also check distance sensors (left/right) in case lane detection fails
        if distances and len(distances) >= 2:
            side_distances = np.asarray(distances, dtype=np.float32)
            side_min = float(np.min(side_distances))
            # Hard penalty for going off-road completely (< 0.2m to boundary)
            if side_min < 0.2:
                lane_violation += lane_violation_penalty * 0.5

        # Traffic light detection and rewards
        traffic_light_bonus = 0.0
        traffic_light_penalty = 0.0
        camera_image = sensor_data.get('fisheye')
        traffic_light_state = self._detect_traffic_light_from_camera(camera_image)
        
        # Calculate current speed (0 = stationary, 1 = moving at max speed)
        current_speed = max(0.0, (speed_cmd + 1.0) * 0.5)  # Normalized [0, 1]
        
        if traffic_light_state == 'red':
            # Red light: reward for stopping or slowing down
            if current_speed < 0.2:  # Nearly stopped
                traffic_light_bonus = 0.3  # Good! Stopped at red light
            elif current_speed < 0.5:  # Slowing down
                traffic_light_bonus = 0.1  # Okay, decelerating
            else:
                # Running red light - severe penalty
                traffic_light_penalty = 2.0  # Big penalty for traffic violation
        
        elif traffic_light_state == 'green':
            # Green light: reward for moving
            if current_speed > 0.5:
                traffic_light_bonus = 0.2  # Good! Moving on green
        
        elif traffic_light_state == 'yellow':
            # Yellow light: reward for caution (slowing down or proceeding carefully)
            if current_speed < 0.5:
                traffic_light_bonus = 0.1  # Good! Slowing down on yellow
            elif current_speed > 0.7:
                # Aggressive on yellow - slight penalty
                traffic_light_penalty = 0.3

        reward = (
            base_reward 
            + forward_weight * forward_reward 
            + avoidance_bonus 
            + lane_keeping_bonus
            + traffic_light_bonus
            - steering_penalty 
            - risk_penalty 
            - side_penalty
            - lane_violation
            - traffic_light_penalty
        )

        # Two-tier collision detection: soft (recoverable) vs hard (end episode)
        soft_collision = (lidar_min < self.rl_collision_distance) and (lidar_min >= self.rl_collision_distance_hard)

        # If soft collision persists for many consecutive steps, treat it as a hard collision.
        # This catches "scraping/stuck on barrier" cases where lidar hovers above hard threshold.
        if soft_collision:
            self.rl_soft_collision_streak += 1
        else:
            self.rl_soft_collision_streak = 0

        hard_collision = (
            (lidar_min < self.rl_collision_distance_hard)
            or (dist_min < 0.05)
            or touch_contact
            or (self.rl_soft_collision_streak >= self.rl_hard_collision_soft_streak_steps)
        )
        
        collision_type = 'none'
        if hard_collision:
            collision_type = 'hard'
            reward -= collision_penalty * hard_collision_penalty_multiplier
            if touch_contact:
                reward -= touch_collision_extra_penalty
            self.rl_episode_collision_count += 1
            if self.rl_episode_collision_count == 1:  # Log first collision in episode
                if self.rl_soft_collision_streak >= self.rl_hard_collision_soft_streak_steps:
                    trigger_source = f"sustained soft-contact ({self.rl_soft_collision_streak} steps)"
                elif touch_contact:
                    trigger_source = "touch sensor"
                elif lidar_min < self.rl_collision_distance_hard:
                    trigger_source = "lidar"
                else:
                    trigger_source = "distance sensor"
                print(f"[RL] ⚠️ HARD COLLISION at step {self.step_count} ({trigger_source}: {lidar_min:.3f}m vs threshold {self.rl_collision_distance_hard}m)")
        elif soft_collision:
            collision_type = 'soft'
            reward -= collision_penalty * soft_collision_penalty_multiplier
            self.rl_episode_collision_count += 1
            if self.rl_episode_collision_count == 1:  # Log first collision in episode
                print(f"[RL] ⚠️ SOFT COLLISION at step {self.step_count} (lidar: {lidar_min:.3f}m between {self.rl_collision_distance_hard}m and {self.rl_collision_distance}m)")
        
        # Episode penalty: penalize episodes with multiple collisions to prevent crash-recovery learning
        # Each additional collision in episode reduces final reward
        if self.rl_episode_collision_count > 0:
            reward -= repeat_collision_step_penalty * (self.rl_episode_collision_count - 1)
        
        # End episode on hard collision or step limit (soft collisions allow recovery)
        done = hard_collision or (self.rl_episode_step >= self.rl_max_episode_steps)
        info = {
            'collision': collision_type,
            'collision_count': self.rl_episode_collision_count,
            'lidar_min': lidar_min,
            'dist_min': dist_min,
            'touch_contact': bool(touch_contact),
            'traffic_light': traffic_light_state,
            'lane_offset': float(lateral_offset),
            'lane_confidence': float(lane_confidence),
            'soft_collision_streak': int(self.rl_soft_collision_streak),
        }
        return float(reward), done, info

    def _reset_rl_episode(self):
        """Best-effort episode reset in non-supervisor mode."""
        self.apply_control(0.0, -1.0)

        # In non-supervisor mode we cannot teleport reset; attempt local recovery.
        self._recover_from_collision()

        # Do not call simulationResetPhysics from a Robot controller.
        # Webots allows this only from Supervisor and prints an error otherwise.

        self.rl_episode_idx += 1
        self.rl_episode_step = 0
        self.rl_episode_reward = 0.0
        self.rl_episode_collision_count = 0  # Reset collision counter
        self.rl_episode_lane_detected_steps = 0
        self.rl_episode_lane_centered_steps = 0
        self.rl_episode_lane_abs_offset_sum = 0.0
        self.rl_lane_offset_history = []
        self.rl_lane_assist_prev_offset = 0.0
        self.rl_soft_collision_streak = 0
        self.rl_last_steering = 0.0

    def run_rl(self):
        """Main online RL loop using DDPG and Webots interaction."""
        mode_label = 'evaluation' if self.rl_eval else 'training'
        print(f"[Controller] Starting true deep reinforcement learning loop ({mode_label})...")
        print(f"[RL] Warmup phase: first {self.rl_agent.warmup_steps} steps collect experience without learning")
        print(f"[RL] After warmup, actor/critic networks will update and losses will be printed")

        episode_rewards = []  # Track rewards for progress
        episode_collision_flags = []  # Track whether episode had any collision
        episode_lane_center_rates = []  # Track lane-keeping quality trend
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
            progress = min(1.0, float(self.step_count) / float(self.rl_action_noise_decay_steps))
            current_noise = self.rl_action_noise_initial + progress * (self.rl_action_noise_final - self.rl_action_noise_initial)
            noise_scale = np.array(
                [
                    current_noise * self.rl_action_noise_steer_scale,
                    current_noise * self.rl_action_noise_speed_scale,
                ],
                dtype=np.float32,
            )
            action = self.rl_agent.select_action(state, explore=self.rl_train, noise_scale=noise_scale)

            model_prior_blend = 0.0
            prior_action = None
            if self.rl_use_model_prior and self.rl_train:
                try:
                    prior_steer_angle, prior_speed = self.control_with_ai(sensor_data)
                    max_steer = float(SENSOR_CONFIG.get('max_steering_angle', 1.57))
                    prior_action = np.array(
                        [
                            np.clip(float(prior_steer_angle) / max(max_steer, 1e-6), -1.0, 1.0),
                            np.clip(float(prior_speed), -1.0, 1.0),
                        ],
                        dtype=np.float32,
                    )

                    prior_progress = min(1.0, float(self.step_count) / float(self.rl_model_prior_decay_steps))
                    model_prior_blend = self.rl_model_prior_blend_initial + prior_progress * (
                        self.rl_model_prior_blend_final - self.rl_model_prior_blend_initial
                    )
                    model_prior_blend = float(np.clip(model_prior_blend, 0.0, 1.0))
                    action = (1.0 - model_prior_blend) * action + model_prior_blend * prior_action
                except Exception as exc:
                    if self.step_count % 500 == 0:
                        print(f"[RL] Model-prior action failed, falling back to pure RL action: {exc}")

            # Slew-rate limit in action space to prevent sudden jumps from exploration.
            prev_steer = float(self.rl_last_steering)
            action[0] = float(np.clip(action[0], prev_steer - self.rl_max_steer_delta_per_step, prev_steer + self.rl_max_steer_delta_per_step))
            prev_speed = float(np.clip((state[-3] * 2.0) - 1.0, -1.0, 1.0)) if len(state) >= 3 else 0.0
            action[1] = float(np.clip(action[1], prev_speed - self.rl_max_speed_delta_per_step, prev_speed + self.rl_max_speed_delta_per_step))

            # Early-episode action limiter: reduces violent steering and full throttle at episode start.
            if self.rl_train and self.rl_episode_step < self.rl_early_episode_limit_steps:
                action[0] = float(np.clip(action[0], -self.rl_early_episode_steer_limit, self.rl_early_episode_steer_limit))
                action[1] = float(np.clip(action[1], -1.0, self.rl_early_episode_speed_limit))

            action, assist_lane_offset, assist_lane_conf = self._apply_lane_assist_action(action, sensor_data)

            steering_cmd = float(action[0]) * float(SENSOR_CONFIG.get('max_steering_angle', 1.57))
            speed_cmd = float(action[1])

            # Emergency braking: Safety override (hard constraint, not learned)
            emergency_brake_enabled = bool(SAFETY_CONFIG.get('enable_safety_checks', True))
            if emergency_brake_enabled:
                # Get current sensor reading to decide if we need emergency brake for next step
                emergency_threshold = float(SAFETY_CONFIG.get('emergency_brake_lidar_threshold', 0.35))
                if sensor_data.get('lidar') is not None:
                    lidar_current = float(np.min(np.asarray(sensor_data['lidar'], dtype=np.float32)))
                    if lidar_current < emergency_threshold:
                        speed_cmd = -1.0  # Full reverse (emergency brake)
                        if self.step_count % 500 == 0:
                            print(f"[SAFETY] Emergency brake at step {self.step_count} (lidar={lidar_current:.2f}m)")
                
                # Red light override: Safety rule for traffic lights
                traffic_light = self._detect_traffic_light_from_camera(sensor_data.get('fisheye'))
                if traffic_light == 'red':
                    speed_cmd = -1.0  # Force brake on red light (hard constraint, not learned)
                    if self.step_count % 500 == 0:
                        print(f"[SAFETY] Red light override at step {self.step_count}")

            self.apply_control(steering_cmd, speed_cmd)
            self.rl_last_steering = float(action[0])
            
            # Debug: print action every 200 steps during warmup phase
            if self.step_count < 2500 and self.step_count % 200 == 0:
                learning_status = "WARMUP (no learning yet)" if len(self.rl_agent.replay) < self.rl_agent.warmup_steps else "LEARNING (networks updating)"
                print(
                    f"[RL] step={self.step_count}, {learning_status}, "
                    f"action=[{action[0]:+.3f}, {action[1]:+.3f}], "
                    f"steering_cmd={steering_cmd:+.3f}rad, speed_cmd={speed_cmd:+.3f}, "
                    f"noise={current_noise:.3f}, lane_assist=[off={assist_lane_offset:+.2f}, conf={assist_lane_conf:.2f}], "
                    f"model_prior_blend={model_prior_blend:.2f}, "
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

            lane_conf = float(info.get('lane_confidence', 0.0))
            lane_offset = float(info.get('lane_offset', 0.0))
            if lane_conf >= 0.15:
                self.rl_episode_lane_detected_steps += 1
                self.rl_episode_lane_abs_offset_sum += abs(lane_offset)
                if abs(lane_offset) < 0.3:
                    self.rl_episode_lane_centered_steps += 1

            if self.step_count % 100 == 0:
                loss_msg = ""
                if self.rl_train and self.rl_last_losses is not None:
                    loss_msg = (
                        f", actor_loss={self.rl_last_losses['actor_loss']:.6f}"
                        f", critic_loss={self.rl_last_losses['critic_loss']:.6f}"
                        f", updates={self.rl_last_losses['updates']}"
                    )
                learning_active = "✓LEARNING" if len(self.rl_agent.replay) >= self.rl_agent.warmup_steps else "warming up"
                sensor_info = (
                    f"lidar={info['lidar_min']:.2f}m, dist={info.get('dist_min', 10.0):.2f}m, "
                    f"lane_offset={info.get('lane_offset', 0.0):+.2f}, lane_conf={info.get('lane_confidence', 0.0):.2f}"
                )
                print(
                    f"[RL] [{learning_active}] step={self.step_count}, episode={self.rl_episode_idx}, "
                    f"ep_step={self.rl_episode_step}, ep_reward={self.rl_episode_reward:.2f}, {sensor_info}{loss_msg}"
                )

            if done:
                episode_rewards.append(self.rl_episode_reward)
                episode_collision_flags.append(1 if self.rl_episode_collision_count > 0 else 0)
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                collision_info = f"collisions={self.rl_episode_collision_count}" if self.rl_episode_collision_count > 0 else "collision=none"

                lane_detected_steps = max(1, self.rl_episode_lane_detected_steps)
                lane_center_rate = self.rl_episode_lane_centered_steps / lane_detected_steps
                lane_mean_abs_offset = self.rl_episode_lane_abs_offset_sum / lane_detected_steps
                lane_detect_rate = self.rl_episode_lane_detected_steps / max(1, self.rl_episode_step)
                episode_lane_center_rates.append(lane_center_rate)

                if lane_detect_rate > 0.8 and lane_mean_abs_offset > 0.6:
                    print(
                        "[Lane] WARNING: lane is detected but consistently off-center. "
                        "This suggests bias toward a single lane marking or camera pose mismatch."
                    )
                
                # Reward validation: detect anomalies that suggest reward hacking
                validate_rewards = bool(REWARD_VALIDATION.get('enable_validation', True))
                validation_flags = []
                if validate_rewards:
                    max_reward = float(REWARD_VALIDATION.get('max_single_step_reward', 10.0))
                    if self.rl_episode_reward > max_reward * self.rl_episode_step:
                        validation_flags.append("⚠️ ANOMALY: Reward/step too high (possible hacking)")
                    
                    min_reward = float(REWARD_VALIDATION.get('min_episode_reward', -100.0))
                    if self.rl_episode_reward < min_reward:
                        validation_flags.append("⚠️ ANOMALY: Very negative reward (trapped/crashing)")
                    
                    efficiency = self.rl_episode_reward / max(1, self.rl_episode_step)
                    min_efficiency = float(REWARD_VALIDATION.get('episode_efficiency_min', 0.1))
                    if efficiency < -min_efficiency and self.rl_episode_idx > 5:
                        validation_flags.append(f"⚠️ WARNING: Low efficiency ({efficiency:.3f}/step)")
                    
                    # Collision rate check
                    if self.rl_episode_idx > 0 and len(episode_collision_flags) >= 10:
                        collision_threshold = float(REWARD_VALIDATION.get('collision_rate_threshold', 0.5))
                        collision_count = int(sum(episode_collision_flags[-10:]))
                        if collision_count / 10 > collision_threshold:
                            validation_flags.append(f"⚠️ WARNING: High collision rate ({collision_count}/10 episodes)")

                    # Lane keeping trend check
                    if self.rl_episode_idx > 0 and len(episode_lane_center_rates) >= 10:
                        recent_center_rate = float(np.mean(episode_lane_center_rates[-10:]))
                        if recent_center_rate < 0.35:
                            validation_flags.append(f"⚠️ WARNING: Weak lane keeping (centered {recent_center_rate:.2f})")
                
                validation_str = " | " + " | ".join(validation_flags) if validation_flags else ""
                print(
                    f"[RL] Episode {self.rl_episode_idx} finished: "
                    f"steps={self.rl_episode_step}, reward={self.rl_episode_reward:.2f}, "
                    f"avg_10ep={avg_reward:.2f}, {collision_info}, "
                    f"lane_detect={lane_detect_rate:.2f}, lane_centered={lane_center_rate:.2f}, "
                    f"lane_abs_offset={lane_mean_abs_offset:.2f}{validation_str}"
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
    #              (can optionally blend supervised model prior)
    #   rl_hybrid -> RL training + supervised model prior enabled
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
        use_model = os.environ.get('NAWNAW_USE_MODEL_IN_RL', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        collect_data = False
        rl_train = True
        rl_eval = False
    elif mode == 'rl_hybrid':
        use_model = True
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
