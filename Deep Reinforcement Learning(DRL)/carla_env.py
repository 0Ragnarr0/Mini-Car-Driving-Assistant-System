"""CARLA environment wrapper with a gym-style interface for the TD3 agent.

Exposes:
    env = CarlaEnv(...)
    state = env.reset()
    next_state, reward, done, info = env.step(action)   # action = [steer, throttle] in [-1, 1]
    env.close()

Design choices (these fix the bugs from the old Webots pipeline):
  * Reward uses the vehicle's ACTUAL measured speed projected onto the lane
    direction (real forward progress), not the commanded throttle. No
    "floor-it-against-a-wall" exploit.
  * Lane offset and heading error come from CARLA's ground-truth waypoints
    (map API), not a fragile RGB color filter.
  * The action stored for learning == the action actually applied (the env
    clips once and applies exactly that; no hidden overrides).
  * Runs CARLA in synchronous fixed-step mode for reproducible RL.

State vector (compact, low-dim so the MLP policy learns fast):
    [ lidar_bins (min distance per angular sector, normalized) ,
      speed_norm, lateral_offset_norm, heading_error_norm, prev_steer ]
    -> state_dim = lidar_bins + 4

Camera is attached but NOT used in the state yet. That is the hook for the
perception upgrade (YOLOP/CNN features) later -- see get_camera_image().

Requires the CARLA Python API (`pip install carla`, version matching your
CARLA server) and a running CARLA server (e.g. ./CarlaUE4.sh).
"""

import math
import time
import queue
import random
import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover - only triggers without CARLA
    carla = None
    _CARLA_IMPORT_ERROR = exc


class CarlaEnv:
    def __init__(
        self,
        host="localhost",
        port=2000,
        town="Town03",
        dt=0.05,
        lidar_bins=12,
        lidar_range=50.0,
        max_episode_steps=1000,
        target_speed_kmh=30.0,
        image_size=(84, 84),
        vehicle_filter="vehicle.tesla.model3",
        seed=None,
    ):
        if carla is None:
            raise ImportError(
                "The CARLA Python API is not installed. Install it with "
                "`pip install carla` (matching your CARLA server version) and "
                f"make sure a CARLA server is running.\nOriginal error: {_CARLA_IMPORT_ERROR}"
            )

        self.dt = float(dt)
        self.lidar_bins = int(lidar_bins)
        self.lidar_range = float(lidar_range)
        self.max_episode_steps = int(max_episode_steps)
        self.target_speed = float(target_speed_kmh) / 3.6  # m/s
        self.image_size = image_size
        self.vehicle_filter = vehicle_filter

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Interface dims (used by the agent)
        self.action_dim = 2
        self.state_dim = self.lidar_bins + 4

        # Connect to the simulator
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.map.get_spawn_points()

        # Synchronous fixed-step mode for reproducible RL
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        # Actors / sensors (created on reset)
        self.vehicle = None
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        self._collision = False
        self._lidar_queue = queue.Queue()
        self._camera_queue = queue.Queue()
        self._latest_lidar = None
        self._latest_image = None

        self.episode_step = 0
        self.prev_steer = 0.0

    # ------------------------------------------------------------------ setup
    def _spawn_vehicle(self):
        bp = self.bp_lib.filter(self.vehicle_filter)[0]
        for _ in range(20):
            spawn = random.choice(self.spawn_points)
            actor = self.world.try_spawn_actor(bp, spawn)
            if actor is not None:
                self.vehicle = actor
                return
        raise RuntimeError("Could not spawn a vehicle after 20 attempts.")

    def _attach_collision_sensor(self):
        bp = self.bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self._collision = True

    def _attach_lidar(self):
        bp = self.bp_lib.find("sensor.lidar.ray_cast")
        bp.set_attribute("range", str(self.lidar_range))
        bp.set_attribute("channels", "32")
        bp.set_attribute("points_per_second", "100000")
        bp.set_attribute("rotation_frequency", str(1.0 / self.dt))
        bp.set_attribute("upper_fov", "10")
        bp.set_attribute("lower_fov", "-15")
        transform = carla.Transform(carla.Location(x=0.0, z=2.4))
        self.lidar_sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.lidar_sensor.listen(self._lidar_queue.put)

    def _attach_camera(self):
        bp = self.bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.image_size[0]))
        bp.set_attribute("image_size_y", str(self.image_size[1]))
        bp.set_attribute("fov", "90")
        transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera_sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.camera_sensor.listen(self._camera_queue.put)

    def _destroy_actors(self):
        for sensor in (self.collision_sensor, self.lidar_sensor, self.camera_sensor):
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
        self.collision_sensor = self.lidar_sensor = self.camera_sensor = self.vehicle = None

    # ------------------------------------------------------------------ api
    def reset(self):
        self._destroy_actors()
        self._collision = False
        self.episode_step = 0
        self.prev_steer = 0.0
        with self._lidar_queue.mutex:
            self._lidar_queue.queue.clear()
        with self._camera_queue.mutex:
            self._camera_queue.queue.clear()

        self._spawn_vehicle()
        self._attach_collision_sensor()
        self._attach_lidar()
        self._attach_camera()

        # Let physics settle and sensors produce a first frame.
        for _ in range(4):
            self.world.tick()
        self._pull_sensors()
        return self._build_state()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        steer = float(action[0])
        throttle_brake = float(action[1])

        control = carla.VehicleControl()
        control.steer = steer
        if throttle_brake >= 0.0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
        self.vehicle.apply_control(control)

        self.world.tick()
        self._pull_sensors()
        self.episode_step += 1

        state = self._build_state()
        reward, done, info = self._reward_done(action)
        self.prev_steer = steer
        # The applied action equals `action` (clipped once above), so the caller
        # can safely store `action` in the replay buffer for correct learning.
        info["applied_action"] = action
        return state, reward, done, info

    def close(self):
        self._destroy_actors()
        try:
            self.world.apply_settings(self._original_settings)
        except Exception:
            pass

    # ------------------------------------------------------------- internals
    def _pull_sensors(self):
        # Drain queues; keep the most recent measurement from this tick.
        try:
            while True:
                self._latest_lidar = self._lidar_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                self._latest_image = self._camera_queue.get_nowait()
        except queue.Empty:
            pass

    def _lidar_sectors(self):
        """Min distance per angular sector, normalized to [0, 1]; 1.0 = clear."""
        bins = np.ones(self.lidar_bins, dtype=np.float32)
        meas = self._latest_lidar
        if meas is None:
            return bins
        pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
        if pts.shape[0] == 0:
            return bins
        x, y = pts[:, 0], pts[:, 1]
        dist = np.sqrt(x * x + y * y)
        ground = pts[:, 2] < -1.5  # drop near-ground returns
        valid = (~ground) & (dist > 0.5)
        x, y, dist = x[valid], y[valid], dist[valid]
        if dist.shape[0] == 0:
            return bins
        ang = np.arctan2(y, x)  # [-pi, pi], 0 = forward
        idx = ((ang + math.pi) / (2 * math.pi) * self.lidar_bins).astype(np.int32)
        idx = np.clip(idx, 0, self.lidar_bins - 1)
        for b in range(self.lidar_bins):
            sel = dist[idx == b]
            if sel.size:
                bins[b] = float(np.clip(sel.min() / self.lidar_range, 0.0, 1.0))
        return bins

    def _lane_metrics(self):
        """Signed lateral offset (m) and heading error (rad) vs lane center."""
        tf = self.vehicle.get_transform()
        loc = tf.location
        wp = self.map.get_waypoint(loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0, 0.0

        # Lateral offset: vector from lane center to vehicle, onto lane's right axis.
        wp_loc = wp.transform.location
        wp_yaw = math.radians(wp.transform.rotation.yaw)
        right = np.array([math.cos(wp_yaw + math.pi / 2.0),
                          math.sin(wp_yaw + math.pi / 2.0)])
        delta = np.array([loc.x - wp_loc.x, loc.y - wp_loc.y])
        lateral = float(np.dot(delta, right))

        veh_yaw = math.radians(tf.rotation.yaw)
        heading_err = math.atan2(math.sin(veh_yaw - wp_yaw), math.cos(veh_yaw - wp_yaw))
        return lateral, heading_err

    def _speed(self):
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)  # m/s

    def _forward_speed(self):
        """Speed component along the vehicle's forward axis (real progress)."""
        v = self.vehicle.get_velocity()
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        return float(np.dot([v.x, v.y], fwd))

    def _build_state(self):
        lidar = self._lidar_sectors()
        lateral, heading_err = self._lane_metrics()
        speed_norm = float(np.clip(self._speed() / max(self.target_speed, 0.1), 0.0, 2.0))
        lateral_norm = float(np.clip(lateral / 2.0, -1.0, 1.0))
        heading_norm = float(np.clip(heading_err / math.pi, -1.0, 1.0))
        extras = np.array([speed_norm, lateral_norm, heading_norm, self.prev_steer],
                          dtype=np.float32)
        return np.concatenate([lidar, extras]).astype(np.float32)

    def _reward_done(self, action):
        lateral, heading_err = self._lane_metrics()
        fwd_speed = self._forward_speed()

        # Progress: reward real forward speed, but not exceeding target speed.
        speed_reward = (fwd_speed / self.target_speed) if fwd_speed > 0 else fwd_speed
        if fwd_speed > self.target_speed:
            speed_reward = 1.0 - (fwd_speed - self.target_speed) / self.target_speed
        speed_reward = float(np.clip(speed_reward, -1.0, 1.0))

        lateral_pen = 0.5 * abs(lateral)                 # stay in lane center
        heading_pen = 0.3 * abs(heading_err)             # face down the lane
        steer_pen = 0.05 * abs(float(action[0]) - self.prev_steer)  # smoothness

        reward = speed_reward - lateral_pen - heading_pen - steer_pen

        done = False
        info = {
            "collision": False,
            "fwd_speed": fwd_speed,
            "lateral": lateral,
            "heading_err": heading_err,
        }

        if self._collision:
            reward -= 50.0
            done = True
            info["collision"] = True
        elif abs(lateral) > 3.0:  # ran off the lane / road
            reward -= 10.0
            done = True
            info["off_lane"] = True
        elif self.episode_step >= self.max_episode_steps:
            done = True
            info["timeout"] = True

        return float(reward), done, info

    def get_camera_image(self):
        """Latest RGB frame as (H, W, 3) uint8. Hook for CNN/YOLOP perception."""
        img = self._latest_image
        if img is None:
            return None
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
            (img.height, img.width, 4)
        )
        return arr[:, :, [2, 1, 0]].copy()  # BGRA -> RGB
