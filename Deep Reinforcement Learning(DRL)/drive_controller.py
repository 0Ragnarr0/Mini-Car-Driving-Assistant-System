"""Combined drive controller: perception + policy + safety, one loop.

This is the orchestration layer that will run on the car (Jetson) and in sim. Each
step:
    1. read camera          -> perception (objects, lane, traffic light, lane flags)
    2. read range sensors   -> SensorReading (lidar / ToF / depth clearances + speed)
    3. policy proposes an action from the fused state
    4. SAFETY GOVERNOR overrides it (hard rules)   <-- never bypassed
    5. red traffic light = hard stop
    6. actuate (steering servo + motor)

Hardware-agnostic via the `VehicleIO` interface: implement read_camera /
read_ranges / apply once per platform (a MetaDrive/CARLA adapter, or the real
Jetson with its camera + lidar + ESC/servo), and this controller stays identical.

The policy is optional: with no policy it drives a gentle forward base action and
lets the safety layer handle everything -- useful for bringing up and testing the
sensor + safety wiring before the trained multimodal policy exists.
"""

import numpy as np

from perception import Perception, PerceptionResult
from safety_governor import SafetyGovernor, SensorReading


class VehicleIO:
    """Platform adapter. Subclass for sim or for the real Jetson car."""

    def read_camera(self):
        """Return an RGB image (H, W, 3) or None if no camera."""
        raise NotImplementedError

    def read_ranges(self):
        """Return a SensorReading with obstacle clearances (m) + speed (m/s).
        Lane flags are filled in by the controller from perception."""
        raise NotImplementedError

    def apply(self, steer, throttle):
        """Send (steer, throttle), each in [-1, 1], to the actuators."""
        raise NotImplementedError

    def ok(self):
        """Return False to end the run loop (e.g. sim episode over / shutdown)."""
        return True


def default_state_fn(perc, reading):
    """Build a compact policy input from perception + ranges.

    NOTE: the deployed policy must be TRAINED on this exact vector. Until the
    multimodal policy exists this is just a reasonable default for bring-up.
    """
    def n(d, scale=5.0):
        return float(np.clip(d / scale, 0.0, 1.0))
    return np.array([
        n(reading.front), n(reading.front_left), n(reading.front_right),
        n(reading.left), n(reading.right), n(reading.rear),
        float(np.clip(reading.speed / 5.0, 0.0, 1.0)),
        float(np.clip(perc.lane_offset, -1.0, 1.0)),
        float(np.clip(perc.lane_confidence, 0.0, 1.0)),
        1.0 if perc.can_go_left else 0.0,
        1.0 if perc.can_go_right else 0.0,
    ], dtype=np.float32)


class DriveController:
    def __init__(self, io, perception=None, policy=None, governor=None,
                 state_fn=None, base_throttle=0.5):
        self.io = io
        self.perception = perception if perception is not None else Perception()
        self.policy = policy
        self.governor = governor if governor is not None else SafetyGovernor()
        self.state_fn = state_fn or default_state_fn
        self.base_throttle = float(base_throttle)

    def reset(self):
        self.governor.reset()

    def step(self):
        # 1. Perception from the camera.
        cam = self.io.read_camera()
        perc = self.perception.perceive(cam) if cam is not None else PerceptionResult()

        # 2. Range sensors -> SensorReading; inject perception's lane flags.
        reading = self.io.read_ranges()
        reading.can_go_left = perc.can_go_left
        reading.can_go_right = perc.can_go_right

        # 3. Policy proposes (or a gentle forward base action during bring-up).
        if self.policy is not None:
            state = self.state_fn(perc, reading)
            raw = self.policy.select_action(state, explore=False)
            raw = (float(raw[0]), float(raw[1]))
        else:
            raw = (0.0, self.base_throttle)

        # 4. Safety governor: hard override.
        safe, info = self.governor.filter(raw, reading)

        # 5. Red light = hard stop (always overrides).
        if perc.traffic_light == "red":
            safe = (safe[0], min(safe[1], self.governor.cfg["brake_throttle"]))
            info = {**info, "reason": "RED_LIGHT_STOP"}

        # 6. Actuate.
        self.io.apply(safe[0], safe[1])
        return {
            "action": safe,
            "raw_action": raw,
            "reason": info["reason"],
            "state": self.governor.state.value,
            "traffic_light": perc.traffic_light,
            "perception": perc,
        }

    def run(self, max_steps=None):
        self.reset()
        n = 0
        while self.io.ok():
            self.step()
            n += 1
            if max_steps is not None and n >= max_steps:
                break
        return n
