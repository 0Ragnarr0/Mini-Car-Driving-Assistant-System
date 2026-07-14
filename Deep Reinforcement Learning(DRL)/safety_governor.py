"""Safety-override governor for the autonomous mini car.

This is a HARD-RULE safety layer that wraps the learned policy. It takes the
policy's proposed action plus a normalized sensor view, and returns a *safe*
action that ALWAYS overrides the neural net when needed. This is how real
self-driving stacks work: the learned model proposes, a deterministic safety
layer disposes. Never trust a neural net alone for "don't hit the human."

It encodes the project's safety policy:
  * Slow down as obstacles get closer (speed governor).
  * Steer around an obstacle ONLY if there's clear room AND it does not require
    crossing into a forbidden area / opposite lane (can_go_left / can_go_right
    flags come from perception). Otherwise come to a full STOP.
  * If boxed in (blocked and cannot steer around), run a stop -> reverse -> turn
    -> resume recovery so the car can back out and find another route. This covers
    pedestrian spaces / classrooms where there is no lane to follow.

Framework-agnostic: it works on plain numbers, so the same governor plugs into
MetaDrive, CARLA, or the real car. Distances are in METERS, speed in m/s, and the
action is (steer, throttle) each in [-1, 1] (steer: -1 = full left, +1 = full
right; throttle: >0 forward, <0 brake/reverse).
"""

from dataclasses import dataclass
from enum import Enum


class SafetyState(Enum):
    NORMAL = "normal"
    REVERSING = "reversing"
    TURNING = "turning"


@dataclass
class SensorReading:
    """Normalized obstacle clearances (meters) around the car + current speed."""
    front: float          # nearest obstacle in the forward cone
    front_left: float = 10.0
    front_right: float = 10.0
    left: float = 10.0
    right: float = 10.0
    rear: float = 10.0
    speed: float = 0.0    # current forward speed, m/s (>= 0)
    # Perception flags: is steering that way allowed (NOT into opposite lane /
    # off-limits)? Default True = allowed (e.g. open pedestrian space).
    can_go_left: bool = True
    can_go_right: bool = True


DEFAULT_SAFETY_CONFIG = {
    "hard_distance": 0.15,        # contact-imminent -> emergency stop / reverse
    "stop_distance": 0.30,        # must stop unless it can steer around
    "avoid_distance": 0.70,       # try to steer around within this range
    "slow_distance": 1.00,        # start slowing within this range
    "side_clear_distance": 0.35,  # a side counts as "clear" above this
    "avoid_steer": 0.6,           # steering magnitude used for avoidance/turning
    "brake_throttle": -1.0,       # throttle command to brake/stop
    "reverse_throttle": -0.5,     # throttle while reversing
    "turn_throttle": 0.25,        # gentle forward throttle while turning out
    "stuck_steps_to_reverse": 8,  # blocked this many steps -> start reversing
    "reverse_steps": 12,          # how long to reverse before turning
    "turn_steps": 12,             # how long to turn before resuming
}


class SafetyGovernor:
    """Wraps a policy action with deterministic safety rules."""

    def __init__(self, config=None):
        cfg = dict(DEFAULT_SAFETY_CONFIG)
        if config:
            cfg.update(config)
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = SafetyState.NORMAL
        self.block_counter = 0
        self.timer = 0

    # ------------------------------------------------------------------ helpers
    def _clearer_side(self, r):
        """Return +1 to steer right, -1 to steer left, toward more open space,
        respecting the lane/opposite-lane flags. Returns 0 if neither side allowed."""
        right_ok = r.can_go_right and r.front_right >= self.cfg["side_clear_distance"]
        left_ok = r.can_go_left and r.front_left >= self.cfg["side_clear_distance"]
        if right_ok and left_ok:
            return 1 if r.front_right >= r.front_left else -1
        if right_ok:
            return 1
        if left_ok:
            return -1
        return 0

    def _governed_throttle(self, throttle, front):
        """Cap forward throttle based on front clearance; braking always allowed."""
        stop_d, slow_d = self.cfg["stop_distance"], self.cfg["slow_distance"]
        if front >= slow_d:
            cap = 1.0
        elif front <= stop_d:
            cap = 0.0
        else:
            cap = (front - stop_d) / max(slow_d - stop_d, 1e-6)
        return float(min(throttle, cap))

    # --------------------------------------------------------------------- main
    def filter(self, action, reading):
        """Return (safe_action, info). safe_action = (steer, throttle) in [-1,1]."""
        steer = float(max(-1.0, min(1.0, action[0])))
        throttle = float(max(-1.0, min(1.0, action[1])))
        r = reading
        cfg = self.cfg

        # Active recovery (reverse / turn) takes over completely.
        if self.state in (SafetyState.REVERSING, SafetyState.TURNING):
            return self._handle_recovery(r)

        # --- NORMAL state: layered hard rules ---

        # 1. Contact imminent -> emergency stop, escalate to reverse if stuck.
        if r.front <= cfg["hard_distance"]:
            self.block_counter += 1
            if (self.block_counter >= cfg["stuck_steps_to_reverse"]
                    and r.rear >= cfg["side_clear_distance"]):
                self.state = SafetyState.REVERSING
                self.timer = cfg["reverse_steps"]
                return self._handle_recovery(r)
            return (0.0, cfg["brake_throttle"]), self._info("EMERGENCY_STOP", r)

        # 2. Within stop distance -> steer around if safely possible, else STOP.
        if r.front < cfg["stop_distance"]:
            direction = self._clearer_side(r)
            if direction != 0:
                self.block_counter = 0
                throttle = self._governed_throttle(abs(cfg["turn_throttle"]), r.front)
                return (direction * cfg["avoid_steer"], throttle), self._info("AVOID_STEER", r)
            # Cannot steer away without crossing into forbidden space -> stop.
            self.block_counter += 1
            if (self.block_counter >= cfg["stuck_steps_to_reverse"]
                    and r.rear >= cfg["side_clear_distance"]):
                self.state = SafetyState.REVERSING
                self.timer = cfg["reverse_steps"]
                return self._handle_recovery(r)
            return (0.0, cfg["brake_throttle"]), self._info("STOP_BLOCKED", r)

        # 3. Obstacle ahead but some room -> bias steering to the clear side.
        if r.front < cfg["avoid_distance"]:
            self.block_counter = 0
            direction = self._clearer_side(r)
            throttle = self._governed_throttle(throttle, r.front)
            if direction != 0:
                # Blend policy steer toward the avoidance direction.
                steer = 0.5 * steer + 0.5 * direction * cfg["avoid_steer"]
                return (steer, throttle), self._info("AVOID_BIAS", r)
            return (steer, throttle), self._info("SLOW_NO_PATH", r)

        # 4. Approaching -> just slow down, keep policy steering.
        if r.front < cfg["slow_distance"]:
            self.block_counter = 0
            return (steer, self._governed_throttle(throttle, r.front)), self._info("SLOW", r)

        # 5. Clear road -> pass the policy action through unchanged.
        self.block_counter = 0
        return (steer, throttle), self._info("CLEAR", r)

    def _handle_recovery(self, r):
        cfg = self.cfg
        if self.state == SafetyState.REVERSING:
            self.timer -= 1
            # Stop reversing if rear gets blocked or time is up.
            if r.rear <= cfg["hard_distance"] or self.timer <= 0:
                self.state = SafetyState.TURNING
                self.timer = cfg["turn_steps"]
                return (0.0, 0.0), self._info("REVERSE_TO_TURN", r)
            # Angle the back end toward the clearer side while backing up.
            direction = self._clearer_side(r)
            return (direction * cfg["avoid_steer"], cfg["reverse_throttle"]), \
                self._info("REVERSING", r)

        # TURNING: creep forward steering toward open space until the front clears.
        self.timer -= 1
        if r.front > cfg["avoid_distance"] or self.timer <= 0:
            self.state = SafetyState.NORMAL
            self.block_counter = 0
            return (0.0, 0.0), self._info("TURN_TO_NORMAL", r)
        direction = self._clearer_side(r) or 1
        return (direction * cfg["avoid_steer"], cfg["turn_throttle"]), \
            self._info("TURNING", r)

    def _info(self, reason, r):
        return {
            "reason": reason,
            "state": self.state.value,
            "block_counter": self.block_counter,
            "front": r.front,
        }
