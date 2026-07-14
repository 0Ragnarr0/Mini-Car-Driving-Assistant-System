"""End-to-end wiring test for DriveController (no simulator/GPU/model needed).

Uses a mock vehicle + a scripted scene to confirm perception + safety + actuation
are wired correctly. Run: python test_drive_controller.py
"""

import numpy as np

from drive_controller import DriveController, VehicleIO
from safety_governor import SensorReading
from perception import PerceptionResult


class MockIO(VehicleIO):
    """Scriptable mock car: feed it a list of SensorReadings and an optional camera."""

    def __init__(self, readings, camera=None):
        self.readings = readings
        self.camera = camera
        self.i = 0
        self.applied = []  # record (steer, throttle) each step

    def read_camera(self):
        return self.camera

    def read_ranges(self):
        r = self.readings[min(self.i, len(self.readings) - 1)]
        return r

    def apply(self, steer, throttle):
        self.applied.append((steer, throttle))
        self.i += 1

    def ok(self):
        return self.i < len(self.readings)


class FakePerception:
    """Returns a fixed PerceptionResult (e.g. to simulate a red light)."""

    def __init__(self, result):
        self.result = result

    def perceive(self, image):
        return self.result


def test_forward_when_clear_then_stop_when_blocked():
    readings = [
        SensorReading(front=5.0, speed=1.0),   # clear -> forward
        SensorReading(front=5.0, speed=1.0),
        SensorReading(front=0.12, rear=5.0),   # blocked -> stop
        SensorReading(front=0.12, rear=5.0),
    ]
    io = MockIO(readings)
    ctrl = DriveController(io, policy=None, base_throttle=0.6)
    ctrl.run()

    # First steps: forward (positive throttle). Later steps: braking (<= 0).
    assert io.applied[0][1] > 0.0, f"expected forward, got {io.applied[0]}"
    assert io.applied[2][1] <= 0.0, f"expected brake, got {io.applied[2]}"
    print("  [ok] drives forward when clear, stops when blocked")


def test_red_light_forces_stop():
    red = PerceptionResult(traffic_light="red", image_shape=(90, 160))
    io = MockIO([SensorReading(front=5.0, speed=1.0)], camera=np.zeros((90, 160, 3)))
    ctrl = DriveController(io, perception=FakePerception(red), policy=None,
                           base_throttle=0.8)
    out = ctrl.step()
    assert out["reason"] == "RED_LIGHT_STOP"
    assert io.applied[-1][1] <= 0.0, "red light should force braking"
    print("  [ok] red traffic light forces a stop even on a clear road")


def test_policy_action_is_safety_filtered():
    # Policy that always floors it; obstacle dead ahead -> safety must override.
    class FlooringPolicy:
        def select_action(self, state, explore=False):
            return np.array([0.0, 1.0], dtype=np.float32)

    io = MockIO([SensorReading(front=0.12, rear=5.0)])
    ctrl = DriveController(io, policy=FlooringPolicy())
    out = ctrl.step()
    assert out["raw_action"][1] == 1.0           # policy wanted full throttle
    assert io.applied[-1][1] <= 0.0              # safety braked instead
    assert out["reason"] == "EMERGENCY_STOP"
    print("  [ok] safety governor overrides a reckless policy action")


if __name__ == "__main__":
    print("=== DriveController wiring tests ===")
    test_forward_when_clear_then_stop_when_blocked()
    test_red_light_forces_stop()
    test_policy_action_is_safety_filtered()
    print("ALL DRIVE CONTROLLER TESTS PASSED")
