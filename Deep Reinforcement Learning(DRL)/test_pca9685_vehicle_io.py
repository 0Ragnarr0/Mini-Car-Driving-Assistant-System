"""Tests for the PCA9685 VehicleIO mapping (stub mode, no hardware needed).

Verifies steer/throttle -> servo-angle/ESC mapping, the indoor throttle cap, the
reverse requirement, and that it plugs into DriveController. Run:
    python test_pca9685_vehicle_io.py
"""

import numpy as np
from pca9685_vehicle_io import PCA9685VehicleIO
from drive_controller import DriveController
from safety_governor import SensorReading


def test_steer_mapping():
    io = PCA9685VehicleIO(steer_center_deg=90, steer_max_deg=35, steer_sign=1)
    assert abs(io.steer_to_angle(0.0) - 90) < 1e-6            # center
    assert abs(io.steer_to_angle(1.0) - 125) < 1e-6           # full right
    assert abs(io.steer_to_angle(-1.0) - 55) < 1e-6           # full left
    # sign flip reverses direction
    io2 = PCA9685VehicleIO(steer_sign=-1)
    assert io2.steer_to_angle(1.0) < io2.steer_center_deg
    # never exceeds physical servo bounds
    io3 = PCA9685VehicleIO(steer_center_deg=170, steer_max_deg=40)
    assert 0.0 <= io3.steer_to_angle(1.0) <= 180.0
    print("  [ok] steer -> servo angle mapping (center/limits/sign/clamp)")


def test_throttle_cap_and_reverse():
    io = PCA9685VehicleIO(max_throttle=0.3, reverse_enabled=True)
    assert abs(io.throttle_to_esc(1.0) - 0.3) < 1e-6     # forward capped at 30%
    assert abs(io.throttle_to_esc(-1.0) + 0.3) < 1e-6    # reverse capped at -30%
    assert abs(io.throttle_to_esc(0.0)) < 1e-6           # neutral
    # reverse disabled -> no negative command (but then safety reverse won't work!)
    io_nr = PCA9685VehicleIO(max_throttle=0.3, reverse_enabled=False)
    assert io_nr.throttle_to_esc(-1.0) == 0.0
    print("  [ok] throttle cap + reverse handling")


def test_plugs_into_drive_controller():
    # No policy, no camera; supply a scripted range source. Safety should stop it
    # when an obstacle is dead ahead.
    readings = iter([
        SensorReading(front=5.0, speed=1.0),    # clear
        SensorReading(front=0.12, rear=5.0),    # blocked -> stop
    ])
    applied = []

    class TestIO(PCA9685VehicleIO):
        def read_ranges(self):
            return next(readings)
        def apply(self, steer, throttle):
            applied.append((self.steer_to_angle(steer), self.throttle_to_esc(throttle)))

    io = TestIO(range_source=lambda: None)  # stub kit
    ctrl = DriveController(io, policy=None, base_throttle=0.8)
    ctrl.step()  # clear -> forward
    ctrl.step()  # blocked -> brake
    assert applied[0][1] > 0.0, f"expected forward ESC, got {applied[0]}"
    assert applied[1][1] <= 0.0, f"expected brake/neutral ESC, got {applied[1]}"
    print("  [ok] PCA9685 adapter drives DriveController (forward then stop)")


if __name__ == "__main__":
    print("=== PCA9685 VehicleIO tests ===")
    test_steer_mapping()
    test_throttle_cap_and_reverse()
    test_plugs_into_drive_controller()
    print("ALL PCA9685 ADAPTER TESTS PASSED")
