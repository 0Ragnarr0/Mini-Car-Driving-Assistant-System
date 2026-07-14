"""PCA9685 VehicleIO adapter for the LDRC 18401 PRO (or any RC car).

This is the concrete `VehicleIO` implementation that runs on the Raspberry Pi and
drives the physical car:
  * steering servo  <- PWM channel on a PCA9685 board (I2C)
  * ESC (drive motor) <- PWM channel on the PCA9685 (RC ESC is servo-style PWM)

It plugs straight into drive_controller.py: DriveController(io=PCA9685VehicleIO(...)).

Actuation (`apply`) is fully implemented here. Sensing is pluggable:
  * read_camera(): OpenCV capture if available, else a supplied callable.
  * read_ranges(): you MUST supply a range_source (lidar/ToF) before real driving;
    the default returns a "clear" reading and prints a loud warning, because
    without range sensing the safety layer is blind.

Hardware libs (adafruit-circuitpython-servokit, opencv) are imported lazily, so
this file imports/compiles fine on a dev machine without them (stub mode) for
testing the mapping logic.

On the Pi you'll need:
    pip install adafruit-circuitpython-servokit opencv-python
    # and I2C enabled (raspi-config)
"""

import numpy as np

try:
    from adafruit_servokit import ServoKit
except Exception:  # pragma: no cover - not present on dev machine
    ServoKit = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from drive_controller import VehicleIO
from safety_governor import SensorReading


class PCA9685VehicleIO(VehicleIO):
    def __init__(
        self,
        steer_channel=0,
        throttle_channel=1,
        i2c_address=0x40,
        pwm_freq=50,
        # Steering calibration (tune to the car's linkage):
        steer_center_deg=90.0,
        steer_max_deg=35.0,     # max deflection each side from center
        steer_sign=1,           # flip to -1 if left/right are reversed
        # Throttle safety cap (indoor): fraction of full ESC range used.
        max_throttle=0.30,      # cap forward/reverse magnitude to 30%
        reverse_enabled=True,   # REQUIRED for the safety layer's reverse recovery
        # Sensing hooks:
        camera_index=0,
        camera_source=None,     # callable() -> RGB image (overrides OpenCV)
        range_source=None,      # callable() -> SensorReading (lidar/ToF)
    ):
        self.steer_channel = steer_channel
        self.throttle_channel = throttle_channel
        self.steer_center_deg = float(steer_center_deg)
        self.steer_max_deg = float(steer_max_deg)
        self.steer_sign = 1 if steer_sign >= 0 else -1
        self.max_throttle = float(np.clip(max_throttle, 0.0, 1.0))
        self.reverse_enabled = bool(reverse_enabled)
        self.camera_index = camera_index
        self.camera_source = camera_source
        self.range_source = range_source
        self._warned_no_ranges = False

        # PCA9685 via ServoKit (graceful stub if the lib isn't installed).
        if ServoKit is None:
            self.kit = None
            print("[PCA9685] adafruit-servokit not installed -> STUB mode "
                  "(apply() logs instead of driving hardware).")
        else:
            self.kit = ServoKit(channels=16, address=i2c_address, frequency=pwm_freq)
            # Arm the ESC at neutral.
            self.kit.continuous_servo[self.throttle_channel].throttle = 0.0
            self.kit.servo[self.steer_channel].angle = self.steer_center_deg

        # OpenCV camera capture (opened lazily).
        self._cap = None

    # ---------------------------------------------------- pure mapping (tested)
    def steer_to_angle(self, steer):
        """steer in [-1,1] -> servo angle in degrees (0..180)."""
        steer = float(np.clip(steer, -1.0, 1.0))
        angle = self.steer_center_deg + self.steer_sign * steer * self.steer_max_deg
        return float(np.clip(angle, 0.0, 180.0))

    def throttle_to_esc(self, throttle):
        """throttle in [-1,1] -> ESC command in [-1,1], capped for safety."""
        throttle = float(np.clip(throttle, -1.0, 1.0))
        cmd = throttle * self.max_throttle
        if not self.reverse_enabled:
            cmd = max(0.0, cmd)
        return float(np.clip(cmd, -1.0, 1.0))

    # ------------------------------------------------------------ VehicleIO API
    def apply(self, steer, throttle):
        angle = self.steer_to_angle(steer)
        esc = self.throttle_to_esc(throttle)
        if self.kit is None:
            return  # stub mode
        self.kit.servo[self.steer_channel].angle = angle
        self.kit.continuous_servo[self.throttle_channel].throttle = esc

    def read_camera(self):
        if self.camera_source is not None:
            return self.camera_source()
        if cv2 is None:
            return None
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_index)
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        return frame_bgr[:, :, ::-1].copy()  # BGR -> RGB

    def read_ranges(self):
        if self.range_source is not None:
            return self.range_source()
        # No range sensor wired: safety layer would be blind. Warn once and
        # report "clear" so bring-up can proceed -- REPLACE before real driving.
        if not self._warned_no_ranges:
            self._warned_no_ranges = True
            print("[PCA9685] WARNING: no range_source supplied. Lidar/ToF must be "
                  "wired in before autonomous driving -- the safety layer is blind "
                  "without it. Returning a 'clear' placeholder reading.")
        return SensorReading(front=10.0, speed=0.0)

    def ok(self):
        return True

    def stop(self):
        """Cut throttle + center steering (call on shutdown / emergency)."""
        if self.kit is not None:
            self.kit.continuous_servo[self.throttle_channel].throttle = 0.0
            self.kit.servo[self.steer_channel].angle = self.steer_center_deg

    def close(self):
        self.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
