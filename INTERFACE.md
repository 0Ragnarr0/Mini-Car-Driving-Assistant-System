# U22-Car — Hardware Interface Spec (software ↔ body)

This defines the boundary between the **AI/software** (Jetson + sensors, owned by the
programmer) and the **body/driving system** (chassis, steering, motors, power, owned
by the hardware team). Build to this contract so the two halves fit together.

The software boundary is the `VehicleIO` class in
`Deep Reinforcement Learning(DRL)/drive_controller.py`:
`read_camera()`, `read_ranges()`, `apply(steer, throttle)`.

---

## PART A — Hardware the SOFTWARE owns (programmer buys these)
These plug into the Jetson; the body must leave room + mounts for them.

| # | Item | Interface to Jetson | Purpose |
|---|---|---|---|
| 1 | **NVIDIA Jetson Orin Nano 8GB Dev Kit** | — (the brain) | runs perception + policy + safety |
| 2 | microSD ≥128GB (or NVMe SSD) | onboard | OS + models |
| 3 | **Fisheye / wide camera** (CSI or USB) | CSI ribbon / USB | object detection + lanes |
| 4 | **2D lidar** (RPLidar A1/C1) | USB | obstacle ranging, emergency stop, mapping |
| 5 | **ToF distance sensors ×2–4** (VL53L1X) | I2C | close-range emergency-stop backstop |
| 6 | **IMU** (BNO055) | I2C | heading / turn-rate |

---

## PART B — What the BODY must provide (hardware team builds to this)
These are the REQUIREMENTS the driving system must meet, or the software won't work.

### B1. Steering  (REQUIRED: Ackermann)
- A front **steering servo** (or steering actuator) the Jetson can command.
- Must accept a steering command that maps to **steer ∈ [-1, 1]**
  (-1 = full left, 0 = straight, +1 = full right).
- Needed for lane-keeping, overtaking, and **auto-parking** (differential/skid-steer
  cannot park like a car — Ackermann required).

### B2. Drive motor  (REQUIRED: forward + reverse + brake)
- A drive motor + **ESC / motor driver** the Jetson can command.
- Must accept **throttle ∈ [-1, 1]**: `>0` forward, `0` coast, `<0` brake/reverse.
- **Reverse MUST work** — the safety layer does stop → reverse → re-route when boxed
  in (pedestrian spaces / classrooms). No reverse = a core safety behavior is dead.

### B3. Speed / odometry feedback  (REQUIRED)
- **Wheel encoders** (or hall sensors) reporting wheel speed back to the Jetson.
- Software needs actual speed (m/s) for `SensorReading.speed` and control. Don't rely
  on commanded throttle as a speed estimate.

### B4. Motor-control bridge  (REQUIRED)
- A controller the Jetson talks to: **microcontroller (Arduino/ESP32) or VESC**, over
  **USB-serial / PWM / CAN**.
- Must accept commands at **≥20 Hz** (control loop rate) and ideally return encoder
  speed on the same link.
- This board is what `VehicleIO.apply(steer, throttle)` writes to.

### B5. Power  (REQUIRED: clean, separate)
- A **regulated 5V, ≥4A** supply for the Jetson — **separate from the motor rail.**
  Motor current spikes will brown out / crash the Jetson if shared. (UBEC or dedicated
  regulator off the main battery is fine.)
- Battery sized for run time; motors and Jetson on separate regulated lines.

### B6. Mounting + sensor placement  (REQUIRED)
- **Camera:** fixed forward mount, known height + tilt angle (needed for calibration).
  Wide/fisheye view of the road/path ahead, unobstructed by the body.
- **Lidar:** mounted with a clear ~360° (or at least forward 180°) horizontal view —
  the body must NOT block its scan plane.
- **ToF sensors:** at bumper corners (front-left, front-right, + optional rear) facing
  outward for close-range detection.
- **IMU:** mounted rigidly, **away from the motors** (motor magnetic fields disturb it).

---

## PART C — The command/feedback contract (agree on exact values)
| Signal | Direction | Range / units | Meaning |
|---|---|---|---|
| `steer` | Jetson → body | [-1, 1] | -1 left, +1 right (map to servo limits) |
| `throttle` | Jetson → body | [-1, 1] | >0 forward, <0 brake/reverse |
| `speed` | body → Jetson | m/s (≥0) | measured forward speed from encoders |
| (optional) `wheel_ticks` | body → Jetson | counts | odometry |

Update rate: ≥20 Hz both ways. Define the physical steering limits (max left/right
angle) and max speed so the [-1,1] range maps correctly.

---

## PART D — Team checklist (the "won't conflict" gate)
- [ ] Ackermann steering with a Jetson-commandable servo (B1)
- [ ] Motor + ESC that does forward, reverse, AND brake (B2)
- [ ] Wheel encoders wired to report speed (B3)
- [ ] Microcontroller/VESC bridge on USB-serial/PWM/CAN, ≥20 Hz (B4)
- [ ] Separate regulated 5V ≥4A for the Jetson, isolated from motor rail (B5)
- [ ] Mounts: camera (fwd, known angle), lidar (clear scan), ToF (corners), IMU (away
      from motors) (B6)
- [ ] Documented steering angle limits + max speed for the [-1,1] mapping (C)

Once these are met, the software side drops in via one `VehicleIO` adapter and
nothing above it changes.
