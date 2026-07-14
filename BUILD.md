# U22-Car — Mini Autonomous Car Build Guide

## ✅ SELECTED BUILD (decided 2026-06-29)
For the programmer's own autonomous build (budget ~¥20k for the car, in Japan):

- **Car: LDRC 18401 PRO** (1/18, 4WD, ~¥12.7k). Chosen for **FOC sensored brushless
  motor (1700kv) with low-speed torque compensation** = smooth precise control at
  crawl speed (the hard part), **15 km/h** (ideal for indoor), **full proportional**
  steering + throttle, ~30 min runtime. It's a rock-crawler chassis (fine on a flat
  indoor track; great for parking). TX needs 2×AA (not included). Buy spare batteries.
- **Control: PCA9685 PWM board** (I2C) driving the steering servo + ESC from the Pi.
  Adapter is written: `Deep Reinforcement Learning(DRL)/pca9685_vehicle_io.py`
  (plugs into `drive_controller.py`; tested). Install on the Pi:
  `pip install adafruit-circuitpython-servokit opencv-python`, enable I2C.
- **Odometry:** stock ESC won't expose speed. Options: add external wheel encoders,
  OR (because the motor is **sensored**) swap the 10A ESC for a small **VESC** later
  to get smooth FOC control + clean wheel odometry.
- **Still to buy:** Raspberry Pi 5 (+ Hailo AI HAT later for real-time YOLO), camera,
  ToF sensors, lidar (when budget allows), IMU (BNO055), PCA9685 board, spare batteries.
- **Calibration TODO on assembly:** `steer_center_deg`, `steer_max_deg`, `steer_sign`,
  and `max_throttle` in the PCA9685 adapter; ESC arming/neutral.

(General guidance below still applies; this section is the concrete pick.)

---


Target capabilities (from the project policies): strong real-car-like autonomy,
safety-first obstacle avoidance + emergency stop, stop/reverse/reroute in
pedestrian spaces, object detection (lanes, traffic lights, humans, poles, curbs),
lane keeping, overtaking, auto-parking. Runs indoors on a mini track + classrooms.

This drives the parts: the car needs **onboard real-time camera perception (a GPU)**,
**lidar + close-range distance sensors for safety**, **Ackermann steering** (so it
drives/parks like a real car), and **wheel odometry + IMU** for motion/heading.

---

## 1. Brain (compute) + OS  — the most important choice

**Recommended: NVIDIA Jetson Orin Nano (8 GB) Developer Kit.**
- Has a real CUDA GPU -> runs CNN object detection (YOLO) + the RL policy in real
  time. A Raspberry Pi alone cannot do the vision in real time.
- **OS: Ubuntu (via NVIDIA JetPack)**, which includes CUDA, cuDNN, TensorRT.
- **Middleware: ROS 2 (Humble)** — the standard robotics framework to connect
  sensors, perception, planning, and motor control as modular "nodes."
- Inference: PyTorch + **TensorRT** (TensorRT makes models run much faster on Jetson).

Alternatives: Jetson Orin NX (more power, pricier); Raspberry Pi 5 + a **Hailo-8L**
or **Google Coral** AI accelerator (works, but more integration effort than Jetson).
Avoid plain Raspberry Pi / ESP32 as the main brain for this vision workload.

---

## 2. Chassis + drivetrain — make it drive like a real car

**Recommended: a 1/10-scale RC car chassis with Ackermann steering**
(front steering servo + rear drive motor). Ackermann = real-car-like turning, and
it's what makes realistic lane driving + parallel/auto parking possible. A
differential-drive bot (tank-style) cannot park or corner like a car.

Good starting points / kits:
- **F1TENTH** build (1/10, Ackermann, Jetson + lidar, ROS 2) — the gold-standard
  research platform for exactly this kind of car. Designed for camera+lidar autonomy.
- **Waveshare JetRacer Pro** (Jetson-based, Ackermann) — more turnkey.
- Any hobby 1/10 RC car + add a steering servo, ESC, and sensor mounts.

You need: a **steering servo**, an **ESC** (electronic speed controller) + drive
motor (brushed is simplest; brushless is faster), and a chassis with room for the
Jetson + battery + sensors. Keep top speed LOW for indoor/classroom safety.

---

## 3. Sensors

| Sensor | Recommended part | Why / used for |
|---|---|---|
| **Main camera** | Wide-angle / **fisheye CSI camera** (IMX219 ~160°) | Lane detection, traffic lights, object detection. Wide FOV matches your fisheye plan. |
| **Depth camera** (optional, strong) | **Intel RealSense D435i** | Gives true distance to obstacles ("stop before hitting") + has a **built-in IMU**. Big help for safety. |
| **Lidar (2D 360°)** | **RPLidar A1 / C1** | 360° obstacle ranging, detecting big blockages, mapping the mini track, reroute decisions. ROS-supported. |
| **Close-range distance** | **VL53L1X ToF** (or HC-SR04 ultrasonic) | Cheap bumper-level backstop for **emergency stop** when something is very close. Add 2-4 around the car. |
| **IMU** | **BNO055** (fused orientation) or the RealSense's IMU | Heading + turn-rate for smooth steering and odometry. |
| **Wheel encoders / speedometer** | Hall-effect or magnetic encoders (or an encoder motor) | Actual speed + odometry (how far it moved) — needed for control + parking. |
| **GPS** | **Skip for indoor.** (u-blox NEO-M8 only if outdoor) | GPS has NO signal indoors and is accurate to ~meters — useless at mini-car scale. Use wheel odometry + IMU + lidar instead for indoor localization. |

**Minimum safe set:** fisheye camera + RPLidar + 2-4 ToF/ultrasonic + IMU + wheel
encoders. The RealSense depth camera is the highest-value optional add for safety.

---

## 4. Power
- **LiPo battery** (2S or 3S) for the drive motor/ESC.
- A separate regulated **5V/≥4A UBEC** (or the Jetson's recommended supply) for the
  Jetson — don't power the Jetson directly off the motor rail (motor spikes crash it).
- Budget the Jetson Orin Nano at ~7-15 W; size the battery for your run time.

---

## 5. Software stack (on the Jetson)
- Ubuntu (JetPack) + **ROS 2 Humble**
- **PyTorch + TensorRT**, OpenCV
- Perception: **YOLO (v8/v11 nano)** for object detection (humans, poles, lights,
  cars) + a lightweight **lane segmentation** model (or YOLOP/HybridNets doing
  detection + lane + drivable-area in one network).
- Control: the trained **TD3 policy** + a safety layer (hard emergency-stop rules
  from lidar/ToF that override the policy).

---

## 6. Honest expectations (so the plan stays realistic)
- **"Zero retraining" sim-to-real is very hard.** Even excellent simulators have a
  reality gap (lighting, textures, tire friction, sensor noise). Realistic plan:
  train mostly in sim with **domain randomization**, then expect a **small amount of
  real-world fine-tuning / calibration**. Your fallback (train on the real car if
  needed) is the right mindset.
- **Build capabilities incrementally**, in this safety-first order:
  1. Emergency stop + basic obstacle avoidance (lidar/ToF hard rules)
  2. Lane keeping
  3. Object detection (YOLO)
  4. Stop / reverse / reroute in open (non-lane) spaces
  5. Overtaking
  6. Auto-parking
  Trying to train all of these at once will stall; each is its own milestone.
- **Safety layer is separate from the learned policy.** The emergency stop should be
  hard-coded rules on lidar/ToF distance that ALWAYS override the neural net — never
  rely solely on a learned model for "don't hit the human."
