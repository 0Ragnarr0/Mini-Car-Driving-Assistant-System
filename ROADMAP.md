# U22-Car — Improvement Roadmap

Goal: a strong, high-quality autonomous-driving model with real perception
(object recognition + lane detection) feeding the driving policy.

## STATUS (implemented so far)
- [x] Algorithm upgraded DDPG -> **TD3** in `rl_agent.py` (twin critics,
      target-policy smoothing, delayed actor updates). Interface unchanged;
      `DDPGAgent` is now an alias of `TD3Agent`.
- [x] TD3 hyperparameters added to `config.py` (policy_delay, target_noise,
      noise_clip).
- [x] Standalone algorithm test `test_rl_agent.py` (pure numpy, no simulator).
      PASSES: learning test return improved ~ -275 -> +6.5.
- [x] **CARLA** environment wrapper `carla_env.py` (gym-style reset/step/close,
      synchronous mode, lidar sectors + ground-truth lane metrics, reward on
      real forward speed, action stored == action applied).
- [x] CARLA training loop `train_carla.py` (warmup + TD3 + checkpoints).
- [x] **MetaDrive** env wrapper `metadrive_env.py` + loop `train_metadrive.py`
      (lightweight, runs on integrated GPU/CPU; same TD3 agent).
- [x] **MetaDrive validation run** (272k steps): TD3 learns to drive; but trained on
      1 map -> overfit. `num_scenarios` added (train default 1000) for generalization.
- [x] **Safety-override layer** `safety_governor.py` (+ tests): hard-rule emergency
      stop, in-lane avoidance, refuses opposite-lane swerve, stop->reverse->turn
      recovery. Always overrides the policy. See HARDWARE/north-star policies.
- [x] **Perception scaffold** `perception.py` (+ tests, `PERCEPTION.md`): YOLO object
      detection + traffic-light state + lane hook + lane flags. Stub mode now; train
      on BDD100K + mini-track frames on the RTX 5060.
- [x] `BUILD.md` — mini-car hardware build guide (Jetson Orin Nano, Ackermann chassis,
      sensors; skip GPS indoors).
- [x] **Combined controller** `drive_controller.py` (+ tests): one loop wiring
      perception + policy + safety + red-light stop, via a `VehicleIO` adapter so the
      SAME controller runs in sim and on the Jetson. Policy optional (bring-up mode).

### Quick tests (no simulator/GPU needed)
    cd "Deep Reinforcement Learning(DRL)"
    python test_rl_agent.py            # TD3 algorithm
    python test_safety_governor.py     # safety override layer
    python test_perception.py          # perception plumbing
    python test_drive_controller.py    # full perception+safety+policy wiring

## PLAN: two-track (CARLA needs a dedicated 6-8 GB GPU; integrated GPU can't run it)
  * **Track 1 (now, local):** MetaDrive on this machine to develop/validate the
    TD3 driving policy. Free, fast, runs on integrated GPU.
  * **Track 2 (later, cloud):** CARLA on a RunPod GPU for realistic camera +
    perfect-label data, used for the YOLOP/CNN perception upgrade.

### Run locally now (MetaDrive) — with live learning progress
    pip install metadrive-simulator matplotlib
    cd "Deep Reinforcement Learning(DRL)"
    # Terminal 1 — train (headless; writes a CSV + auto-updating progress PNG):
    python train_metadrive.py --episodes 2000 --traffic-density 0 --num-scenarios 30
    # Terminal 2 — watch the learning curve live:
    python plot_progress.py
    # Progress PNG + CSV are saved in the logs/ folder each run.
    # Watch the trained car drive:
    python eval_metadrive.py --ckpt td3_metadrive_ep_100 --episodes 3 --render

### Run later on RunPod (CARLA)
    # 1. start server:   ./CarlaUE4.sh -quality-level=Low
    # 2. pip install carla   (version must match the server)
    # 3. cd "Deep Reinforcement Learning(DRL)" && python train_carla.py --episodes 500

### Verify the algorithm anytime (no simulator needed)
    cd "Deep Reinforcement Learning(DRL)" && python test_rl_agent.py


---

## PART 1 — Step-by-step plan to fix the current RL pipeline

These fixes make the *existing* DDPG setup actually learn correctly. Do these
first; they are cheap and high-impact. (Each step lists the file to touch.)

### Phase A — Correctness bugs (do first)

**Step 1. Learn from the action you actually executed.**
File: `controllers/nawnaw_robot/nawnaw_robot.py` (`run_rl`)
- Today, emergency-brake / red-light overrides change a local `speed_cmd`, and
  `apply_control` reduces speed when steering — but the replay buffer stores the
  ORIGINAL `action`. The agent learns from actions it never took.
- Fix: build the final executed action vector AFTER all overrides, and use THAT
  for both `store_transition(...)` and `_compute_rl_reward_done(...)`.
  Concretely: have `apply_control` return the (steer_norm, speed_norm) it really
  applied, and store/reward that.

**Step 2. Reward real motion, not the throttle command.**
File: `nawnaw_robot.py` (`_compute_rl_reward_done`)
- Replace `forward_reward = (speed_cmd+1)/2` with progress measured from GPS:
  `progress = distance(gps_now, gps_prev)` (clip to a sane max per step).
  Keep a small speed-command term only as a tie-breaker.
- This kills the "floor the throttle against a wall" exploit.

**Step 3. Make episodes independent (Supervisor reset).**
- Convert the controller to a Webots **Supervisor** (or add a small supervisor
  controller) so after a crash you teleport the car to a start pose +
  `simulationResetPhysics()`. Today it just reverses a little, so the replay
  buffer fills with "stuck near wall" transitions.
- This single change usually improves final quality a lot.

### Phase B — Algorithm upgrade

**Step 4. DDPG -> TD3.**
File: `Deep Reinforcement Learning(DRL)/rl_agent.py`
- Add: twin critics (Q1, Q2, take the min for the target), target-policy
  smoothing (add clipped noise to the target action), and delayed actor updates
  (update actor every 2 critic updates). ~40 lines of change, big stability win.
- Optional next step: SAC (entropy-regularized) for even better exploration.

**Step 5. Fix smaller issues.**
- IMU vs Gyro: code treats InertialUnit yaw *angle* and Gyro yaw *rate* the same.
  Use angular velocity consistently (prefer the Gyro device, or differentiate the
  IMU yaw).
- Model net: change `Linear -> ReLU -> Dropout -> BatchNorm` to
  `Linear -> BatchNorm -> ReLU -> Dropout` (or use LayerNorm to avoid batch-size-1
  crashes).
- Drop the supervised "model prior" until you have a real teacher (see Part 2),
  because today it imitates the trivial rule-based controller.

### Phase C — Verify
- Train ~200 episodes, watch `ep_reward` trend up and collision rate trend down.
- Then move to the perception upgrade (Part 2).

---

## PART 2 — Perception: object recognition + lane detection

### The key idea
Object detection and lane detection are BOTH solved with a CNN, but with
different "heads":

| Task                 | Technique                         | Output                         |
|----------------------|-----------------------------------|--------------------------------|
| Object recognition   | **Object detection** (YOLO)       | boxes + class (car, person...) |
| Lane recognition     | **Semantic segmentation**         | per-pixel "is this lane?" mask |
| Free / drivable area | **Semantic segmentation**         | per-pixel "can I drive here?"  |

Lanes are usually NOT done with bounding boxes — they're thin and curved, so you
use segmentation or a lane-specific model.

### Best techniques (practical ranking)

**Object detection (for obstacles / cars / pedestrians):**
- **YOLO (v8 / v11, Ultralytics)** — best real-time accuracy/speed tradeoff,
  easy to train, tons of tutorials. **Start here.**
- Alternatives: SSD (lighter, weaker), Faster R-CNN (more accurate, slower).

**Lane detection:**
- **Ultra-Fast-Lane-Detection (v2)** or **CLRNet** — fast, lane-specialized,
  strong on TuSimple/CULane.
- Generic: **DeepLabv3+ / SegFormer / BiSeNet** semantic segmentation.
- Classic baseline: LaneNet.

**Best overall for a car (recommended):**
- **YOLOP** or **HybridNets** — a SINGLE network with a shared backbone and
  THREE heads: object detection + drivable-area segmentation + lane-line
  segmentation, all in real time. This is exactly your use case and is the
  cleanest path to "object recognition used in both obstacle + lane." Pretrained
  on BDD100K and easy to fine-tune.

### How perception feeds driving
1. Camera frame -> YOLOP/HybridNets -> {object boxes, lane mask, drivable mask}.
2. Turn that into compact features: nearest-obstacle distance/angle per sector,
   lane offset + heading error, drivable-area width ahead.
3. Feed those features (plus lidar) into the RL policy (TD3) OR into a planning
   layer. This replaces the hand-crafted color detector with a learned one.

---

## PART 3 — Training data

### Do you need data? Yes.
For the perception network (Part 2) you need labeled images. Three sources:

**A. Public real-world datasets (best quality, free):**
- **BDD100K** — 100k driving images with object boxes + lane lines + drivable
  area. PERFECT for YOLOP/HybridNets. **Top recommendation.**
- **KITTI** — detection + lidar, classic benchmark.
- **Cityscapes / Mapillary Vistas** — segmentation.
- **TuSimple / CULane** — lane detection specifically.
- **nuScenes / Waymo Open** — large, multi-sensor, more advanced.

**B. Simulator-generated labeled data (free, perfect labels):**
- A simulator can output ground-truth segmentation masks and exact object
  positions, so you get perfectly labeled data automatically — great for
  closing the sim-to-real gap or augmenting real data.

**C. Pretrained + fine-tune (fastest):**
- Use a YOLO / YOLOP model pretrained on BDD100K/COCO, then fine-tune on a few
  hundred of YOUR simulator frames. Far less data needed than training from
  scratch.

For the DRIVING policy (RL), you don't need a labeled dataset — it generates its
own experience by driving in the simulator.

---

## PART 4 — Simulator options (you asked for better than Udacity)

Your current options:
- **Webots** — fine for robotics, weak for realistic driving perception.
- **Udacity sim** — camera-only, steering-only (behavioral cloning). No lidar,
  no objects, no labels. Too limited for object recognition.

Better options:
- **CARLA** *(strongly recommended)* — the industry-standard open-source AD
  simulator. Gives RGB + depth + **semantic-segmentation cameras** + lidar +
  radar + GPS/IMU, plus exact ground-truth labels for every object. Realistic
  towns, traffic, pedestrians, weather. Free. This is the best single upgrade for
  your perception + RL goals.
- **MetaDrive** — lightweight, very fast, built for RL research (great if you
  want thousands of episodes quickly).
- **AirSim / Cosys-AirSim** — photorealistic (Unreal), good sensor suite.
- **LGSVL** — discontinued (avoid for new work).

Recommendation: keep Webots for quick iteration, but move the serious
perception + policy work to **CARLA** (perfect labels + realistic sensors), and
use **BDD100K** for the perception network.

---

## PART 5 — Suggested order of work

1. Fix RL correctness (Part 1, Phase A: steps 1-3).
2. Upgrade DDPG -> TD3 (step 4) and clean up (step 5).
3. Stand up perception: fine-tune **YOLOP/HybridNets** on **BDD100K**
   (+ optionally simulator frames).
4. Replace the hand-crafted lane/obstacle features with perception outputs.
5. Move to **CARLA** for realistic training + auto-labeled data.
6. Train, evaluate, iterate.
