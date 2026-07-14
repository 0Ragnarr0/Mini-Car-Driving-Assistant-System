# U22-Car — Perception Guide

How the camera perception (`Deep Reinforcement Learning(DRL)/perception.py`) works,
what to train it on, and how it fuses with lidar + the policy + the safety layer.

## What perception produces (camera only)
- Object detections with class labels (person, car, traffic light, pole, ...).
- Traffic-light state: red / yellow / green.
- Lane offset + confidence (hook for a trained lane-segmentation model).
- Lane flags `can_go_left` / `can_go_right` (so the safety layer never swerves into
  the opposite lane).

Distances to obstacles do NOT come from the camera (monocular distance is
unreliable). They come from **lidar / ToF / depth** and feed `SensorReading` in
`safety_governor.py`. Camera + range sensors fuse at the policy/safety layer:

```
   camera ──► perception.py ──► objects, lane, traffic light, lane-flags ─┐
                                                                          ├─► policy (TD3) ─► action
   lidar/ToF/depth ──► SensorReading (front/side/rear clearances) ────────┘            │
                                                                                        ▼
                                                          safety_governor.py (hard override)
                                                                                        │
                                                                                        ▼
                                                                              steering servo + motor
```

## Models to use
- **Object detection: YOLO (v8/v11 nano)** via `ultralytics`. Nano size runs in real
  time on the Jetson. COCO-pretrained already knows person, car, bicycle, traffic
  light, stop sign.
- **Lanes + drivable area: a segmentation model** (or YOLOP / HybridNets, which do
  detection + lane + drivable-area in ONE network). Train on BDD100K.
- Custom classes you need that COCO lacks (pole, curb/sidebar): add them via custom
  training (BDD100K / your own labeled mini-track frames).

## Training (do this on the RTX 5060)
1. Install: `pip install ultralytics`
2. Quick start (COCO-pretrained, fine-tune on your data):
   ```
   yolo detect train model=yolov8n.pt data=your_dataset.yaml epochs=100 imgsz=640
   ```
3. Datasets:
   - **BDD100K** — driving images with object boxes + lane lines + drivable area
     (best single source for this car).
   - **Your mini-track frames** — capture a few hundred images from the actual
     fisheye camera on the real track and label them (Roboflow / CVAT). This is the
     highest-value data for closing the sim-to-real gap.
4. Point `Perception(model_path=...)` at your trained `best.pt`.

## Lane model hook
`Perception(lane_model=callable)` — pass a callable that takes an RGB image and
returns `(lane_offset in [-1,1], confidence in [0,1])`. Plug a trained lane-seg
model here; until then it reports confidence 0 (unknown) and the car treats the
area as open space.

## Fisheye note
The fisheye camera distorts straight lines. Either (a) undistort frames with a
calibrated camera matrix before perception, or (b) train the detector/lane model
directly on fisheye frames (often simpler and works well). Calibrate the fisheye
once with a checkerboard (OpenCV `cv2.fisheye`).

## Status
Scaffold built + smoke-tested in stub mode (no model). Real detection turns on as
soon as `ultralytics` is installed and a model is provided. Training is GPU work for
the laptop.
