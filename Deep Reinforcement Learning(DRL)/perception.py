"""Camera perception for the mini car: object detection + traffic light + lane hook.

This is the CAMERA half of perception. It produces:
  * object detections with class labels (person, car, traffic light, pole, ...),
  * traffic-light state (red / yellow / green),
  * lane offset + confidence (hook for a trained lane-segmentation model),
  * lane flags (can_go_left / can_go_right) for the safety governor.

Distances to obstacles are NOT estimated from the monocular camera (unreliable) --
those come from lidar / ToF / depth and feed SensorReading in safety_governor.py.
The two are fused at the policy/safety layer.

Runs in two modes:
  * Real mode: if `ultralytics` (YOLO) is installed, loads a detector. Fine-tune on
    BDD100K + your mini-track frames on the RTX 5060 (see PERCEPTION.md).
  * Stub mode: if YOLO is not installed, returns empty detections so the rest of the
    pipeline (fusion, safety, policy) can still be developed and tested.

Install for real mode:  pip install ultralytics
"""

from dataclasses import dataclass, field
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    _YOLO_IMPORT_ERROR = exc


# Classes the car cares about. COCO (default YOLO) already covers person, car,
# bicycle, traffic light, stop sign, etc. Lanes, poles, and curbs/sidebars need a
# custom-trained model (BDD100K has lanes/drivable area) -- see PERCEPTION.md.
TARGET_CLASSES = {
    "person", "car", "bicycle", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "pole", "curb", "sidebar",
}


@dataclass
class Detection:
    cls_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels


@dataclass
class PerceptionResult:
    detections: list = field(default_factory=list)
    image_shape: tuple = (0, 0)        # (H, W)
    lane_offset: float = 0.0           # [-1, +1], 0 = centered
    lane_confidence: float = 0.0       # [0, 1]
    traffic_light: str = "none"        # red / yellow / green / none
    can_go_left: bool = True
    can_go_right: bool = True

    def has(self, cls_name):
        return any(d.cls_name == cls_name for d in self.detections)

    def nearest_in_path(self):
        """Return the detection whose box is largest+lowest (roughly closest ahead),
        or None. A coarse 'something blocking the path' cue for the camera."""
        if not self.detections:
            return None
        H, W = self.image_shape if self.image_shape != (0, 0) else (1, 1)
        def score(d):
            x1, y1, x2, y2 = d.bbox
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            return area * (y2 / max(H, 1))  # bigger + lower = closer
        return max(self.detections, key=score)


class Perception:
    def __init__(self, model_path="yolov8n.pt", device="cpu", conf=0.35,
                 lane_model=None):
        self.device = device
        self.conf = float(conf)
        self.lane_model = lane_model  # optional trained lane-seg callable
        if YOLO is None:
            self.model = None
            print("[Perception] ultralytics not installed -> STUB mode "
                  "(no detections). Install with `pip install ultralytics`.")
        else:
            self.model = YOLO(model_path)
            print(f"[Perception] YOLO loaded: {model_path} on {device}")

    # ----------------------------------------------------------------- detect
    def perceive(self, image_rgb):
        """Run full perception on an RGB image (H, W, 3) -> PerceptionResult."""
        img = np.asarray(image_rgb)
        H, W = img.shape[:2]
        detections = self._detect(img)
        lane_offset, lane_conf = self._detect_lane(img)
        can_left, can_right = self._lane_flags(lane_offset, lane_conf)
        tl = self._traffic_light_state(detections, img)
        return PerceptionResult(
            detections=detections,
            image_shape=(H, W),
            lane_offset=lane_offset,
            lane_confidence=lane_conf,
            traffic_light=tl,
            can_go_left=can_left,
            can_go_right=can_right,
        )

    def _detect(self, img):
        if self.model is None:
            return []  # stub mode
        results = self.model.predict(img, conf=self.conf, device=self.device,
                                     verbose=False)
        dets = []
        for res in results:
            names = res.names
            for box in res.boxes:
                cls_name = names[int(box.cls[0])]
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                dets.append(Detection(cls_name, float(box.conf[0]), xyxy))
        return dets

    def _detect_lane(self, img):
        """Lane offset + confidence. Hook for a trained lane-segmentation model;
        falls back to 0 confidence (unknown) in the scaffold."""
        if self.lane_model is not None:
            try:
                return self.lane_model(img)  # expected to return (offset, conf)
            except Exception:
                return 0.0, 0.0
        return 0.0, 0.0

    def _lane_flags(self, lane_offset, lane_conf):
        """Derive can_go_left/right from lane geometry. With a confident lane,
        crossing toward the opposite side is disallowed (don't enter oncoming).
        Without lane info (open pedestrian space) both sides are allowed."""
        if lane_conf < 0.3:
            return True, True  # no lane / open space -> free to maneuver
        # In-lane: allow nudging toward center, disallow crossing the far boundary.
        can_left = lane_offset > -0.5   # already far left -> don't go further left
        can_right = lane_offset < 0.5
        return can_left, can_right

    def _traffic_light_state(self, detections, img):
        """Classify the color of the most confident traffic-light detection."""
        lights = [d for d in detections if d.cls_name == "traffic light"]
        if not lights:
            return "none"
        d = max(lights, key=lambda x: x.confidence)
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        crop = np.asarray(img)[max(y1, 0):y2, max(x1, 0):x2, :3].astype(np.float32)
        if crop.size == 0:
            return "none"
        r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
        red = np.sum((r > 150) & (g < 110) & (b < 110))
        green = np.sum((g > 150) & (r < 110) & (b < 130))
        yellow = np.sum((r > 150) & (g > 150) & (b < 110))
        counts = {"red": red, "green": green, "yellow": yellow}
        best = max(counts, key=counts.get)
        return best if counts[best] > 5 else "none"
