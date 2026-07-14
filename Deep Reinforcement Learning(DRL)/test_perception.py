"""Smoke tests for the perception scaffold (stub mode, no model/GPU needed).

Verifies the plumbing: perceive() returns a well-formed result, traffic-light
color logic works, and lane flags behave per policy. Run: python test_perception.py
"""

import numpy as np
from perception import Perception, PerceptionResult, Detection


def test_perceive_returns_result_in_stub_mode():
    p = Perception()  # no ultralytics -> stub
    img = np.zeros((90, 160, 3), dtype=np.uint8)
    res = p.perceive(img)
    assert isinstance(res, PerceptionResult)
    assert res.image_shape == (90, 160)
    assert res.detections == []          # stub -> no detections
    assert res.traffic_light == "none"
    assert res.can_go_left and res.can_go_right  # no lane -> open space
    print("  [ok] perceive() returns a valid result in stub mode")


def test_traffic_light_color_classification():
    p = Perception()
    img = np.zeros((90, 160, 3), dtype=np.uint8)
    # Paint a red blob and pretend YOLO detected a traffic light there.
    img[10:30, 70:90] = [220, 20, 20]
    det = Detection("traffic light", 0.9, (70, 10, 90, 30))
    assert p._traffic_light_state([det], img) == "red"
    # Green case
    img2 = np.zeros((90, 160, 3), dtype=np.uint8)
    img2[10:30, 70:90] = [20, 220, 40]
    det2 = Detection("traffic light", 0.9, (70, 10, 90, 30))
    assert p._traffic_light_state([det2], img2) == "green"
    print("  [ok] traffic-light color classification (red/green)")


def test_lane_flags_policy():
    p = Perception()
    # No confident lane -> both directions allowed (open pedestrian space).
    assert p._lane_flags(0.0, 0.1) == (True, True)
    # Confident lane, car far to the right -> must not go further right.
    can_left, can_right = p._lane_flags(0.7, 0.9)
    assert can_left and not can_right
    # Confident lane, car far to the left -> must not go further left.
    can_left, can_right = p._lane_flags(-0.7, 0.9)
    assert can_right and not can_left
    print("  [ok] lane flags enforce 'don't cross the far lane boundary'")


def test_nearest_in_path():
    p = Perception()
    res = PerceptionResult(
        detections=[
            Detection("car", 0.8, (10, 10, 30, 30)),       # small, high
            Detection("person", 0.9, (50, 40, 120, 88)),   # big, low -> closest
        ],
        image_shape=(90, 160),
    )
    nearest = res.nearest_in_path()
    assert nearest.cls_name == "person"
    print("  [ok] nearest_in_path picks the closest (big+low) object")


if __name__ == "__main__":
    print("=== Perception scaffold tests ===")
    test_perceive_returns_result_in_stub_mode()
    test_traffic_light_color_classification()
    test_lane_flags_policy()
    test_nearest_in_path()
    print("ALL PERCEPTION TESTS PASSED")
