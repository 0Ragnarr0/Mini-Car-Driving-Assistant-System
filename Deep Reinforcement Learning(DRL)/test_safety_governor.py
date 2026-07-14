"""Unit tests for the safety-override governor (no simulator needed).

Run:  python test_safety_governor.py
"""

from safety_governor import SafetyGovernor, SensorReading, SafetyState


def test_clear_road_passes_through():
    gov = SafetyGovernor()
    action = (0.3, 0.8)
    (steer, throttle), info = gov.filter(action, SensorReading(front=5.0, speed=2.0))
    assert info["reason"] == "CLEAR"
    assert abs(steer - 0.3) < 1e-6 and abs(throttle - 0.8) < 1e-6
    print("  [ok] clear road passes policy action through unchanged")


def test_speed_governor_slows_near_obstacle():
    gov = SafetyGovernor()
    # Obstacle within slow range but outside avoid range -> throttle capped < 0.8.
    (steer, throttle), info = gov.filter((0.0, 0.8), SensorReading(front=0.65))
    assert info["reason"] in ("SLOW", "AVOID_BIAS", "SLOW_NO_PATH")
    assert throttle < 0.8, f"expected throttle capped, got {throttle}"
    print(f"  [ok] speed governor slows near obstacle (throttle={throttle:.2f})")


def test_emergency_stop_overrides_full_throttle():
    gov = SafetyGovernor()
    # Something right in front; policy wants full throttle -> must brake.
    (steer, throttle), info = gov.filter((0.0, 1.0),
                                         SensorReading(front=0.12, rear=5.0))
    assert info["reason"] == "EMERGENCY_STOP"
    assert throttle <= 0.0, f"expected braking, got {throttle}"
    print("  [ok] emergency stop overrides full-throttle command")


def test_steers_around_when_side_clear_and_allowed():
    gov = SafetyGovernor()
    # Blocked ahead, right side clear and allowed -> steer right (+).
    r = SensorReading(front=0.25, front_right=2.0, front_left=0.1,
                      can_go_right=True, can_go_left=True)
    (steer, throttle), info = gov.filter((0.0, 0.8), r)
    assert info["reason"] == "AVOID_STEER"
    assert steer > 0.2, f"expected steer right, got {steer}"
    print(f"  [ok] steers around obstacle toward clear side (steer={steer:+.2f})")


def test_does_not_swerve_into_forbidden_lane():
    gov = SafetyGovernor()
    # Blocked ahead; the only clear side (left) is forbidden (opposite lane).
    # Right is blocked. Must STOP, not swerve into oncoming.
    r = SensorReading(front=0.25, front_left=3.0, front_right=0.1,
                      can_go_left=False, can_go_right=True)
    (steer, throttle), info = gov.filter((0.0, 0.8), r)
    assert info["reason"] == "STOP_BLOCKED", info
    assert throttle <= 0.0 and abs(steer) < 1e-6
    print("  [ok] refuses to swerve into a forbidden/opposite lane -> stops instead")


def test_stuck_triggers_reverse_then_turn_recovery():
    gov = SafetyGovernor()
    # Fully boxed in front + both sides forbidden/blocked, but rear is clear.
    boxed = SensorReading(front=0.12, front_left=0.1, front_right=0.1,
                          can_go_left=False, can_go_right=False, rear=5.0)
    reason = None
    for _ in range(20):
        _, info = gov.filter((0.0, 1.0), boxed)
        reason = info["reason"]
        if gov.state == SafetyState.REVERSING:
            break
    assert gov.state == SafetyState.REVERSING, f"never entered reverse (last={reason})"
    print("  [ok] stays stuck then enters REVERSING recovery")

    # Now give it a clear front while reversing -> should progress to TURNING then NORMAL.
    saw_turn = False
    for _ in range(40):
        clearing = SensorReading(front=2.0, front_right=2.0, rear=5.0)
        _, info = gov.filter((0.0, 0.0), clearing)
        if gov.state == SafetyState.TURNING:
            saw_turn = True
        if gov.state == SafetyState.NORMAL and saw_turn:
            break
    assert saw_turn, "never reached TURNING"
    assert gov.state == SafetyState.NORMAL, "did not resume NORMAL after recovery"
    print("  [ok] reverse -> turn -> resume NORMAL recovery completes")


if __name__ == "__main__":
    print("=== Safety governor tests ===")
    test_clear_road_passes_through()
    test_speed_governor_slows_near_obstacle()
    test_emergency_stop_overrides_full_throttle()
    test_steers_around_when_side_clear_and_allowed()
    test_does_not_swerve_into_forbidden_lane()
    test_stuck_triggers_reverse_then_turn_recovery()
    print("ALL SAFETY TESTS PASSED")
