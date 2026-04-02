"""Tests for safety metrics."""

import math

import pytest

from av_sim_arena.metrics.safety import MetricResult, SafetyMetrics, TrajectoryPoint


def _make_trajectory(
    n: int = 10, speed: float = 10.0, dt: float = 0.1, y: float = 0.0
) -> list[TrajectoryPoint]:
    """Create a straight trajectory."""
    return [
        TrajectoryPoint(
            x=speed * i * dt,
            y=y,
            heading=0.0,
            speed=speed,
            time=i * dt,
        )
        for i in range(n)
    ]


class TestMetricResult:
    def test_defaults(self):
        r = MetricResult()
        assert r.ttc_min == float("inf")
        assert r.collision_count == 0

    def test_to_dict(self):
        r = MetricResult(ttc_min=3.0, collision_count=1)
        d = r.to_dict()
        assert d["ttc_min"] == 3.0
        assert d["collision_count"] == 1
        assert "max_jerk" in d


class TestSafetyMetrics:
    def test_ttc_no_npcs(self):
        sm = SafetyMetrics()
        ego = _make_trajectory()
        ttc = sm.compute_ttc(ego, [])
        assert ttc == []

    def test_ttc_with_approaching_npc(self):
        sm = SafetyMetrics()
        ego = _make_trajectory(speed=10.0)
        npc = [
            TrajectoryPoint(x=20.0, y=0.0, heading=math.pi, speed=5.0, time=i * 0.1)
            for i in range(10)
        ]
        ttc = sm.compute_ttc(ego, [npc])
        assert len(ttc) > 0
        assert any(t < float("inf") for t in ttc)

    def test_collision_detection(self):
        sm = SafetyMetrics(collision_threshold=2.0)
        ego = _make_trajectory(speed=10.0)
        npc = [
            TrajectoryPoint(x=10.0 * 0.1 * i + 0.5, y=0.0, heading=0.0, speed=10.0, time=i * 0.1)
            for i in range(10)
        ]
        collisions = sm.compute_collisions(ego, [npc])
        assert collisions >= 1

    def test_no_collision(self):
        sm = SafetyMetrics(collision_threshold=2.0)
        ego = _make_trajectory(speed=10.0, y=0.0)
        npc = _make_trajectory(speed=10.0, y=50.0)
        collisions = sm.compute_collisions(ego, [npc])
        assert collisions == 0

    def test_jerk_computation(self):
        sm = SafetyMetrics()
        traj = []
        for i in range(20):
            t = i * 0.1
            speed = 10.0 + 2.0 * t
            traj.append(TrajectoryPoint(x=t * speed, y=0.0, heading=0.0, speed=speed, time=t))
        jerks = sm.compute_jerk(traj)
        assert len(jerks) > 0

    def test_jerk_constant_speed(self):
        sm = SafetyMetrics()
        traj = _make_trajectory(n=20, speed=10.0)
        jerks = sm.compute_jerk(traj)
        for j in jerks:
            assert abs(j) < 0.01

    def test_lateral_deviation(self):
        sm = SafetyMetrics()
        ref = [TrajectoryPoint(x=i, y=0.0, heading=0.0, speed=10.0, time=i * 0.1) for i in range(10)]
        ego = [TrajectoryPoint(x=i, y=1.5, heading=0.0, speed=10.0, time=i * 0.1) for i in range(10)]
        devs = sm.compute_lateral_deviation(ego, ref)
        assert all(abs(d - 1.5) < 0.01 for d in devs)

    def test_heading_error(self):
        sm = SafetyMetrics()
        ref = [TrajectoryPoint(x=i, y=0.0, heading=0.0, speed=10.0, time=i * 0.1) for i in range(10)]
        ego = [TrajectoryPoint(x=i, y=0.0, heading=0.1, speed=10.0, time=i * 0.1) for i in range(10)]
        errs = sm.compute_heading_error(ego, ref)
        assert all(abs(e - 0.1) < 0.01 for e in errs)

    def test_compute_all(self):
        sm = SafetyMetrics()
        ego = _make_trajectory(n=20, speed=10.0)
        npc = _make_trajectory(n=20, speed=10.0, y=20.0)
        ref = _make_trajectory(n=20, speed=10.0)
        result = sm.compute_all(ego, [npc], ref)
        assert isinstance(result, MetricResult)
        assert result.collision_count == 0
        assert result.max_lateral_deviation >= 0

    def test_compute_all_empty_trajectory(self):
        sm = SafetyMetrics()
        result = sm.compute_all([], [], None)
        assert result.ttc_min == float("inf")

    def test_pet_computation(self):
        sm = SafetyMetrics(collision_threshold=3.0)
        ego = [TrajectoryPoint(x=5.0, y=0.0, heading=0.0, speed=10.0, time=1.0)]
        npc = [TrajectoryPoint(x=5.5, y=0.0, heading=0.0, speed=10.0, time=2.0)]
        pet = sm.compute_pet(ego, [npc])
        assert len(pet) > 0
        assert pet[0] == pytest.approx(1.0)

    def test_normalize_angle(self):
        assert SafetyMetrics._normalize_angle(0.0) == 0.0
        assert abs(SafetyMetrics._normalize_angle(2 * math.pi)) < 0.01
        assert abs(SafetyMetrics._normalize_angle(-2 * math.pi)) < 0.01
