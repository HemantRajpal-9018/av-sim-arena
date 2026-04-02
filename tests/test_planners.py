"""Tests for planner implementations."""

import math

import pytest

from av_sim_arena.planners.base import BasePlanner, PlannerOutput
from av_sim_arena.planners.lattice import LatticePlanner
from av_sim_arena.planners.mpc import MPCPlanner
from av_sim_arena.planners.rl_planner import RLPlanner
from av_sim_arena.planners.rrt_star import RRTStarPlanner
from av_sim_arena.scenarios.models import VehicleState, Waypoint


def _make_path(length: float = 100.0, spacing: float = 5.0) -> list[Waypoint]:
    n = int(length / spacing)
    return [Waypoint(x=i * spacing, y=0.0, heading=0.0) for i in range(n + 1)]


def _make_ego(x: float = 0.0, speed: float = 10.0) -> VehicleState:
    return VehicleState(x=x, y=0.0, heading=0.0, speed=speed)


class TestBasePlanner:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BasePlanner(name="test")

    def test_planner_output_defaults(self):
        out = PlannerOutput()
        assert out.acceleration == 0.0
        assert out.steering == 0.0
        assert out.trajectory is None


class TestLatticePlanner:
    def test_plan_basic(self):
        planner = LatticePlanner()
        ego = _make_ego()
        path = _make_path()
        result = planner.plan(ego, path, [])
        assert isinstance(result, PlannerOutput)

    def test_plan_with_obstacle(self):
        planner = LatticePlanner()
        ego = _make_ego()
        path = _make_path()
        obstacle = VehicleState(x=15.0, y=0.0, speed=5.0)
        result = planner.plan(ego, path, [obstacle])
        assert isinstance(result, PlannerOutput)

    def test_plan_empty_path(self):
        planner = LatticePlanner()
        ego = _make_ego()
        result = planner.plan(ego, [], [])
        assert result.acceleration == 0.0

    def test_repr(self):
        planner = LatticePlanner()
        assert "lattice" in repr(planner)

    def test_reset(self):
        planner = LatticePlanner()
        planner.reset()  # Should not raise


class TestRRTStarPlanner:
    def test_plan_basic(self):
        planner = RRTStarPlanner(max_iterations=50, seed=42)
        ego = _make_ego()
        path = _make_path()
        result = planner.plan(ego, path, [])
        assert isinstance(result, PlannerOutput)

    def test_plan_with_obstacles(self):
        planner = RRTStarPlanner(max_iterations=50, seed=42)
        ego = _make_ego()
        path = _make_path()
        obstacles = [VehicleState(x=20.0, y=1.0, speed=8.0)]
        result = planner.plan(ego, path, obstacles)
        assert isinstance(result, PlannerOutput)

    def test_plan_empty_path(self):
        planner = RRTStarPlanner()
        ego = _make_ego()
        result = planner.plan(ego, [], [])
        assert result.acceleration == 0.0


class TestMPCPlanner:
    def test_plan_basic(self):
        planner = MPCPlanner(num_accel_samples=3, num_steer_samples=3)
        ego = _make_ego()
        path = _make_path()
        result = planner.plan(ego, path, [])
        assert isinstance(result, PlannerOutput)
        assert -3.0 <= result.acceleration <= 3.0
        assert -0.5 <= result.steering <= 0.5

    def test_plan_with_obstacle(self):
        planner = MPCPlanner(num_accel_samples=3, num_steer_samples=3)
        ego = _make_ego()
        path = _make_path()
        obstacle = VehicleState(x=10.0, y=0.0, speed=5.0)
        result = planner.plan(ego, path, [obstacle])
        assert isinstance(result, PlannerOutput)

    def test_plan_empty_path(self):
        planner = MPCPlanner()
        ego = _make_ego()
        result = planner.plan(ego, [], [])
        assert result.acceleration == 0.0


class TestRLPlanner:
    def test_plan_basic(self):
        planner = RLPlanner(seed=42)
        ego = _make_ego()
        path = _make_path()
        result = planner.plan(ego, path, [])
        assert isinstance(result, PlannerOutput)
        assert -3.0 <= result.acceleration <= 3.0
        assert -0.5 <= result.steering <= 0.5

    def test_plan_with_obstacles(self):
        planner = RLPlanner(seed=42)
        ego = _make_ego()
        path = _make_path()
        obstacles = [VehicleState(x=15.0, y=2.0, speed=8.0)]
        result = planner.plan(ego, path, obstacles)
        assert isinstance(result, PlannerOutput)

    def test_plan_empty_path(self):
        planner = RLPlanner()
        ego = _make_ego()
        result = planner.plan(ego, [], [])
        assert result.acceleration == 0.0

    def test_deterministic(self):
        p1 = RLPlanner(seed=42)
        p2 = RLPlanner(seed=42)
        ego = _make_ego()
        path = _make_path()
        r1 = p1.plan(ego, path, [])
        r2 = p2.plan(ego, path, [])
        assert r1.acceleration == pytest.approx(r2.acceleration)
        assert r1.steering == pytest.approx(r2.steering)
