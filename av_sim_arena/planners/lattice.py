"""Lattice planner implementation."""

from __future__ import annotations

import math

from av_sim_arena.planners.base import BasePlanner, PlannerOutput
from av_sim_arena.scenarios.models import VehicleState, Waypoint


class LatticePlanner(BasePlanner):
    """State lattice planner that samples trajectories in a Frenet frame."""

    def __init__(
        self,
        num_lateral_samples: int = 5,
        num_longitudinal_samples: int = 5,
        planning_horizon: float = 3.0,
        lateral_range: float = 3.7,
        weight_progress: float = 1.0,
        weight_deviation: float = 2.0,
        weight_obstacle: float = 5.0,
    ):
        super().__init__(name="lattice")
        self.num_lateral_samples = num_lateral_samples
        self.num_longitudinal_samples = num_longitudinal_samples
        self.planning_horizon = planning_horizon
        self.lateral_range = lateral_range
        self.weight_progress = weight_progress
        self.weight_deviation = weight_deviation
        self.weight_obstacle = weight_obstacle

    def plan(
        self,
        ego_state: VehicleState,
        reference_path: list[Waypoint],
        obstacles: list[VehicleState],
        dt: float = 0.1,
    ) -> PlannerOutput:
        if not reference_path:
            return PlannerOutput()

        target_wp = self._find_target_waypoint(ego_state, reference_path)

        candidates = self._generate_candidates(ego_state, target_wp, dt)

        best_candidate = None
        best_cost = float("inf")
        for accel, steer in candidates:
            cost = self._evaluate_candidate(
                ego_state, accel, steer, target_wp, obstacles, dt
            )
            if cost < best_cost:
                best_cost = cost
                best_candidate = (accel, steer)

        if best_candidate is None:
            return PlannerOutput()

        return PlannerOutput(acceleration=best_candidate[0], steering=best_candidate[1])

    def _find_target_waypoint(
        self, ego: VehicleState, path: list[Waypoint]
    ) -> Waypoint:
        """Find the nearest waypoint ahead of the ego vehicle."""
        min_dist = float("inf")
        best_idx = 0
        for i, wp in enumerate(path):
            dx = wp.x - ego.x
            dy = wp.y - ego.y
            dist = math.sqrt(dx**2 + dy**2)
            ahead = dx * math.cos(ego.heading) + dy * math.sin(ego.heading)
            if ahead > 0 and dist < min_dist:
                min_dist = dist
                best_idx = i
        lookahead_idx = min(best_idx + 3, len(path) - 1)
        return path[lookahead_idx]

    def _generate_candidates(
        self, ego: VehicleState, target: Waypoint, dt: float
    ) -> list[tuple[float, float]]:
        """Generate candidate (acceleration, steering) pairs."""
        candidates = []
        accel_range = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        steer_range = [
            -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3
        ]
        for a in accel_range:
            for s in steer_range:
                candidates.append((a, s))
        return candidates

    def _evaluate_candidate(
        self,
        ego: VehicleState,
        accel: float,
        steer: float,
        target: Waypoint,
        obstacles: list[VehicleState],
        dt: float,
    ) -> float:
        """Evaluate cost of a candidate action."""
        new_speed = max(0.0, ego.speed + accel * dt)
        new_heading = ego.heading + steer * dt
        new_x = ego.x + new_speed * math.cos(new_heading) * dt
        new_y = ego.y + new_speed * math.sin(new_heading) * dt

        dx = target.x - new_x
        dy = target.y - new_y
        dist_to_target = math.sqrt(dx**2 + dy**2)

        lateral_dev = abs(
            -math.sin(target.heading) * (new_x - target.x)
            + math.cos(target.heading) * (new_y - target.y)
        )

        min_obs_dist = float("inf")
        for obs in obstacles:
            d = math.sqrt((new_x - obs.x) ** 2 + (new_y - obs.y) ** 2)
            min_obs_dist = min(min_obs_dist, d)
        obstacle_cost = 0.0
        if min_obs_dist < 5.0:
            obstacle_cost = (5.0 - min_obs_dist) ** 2

        cost = (
            self.weight_progress * dist_to_target
            + self.weight_deviation * lateral_dev
            + self.weight_obstacle * obstacle_cost
        )
        return cost

    def reset(self) -> None:
        pass
