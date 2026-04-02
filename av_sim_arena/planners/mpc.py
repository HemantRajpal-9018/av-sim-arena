"""Model Predictive Control (MPC) planner implementation."""

from __future__ import annotations

import math

import numpy as np

from av_sim_arena.planners.base import BasePlanner, PlannerOutput
from av_sim_arena.scenarios.models import VehicleState, Waypoint


class MPCPlanner(BasePlanner):
    """MPC-based planner using a bicycle model and quadratic cost."""

    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
        wheelbase: float = 2.5,
        max_accel: float = 3.0,
        max_steer: float = 0.5,
        weight_tracking: float = 1.0,
        weight_heading: float = 0.5,
        weight_speed: float = 0.3,
        weight_accel: float = 0.1,
        weight_steer: float = 0.1,
        num_accel_samples: int = 7,
        num_steer_samples: int = 9,
    ):
        super().__init__(name="mpc")
        self.horizon = horizon
        self.dt_mpc = dt
        self.wheelbase = wheelbase
        self.max_accel = max_accel
        self.max_steer = max_steer
        self.weight_tracking = weight_tracking
        self.weight_heading = weight_heading
        self.weight_speed = weight_speed
        self.weight_accel = weight_accel
        self.weight_steer = weight_steer
        self.num_accel_samples = num_accel_samples
        self.num_steer_samples = num_steer_samples

    def plan(
        self,
        ego_state: VehicleState,
        reference_path: list[Waypoint],
        obstacles: list[VehicleState],
        dt: float = 0.1,
    ) -> PlannerOutput:
        if not reference_path:
            return PlannerOutput()

        ref_points = self._select_reference_points(ego_state, reference_path)

        best_cost = float("inf")
        best_accel = 0.0
        best_steer = 0.0

        accel_candidates = np.linspace(
            -self.max_accel, self.max_accel, self.num_accel_samples
        )
        steer_candidates = np.linspace(
            -self.max_steer, self.max_steer, self.num_steer_samples
        )

        for accel in accel_candidates:
            for steer in steer_candidates:
                cost = self._simulate_and_cost(
                    ego_state, float(accel), float(steer), ref_points, obstacles
                )
                if cost < best_cost:
                    best_cost = cost
                    best_accel = float(accel)
                    best_steer = float(steer)

        return PlannerOutput(acceleration=best_accel, steering=best_steer)

    def _select_reference_points(
        self, ego: VehicleState, path: list[Waypoint]
    ) -> list[Waypoint]:
        """Select reference points ahead of the ego vehicle for the MPC horizon."""
        closest_idx = 0
        min_dist = float("inf")
        for i, wp in enumerate(path):
            d = math.sqrt((wp.x - ego.x) ** 2 + (wp.y - ego.y) ** 2)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        points = []
        for i in range(self.horizon):
            idx = min(closest_idx + i + 1, len(path) - 1)
            points.append(path[idx])
        return points

    def _simulate_and_cost(
        self,
        ego: VehicleState,
        accel: float,
        steer: float,
        ref_points: list[Waypoint],
        obstacles: list[VehicleState],
    ) -> float:
        """Forward-simulate the bicycle model and compute cost."""
        x, y, heading, speed = ego.x, ego.y, ego.heading, ego.speed
        total_cost = 0.0

        for i in range(min(self.horizon, len(ref_points))):
            speed = max(0.0, speed + accel * self.dt_mpc)
            heading += (speed / self.wheelbase) * math.tan(steer) * self.dt_mpc
            x += speed * math.cos(heading) * self.dt_mpc
            y += speed * math.sin(heading) * self.dt_mpc

            ref = ref_points[i]
            tracking_err = (x - ref.x) ** 2 + (y - ref.y) ** 2
            heading_err = self._normalize_angle(heading - ref.heading) ** 2
            speed_err = (speed - ref.speed_limit) ** 2

            total_cost += (
                self.weight_tracking * tracking_err
                + self.weight_heading * heading_err
                + self.weight_speed * speed_err
            )

            for obs in obstacles:
                obs_dist = math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
                if obs_dist < 3.0:
                    total_cost += 1000.0
                elif obs_dist < 6.0:
                    total_cost += 50.0 / max(obs_dist, 0.1)

        total_cost += self.weight_accel * accel**2
        total_cost += self.weight_steer * steer**2

        return total_cost

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset(self) -> None:
        pass
