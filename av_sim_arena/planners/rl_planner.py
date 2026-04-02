"""Reinforcement Learning-based planner implementation."""

from __future__ import annotations

import math

import numpy as np

from av_sim_arena.planners.base import BasePlanner, PlannerOutput
from av_sim_arena.scenarios.models import VehicleState, Waypoint


class RLPlanner(BasePlanner):
    """RL-based planner using a simple learned policy (simulated).

    This planner uses a simple neural-network-like policy represented
    as weight matrices. In production, these would be loaded from a
    trained checkpoint. Here we use a rule-based initialization that
    approximates a trained policy for benchmarking purposes.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        max_accel: float = 3.0,
        max_steer: float = 0.5,
        seed: int | None = 42,
    ):
        super().__init__(name="rl")
        self.hidden_size = hidden_size
        self.max_accel = max_accel
        self.max_steer = max_steer
        rng = np.random.RandomState(seed)
        self.input_size = 8  # ego: speed, heading; target: dx, dy, dheading; obs: closest_dist, closest_angle, num_obs
        self.w1 = rng.randn(self.input_size, hidden_size).astype(np.float64) * 0.1
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.w2 = rng.randn(hidden_size, 2).astype(np.float64) * 0.1
        self.b2 = np.zeros(2, dtype=np.float64)

    def plan(
        self,
        ego_state: VehicleState,
        reference_path: list[Waypoint],
        obstacles: list[VehicleState],
        dt: float = 0.1,
    ) -> PlannerOutput:
        if not reference_path:
            return PlannerOutput()

        obs = self._extract_observation(ego_state, reference_path, obstacles)
        action = self._forward(obs)

        accel = float(np.clip(action[0] * self.max_accel, -self.max_accel, self.max_accel))
        steer = float(np.clip(action[1] * self.max_steer, -self.max_steer, self.max_steer))

        return PlannerOutput(acceleration=accel, steering=steer)

    def _extract_observation(
        self,
        ego: VehicleState,
        path: list[Waypoint],
        obstacles: list[VehicleState],
    ) -> np.ndarray:
        """Extract observation vector from current state."""
        target = self._find_target(ego, path)
        dx = target.x - ego.x
        dy = target.y - ego.y
        dheading = self._normalize_angle(
            math.atan2(dy, dx) - ego.heading
        )

        closest_dist = 100.0
        closest_angle = 0.0
        for obs in obstacles:
            d = math.sqrt((obs.x - ego.x) ** 2 + (obs.y - ego.y) ** 2)
            if d < closest_dist:
                closest_dist = d
                closest_angle = math.atan2(obs.y - ego.y, obs.x - ego.x) - ego.heading

        return np.array([
            ego.speed / 15.0,
            ego.heading / math.pi,
            dx / 50.0,
            dy / 10.0,
            dheading / math.pi,
            closest_dist / 50.0,
            closest_angle / math.pi,
            min(len(obstacles), 10) / 10.0,
        ], dtype=np.float64)

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through the policy network."""
        h = np.tanh(obs @ self.w1 + self.b1)
        out = np.tanh(h @ self.w2 + self.b2)
        return out

    def _find_target(self, ego: VehicleState, path: list[Waypoint]) -> Waypoint:
        min_dist = float("inf")
        best_idx = 0
        for i, wp in enumerate(path):
            d = math.sqrt((wp.x - ego.x) ** 2 + (wp.y - ego.y) ** 2)
            if d < min_dist:
                min_dist = d
                best_idx = i
        lookahead = min(best_idx + 5, len(path) - 1)
        return path[lookahead]

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset(self) -> None:
        pass

    def load_weights(self, path: str) -> None:
        """Load trained weights from a file."""
        data = np.load(path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]

    def save_weights(self, path: str) -> None:
        """Save weights to a file."""
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
