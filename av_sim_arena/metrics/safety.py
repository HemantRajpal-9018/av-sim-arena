"""Safety metrics for evaluating autonomous vehicle planners."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""

    x: float
    y: float
    heading: float
    speed: float
    time: float


@dataclass
class MetricResult:
    """Container for computed metric values."""

    ttc_min: float = float("inf")
    ttc_mean: float = float("inf")
    pet_min: float = float("inf")
    collision_count: int = 0
    collision_rate: float = 0.0
    max_jerk: float = 0.0
    mean_jerk: float = 0.0
    max_lateral_deviation: float = 0.0
    mean_lateral_deviation: float = 0.0
    max_heading_error: float = 0.0
    mean_heading_error: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "ttc_min": self.ttc_min,
            "ttc_mean": self.ttc_mean,
            "pet_min": self.pet_min,
            "collision_count": self.collision_count,
            "collision_rate": self.collision_rate,
            "max_jerk": self.max_jerk,
            "mean_jerk": self.mean_jerk,
            "max_lateral_deviation": self.max_lateral_deviation,
            "mean_lateral_deviation": self.mean_lateral_deviation,
            "max_heading_error": self.max_heading_error,
            "mean_heading_error": self.mean_heading_error,
        }


class SafetyMetrics:
    """Compute safety metrics from ego and NPC trajectories."""

    def __init__(self, collision_threshold: float = 2.0):
        self.collision_threshold = collision_threshold

    def compute_all(
        self,
        ego_trajectory: list[TrajectoryPoint],
        npc_trajectories: list[list[TrajectoryPoint]],
        reference_path: list[TrajectoryPoint] | None = None,
    ) -> MetricResult:
        """Compute all safety metrics."""
        result = MetricResult()

        if len(ego_trajectory) < 2:
            return result

        ttc_values = self.compute_ttc(ego_trajectory, npc_trajectories)
        if ttc_values:
            result.ttc_min = min(ttc_values)
            result.ttc_mean = float(np.mean(ttc_values))

        pet_values = self.compute_pet(ego_trajectory, npc_trajectories)
        if pet_values:
            result.pet_min = min(pet_values)

        collisions = self.compute_collisions(ego_trajectory, npc_trajectories)
        result.collision_count = collisions
        result.collision_rate = collisions / max(len(ego_trajectory), 1)

        jerk_values = self.compute_jerk(ego_trajectory)
        if jerk_values:
            result.max_jerk = max(abs(j) for j in jerk_values)
            result.mean_jerk = float(np.mean(np.abs(jerk_values)))

        if reference_path:
            lat_dev = self.compute_lateral_deviation(ego_trajectory, reference_path)
            if lat_dev:
                result.max_lateral_deviation = max(lat_dev)
                result.mean_lateral_deviation = float(np.mean(lat_dev))

            heading_err = self.compute_heading_error(ego_trajectory, reference_path)
            if heading_err:
                result.max_heading_error = max(heading_err)
                result.mean_heading_error = float(np.mean(heading_err))

        return result

    def compute_ttc(
        self,
        ego: list[TrajectoryPoint],
        npcs: list[list[TrajectoryPoint]],
    ) -> list[float]:
        """Compute time-to-collision for each timestep."""
        ttc_values = []
        for i, ego_pt in enumerate(ego):
            min_ttc = float("inf")
            for npc_traj in npcs:
                if i >= len(npc_traj):
                    continue
                npc_pt = npc_traj[i]
                ttc = self._point_ttc(ego_pt, npc_pt)
                min_ttc = min(min_ttc, ttc)
            if min_ttc < float("inf"):
                ttc_values.append(min_ttc)
        return ttc_values

    def _point_ttc(self, ego: TrajectoryPoint, npc: TrajectoryPoint) -> float:
        """Compute TTC between two trajectory points."""
        dx = npc.x - ego.x
        dy = npc.y - ego.y
        dist = math.sqrt(dx**2 + dy**2)

        ego_vx = ego.speed * math.cos(ego.heading)
        ego_vy = ego.speed * math.sin(ego.heading)
        npc_vx = npc.speed * math.cos(npc.heading)
        npc_vy = npc.speed * math.sin(npc.heading)

        rel_vx = ego_vx - npc_vx
        rel_vy = ego_vy - npc_vy

        closing_speed = (dx * rel_vx + dy * rel_vy) / max(dist, 1e-6)

        if closing_speed <= 0:
            return float("inf")

        return (dist - self.collision_threshold) / closing_speed

    def compute_pet(
        self,
        ego: list[TrajectoryPoint],
        npcs: list[list[TrajectoryPoint]],
    ) -> list[float]:
        """Compute post-encroachment time."""
        pet_values = []
        for npc_traj in npcs:
            for i, ego_pt in enumerate(ego):
                for j, npc_pt in enumerate(npc_traj):
                    dist = math.sqrt(
                        (ego_pt.x - npc_pt.x) ** 2 + (ego_pt.y - npc_pt.y) ** 2
                    )
                    if dist < self.collision_threshold:
                        pet = abs(ego_pt.time - npc_pt.time)
                        if pet > 0:
                            pet_values.append(pet)
        return pet_values

    def compute_collisions(
        self,
        ego: list[TrajectoryPoint],
        npcs: list[list[TrajectoryPoint]],
    ) -> int:
        """Count number of collision events."""
        collision_count = 0
        in_collision = False
        for i, ego_pt in enumerate(ego):
            colliding = False
            for npc_traj in npcs:
                if i >= len(npc_traj):
                    continue
                npc_pt = npc_traj[i]
                dist = math.sqrt(
                    (ego_pt.x - npc_pt.x) ** 2 + (ego_pt.y - npc_pt.y) ** 2
                )
                if dist < self.collision_threshold:
                    colliding = True
                    break
            if colliding and not in_collision:
                collision_count += 1
            in_collision = colliding
        return collision_count

    def compute_jerk(self, ego: list[TrajectoryPoint]) -> list[float]:
        """Compute longitudinal jerk (derivative of acceleration)."""
        if len(ego) < 3:
            return []
        speeds = [p.speed for p in ego]
        times = [p.time for p in ego]

        accelerations = []
        for i in range(1, len(speeds)):
            dt = times[i] - times[i - 1]
            if dt > 0:
                accelerations.append((speeds[i] - speeds[i - 1]) / dt)

        jerks = []
        for i in range(1, len(accelerations)):
            dt = times[i + 1] - times[i]
            if dt > 0:
                jerks.append((accelerations[i] - accelerations[i - 1]) / dt)

        return jerks

    def compute_lateral_deviation(
        self,
        ego: list[TrajectoryPoint],
        reference: list[TrajectoryPoint],
    ) -> list[float]:
        """Compute lateral deviation from reference path."""
        deviations = []
        ref_points = np.array([[p.x, p.y] for p in reference])
        for ego_pt in ego:
            ego_pos = np.array([ego_pt.x, ego_pt.y])
            dists = np.linalg.norm(ref_points - ego_pos, axis=1)
            closest_idx = int(np.argmin(dists))

            ref_pt = reference[closest_idx]
            dx = ego_pt.x - ref_pt.x
            dy = ego_pt.y - ref_pt.y
            lateral = abs(-math.sin(ref_pt.heading) * dx + math.cos(ref_pt.heading) * dy)
            deviations.append(lateral)
        return deviations

    def compute_heading_error(
        self,
        ego: list[TrajectoryPoint],
        reference: list[TrajectoryPoint],
    ) -> list[float]:
        """Compute heading error relative to reference path."""
        errors = []
        ref_points = np.array([[p.x, p.y] for p in reference])
        for ego_pt in ego:
            ego_pos = np.array([ego_pt.x, ego_pt.y])
            dists = np.linalg.norm(ref_points - ego_pos, axis=1)
            closest_idx = int(np.argmin(dists))
            ref_heading = reference[closest_idx].heading
            error = abs(self._normalize_angle(ego_pt.heading - ref_heading))
            errors.append(error)
        return errors

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
