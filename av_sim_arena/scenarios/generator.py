"""Configurable scenario generator for AV simulation benchmarks."""

from __future__ import annotations

import math
import random
from typing import Any

import yaml

from av_sim_arena.scenarios.models import (
    EdgeCase,
    PedestrianBehavior,
    Scenario,
    TrafficDensity,
    VehicleState,
    Waypoint,
    WeatherCondition,
)


class ScenarioGenerator:
    """Generates simulation scenarios from configuration or randomly."""

    NPC_DENSITY_MAP = {
        TrafficDensity.LOW: (1, 3),
        TrafficDensity.MEDIUM: (3, 7),
        TrafficDensity.HIGH: (7, 15),
    }

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def from_yaml(self, path: str) -> Scenario:
        """Load a scenario from a YAML configuration file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return self.from_dict(config)

    def from_dict(self, config: dict[str, Any]) -> Scenario:
        """Create a scenario from a dictionary configuration."""
        weather = WeatherCondition(config.get("weather", "clear"))
        traffic_density = TrafficDensity(config.get("traffic_density", "medium"))

        pedestrian_behaviors = [
            PedestrianBehavior(p) for p in config.get("pedestrian_behaviors", [])
        ]
        edge_cases = [EdgeCase(e) for e in config.get("edge_cases", [])]

        ego_start_cfg = config.get("ego_start", {})
        ego_start = VehicleState(
            x=ego_start_cfg.get("x", 0.0),
            y=ego_start_cfg.get("y", 0.0),
            heading=ego_start_cfg.get("heading", 0.0),
            speed=ego_start_cfg.get("speed", 0.0),
        )

        ego_goal_cfg = config.get("ego_goal", {})
        ego_goal = Waypoint(
            x=ego_goal_cfg.get("x", 100.0),
            y=ego_goal_cfg.get("y", 0.0),
            heading=ego_goal_cfg.get("heading", 0.0),
        )

        reference_path = []
        for wp in config.get("reference_path", []):
            reference_path.append(
                Waypoint(
                    x=wp["x"],
                    y=wp["y"],
                    heading=wp.get("heading", 0.0),
                    speed_limit=wp.get("speed_limit", 13.9),
                )
            )

        if not reference_path:
            reference_path = self._generate_straight_path(ego_start, ego_goal)

        return Scenario(
            name=config.get("name", "unnamed"),
            description=config.get("description", ""),
            weather=weather,
            traffic_density=traffic_density,
            pedestrian_behaviors=pedestrian_behaviors,
            edge_cases=edge_cases,
            ego_start=ego_start,
            ego_goal=ego_goal,
            reference_path=reference_path,
            npc_vehicles=config.get("npc_vehicles", []),
            npc_pedestrians=config.get("npc_pedestrians", []),
            duration=config.get("duration", 30.0),
            dt=config.get("dt", 0.1),
            road_width=config.get("road_width", 3.7),
            num_lanes=config.get("num_lanes", 2),
            speed_limit=config.get("speed_limit", 13.9),
        )

    def generate_random(
        self,
        name: str = "random_scenario",
        weather: WeatherCondition | None = None,
        traffic_density: TrafficDensity | None = None,
        edge_cases: list[EdgeCase] | None = None,
    ) -> Scenario:
        """Generate a random scenario with optional constraints."""
        if weather is None:
            weather = self.rng.choice(list(WeatherCondition))
        if traffic_density is None:
            traffic_density = self.rng.choice(list(TrafficDensity))
        if edge_cases is None:
            num_edge = self.rng.randint(0, 2)
            edge_cases = self.rng.sample(list(EdgeCase), min(num_edge, len(EdgeCase)))

        ped_behaviors = self.rng.sample(
            list(PedestrianBehavior),
            self.rng.randint(0, 2),
        )

        ego_start = VehicleState(x=0.0, y=0.0, heading=0.0, speed=10.0)
        goal_x = self.rng.uniform(80.0, 200.0)
        ego_goal = Waypoint(x=goal_x, y=0.0)
        reference_path = self._generate_straight_path(ego_start, ego_goal)

        npc_min, npc_max = self.NPC_DENSITY_MAP[traffic_density]
        num_npcs = self.rng.randint(npc_min, npc_max)
        npc_vehicles = self._generate_npc_vehicles(num_npcs, ego_start, ego_goal)

        return Scenario(
            name=name,
            description=f"Random scenario: {weather.value}, {traffic_density.value} traffic",
            weather=weather,
            traffic_density=traffic_density,
            pedestrian_behaviors=ped_behaviors,
            edge_cases=edge_cases,
            ego_start=ego_start,
            ego_goal=ego_goal,
            reference_path=reference_path,
            npc_vehicles=npc_vehicles,
        )

    def _generate_straight_path(
        self, start: VehicleState, goal: Waypoint, spacing: float = 5.0
    ) -> list[Waypoint]:
        """Generate a straight reference path between start and goal."""
        dx = goal.x - start.x
        dy = goal.y - start.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < spacing:
            return [Waypoint(x=start.x, y=start.y), goal]

        heading = math.atan2(dy, dx)
        n_points = max(int(dist / spacing), 2)
        path = []
        for i in range(n_points + 1):
            t = i / n_points
            path.append(
                Waypoint(
                    x=start.x + t * dx,
                    y=start.y + t * dy,
                    heading=heading,
                )
            )
        return path

    def _generate_npc_vehicles(
        self,
        count: int,
        ego_start: VehicleState,
        ego_goal: Waypoint,
    ) -> list[dict[str, Any]]:
        """Generate NPC vehicle configurations."""
        npcs = []
        for i in range(count):
            lane_offset = self.rng.choice([-3.7, 0.0, 3.7])
            x_offset = self.rng.uniform(20.0, ego_goal.x - 10)
            npcs.append({
                "id": f"npc_{i}",
                "x": ego_start.x + x_offset,
                "y": ego_start.y + lane_offset,
                "heading": 0.0,
                "speed": self.rng.uniform(8.0, 15.0),
                "behavior": self.rng.choice(["follow", "yield", "lane_change", "aggressive"]),
            })
        return npcs
