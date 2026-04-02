"""Data models for scenario configuration."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class WeatherCondition(enum.Enum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    NIGHT = "night"


class TrafficDensity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PedestrianBehavior(enum.Enum):
    NORMAL = "normal"
    JAYWALKING = "jaywalking"
    DISTRACTED = "distracted"
    RUNNING = "running"


class EdgeCase(enum.Enum):
    CUT_IN = "cut_in"
    SUDDEN_BRAKE = "sudden_brake"
    JAYWALKER = "jaywalker"
    ROAD_DEBRIS = "road_debris"
    WRONG_WAY_DRIVER = "wrong_way_driver"
    EMERGENCY_VEHICLE = "emergency_vehicle"


@dataclass
class VehicleState:
    """State of a vehicle at a point in time."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians
    speed: float = 0.0  # m/s
    acceleration: float = 0.0  # m/s^2
    yaw_rate: float = 0.0  # rad/s
    width: float = 2.0
    length: float = 4.5


@dataclass
class Waypoint:
    """A point along a reference path."""

    x: float
    y: float
    heading: float = 0.0
    speed_limit: float = 13.9  # 50 km/h default


@dataclass
class Scenario:
    """Complete scenario definition for simulation."""

    name: str
    description: str = ""
    weather: WeatherCondition = WeatherCondition.CLEAR
    traffic_density: TrafficDensity = TrafficDensity.MEDIUM
    pedestrian_behaviors: list[PedestrianBehavior] = field(default_factory=list)
    edge_cases: list[EdgeCase] = field(default_factory=list)
    ego_start: VehicleState = field(default_factory=VehicleState)
    ego_goal: Waypoint = field(default_factory=lambda: Waypoint(x=100.0, y=0.0))
    reference_path: list[Waypoint] = field(default_factory=list)
    npc_vehicles: list[dict[str, Any]] = field(default_factory=list)
    npc_pedestrians: list[dict[str, Any]] = field(default_factory=list)
    duration: float = 30.0  # seconds
    dt: float = 0.1  # time step
    road_width: float = 3.7  # lane width in meters
    num_lanes: int = 2
    speed_limit: float = 13.9  # m/s (~50 km/h)

    @property
    def num_steps(self) -> int:
        return int(self.duration / self.dt)
