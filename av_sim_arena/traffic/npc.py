"""NPC vehicle and pedestrian agents with behavior tree control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from av_sim_arena.scenarios.models import VehicleState
from av_sim_arena.traffic.behavior_tree import (
    AggressiveDriver,
    BehaviorTree,
    FollowBehavior,
    LaneChangeBehavior,
    Selector,
    YieldBehavior,
)


class NPCVehicle:
    """NPC vehicle controlled by a behavior tree."""

    BEHAVIOR_MAP = {
        "follow": lambda: BehaviorTree(FollowBehavior()),
        "yield": lambda: BehaviorTree(YieldBehavior()),
        "lane_change": lambda: BehaviorTree(LaneChangeBehavior()),
        "aggressive": lambda: BehaviorTree(AggressiveDriver()),
        "default": lambda: BehaviorTree(
            Selector([YieldBehavior(), FollowBehavior()])
        ),
    }

    def __init__(
        self,
        vehicle_id: str,
        state: VehicleState,
        behavior: str = "follow",
    ):
        self.vehicle_id = vehicle_id
        self.state = state
        factory = self.BEHAVIOR_MAP.get(behavior, self.BEHAVIOR_MAP["default"])
        self.behavior_tree = factory()
        self.history: list[VehicleState] = [self._copy_state()]

    def step(self, context: dict, dt: float = 0.1) -> VehicleState:
        """Advance one simulation step."""
        context["dt"] = dt
        self.behavior_tree.tick(self.state, context)

        accel = context.get("acceleration", 0.0)
        steer = context.get("steering", 0.0)

        self.state.speed = max(0.0, self.state.speed + accel * dt)
        self.state.heading += steer * dt
        self.state.x += self.state.speed * math.cos(self.state.heading) * dt
        self.state.y += self.state.speed * math.sin(self.state.heading) * dt
        self.state.acceleration = accel

        self.history.append(self._copy_state())
        return self.state

    def _copy_state(self) -> VehicleState:
        return VehicleState(
            x=self.state.x,
            y=self.state.y,
            heading=self.state.heading,
            speed=self.state.speed,
            acceleration=self.state.acceleration,
        )


@dataclass
class PedestrianState:
    """State of a pedestrian."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 1.4  # average walking speed m/s


class NPCPedestrian:
    """NPC pedestrian with simple behavior patterns."""

    def __init__(
        self,
        pedestrian_id: str,
        state: PedestrianState,
        behavior: str = "normal",
    ):
        self.pedestrian_id = pedestrian_id
        self.state = state
        self.behavior = behavior
        self.history: list[PedestrianState] = [self._copy_state()]

        if behavior == "jaywalking":
            self.state.heading = math.pi / 2  # cross perpendicular
            self.state.speed = 1.8
        elif behavior == "running":
            self.state.speed = 3.0
        elif behavior == "distracted":
            self.state.speed = 0.8

    def step(self, context: dict | None = None, dt: float = 0.1) -> PedestrianState:
        """Advance one simulation step."""
        self.state.x += self.state.speed * math.cos(self.state.heading) * dt
        self.state.y += self.state.speed * math.sin(self.state.heading) * dt

        self.history.append(self._copy_state())
        return self.state

    def _copy_state(self) -> PedestrianState:
        return PedestrianState(
            x=self.state.x,
            y=self.state.y,
            heading=self.state.heading,
            speed=self.state.speed,
        )
