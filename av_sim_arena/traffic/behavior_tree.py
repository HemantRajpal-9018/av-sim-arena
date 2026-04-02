"""Behavior trees for NPC vehicles and pedestrians."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum

from av_sim_arena.scenarios.models import VehicleState


class NodeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorNode(ABC):
    """Base class for behavior tree nodes."""

    @abstractmethod
    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        ...


class BehaviorTree:
    """Behavior tree that manages NPC decision-making."""

    def __init__(self, root: BehaviorNode):
        self.root = root

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        return self.root.tick(state, context)


class Sequence(BehaviorNode):
    """Runs children in order; fails on first failure."""

    def __init__(self, children: list[BehaviorNode]):
        self.children = children

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        for child in self.children:
            status = child.tick(state, context)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS


class Selector(BehaviorNode):
    """Runs children in order; succeeds on first success."""

    def __init__(self, children: list[BehaviorNode]):
        self.children = children

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        for child in self.children:
            status = child.tick(state, context)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE


class Condition(BehaviorNode):
    """Checks a condition from context."""

    def __init__(self, key: str, check: callable):
        self.key = key
        self.check = check

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        value = context.get(self.key)
        if self.check(value):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class FollowBehavior(BehaviorNode):
    """Follow the vehicle ahead at a safe distance."""

    def __init__(self, desired_gap: float = 10.0, kp: float = 0.5):
        self.desired_gap = desired_gap
        self.kp = kp

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        lead_vehicle: VehicleState | None = context.get("lead_vehicle")
        if lead_vehicle is None:
            context["acceleration"] = 0.5
            context["steering"] = 0.0
            return NodeStatus.SUCCESS

        dx = lead_vehicle.x - state.x
        dy = lead_vehicle.y - state.y
        gap = math.sqrt(dx**2 + dy**2)

        accel = self.kp * (gap - self.desired_gap)
        accel = max(-4.0, min(3.0, accel))
        context["acceleration"] = accel
        context["steering"] = 0.0
        return NodeStatus.SUCCESS


class YieldBehavior(BehaviorNode):
    """Yield to approaching vehicles or pedestrians."""

    def __init__(self, yield_distance: float = 15.0):
        self.yield_distance = yield_distance

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        approaching: list[VehicleState] = context.get("approaching_vehicles", [])
        should_yield = False
        for vehicle in approaching:
            dist = math.sqrt((vehicle.x - state.x) ** 2 + (vehicle.y - state.y) ** 2)
            if dist < self.yield_distance:
                should_yield = True
                break

        if should_yield:
            context["acceleration"] = -2.0
            context["steering"] = 0.0
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class LaneChangeBehavior(BehaviorNode):
    """Execute a lane change maneuver."""

    def __init__(self, lane_width: float = 3.7, duration: float = 3.0):
        self.lane_width = lane_width
        self.duration = duration
        self.progress = 0.0

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        direction = context.get("lane_change_direction", 1)  # 1=left, -1=right
        dt = context.get("dt", 0.1)

        self.progress += dt
        if self.progress >= self.duration:
            self.progress = 0.0
            context["acceleration"] = 0.0
            context["steering"] = 0.0
            return NodeStatus.SUCCESS

        t = self.progress / self.duration
        steer = direction * 0.1 * math.sin(math.pi * t)
        context["acceleration"] = 0.0
        context["steering"] = steer
        return NodeStatus.RUNNING


class AggressiveDriver(BehaviorNode):
    """Aggressive driving behavior: tailgating, frequent lane changes."""

    def __init__(self, desired_gap: float = 3.0, max_accel: float = 4.0):
        self.desired_gap = desired_gap
        self.max_accel = max_accel

    def tick(self, state: VehicleState, context: dict) -> NodeStatus:
        lead_vehicle: VehicleState | None = context.get("lead_vehicle")

        if lead_vehicle is None:
            context["acceleration"] = self.max_accel * 0.5
            context["steering"] = 0.0
            return NodeStatus.SUCCESS

        dx = lead_vehicle.x - state.x
        dy = lead_vehicle.y - state.y
        gap = math.sqrt(dx**2 + dy**2)

        if gap < self.desired_gap:
            context["acceleration"] = -3.0
        else:
            context["acceleration"] = min(self.max_accel, (gap - self.desired_gap) * 0.8)

        context["steering"] = 0.0
        return NodeStatus.SUCCESS
