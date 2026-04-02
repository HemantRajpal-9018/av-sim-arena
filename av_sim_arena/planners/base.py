"""Abstract base class for AV planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from av_sim_arena.scenarios.models import VehicleState, Waypoint


@dataclass
class PlannerOutput:
    """Output from a planner at each timestep."""

    acceleration: float = 0.0  # m/s^2
    steering: float = 0.0  # radians
    trajectory: list[VehicleState] | None = None


class BasePlanner(ABC):
    """Abstract base class that all planners must implement."""

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def plan(
        self,
        ego_state: VehicleState,
        reference_path: list[Waypoint],
        obstacles: list[VehicleState],
        dt: float = 0.1,
    ) -> PlannerOutput:
        """Compute the next control action.

        Args:
            ego_state: Current state of the ego vehicle.
            reference_path: Reference waypoints to follow.
            obstacles: States of surrounding vehicles/obstacles.
            dt: Time step in seconds.

        Returns:
            PlannerOutput with acceleration and steering commands.
        """
        ...

    def reset(self) -> None:
        """Reset internal planner state."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
