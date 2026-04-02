"""Base interface for simulator connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from av_sim_arena.scenarios.models import Scenario, VehicleState


class BaseConnector(ABC):
    """Abstract interface for connecting to external simulators."""

    def __init__(self, host: str = "localhost", port: int = 2000):
        self.host = host
        self.port = port
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the simulator."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the simulator."""
        ...

    @abstractmethod
    def load_scenario(self, scenario: Scenario) -> bool:
        """Load a scenario into the simulator."""
        ...

    @abstractmethod
    def step(self, dt: float = 0.1) -> dict[str, Any]:
        """Advance the simulation by one timestep."""
        ...

    @abstractmethod
    def get_ego_state(self) -> VehicleState:
        """Get the current ego vehicle state."""
        ...

    @abstractmethod
    def set_ego_control(self, acceleration: float, steering: float) -> None:
        """Apply control inputs to the ego vehicle."""
        ...

    @abstractmethod
    def get_npc_states(self) -> list[VehicleState]:
        """Get states of all NPC vehicles."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulator to initial state."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(host={self.host!r}, port={self.port})"
