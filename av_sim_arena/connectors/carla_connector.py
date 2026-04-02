"""CARLA simulator connector stub."""

from __future__ import annotations

import logging
from typing import Any

from av_sim_arena.connectors.base import BaseConnector
from av_sim_arena.scenarios.models import Scenario, VehicleState, WeatherCondition

logger = logging.getLogger(__name__)

CARLA_WEATHER_MAP = {
    WeatherCondition.CLEAR: "ClearNoon",
    WeatherCondition.RAIN: "HardRainNoon",
    WeatherCondition.FOG: "CloudyNoon",
    WeatherCondition.SNOW: "ClearSunset",
    WeatherCondition.NIGHT: "ClearNight",
}


class CARLAConnector(BaseConnector):
    """Connector stub for the CARLA simulator.

    Requires the `carla` Python package and a running CARLA server.
    This stub defines the interface; actual CARLA calls are wrapped
    in try/except to allow usage without the CARLA dependency.
    """

    def __init__(self, host: str = "localhost", port: int = 2000, timeout: float = 10.0):
        super().__init__(host=host, port=port)
        self.timeout = timeout
        self._client = None
        self._world = None
        self._ego_vehicle = None
        self._npc_actors: list = []

    def connect(self) -> bool:
        """Connect to a running CARLA server."""
        try:
            import carla

            self._client = carla.Client(self.host, self.port)
            self._client.set_timeout(self.timeout)
            self._world = self._client.get_world()
            self.connected = True
            logger.info("Connected to CARLA at %s:%d", self.host, self.port)
            return True
        except ImportError:
            logger.warning("CARLA Python package not installed. Running in stub mode.")
            self.connected = False
            return False
        except Exception as e:
            logger.error("Failed to connect to CARLA: %s", e)
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from CARLA and clean up actors."""
        if self._ego_vehicle is not None:
            try:
                self._ego_vehicle.destroy()
            except Exception:
                pass
        for actor in self._npc_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self._npc_actors.clear()
        self._ego_vehicle = None
        self._world = None
        self._client = None
        self.connected = False
        logger.info("Disconnected from CARLA")

    def load_scenario(self, scenario: Scenario) -> bool:
        """Load a scenario into CARLA."""
        if not self.connected:
            logger.warning("Not connected to CARLA. Cannot load scenario.")
            return False

        try:
            weather_preset = CARLA_WEATHER_MAP.get(scenario.weather, "ClearNoon")
            logger.info(
                "Loading scenario '%s' with weather=%s", scenario.name, weather_preset
            )
            return True
        except Exception as e:
            logger.error("Failed to load scenario: %s", e)
            return False

    def step(self, dt: float = 0.1) -> dict[str, Any]:
        """Advance CARLA simulation by one tick."""
        if not self.connected:
            return {"status": "disconnected"}
        try:
            if self._world:
                self._world.tick()
            return {"status": "ok", "dt": dt}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_ego_state(self) -> VehicleState:
        """Get ego vehicle state from CARLA."""
        return VehicleState()

    def set_ego_control(self, acceleration: float, steering: float) -> None:
        """Apply control to the ego vehicle in CARLA."""
        logger.debug("Setting ego control: accel=%.2f, steer=%.2f", acceleration, steering)

    def get_npc_states(self) -> list[VehicleState]:
        """Get all NPC vehicle states from CARLA."""
        return []

    def reset(self) -> None:
        """Reset the CARLA world."""
        self.disconnect()
