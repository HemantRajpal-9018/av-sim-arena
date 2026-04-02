"""SUMO (Simulation of Urban Mobility) connector stub."""

from __future__ import annotations

import logging
from typing import Any

from av_sim_arena.connectors.base import BaseConnector
from av_sim_arena.scenarios.models import Scenario, VehicleState

logger = logging.getLogger(__name__)


class SUMOConnector(BaseConnector):
    """Connector stub for the SUMO traffic simulator.

    Requires the `traci` Python package and a SUMO installation.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8813,
        sumo_cfg: str | None = None,
        gui: bool = False,
    ):
        super().__init__(host=host, port=port)
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self._connection = None
        self._ego_id = "ego"

    def connect(self) -> bool:
        """Start SUMO and connect via TraCI."""
        try:
            import traci

            sumo_binary = "sumo-gui" if self.gui else "sumo"
            if self.sumo_cfg:
                traci.start([sumo_binary, "-c", self.sumo_cfg, "--start"])
            self._connection = traci
            self.connected = True
            logger.info("Connected to SUMO")
            return True
        except ImportError:
            logger.warning("TraCI not installed. Running in stub mode.")
            self.connected = False
            return False
        except Exception as e:
            logger.error("Failed to connect to SUMO: %s", e)
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Close the SUMO connection."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None
        self.connected = False
        logger.info("Disconnected from SUMO")

    def load_scenario(self, scenario: Scenario) -> bool:
        """Configure SUMO for the given scenario."""
        if not self.connected:
            logger.warning("Not connected to SUMO. Cannot load scenario.")
            return False
        logger.info("Loading scenario '%s' into SUMO", scenario.name)
        return True

    def step(self, dt: float = 0.1) -> dict[str, Any]:
        """Advance SUMO simulation by one step."""
        if not self.connected:
            return {"status": "disconnected"}
        try:
            if self._connection:
                self._connection.simulationStep()
            return {"status": "ok", "dt": dt}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_ego_state(self) -> VehicleState:
        """Get ego vehicle state from SUMO."""
        if not self.connected or self._connection is None:
            return VehicleState()
        try:
            x, y = self._connection.vehicle.getPosition(self._ego_id)
            speed = self._connection.vehicle.getSpeed(self._ego_id)
            angle = self._connection.vehicle.getAngle(self._ego_id)
            return VehicleState(x=x, y=y, speed=speed, heading=angle)
        except Exception:
            return VehicleState()

    def set_ego_control(self, acceleration: float, steering: float) -> None:
        """Apply control to the ego vehicle in SUMO."""
        if not self.connected or self._connection is None:
            return
        try:
            current_speed = self._connection.vehicle.getSpeed(self._ego_id)
            new_speed = max(0, current_speed + acceleration * 0.1)
            self._connection.vehicle.setSpeed(self._ego_id, new_speed)
        except Exception as e:
            logger.error("Failed to set ego control: %s", e)

    def get_npc_states(self) -> list[VehicleState]:
        """Get all NPC vehicle states from SUMO."""
        if not self.connected or self._connection is None:
            return []
        try:
            states = []
            vehicle_ids = self._connection.vehicle.getIDList()
            for vid in vehicle_ids:
                if vid == self._ego_id:
                    continue
                x, y = self._connection.vehicle.getPosition(vid)
                speed = self._connection.vehicle.getSpeed(vid)
                angle = self._connection.vehicle.getAngle(vid)
                states.append(VehicleState(x=x, y=y, speed=speed, heading=angle))
            return states
        except Exception:
            return []

    def reset(self) -> None:
        """Reset the SUMO simulation."""
        self.disconnect()
