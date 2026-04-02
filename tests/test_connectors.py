"""Tests for simulator connectors."""

import pytest

from av_sim_arena.connectors.base import BaseConnector
from av_sim_arena.connectors.carla_connector import CARLAConnector
from av_sim_arena.connectors.sumo_connector import SUMOConnector
from av_sim_arena.scenarios.models import Scenario, VehicleState


class TestBaseConnector:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseConnector()


class TestCARLAConnector:
    def test_creation(self):
        conn = CARLAConnector(host="localhost", port=2000)
        assert conn.host == "localhost"
        assert conn.port == 2000
        assert conn.connected is False

    def test_connect_stub_mode(self):
        conn = CARLAConnector()
        result = conn.connect()
        assert result is False  # CARLA not installed
        assert conn.connected is False

    def test_disconnect(self):
        conn = CARLAConnector()
        conn.disconnect()
        assert conn.connected is False

    def test_step_disconnected(self):
        conn = CARLAConnector()
        result = conn.step()
        assert result["status"] == "disconnected"

    def test_get_ego_state(self):
        conn = CARLAConnector()
        state = conn.get_ego_state()
        assert isinstance(state, VehicleState)

    def test_get_npc_states(self):
        conn = CARLAConnector()
        states = conn.get_npc_states()
        assert states == []

    def test_load_scenario_disconnected(self):
        conn = CARLAConnector()
        scenario = Scenario(name="test")
        assert conn.load_scenario(scenario) is False

    def test_repr(self):
        conn = CARLAConnector()
        assert "CARLAConnector" in repr(conn)


class TestSUMOConnector:
    def test_creation(self):
        conn = SUMOConnector(host="localhost", port=8813)
        assert conn.port == 8813
        assert conn.connected is False

    def test_connect_stub_mode(self):
        conn = SUMOConnector()
        result = conn.connect()
        assert result is False  # TraCI not installed
        assert conn.connected is False

    def test_disconnect(self):
        conn = SUMOConnector()
        conn.disconnect()
        assert conn.connected is False

    def test_step_disconnected(self):
        conn = SUMOConnector()
        result = conn.step()
        assert result["status"] == "disconnected"

    def test_get_ego_state_disconnected(self):
        conn = SUMOConnector()
        state = conn.get_ego_state()
        assert isinstance(state, VehicleState)

    def test_get_npc_states_disconnected(self):
        conn = SUMOConnector()
        assert conn.get_npc_states() == []

    def test_set_ego_control_disconnected(self):
        conn = SUMOConnector()
        conn.set_ego_control(1.0, 0.1)  # Should not raise

    def test_reset(self):
        conn = SUMOConnector()
        conn.reset()
        assert conn.connected is False
