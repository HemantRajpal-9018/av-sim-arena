"""Tests for scenario generation."""

import math
import os
import tempfile

import pytest
import yaml

from av_sim_arena.scenarios.generator import ScenarioGenerator
from av_sim_arena.scenarios.models import (
    EdgeCase,
    PedestrianBehavior,
    Scenario,
    TrafficDensity,
    VehicleState,
    Waypoint,
    WeatherCondition,
)


class TestScenarioModels:
    def test_weather_conditions(self):
        for w in WeatherCondition:
            assert w.value in ("clear", "rain", "fog", "snow", "night")

    def test_traffic_density(self):
        for td in TrafficDensity:
            assert td.value in ("low", "medium", "high")

    def test_pedestrian_behavior(self):
        assert len(PedestrianBehavior) == 4

    def test_edge_cases(self):
        assert EdgeCase.CUT_IN.value == "cut_in"
        assert EdgeCase.SUDDEN_BRAKE.value == "sudden_brake"
        assert EdgeCase.JAYWALKER.value == "jaywalker"

    def test_vehicle_state_defaults(self):
        vs = VehicleState()
        assert vs.x == 0.0
        assert vs.speed == 0.0
        assert vs.width == 2.0
        assert vs.length == 4.5

    def test_waypoint(self):
        wp = Waypoint(x=10.0, y=5.0, heading=0.5)
        assert wp.x == 10.0
        assert wp.speed_limit == 13.9

    def test_scenario_defaults(self):
        s = Scenario(name="test")
        assert s.weather == WeatherCondition.CLEAR
        assert s.traffic_density == TrafficDensity.MEDIUM
        assert s.dt == 0.1
        assert s.num_steps == 300

    def test_scenario_num_steps(self):
        s = Scenario(name="test", duration=10.0, dt=0.05)
        assert s.num_steps == 200


class TestScenarioGenerator:
    def test_generate_random(self):
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate_random()
        assert scenario.name == "random_scenario"
        assert isinstance(scenario.weather, WeatherCondition)
        assert isinstance(scenario.traffic_density, TrafficDensity)
        assert len(scenario.reference_path) > 0

    def test_generate_random_with_constraints(self):
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate_random(
            name="constrained",
            weather=WeatherCondition.RAIN,
            traffic_density=TrafficDensity.HIGH,
        )
        assert scenario.weather == WeatherCondition.RAIN
        assert scenario.traffic_density == TrafficDensity.HIGH
        assert len(scenario.npc_vehicles) >= 7

    def test_generate_random_deterministic(self):
        gen1 = ScenarioGenerator(seed=123)
        gen2 = ScenarioGenerator(seed=123)
        s1 = gen1.generate_random()
        s2 = gen2.generate_random()
        assert s1.weather == s2.weather
        assert s1.traffic_density == s2.traffic_density

    def test_from_dict(self):
        config = {
            "name": "test_scenario",
            "weather": "fog",
            "traffic_density": "low",
            "ego_start": {"x": 5.0, "y": 1.0, "heading": 0.1, "speed": 10.0},
            "ego_goal": {"x": 100.0, "y": 0.0},
            "duration": 20.0,
            "edge_cases": ["cut_in"],
        }
        gen = ScenarioGenerator()
        scenario = gen.from_dict(config)
        assert scenario.name == "test_scenario"
        assert scenario.weather == WeatherCondition.FOG
        assert scenario.traffic_density == TrafficDensity.LOW
        assert scenario.ego_start.x == 5.0
        assert scenario.duration == 20.0
        assert EdgeCase.CUT_IN in scenario.edge_cases

    def test_from_yaml(self):
        config = {
            "name": "yaml_test",
            "weather": "snow",
            "traffic_density": "medium",
            "ego_start": {"x": 0, "y": 0, "speed": 5},
            "ego_goal": {"x": 50, "y": 0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            path = f.name
        try:
            gen = ScenarioGenerator()
            scenario = gen.from_yaml(path)
            assert scenario.name == "yaml_test"
            assert scenario.weather == WeatherCondition.SNOW
        finally:
            os.unlink(path)

    def test_straight_path_generation(self):
        gen = ScenarioGenerator()
        start = VehicleState(x=0.0, y=0.0)
        goal = Waypoint(x=50.0, y=0.0)
        path = gen._generate_straight_path(start, goal)
        assert len(path) >= 2
        assert path[0].x == 0.0
        assert path[-1].x == 50.0

    def test_npc_vehicle_generation(self):
        gen = ScenarioGenerator(seed=42)
        start = VehicleState(x=0.0, y=0.0)
        goal = Waypoint(x=100.0, y=0.0)
        npcs = gen._generate_npc_vehicles(5, start, goal)
        assert len(npcs) == 5
        for npc in npcs:
            assert "id" in npc
            assert "speed" in npc
            assert "behavior" in npc
