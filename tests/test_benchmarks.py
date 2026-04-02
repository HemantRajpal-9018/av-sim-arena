"""Tests for benchmark configuration files."""

import os

import pytest
import yaml

from av_sim_arena.scenarios.generator import ScenarioGenerator
from av_sim_arena.scenarios.models import Scenario

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
BENCHMARK_FILES = [
    "highway_merge.yaml",
    "intersection.yaml",
    "parking.yaml",
    "emergency_stop.yaml",
    "roundabout.yaml",
]


class TestBenchmarkConfigs:
    @pytest.mark.parametrize("filename", BENCHMARK_FILES)
    def test_yaml_loads(self, filename):
        path = os.path.join(BENCHMARK_DIR, filename)
        with open(path) as f:
            config = yaml.safe_load(f)
        assert "name" in config
        assert "weather" in config
        assert "ego_start" in config

    @pytest.mark.parametrize("filename", BENCHMARK_FILES)
    def test_scenario_from_yaml(self, filename):
        path = os.path.join(BENCHMARK_DIR, filename)
        gen = ScenarioGenerator()
        scenario = gen.from_yaml(path)
        assert isinstance(scenario, Scenario)
        assert scenario.name != ""
        assert scenario.duration > 0
        assert scenario.dt > 0

    @pytest.mark.parametrize("filename", BENCHMARK_FILES)
    def test_reference_path_exists(self, filename):
        path = os.path.join(BENCHMARK_DIR, filename)
        gen = ScenarioGenerator()
        scenario = gen.from_yaml(path)
        assert len(scenario.reference_path) >= 2

    @pytest.mark.parametrize("filename", BENCHMARK_FILES)
    def test_ego_start_valid(self, filename):
        path = os.path.join(BENCHMARK_DIR, filename)
        gen = ScenarioGenerator()
        scenario = gen.from_yaml(path)
        assert scenario.ego_start is not None

    def test_all_benchmarks_exist(self):
        for filename in BENCHMARK_FILES:
            path = os.path.join(BENCHMARK_DIR, filename)
            assert os.path.exists(path), f"Benchmark file missing: {filename}"
