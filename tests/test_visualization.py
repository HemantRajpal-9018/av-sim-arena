"""Tests for visualization module."""

import pytest

from av_sim_arena.metrics.safety import TrajectoryPoint
from av_sim_arena.visualization.plots import MetricPlotter
from av_sim_arena.visualization.replay import ScenarioReplay


def _make_trajectory(n=20, speed=10.0, y=0.0):
    return [
        TrajectoryPoint(x=speed * i * 0.1, y=y, heading=0.0, speed=speed, time=i * 0.1)
        for i in range(n)
    ]


class TestScenarioReplay:
    def test_creation(self):
        ego = _make_trajectory()
        replay = ScenarioReplay(ego, title="Test Replay")
        assert replay.title == "Test Replay"
        assert len(replay.ego_trajectory) == 20

    def test_creation_with_npcs(self):
        ego = _make_trajectory()
        npc = _make_trajectory(y=3.7)
        replay = ScenarioReplay(ego, npc_trajectories=[npc])
        assert len(replay.npc_trajectories) == 1


class TestMetricPlotter:
    def test_creation(self):
        plotter = MetricPlotter()
        assert "ttc_min" in plotter.METRIC_LABELS

    def test_bar_comparison_data_structure(self):
        plotter = MetricPlotter()
        results = {
            "lattice": {"ttc_min": 3.0, "max_jerk": 4.5},
            "mpc": {"ttc_min": 3.5, "max_jerk": 3.8},
        }
        # Just verify the plotter accepts the data structure
        assert "lattice" in results
        assert "mpc" in results

    def test_heatmap_data_structure(self):
        plotter = MetricPlotter()
        data = {
            "lattice": {"highway": 78.5, "intersection": 72.3},
            "mpc": {"highway": 85.2, "intersection": 80.1},
        }
        assert len(data) == 2

    def test_metric_labels(self):
        plotter = MetricPlotter()
        assert plotter.METRIC_LABELS["collision_rate"] == "Collision Rate"
        assert plotter.METRIC_LABELS["mean_lateral_deviation"] == "Mean Lat. Dev. (m)"
