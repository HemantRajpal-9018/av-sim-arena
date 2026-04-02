"""Tests for the leaderboard system."""

import os
import tempfile

import pytest

from av_sim_arena.leaderboard.database import LeaderboardDB, LeaderboardEntry


@pytest.fixture
def db(tmp_path):
    """Create a temporary leaderboard database."""
    db_path = str(tmp_path / "test_leaderboard.db")
    return LeaderboardDB(db_path)


@pytest.fixture
def sample_entry():
    return LeaderboardEntry(
        id=None,
        planner_name="test_planner",
        scenario_name="highway_merge",
        ttc_min=3.5,
        pet_min=2.0,
        collision_count=0,
        collision_rate=0.0,
        max_jerk=4.0,
        mean_lateral_deviation=0.3,
        mean_heading_error=0.05,
        overall_score=80.0,
    )


class TestLeaderboardDB:
    def test_submit_and_retrieve(self, db, sample_entry):
        entry_id = db.submit(sample_entry)
        assert entry_id > 0
        retrieved = db.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved.planner_name == "test_planner"

    def test_submit_auto_timestamp(self, db, sample_entry):
        entry_id = db.submit(sample_entry)
        retrieved = db.get_entry(entry_id)
        assert retrieved.submitted_at is not None

    def test_get_rankings(self, db):
        for i, score in enumerate([80.0, 90.0, 70.0]):
            entry = LeaderboardEntry(
                id=None,
                planner_name=f"planner_{i}",
                scenario_name="test",
                ttc_min=3.0,
                pet_min=2.0,
                collision_count=0,
                collision_rate=0.0,
                max_jerk=3.0,
                mean_lateral_deviation=0.3,
                mean_heading_error=0.05,
                overall_score=score,
            )
            db.submit(entry)

        rankings = db.get_rankings()
        assert len(rankings) == 3
        assert rankings[0].overall_score >= rankings[1].overall_score

    def test_get_rankings_by_scenario(self, db):
        for scenario in ["highway", "parking"]:
            entry = LeaderboardEntry(
                id=None,
                planner_name="planner_a",
                scenario_name=scenario,
                ttc_min=3.0, pet_min=2.0, collision_count=0,
                collision_rate=0.0, max_jerk=3.0,
                mean_lateral_deviation=0.3, mean_heading_error=0.05,
                overall_score=80.0,
            )
            db.submit(entry)

        rankings = db.get_rankings(scenario_name="highway")
        assert len(rankings) == 1
        assert rankings[0].scenario_name == "highway"

    def test_get_planner_results(self, db, sample_entry):
        db.submit(sample_entry)
        results = db.get_planner_results("test_planner")
        assert len(results) == 1

    def test_get_planner_results_empty(self, db):
        results = db.get_planner_results("nonexistent")
        assert len(results) == 0

    def test_get_entry_not_found(self, db):
        assert db.get_entry(9999) is None

    def test_delete_entry(self, db, sample_entry):
        entry_id = db.submit(sample_entry)
        assert db.delete_entry(entry_id) is True
        assert db.get_entry(entry_id) is None

    def test_delete_nonexistent(self, db):
        assert db.delete_entry(9999) is False

    def test_compute_overall_score(self, db):
        score = db.compute_overall_score(
            ttc_min=5.0,
            pet_min=3.0,
            collision_count=0,
            collision_rate=0.0,
            max_jerk=3.0,
            mean_lateral_deviation=0.3,
            mean_heading_error=0.05,
        )
        assert 0.0 <= score <= 100.0
        assert score > 50.0  # Good metrics should give high score

    def test_compute_overall_score_with_collision(self, db):
        score_no_collision = db.compute_overall_score(
            ttc_min=5.0, pet_min=3.0, collision_count=0,
            collision_rate=0.0, max_jerk=3.0,
            mean_lateral_deviation=0.3, mean_heading_error=0.05,
        )
        score_collision = db.compute_overall_score(
            ttc_min=5.0, pet_min=3.0, collision_count=2,
            collision_rate=0.01, max_jerk=3.0,
            mean_lateral_deviation=0.3, mean_heading_error=0.05,
        )
        assert score_no_collision > score_collision

    def test_entry_to_dict(self, sample_entry):
        d = sample_entry.to_dict()
        assert d["planner_name"] == "test_planner"
        assert d["ttc_min"] == 3.5

    def test_metadata_serialization(self, db):
        entry = LeaderboardEntry(
            id=None,
            planner_name="meta_planner",
            scenario_name="test",
            ttc_min=3.0, pet_min=2.0, collision_count=0,
            collision_rate=0.0, max_jerk=3.0,
            mean_lateral_deviation=0.3, mean_heading_error=0.05,
            overall_score=75.0,
            metadata={"version": "1.0", "notes": "test run"},
        )
        entry_id = db.submit(entry)
        retrieved = db.get_entry(entry_id)
        assert retrieved.metadata["version"] == "1.0"
