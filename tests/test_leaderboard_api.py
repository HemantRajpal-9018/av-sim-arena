"""Tests for the leaderboard FastAPI application."""

import tempfile

import pytest
from fastapi.testclient import TestClient

from av_sim_arena.leaderboard.api import create_app


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test_api.db")
    app = create_app(db_path)
    return TestClient(app)


@pytest.fixture
def submit_payload():
    return {
        "planner_name": "test_planner",
        "scenario_name": "highway_merge",
        "ttc_min": 3.5,
        "pet_min": 2.0,
        "collision_count": 0,
        "collision_rate": 0.0,
        "max_jerk": 4.0,
        "mean_lateral_deviation": 0.3,
        "mean_heading_error": 0.05,
    }


class TestLeaderboardAPI:
    def test_submit_result(self, client, submit_payload):
        resp = client.post("/api/v1/submit", json=submit_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "overall_score" in data
        assert "rank" in data
        assert data["overall_score"] > 0

    def test_get_rankings_empty(self, client):
        resp = client.get("/api/v1/rankings")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_rankings_after_submit(self, client, submit_payload):
        client.post("/api/v1/submit", json=submit_payload)
        resp = client.get("/api/v1/rankings")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["planner_name"] == "test_planner"

    def test_get_rankings_filtered(self, client, submit_payload):
        client.post("/api/v1/submit", json=submit_payload)
        resp = client.get("/api/v1/rankings?scenario=highway_merge")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        resp = client.get("/api/v1/rankings?scenario=parking")
        assert resp.status_code == 200
        assert len(resp.json()) == 0

    def test_get_planner_results(self, client, submit_payload):
        client.post("/api/v1/submit", json=submit_payload)
        resp = client.get("/api/v1/planner/test_planner")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_planner_not_found(self, client):
        resp = client.get("/api/v1/planner/nonexistent")
        assert resp.status_code == 404

    def test_get_entry(self, client, submit_payload):
        submit_resp = client.post("/api/v1/submit", json=submit_payload)
        entry_id = submit_resp.json()["id"]
        resp = client.get(f"/api/v1/entry/{entry_id}")
        assert resp.status_code == 200
        assert resp.json()["planner_name"] == "test_planner"

    def test_get_entry_not_found(self, client):
        resp = client.get("/api/v1/entry/9999")
        assert resp.status_code == 404

    def test_delete_entry(self, client, submit_payload):
        submit_resp = client.post("/api/v1/submit", json=submit_payload)
        entry_id = submit_resp.json()["id"]
        resp = client.delete(f"/api/v1/entry/{entry_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_entry_not_found(self, client):
        resp = client.delete("/api/v1/entry/9999")
        assert resp.status_code == 404

    def test_submit_with_metadata(self, client, submit_payload):
        submit_payload["metadata"] = {"version": "1.0"}
        resp = client.post("/api/v1/submit", json=submit_payload)
        assert resp.status_code == 200
