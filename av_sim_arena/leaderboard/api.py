"""FastAPI application for the leaderboard API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from av_sim_arena.leaderboard.database import LeaderboardDB, LeaderboardEntry


class SubmitRequest(BaseModel):
    """Request body for submitting results."""

    planner_name: str
    scenario_name: str
    ttc_min: float
    pet_min: float
    collision_count: int = 0
    collision_rate: float = 0.0
    max_jerk: float = 0.0
    mean_lateral_deviation: float = 0.0
    mean_heading_error: float = 0.0
    metadata: dict | None = None


class SubmitResponse(BaseModel):
    """Response body after submitting results."""

    id: int
    overall_score: float
    rank: int


class RankingEntry(BaseModel):
    """Single entry in ranking response."""

    rank: int
    planner_name: str
    scenario_name: str
    overall_score: float
    ttc_min: float
    collision_count: int
    max_jerk: float
    mean_lateral_deviation: float


def create_app(db_path: str = "leaderboard.db") -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AV-Sim-Arena Leaderboard",
        description="Benchmark leaderboard for autonomous vehicle planners",
        version="0.1.0",
    )
    db = LeaderboardDB(db_path)

    @app.post("/api/v1/submit", response_model=SubmitResponse)
    def submit_result(req: SubmitRequest) -> SubmitResponse:
        """Submit a benchmark result."""
        overall_score = db.compute_overall_score(
            ttc_min=req.ttc_min,
            pet_min=req.pet_min,
            collision_count=req.collision_count,
            collision_rate=req.collision_rate,
            max_jerk=req.max_jerk,
            mean_lateral_deviation=req.mean_lateral_deviation,
            mean_heading_error=req.mean_heading_error,
        )

        entry = LeaderboardEntry(
            id=None,
            planner_name=req.planner_name,
            scenario_name=req.scenario_name,
            ttc_min=req.ttc_min,
            pet_min=req.pet_min,
            collision_count=req.collision_count,
            collision_rate=req.collision_rate,
            max_jerk=req.max_jerk,
            mean_lateral_deviation=req.mean_lateral_deviation,
            mean_heading_error=req.mean_heading_error,
            overall_score=overall_score,
            metadata=req.metadata,
        )

        entry_id = db.submit(entry)

        rankings = db.get_rankings()
        rank = 1
        for i, r in enumerate(rankings):
            if r.id == entry_id:
                rank = i + 1
                break

        return SubmitResponse(id=entry_id, overall_score=overall_score, rank=rank)

    @app.get("/api/v1/rankings")
    def get_rankings(
        scenario: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RankingEntry]:
        """Get the current leaderboard rankings."""
        entries = db.get_rankings(scenario_name=scenario, limit=limit, offset=offset)
        results = []
        for i, entry in enumerate(entries):
            results.append(
                RankingEntry(
                    rank=offset + i + 1,
                    planner_name=entry.planner_name,
                    scenario_name=entry.scenario_name,
                    overall_score=entry.overall_score,
                    ttc_min=entry.ttc_min,
                    collision_count=entry.collision_count,
                    max_jerk=entry.max_jerk,
                    mean_lateral_deviation=entry.mean_lateral_deviation,
                )
            )
        return results

    @app.get("/api/v1/planner/{planner_name}")
    def get_planner_results(planner_name: str) -> list[dict]:
        """Get all results for a specific planner."""
        entries = db.get_planner_results(planner_name)
        if not entries:
            raise HTTPException(status_code=404, detail="Planner not found")
        return [e.to_dict() for e in entries]

    @app.get("/api/v1/entry/{entry_id}")
    def get_entry(entry_id: int) -> dict:
        """Get a specific leaderboard entry."""
        entry = db.get_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry.to_dict()

    @app.delete("/api/v1/entry/{entry_id}")
    def delete_entry(entry_id: int) -> dict:
        """Delete a leaderboard entry."""
        if db.delete_entry(entry_id):
            return {"status": "deleted", "id": entry_id}
        raise HTTPException(status_code=404, detail="Entry not found")

    return app
