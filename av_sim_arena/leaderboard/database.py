"""SQLite backend for the leaderboard system."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LeaderboardEntry:
    """A single leaderboard entry."""

    id: int | None
    planner_name: str
    scenario_name: str
    ttc_min: float
    pet_min: float
    collision_count: int
    collision_rate: float
    max_jerk: float
    mean_lateral_deviation: float
    mean_heading_error: float
    overall_score: float
    metadata: dict | None = None
    submitted_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "planner_name": self.planner_name,
            "scenario_name": self.scenario_name,
            "ttc_min": self.ttc_min,
            "pet_min": self.pet_min,
            "collision_count": self.collision_count,
            "collision_rate": self.collision_rate,
            "max_jerk": self.max_jerk,
            "mean_lateral_deviation": self.mean_lateral_deviation,
            "mean_heading_error": self.mean_heading_error,
            "overall_score": self.overall_score,
            "metadata": self.metadata,
            "submitted_at": self.submitted_at,
        }


class LeaderboardDB:
    """SQLite-backed leaderboard database."""

    def __init__(self, db_path: str = "leaderboard.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    planner_name TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    ttc_min REAL,
                    pet_min REAL,
                    collision_count INTEGER DEFAULT 0,
                    collision_rate REAL DEFAULT 0.0,
                    max_jerk REAL,
                    mean_lateral_deviation REAL,
                    mean_heading_error REAL,
                    overall_score REAL,
                    metadata TEXT,
                    submitted_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_planner ON results(planner_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scenario ON results(scenario_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_score ON results(overall_score)
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def submit(self, entry: LeaderboardEntry) -> int:
        """Submit a new result to the leaderboard."""
        if entry.submitted_at is None:
            entry.submitted_at = datetime.now(timezone.utc).isoformat()

        metadata_json = json.dumps(entry.metadata) if entry.metadata else None

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO results (
                    planner_name, scenario_name, ttc_min, pet_min,
                    collision_count, collision_rate, max_jerk,
                    mean_lateral_deviation, mean_heading_error,
                    overall_score, metadata, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.planner_name,
                    entry.scenario_name,
                    entry.ttc_min,
                    entry.pet_min,
                    entry.collision_count,
                    entry.collision_rate,
                    entry.max_jerk,
                    entry.mean_lateral_deviation,
                    entry.mean_heading_error,
                    entry.overall_score,
                    metadata_json,
                    entry.submitted_at,
                ),
            )
            return cursor.lastrowid

    def get_rankings(
        self,
        scenario_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[LeaderboardEntry]:
        """Get ranked results, optionally filtered by scenario."""
        query = "SELECT * FROM results"
        params: list = []

        if scenario_name:
            query += " WHERE scenario_name = ?"
            params.append(scenario_name)

        query += " ORDER BY overall_score DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_planner_results(self, planner_name: str) -> list[LeaderboardEntry]:
        """Get all results for a specific planner."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM results WHERE planner_name = ? ORDER BY overall_score DESC",
                (planner_name,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_entry(self, entry_id: int) -> LeaderboardEntry | None:
        """Get a specific entry by ID."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM results WHERE id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def compute_overall_score(
        self,
        ttc_min: float,
        pet_min: float,
        collision_count: int,
        collision_rate: float,
        max_jerk: float,
        mean_lateral_deviation: float,
        mean_heading_error: float,
    ) -> float:
        """Compute an overall score from individual metrics.

        Higher is better. Penalizes collisions and poor safety metrics.
        """
        safety_score = min(ttc_min, 10.0) / 10.0 * 30.0
        pet_score = min(pet_min, 5.0) / 5.0 * 15.0
        collision_penalty = collision_count * 20.0
        jerk_score = max(0, 15.0 - max_jerk)
        tracking_score = max(0, 20.0 - mean_lateral_deviation * 10.0)
        heading_score = max(0, 20.0 - mean_heading_error * 10.0)

        total = (
            safety_score
            + pet_score
            - collision_penalty
            + jerk_score
            + tracking_score
            + heading_score
        )
        return max(0.0, min(100.0, total))

    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry by ID."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM results WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    def _row_to_entry(self, row: sqlite3.Row) -> LeaderboardEntry:
        metadata = None
        if row["metadata"]:
            metadata = json.loads(row["metadata"])
        return LeaderboardEntry(
            id=row["id"],
            planner_name=row["planner_name"],
            scenario_name=row["scenario_name"],
            ttc_min=row["ttc_min"],
            pet_min=row["pet_min"],
            collision_count=row["collision_count"],
            collision_rate=row["collision_rate"],
            max_jerk=row["max_jerk"],
            mean_lateral_deviation=row["mean_lateral_deviation"],
            mean_heading_error=row["mean_heading_error"],
            overall_score=row["overall_score"],
            metadata=metadata,
            submitted_at=row["submitted_at"],
        )
