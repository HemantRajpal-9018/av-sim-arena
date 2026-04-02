# AV-Sim-Arena

Multi-scenario simulation benchmark for autonomous vehicle decision-making.

## Architecture

```
av_sim_arena/
├── scenarios/          # Scenario generation and data models
│   ├── models.py       # WeatherCondition, TrafficDensity, Scenario, VehicleState
│   └── generator.py    # ScenarioGenerator (from YAML, random, configurable)
├── metrics/            # Safety metrics computation
│   └── safety.py       # TTC, PET, collision rate, jerk, lateral deviation, heading error
├── planners/           # Planner interfaces and implementations
│   ├── base.py         # BasePlanner abstract class
│   ├── lattice.py      # State lattice planner (Frenet frame sampling)
│   ├── rrt_star.py     # RRT* motion planner
│   ├── mpc.py          # Model Predictive Control (bicycle model)
│   └── rl_planner.py   # RL-based planner (policy network)
├── traffic/            # Multi-agent traffic simulation
│   ├── behavior_tree.py # Behavior trees: follow, yield, lane change, aggressive
│   └── npc.py          # NPCVehicle and NPCPedestrian agents
├── connectors/         # Simulator connectors
│   ├── base.py         # BaseConnector interface
│   ├── carla_connector.py  # CARLA simulator stub
│   └── sumo_connector.py   # SUMO simulator stub
├── leaderboard/        # Benchmark leaderboard system
│   ├── database.py     # SQLite backend with ranking
│   └── api.py          # FastAPI REST API
└── visualization/      # Plotting and replay tools
    ├── replay.py       # Matplotlib animation for scenario replay
    └── plots.py        # Bar charts, radar charts, heatmaps
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start the leaderboard API
uvicorn av_sim_arena.leaderboard.api:app --reload --port 8000
```

## Benchmark Scenarios

| Scenario | Description | Difficulty |
|----------|-------------|------------|
| `highway_merge` | Merge from on-ramp into flowing highway traffic | High |
| `intersection` | Navigate signalized intersection with cross-traffic | Medium |
| `parking` | Low-speed maneuvering into a parking spot | Low |
| `emergency_stop` | Sudden brake + road debris evasive action | High |
| `roundabout` | Enter/exit roundabout with circulating traffic | Medium |

## Planners

| Planner | Type | Description |
|---------|------|-------------|
| Lattice | Sampling | Samples trajectories in Frenet frame, evaluates cost |
| RRT* | Sampling | Rapidly-exploring random tree with rewiring |
| MPC | Optimization | Model predictive control with bicycle model |
| RL | Learning | Neural network policy (simulated weights) |

## Safety Metrics

- **TTC** — Time-to-collision: time until potential collision with nearest object
- **PET** — Post-encroachment time: temporal gap at conflict points
- **Collision Rate** — Fraction of timesteps with active collisions
- **Jerk** — Rate of change of acceleration (comfort metric)
- **Lateral Deviation** — Distance from reference path centerline
- **Heading Error** — Angular deviation from reference heading

## Example Results

Average overall scores across all benchmark scenarios (higher is better):

| Rank | Planner | Avg Score | Best Scenario | Worst Scenario |
|------|---------|-----------|---------------|----------------|
| 1 | MPC | 80.0 | highway_merge (85.2) | emergency_stop (72.0) |
| 2 | Lattice | 70.3 | highway_merge (78.5) | emergency_stop (52.8) |
| 3 | RRT* | 64.3 | highway_merge (71.2) | emergency_stop (47.2) |
| 4 | RL | 54.2 | highway_merge (68.9) | emergency_stop (30.5) |

## Leaderboard API

```bash
# Submit a result
curl -X POST http://localhost:8000/api/v1/submit \
  -H "Content-Type: application/json" \
  -d '{"planner_name": "my_planner", "scenario_name": "highway_merge", "ttc_min": 3.5, "pet_min": 2.0, "collision_count": 0, "collision_rate": 0.0, "max_jerk": 4.0, "mean_lateral_deviation": 0.3, "mean_heading_error": 0.05}'

# Get rankings
curl http://localhost:8000/api/v1/rankings

# Filter by scenario
curl http://localhost:8000/api/v1/rankings?scenario=highway_merge
```

## Docker

```bash
docker compose up -d
# API available at http://localhost:8000
```

## Development

```bash
make install-dev   # Install with dev dependencies
make test          # Run tests
make lint          # Run linter
make format        # Format code
make clean         # Clean build artifacts
```

## License

MIT
