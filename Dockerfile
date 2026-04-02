FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY av_sim_arena/ av_sim_arena/

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "av_sim_arena.leaderboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
