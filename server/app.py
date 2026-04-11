"""
server/app.py — FastAPI Server
================================
OpenEnv-compliant endpoints + two unique endpoints:
  POST /execute    — run your optimized query against real DuckDB data,
                     see actual speedup + result correctness instantly
  GET  /leaderboard — see best scores + speedups across all tasks
"""

import json
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import SQLOptimEnv
from executor import get_executor
from graders import grade
from leaderboard import get_board
from models import (
    Action,
    EnvironmentState,
    ExecutionResult,
    Observation,
    StepResult,
)
from tasks import TASKS, get_task_list


# ── Lifespan: pre-warm DuckDB on startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build all 4 synthetic tables before first request
    get_executor()
    yield


app = FastAPI(
    title="SQL Query Optimization Environment",
    description=(
        "OpenEnv-compliant RL environment where AI agents learn to diagnose "
        "and optimize SQL queries. Uniquely, optimized queries are EXECUTED "
        "against real DuckDB data — reward is based on actual speedup + "
        "result correctness, not keyword heuristics."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SQLOptimEnv()


# ── Standard OpenEnv endpoints ────────────────────────────────────────────

@app.get("/")
def root():
    ex = get_executor()
    return {
        "status":      "ok",
        "environment": "sql-optim-env",
        "version":     "2.0.0",
        "unique_feature": "Execution-grounded rewards via DuckDB",
        "table_stats": ex.table_stats,
        "tasks":       [t["task_id"] for t in get_task_list()],
    }


@app.post("/reset", response_model=Observation)
async def reset(request: Request):
    """Start a new episode. Body: {"task_id": "..."}  (optional)."""
    try:
        body = await request.body()
        task_id = "task_1_basic_antipatterns"
        if body:
            try:
                data = json.loads(body)
                task_id = data.get("task_id", task_id) or task_id
            except Exception:
                pass
        return env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Submit an optimization action; get real execution feedback."""
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=EnvironmentState)
def state():
    return env.state()


@app.get("/tasks")
def tasks():
    return {"tasks": get_task_list()}


@app.post("/grader")
def grader(action: Action):
    """Grade an action against the current task without advancing the episode."""
    if env._task_data is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return grade(env._task_data, action)


@app.post("/baseline")
def baseline():
    """Run the baseline inference script and return output."""
    import subprocess
    try:
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return {
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {exc}")


# ── Unique endpoints (no other team has these) ────────────────────────────

@app.post("/execute", response_model=ExecutionResult)
async def execute(request: Request):
    """
    🚀 UNIQUE ENDPOINT — Execute your optimized query against real DuckDB data.

    Body:
      {
        "task_id": "task_1_basic_antipatterns",
        "optimized_query": "SELECT id, customer_id ... WHERE customer_id = 5000 ..."
      }

    Returns actual execution timing, speedup ratio, result correctness,
    and an EXPLAIN plan — no other OpenEnv environment does this.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Body required: {task_id, optimized_query}")
    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    task_id = data.get("task_id", "task_1_basic_antipatterns")
    optimized_query = (data.get("optimized_query") or "").strip()

    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    if not optimized_query:
        raise HTTPException(status_code=400, detail="optimized_query is required")

    original_query = TASKS[task_id]["sql_query"]
    ex = get_executor()

    try:
        result = ex.compare(original_query, optimized_query)
        explain = ex.explain(optimized_query)
        return ExecutionResult(
            original_ms=result["original_ms"],
            optimized_ms=result["optimized_ms"],
            speedup=result["speedup"],
            results_match=result["results_match"],
            original_rows=result["original_rows"],
            optimized_rows=result["optimized_rows"],
            original_error=result.get("original_error"),
            optimized_error=result.get("optimized_error"),
            verdict=result["verdict"],
            explain_plan=explain,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/leaderboard")
def leaderboard():
    """
    🏆 UNIQUE ENDPOINT — Real-time leaderboard of best execution scores.

    Shows per-task: best score, best speedup achieved, total attempts,
    how many optimized queries produced correct results.
    """
    return {
        "leaderboard": get_board(),
        "description": (
            "Scores are based on real DuckDB execution: "
            "speedup ratio (35%) + result correctness (20%) + issue detection (25%) + other (20%)"
        ),
    }


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
