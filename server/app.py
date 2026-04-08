from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import SQLOptimEnv
from models import Action, StepResult, EnvironmentState, Observation
from tasks import get_task_list
from graders import grade

app = FastAPI(
    title="SQL Query Optimization Environment",
    description=(
        "OpenEnv-compliant RL environment where AI agents learn to analyze, "
        "diagnose, and optimize SQL queries across three difficulty levels."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SQLOptimEnv()


@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "sql-optim-env",
        "version": "1.0.0",
        "tasks": [t["task_id"] for t in get_task_list()],
    }


@app.post("/reset", response_model=Observation)
async def reset(request: Request):
    """
    Start a new episode. Optionally pass {"task_id": "..."} in the body.
    Defaults to task_1_basic_antipatterns.
    """
    try:
        body = await request.body()
        task_id = "task_1_basic_antipatterns"
        if body:
            try:
                data = json.loads(body)
                task_id = data.get("task_id", task_id) or task_id
            except Exception:
                pass
        obs = env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Take one action (submit SQL analysis + optimized query)."""
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvironmentState)
def state():
    """Get current environment state without advancing the episode."""
    return env.state()


@app.get("/tasks")
def tasks():
    """List all available tasks with descriptions and action schema."""
    return {"tasks": get_task_list()}


@app.post("/grader")
def grader(action: Action):
    """Grade an action against the current task without advancing the episode."""
    if env._task_data is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    reward = grade(env._task_data, action)
    return reward


@app.post("/baseline")
def baseline():
    """Run the baseline agent and return scores for all tasks."""
    try:
        import subprocess
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")



def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
