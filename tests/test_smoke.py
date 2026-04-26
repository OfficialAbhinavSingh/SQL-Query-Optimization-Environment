"""Fast smoke tests for CI (DuckDB warm-up once per process)."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import SQLOptimEnv  # noqa: E402
from graders import GradeMask, grade  # noqa: E402
from models import Action  # noqa: E402
from tasks import TASKS  # noqa: E402


@pytest.fixture(scope="module")
def executor():
    from executor import get_executor

    return get_executor()


def test_executor_compare_task1(executor):
    task = TASKS["task_1_basic_antipatterns"]
    original = task["sql_query"]
    optimized = (
        "SELECT id, customer_id, product_id, status, total, created_at FROM orders "
        "WHERE customer_id = 5000 "
        "AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"
    )
    r = executor.compare(original, optimized)
    assert r["speedup"] >= 1.0
    assert r["results_match"] is True


def test_grade_mask_changes_total():
    task = TASKS["task_1_basic_antipatterns"]
    action = Action(
        suggestions=[
            {
                "issue_type": "select_star",
                "line": 1,
                "description": "SELECT * on large table",
                "severity": "high",
                "fix": "project columns",
            }
        ],
        optimized_query=(
            "SELECT id, customer_id, product_id, status, total, created_at FROM orders "
            "WHERE customer_id = 5000 AND created_at >= DATE '2024-01-01' "
            "AND created_at < DATE '2025-01-01'"
        ),
        summary="x" * 130,
        estimated_improvement="5x",
        approved=False,
    )
    full = grade(task, action).score
    no_exec = grade(
        task, action, mask=GradeMask(execution_speedup=False, result_correctness=False)
    ).score
    assert no_exec < full


def test_fastapi_reset_step():
    from fastapi.testclient import TestClient

    from server.app import app

    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["environment"] == "sql-optim-env"

    obs = client.post("/reset", json={"task_id": "task_1_basic_antipatterns"}).json()
    assert obs["task_id"] == "task_1_basic_antipatterns"

    step = client.post(
        "/step",
        json={
            "suggestions": [
                {
                    "issue_type": "select_star",
                    "line": 1,
                    "description": "avoid star",
                    "severity": "high",
                    "fix": "cols",
                }
            ],
            "optimized_query": (
                "SELECT id, customer_id, product_id, status, total, created_at FROM orders "
                "WHERE customer_id = 5000 "
                "AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"
            ),
            "summary": "Rewrite removes anti-patterns and uses a sargable date range.",
            "estimated_improvement": "4x",
            "approved": False,
        },
    )
    assert step.status_code == 200
    body = step.json()
    assert "reward" in body
    assert body["reward"]["score"] > 0.5


def test_sqoptim_env_reset_step():
    env = SQLOptimEnv()
    obs = env.reset(task_id="task_1_basic_antipatterns")
    assert obs.step_count == 0
    result = env.step(
        Action(
            suggestions=[
                {
                    "issue_type": "select_star",
                    "line": 1,
                    "description": "SELECT *",
                    "severity": "high",
                    "fix": "list columns",
                }
            ],
            optimized_query=(
                "SELECT id, customer_id, product_id, status, total, created_at FROM orders "
                "WHERE customer_id = 5000 "
                "AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"
            ),
            summary="A" * 130,
            estimated_improvement="5x",
            approved=False,
        )
    )
    assert result.reward.score > 0.4
