"""
env.py — SQLOptimEnv: Core OpenEnv Environment Class
"""

from typing import Any, Dict, Optional

from executor import get_executor
from graders import grade
from leaderboard import record as lb_record
from models import (
    Action,
    EnvironmentState,
    Observation,
    Reward,
    StepResult,
)
from tasks import TASKS


class SQLOptimEnv:
    """
    OpenEnv-compliant environment for SQL Query Optimization.

    The agent receives a SQL query + schema context, emits an Action
    containing a list of optimization suggestions AND a rewritten
    optimized_query.  The environment executes both queries against
    real DuckDB data, measures the actual speedup, and checks
    result correctness — all fed into the reward function.

    Multi-step:
      • issues_found_so_far accumulates flagged issue types.
      • last_execution carries execution metrics back to the agent
        so it can refine the optimized_query in subsequent steps.
    """

    def __init__(self) -> None:
        self._task_data: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._issues_found: list = []
        self._last_execution: Optional[Dict[str, Any]] = None

    # ── OpenEnv interface ─────────────────────────────────────────────

    def reset(
        self, task_id: str = "task_1_basic_antipatterns"
    ) -> Observation:
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid: {list(TASKS.keys())}"
            )
        self._task_data = TASKS[task_id]
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._issues_found = []
        self._last_execution = None
        return self._make_obs()

    def step(self, action: Action) -> StepResult:
        if self._task_data is None:
            raise RuntimeError("No active episode — call reset() first.")
        if self._done:
            raise RuntimeError("Episode finished — call reset() to start a new one.")

        self._step_count += 1

        # Grade (runs DuckDB internally)
        reward: Reward = grade(self._task_data, action)
        self._cumulative_reward += reward.score

        # Extract execution info from grader feedback for next obs
        opt_q = (action.optimized_query or "").strip()
        if opt_q:
            try:
                ex = get_executor()
                self._last_execution = ex.compare(
                    self._task_data["sql_query"], opt_q
                )
            except Exception:
                self._last_execution = None

        # Track issue types for progressive context
        for s in action.suggestions:
            itype = s.get("issue_type", "")
            if itype and itype not in self._issues_found:
                self._issues_found.append(itype)

        max_steps: int = self._task_data["max_steps"]
        done = self._step_count >= max_steps or reward.score >= 0.95
        self._done = done

        # Update leaderboard
        speedup = (
            self._last_execution.get("speedup", 1.0)
            if self._last_execution else 1.0
        )
        results_match = (
            self._last_execution.get("results_match", False)
            if self._last_execution else False
        )
        lb_record(
            task_id=self._task_data["task_id"],
            speedup=speedup,
            score=reward.score,
            results_match=results_match,
            steps=self._step_count,
        )

        return StepResult(
            observation=self._make_obs(),
            reward=reward,
            done=done,
            info={
                "step":              self._step_count,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "issues_found":      len(self._issues_found),
                "execution":         self._last_execution,
            },
        )

    def state(self) -> EnvironmentState:
        if self._task_data is None:
            return EnvironmentState(
                task_id="none", step_count=0, max_steps=0,
                episode_done=True, cumulative_reward=0.0,
                current_task="No active episode",
            )
        return EnvironmentState(
            task_id=self._task_data["task_id"],
            step_count=self._step_count,
            max_steps=self._task_data["max_steps"],
            episode_done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            current_task=self._task_data["task_name"],
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _make_obs(self) -> Observation:
        d = self._task_data
        return Observation(
            task_id=d["task_id"],
            task_name=d["task_name"],
            task_description=d["task_description"],
            sql_query=d["sql_query"],
            schema_info=d["schema_info"],
            dialect=d.get("dialect", "duckdb/postgresql"),
            difficulty=d["difficulty"],
            step_count=self._step_count,
            max_steps=d["max_steps"],
            issues_found_so_far=list(self._issues_found),
            last_execution=self._last_execution,
        )
