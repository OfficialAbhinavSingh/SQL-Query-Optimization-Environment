from typing import Optional
from models import Observation, Action, Reward, StepResult, EnvironmentState
from tasks import TASKS
from graders import grade


class SQLOptimEnv:
    """
    OpenEnv-compliant environment for SQL Query Optimization.

    An AI agent iteratively analyzes a SQL query, identifies performance issues,
    and submits optimized rewrites. The environment grades each action and tracks
    progress across multiple steps within an episode.
    """

    def __init__(self):
        self._task_data: Optional[dict] = None
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._issues_found: list = []

    def reset(self, task_id: str = "task_1_basic_antipatterns") -> Observation:
        """Start a new episode for the given task."""
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid tasks: {list(TASKS.keys())}"
            )
        self._task_data = TASKS[task_id]
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._issues_found = []

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Process one agent action and return (observation, reward, done, info)."""
        if self._task_data is None:
            raise RuntimeError("Episode not started. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode already finished. Call reset() to start a new episode.")

        self._step_count += 1

        # Grade the action
        reward: Reward = grade(self._task_data, action)
        self._cumulative_reward += reward.score

        # Track issue types found so far
        for s in action.suggestions:
            issue_type = s.get("issue_type", "")
            if issue_type and issue_type not in self._issues_found:
                self._issues_found.append(issue_type)

        # Episode ends when max_steps reached OR agent finds a perfect score
        max_steps = self._task_data["max_steps"]
        done = self._step_count >= max_steps or reward.score >= 0.95

        self._done = done

        obs = self._make_observation()

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self._step_count,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "issues_found_count": len(self._issues_found),
            }
        )

    def state(self) -> EnvironmentState:
        """Return current environment state (for /state endpoint)."""
        if self._task_data is None:
            return EnvironmentState(
                task_id="none",
                step_count=0,
                max_steps=0,
                episode_done=True,
                cumulative_reward=0.0,
                current_task="No active episode"
            )
        return EnvironmentState(
            task_id=self._task_data["task_id"],
            step_count=self._step_count,
            max_steps=self._task_data["max_steps"],
            episode_done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            current_task=self._task_data["task_name"],
        )

    def _make_observation(self) -> Observation:
        d = self._task_data
        return Observation(
            task_id=d["task_id"],
            task_name=d["task_name"],
            task_description=d["task_description"],
            sql_query=d["sql_query"],
            schema_info=d["schema_info"],
            dialect=d.get("dialect", "postgresql"),
            difficulty=d["difficulty"],
            step_count=self._step_count,
            max_steps=d["max_steps"],
            issues_found_so_far=list(self._issues_found),
        )
