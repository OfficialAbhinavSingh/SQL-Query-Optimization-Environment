from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Observation(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    task_name: str = Field(..., description="Human-readable task name")
    task_description: str = Field(..., description="What the agent must do")
    sql_query: str = Field(..., description="The SQL query to analyze/optimize")
    schema_info: str = Field(..., description="Database schema context")
    dialect: str = Field(default="postgresql", description="SQL dialect (postgresql, mysql, sqlite)")
    difficulty: str = Field(..., description="easy | medium | hard")
    step_count: int = Field(default=0, description="Steps taken in this episode")
    max_steps: int = Field(default=5, description="Max steps per episode")
    issues_found_so_far: List[str] = Field(default_factory=list, description="Issues agent has flagged so far")


class OptimizationSuggestion(BaseModel):
    issue_type: str = Field(..., description="Type of issue (e.g. missing_index, n_plus_one, full_table_scan, etc.)")
    line: Optional[int] = Field(None, description="Approximate line number in query")
    description: str = Field(..., description="Detailed description of the issue")
    severity: str = Field(..., description="critical | high | medium | low")
    fix: str = Field(..., description="Suggested fix or rewrite")


class Action(BaseModel):
    suggestions: List[Dict[str, Any]] = Field(
        ...,
        description="List of optimization suggestions. Each: {issue_type, line, description, severity, fix}"
    )
    optimized_query: str = Field(..., description="Rewritten/optimized version of the SQL query")
    summary: str = Field(..., description="Overall analysis summary")
    estimated_improvement: str = Field(..., description="Estimated performance improvement (e.g. '10x faster', '~50% less I/O')")
    approved: bool = Field(..., description="Whether query is already optimal (True) or needs changes (False)")


class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Reward score 0.0-1.0")
    breakdown: Dict[str, float] = Field(..., description="Per-criterion scores")
    feedback: str = Field(..., description="Human-readable feedback on the action")


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class EnvironmentState(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    episode_done: bool
    cumulative_reward: float
    current_task: str
