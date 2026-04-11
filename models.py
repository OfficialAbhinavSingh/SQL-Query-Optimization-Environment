from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Observation(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    task_name: str = Field(..., description="Human-readable task name")
    task_description: str = Field(..., description="What the agent must do")
    sql_query: str = Field(..., description="The SQL query to analyze and optimize")
    schema_info: str = Field(..., description="Database schema, table sizes, and index info")
    dialect: str = Field(default="duckdb/postgresql", description="SQL dialect")
    difficulty: str = Field(..., description="easy | medium | hard | expert")
    step_count: int = Field(default=0, description="Steps taken in this episode")
    max_steps: int = Field(default=5, description="Max steps per episode")
    issues_found_so_far: List[str] = Field(
        default_factory=list,
        description="Issue types flagged in previous steps"
    )
    last_execution: Optional[Dict[str, Any]] = Field(
        None,
        description="Execution comparison result from previous step — "
                    "use this to refine your optimized_query"
    )


class Action(BaseModel):
    suggestions: List[Dict[str, Any]] = Field(
        ...,
        description="List of issues. Each: {issue_type, line, description, severity, fix}"
    )
    optimized_query: str = Field(
        ...,
        description="Complete rewritten SQL — will be EXECUTED against real data to measure speedup"
    )
    summary: str = Field(..., description="Overall analysis and performance profile")
    estimated_improvement: str = Field(
        ...,
        description="Expected speedup (e.g. '10x faster', '~80% I/O reduction')"
    )
    approved: bool = Field(
        ...,
        description="True if query is already optimal, False if it needs changes"
    )


class Reward(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Composite reward 0.0–1.0")
    breakdown: Dict[str, float] = Field(..., description="Per-criterion scores")
    feedback: str = Field(..., description="Human-readable feedback with execution details")


class ExecutionResult(BaseModel):
    """Real DuckDB execution comparison — returned by /execute endpoint."""
    original_ms: float = Field(..., description="Original query median execution time (ms)")
    optimized_ms: float = Field(..., description="Optimized query median execution time (ms)")
    speedup: float = Field(..., description="Speedup ratio (original_ms / optimized_ms)")
    results_match: bool = Field(..., description="Do both queries return identical results?")
    original_rows: int = Field(..., description="Row count from original query")
    optimized_rows: int = Field(..., description="Row count from optimized query")
    original_error: Optional[str] = Field(None, description="Error from original, if any")
    optimized_error: Optional[str] = Field(None, description="Error from optimized, if any")
    verdict: str = Field(..., description="Human-readable verdict")
    explain_plan: Optional[str] = Field(None, description="EXPLAIN output for optimized query")


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
