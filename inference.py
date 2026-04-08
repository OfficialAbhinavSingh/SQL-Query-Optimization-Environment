"""
inference.py — SQL Query Optimization Environment
===================================================
OpenEnv Hackathon Phase 1 Submission

Required environment variables:
  API_BASE_URL   The API endpoint for the LLM (default: HuggingFace router)
  MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your HuggingFace / API key

stdout format (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import sys
from typing import List, Optional
from openai import OpenAI

# ── Resolve paths so we can import env/models from root ──────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from env import SQLOptimEnv
from models import Action

# ── Configuration ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "") or os.environ.get("API_KEY", "")

BENCHMARK    = "sql-optim-env"
TEMPERATURE  = 0.0
MAX_TOKENS   = 1500

TASK_IDS = [
    "task_1_basic_antipatterns",
    "task_2_join_optimization",
    "task_3_advanced_optimization",
]

SYSTEM_PROMPT = """\
You are an expert database engineer and SQL performance specialist with deep knowledge of \
PostgreSQL internals, query planning, and index design.

You will receive a SQL query, its database schema, and a task description. \
Your job is to:
1. Identify ALL performance issues and anti-patterns in the query.
2. Produce an optimized rewrite of the query.
3. Estimate the expected performance improvement.

Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{
  "suggestions": [
    {
      "issue_type": "string (e.g. select_star, non_sargable_predicate, correlated_subquery, missing_index, etc.)",
      "line": <integer line number in the query>,
      "description": "clear explanation of why this is a problem",
      "severity": "critical | high | medium | low",
      "fix": "specific fix or rewritten clause"
    }
  ],
  "optimized_query": "the full rewritten SQL query with all improvements applied",
  "summary": "2-4 sentence overall analysis of the query performance profile",
  "estimated_improvement": "e.g. '10-50x faster on large tables due to index usage', '~80% reduction in I/O'",
  "approved": false
}

Be thorough and precise. Every issue you identify should have a concrete fix.
"""


# ── Logging helpers ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Model interaction ──────────────────────────────────────────────────────

def parse_action(response_text: str) -> dict:
    """Parse JSON from model response, stripping code fences if present."""
    clean = response_text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Drop first and last fence lines
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        if clean.startswith("json"):
            clean = clean[4:].strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "suggestions": [],
            "optimized_query": "",
            "summary": "JSON parse error — model returned malformed output.",
            "estimated_improvement": "unknown",
            "approved": False,
        }


def get_model_action(client: OpenAI, obs) -> tuple[dict, Optional[str]]:
    """Call the LLM and return (parsed_action_dict, error_or_None)."""
    user_content = f"""Task: {obs.task_name}
Difficulty: {obs.difficulty}
SQL Dialect: {obs.dialect}

Instructions:
{obs.task_description}

Database Schema:
{obs.schema_info}

SQL Query to Analyze (step {obs.step_count + 1}/{obs.max_steps}):
```sql
{obs.sql_query}
```

Issues identified in previous steps: {obs.issues_found_so_far if obs.issues_found_so_far else 'None yet'}

Provide your complete analysis and optimized rewrite now.
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        return parse_action(response_text), None
    except Exception as exc:
        error_msg = str(exc)
        return {
            "suggestions": [],
            "optimized_query": "",
            "summary": f"Model call failed: {error_msg}",
            "estimated_improvement": "unknown",
            "approved": False,
        }, error_msg


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is not set.", flush=True)
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    local_env = SQLOptimEnv()
    results = {}

    for task_id in TASK_IDS:
        obs = local_env.reset(task_id=task_id)
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        try:
            for step in range(1, obs.max_steps + 1):
                parsed, error = get_model_action(client, obs)

                action = Action(
                    suggestions=parsed.get("suggestions", []),
                    optimized_query=parsed.get("optimized_query", ""),
                    summary=parsed.get("summary", ""),
                    estimated_improvement=parsed.get("estimated_improvement", ""),
                    approved=parsed.get("approved", False),
                )

                result = local_env.step(action)
                reward = result.reward.score
                done = result.done

                rewards.append(reward)
                steps_taken = step
                obs = result.observation

                action_summary = f"suggestions={len(action.suggestions)},score={reward:.2f}"
                log_step(step=step, action=action_summary, reward=reward, done=done, error=error)

                if done:
                    break

            score = max(rewards) if rewards else 0.0
            success = score >= 0.5

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        results[task_id] = {
            "task_name": obs.task_name,
            "final_score": round(score, 4),
            "steps_taken": steps_taken,
        }

    return results


if __name__ == "__main__":
    main()
