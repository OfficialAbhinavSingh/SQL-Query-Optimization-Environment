"""
inference.py — SQL Query Optimization Environment
===================================================
Multi-step inference loop with execution-feedback awareness.

When the environment returns execution results from a previous step,
the agent uses them to REFINE its optimized query — creating a genuine
iterative optimization loop grounded in real performance data.

stdout format (strictly followed):
  [START] task=<task_id> env=sql-optim-env model=<MODEL_NAME>
  [STEP]  step=<n> action=<summary> reward=<0.00> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<score> rewards=<r1,...,rn>
"""

import json
import os
import sys
from typing import Dict, List, Optional

from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from env import SQLOptimEnv
from models import Action

# ── Config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "") or os.environ.get("API_KEY", "")

BENCHMARK   = "sql-optim-env"
TEMPERATURE = 0.0
MAX_TOKENS  = 2000

TASK_IDS = [
    "task_1_basic_antipatterns",
    "task_2_correlated_subqueries",
    "task_3_wildcard_scan",
    "task_4_implicit_join",
    "task_5_window_functions",
]

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an elite database engineer and SQL performance specialist with expert-level \
knowledge of PostgreSQL/DuckDB internals, query planning, columnar storage, \
and index design.

You will receive a SQL query and its schema. Your job:
1. Identify ALL performance anti-patterns.
2. Produce a complete, correct, optimized rewrite.
3. Your optimized_query will be ACTUALLY EXECUTED against a DuckDB database \
   with realistic data (orders=500k rows, events=1M rows). \
   If it returns wrong results or errors, your score drops.
4. If you receive execution feedback from a previous step, USE IT to refine \
   your rewrite — fix incorrect results first, then improve speed.

Respond ONLY with valid JSON (no markdown, no fences):
{
  "suggestions": [
    {
      "issue_type": "e.g. select_star / correlated_subquery / wildcard_like",
      "line": <integer>,
      "description": "precise explanation of the performance problem",
      "severity": "critical | high | medium | low",
      "fix": "specific rewrite or corrective SQL"
    }
  ],
  "optimized_query": "<complete, executable SQL that produces IDENTICAL results to original>",
  "summary": "2-4 sentence performance profile of the original query",
  "estimated_improvement": "e.g. '15x faster — eliminates N+1 subquery pattern'",
  "approved": false
}
"""

# ── Logging (strict OpenEnv format) ──────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rstr}",
        flush=True,
    )


# ── Model interaction ─────────────────────────────────────────────────────

def parse_action(text: str) -> Dict:
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )
        if clean.startswith("json"):
            clean = clean[4:].strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "suggestions":          [],
            "optimized_query":      "",
            "summary":              "Parse error — model returned malformed JSON.",
            "estimated_improvement": "unknown",
            "approved":             False,
        }


def build_user_prompt(obs) -> str:
    exec_feedback = ""
    if obs.last_execution:
        ex = obs.last_execution
        exec_feedback = (
            f"\n\n⚡ EXECUTION FEEDBACK FROM YOUR LAST OPTIMIZED QUERY:\n"
            f"  Original query  : {ex.get('original_ms', '?'):.1f} ms "
            f"  ({ex.get('original_rows', 0)} rows)\n"
            f"  Your last query : {ex.get('optimized_ms', '?'):.1f} ms "
            f"  ({ex.get('optimized_rows', 0)} rows)\n"
            f"  Speedup achieved: {ex.get('speedup', 1.0):.2f}x\n"
            f"  Results match   : {'✅ YES' if ex.get('results_match') else '❌ NO — fix your WHERE/JOIN logic'}\n"
            f"  Verdict         : {ex.get('verdict', '')}\n"
            f"Refine your optimized_query to fix any correctness issues first, "
            f"then improve speed further."
        )

    issues_ctx = ""
    if obs.issues_found_so_far:
        issues_ctx = (
            f"\nIssue types you've already flagged: {obs.issues_found_so_far}"
        )

    return (
        f"Task        : {obs.task_name}\n"
        f"Difficulty  : {obs.difficulty}\n"
        f"Step        : {obs.step_count + 1} / {obs.max_steps}\n\n"
        f"Instructions:\n{obs.task_description}\n\n"
        f"Database Schema:\n{obs.schema_info}\n\n"
        f"SQL Query to Optimize:\n```sql\n{obs.sql_query}\n```"
        f"{issues_ctx}"
        f"{exec_feedback}\n\n"
        f"Provide your complete analysis and optimized_query now."
    )


def call_model(client: OpenAI, obs) -> tuple:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return parse_action(resp.choices[0].message.content or ""), None
    except Exception as exc:
        return {
            "suggestions": [], "optimized_query": "", "approved": False,
            "summary": f"Model error: {exc}",
            "estimated_improvement": "unknown",
        }, str(exc)


# ── Main loop ─────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN not set.", flush=True)
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    local_env = SQLOptimEnv()
    results: Dict[str, Dict] = {}

    for task_id in TASK_IDS:
        obs = local_env.reset(task_id=task_id)
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        try:
            for step in range(1, obs.max_steps + 1):
                parsed, error = call_model(client, obs)

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

                # Pull execution info for the action summary
                exec_info = result.info.get("execution") or {}
                speedup = exec_info.get("speedup", 1.0)
                correct = exec_info.get("results_match", False)
                action_summary = (
                    f"suggestions={len(action.suggestions)},"
                    f"speedup={speedup:.2f}x,"
                    f"correct={str(correct).lower()}"
                )

                rewards.append(reward)
                steps_taken = step
                obs = result.observation

                log_step(step=step, action=action_summary,
                         reward=reward, done=done, error=error)

                if done:
                    break

            score = max(rewards) if rewards else 0.0
            success = score >= 0.5

        finally:
            log_end(success=success, steps=steps_taken,
                    score=score, rewards=rewards)

        results[task_id] = {
            "task_name":   obs.task_name,
            "final_score": round(score, 4),
            "steps_taken": steps_taken,
        }

    return results


if __name__ == "__main__":
    main()
