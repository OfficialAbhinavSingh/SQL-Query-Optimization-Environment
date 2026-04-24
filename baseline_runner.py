"""
baseline_runner.py - Generate Real Baseline Comparison Results
===============================================================
Runs two policies against all 5 tasks and prints a clean comparison table:
  1. Fallback policy: deterministic rule-based (no LLM required)
  2. LLM policy: uses Qwen2.5-72B via HF Inference Router

Run:
  # Fallback only (no API key needed):
  python baseline_runner.py

  # With LLM comparison:
  HF_TOKEN=hf_xxx python baseline_runner.py
  MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python baseline_runner.py

Results are saved to results/baseline_results.json and printed as a table.
"""

import io
import sys
# Fix Windows console encoding so non-ASCII results don't crash
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from env import SQLOptimEnv
from models import Action
from tasks import TASKS

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_BASE   = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")

TASK_IDS = list(TASKS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Fallback policy: deterministic, hand-crafted, no LLM
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_SOLUTIONS: Dict[str, Dict[str, Any]] = {
    "task_1_basic_antipatterns": {
        "suggestions": [
            {"issue_type": "select_star", "line": 1,
             "description": "SELECT * fetches all columns from 500k rows — use explicit projection.",
             "severity": "high", "fix": "SELECT id, customer_id, status, total, created_at"},
            {"issue_type": "non_sargable_cast", "line": 3,
             "description": "CAST(customer_id AS VARCHAR) prevents integer comparison and pruning.",
             "severity": "critical", "fix": "WHERE customer_id = 5000"},
            {"issue_type": "function_on_date_column", "line": 4,
             "description": "YEAR() on date column forces full scan; use a date range instead.",
             "severity": "high", "fix": "created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"},
        ],
        "optimized_query": (
            "SELECT id, customer_id, status, total, created_at\n"
            "FROM orders\n"
            "WHERE customer_id = 5000\n"
            "  AND created_at >= DATE '2024-01-01'\n"
            "  AND created_at < DATE '2025-01-01';"
        ),
        "summary": (
            "Three anti-patterns: SELECT * over 500k rows wastes bandwidth, "
            "CAST on customer_id prevents pruning, YEAR() forces full date scan. "
            "Explicit column projection + integer comparison + date range fix all three."
        ),
        "estimated_improvement": "3-5x faster — eliminates type-cast and function penalties",
        "approved": False,
    },

    "task_2_correlated_subqueries": {
        "suggestions": [
            {"issue_type": "correlated_subquery_count", "line": 4,
             "description": "Correlated COUNT subquery scans 500k orders per premium user (N+1 pattern).",
             "severity": "critical", "fix": "LEFT JOIN with GROUP BY aggregation"},
            {"issue_type": "correlated_subquery_sum", "line": 7,
             "description": "Correlated SUM subquery -- another full scan per user.",
             "severity": "critical", "fix": "Include in the same LEFT JOIN aggregation"},
            {"issue_type": "correlated_subquery_limit", "line": 11,
             "description": "Correlated ORDER BY LIMIT 1 -- sorted scan per user.",
             "severity": "high", "fix": "Use ROW_NUMBER() window function in a CTE"},
            {"issue_type": "missing_aggregation_join", "line": 16,
             "description": "Single aggregation JOIN replaces all three subqueries in one pass.",
             "severity": "critical", "fix": "LEFT JOIN aggregated subquery ON u.id = agg.customer_id"},
        ],
        "optimized_query": (
            "WITH agg AS (\n"
            "    SELECT\n"
            "        customer_id,\n"
            "        COUNT(*) FILTER (WHERE status = 'completed')              AS completed_orders,\n"
            "        SUM(total) FILTER (WHERE created_at >= DATE '2024-01-01') AS ytd_spend\n"
            "    FROM orders\n"
            "    GROUP BY customer_id\n"
            "),\n"
            "last_order AS (\n"
            "    SELECT customer_id, total AS last_order_amount\n"
            "    FROM (\n"
            "        SELECT customer_id, total,\n"
            "               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) AS rn\n"
            "        FROM orders\n"
            "    ) t WHERE rn = 1\n"
            ")\n"
            "SELECT\n"
            "    u.email,\n"
            "    u.region,\n"
            "    COALESCE(a.completed_orders, 0) AS completed_orders,\n"
            "    a.ytd_spend,\n"
            "    l.last_order_amount\n"
            "FROM users u\n"
            "LEFT JOIN agg a ON u.id = a.customer_id\n"
            "LEFT JOIN last_order l ON u.id = l.customer_id\n"
            "WHERE u.tier = 'premium';"
        ),
        "summary": (
            "Three correlated subqueries each scan 500k orders per premium user (~3300 users). "
            "Worst case: 3 × 3300 × 500k = 5B row reads. "
            "A single CTE with GROUP BY + FILTER aggregates everything in one pass over orders."
        ),
        "estimated_improvement": "10-20x faster — eliminates N+1 pattern with single JOIN",
        "approved": False,
    },

    "task_3_wildcard_scan": {
        "suggestions": [
            {"issue_type": "leading_wildcard_like", "line": 6,
             "description": "LIKE '%purchase%' disables zone-map pruning — forces full 1M row scan.",
             "severity": "critical", "fix": "Use exact equality: event_type = 'purchase'"},
            {"issue_type": "or_expands_to_full_scan", "line": 7,
             "description": "OR with wildcard LIKE '%buy%' is redundant — no 'buy' events exist in schema.",
             "severity": "high", "fix": "Remove OR clause; use IN ('purchase', ...) for multi-type"},
            {"issue_type": "select_star_large_table", "line": 2,
             "description": "SELECT * on 1M rows transfers all columns unnecessarily.",
             "severity": "high", "fix": "SELECT id, user_id, event_type, occurred_at"},
            {"issue_type": "pre_filter_computed_columns", "line": 3,
             "description": "Derived columns (CAST, UPPER) computed on all 1M rows before WHERE.",
             "severity": "medium", "fix": "Apply WHERE first via CTE, then compute derived columns"},
        ],
        "optimized_query": (
            "WITH filtered AS (\n"
            "    SELECT id, user_id, session_id, event_type, occurred_at\n"
            "    FROM events\n"
            "    WHERE event_type = 'purchase'\n"
            ")\n"
            "SELECT\n"
            "    id, user_id, session_id, event_type, occurred_at,\n"
            "    CAST(id AS VARCHAR) || '_' || event_type AS event_key,\n"
            "    UPPER(event_type) AS event_type_upper\n"
            "FROM filtered;"
        ),
        "summary": (
            "Leading-wildcard LIKE patterns on 1M events force full column scans. "
            "Exact equality (event_type = 'purchase') enables zone-map pruning, "
            "reducing the dataset to ~167k rows before computed columns are evaluated."
        ),
        "estimated_improvement": "4-8x faster — exact match + filter pushdown on 1M rows",
        "approved": False,
    },

    "task_4_implicit_join": {
        "suggestions": [
            {"issue_type": "implicit_cross_join", "line": 8,
             "description": "Comma-syntax FROM (implicit join) risks Cartesian product if WHERE fails.",
             "severity": "critical", "fix": "Use explicit INNER JOIN ... ON syntax"},
            {"issue_type": "repeated_scalar_subquery_avg", "line": 6,
             "description": "Scalar subquery AVG(total) re-scans all 500k orders once per GROUP BY group.",
             "severity": "high", "fix": "Pre-compute in a CTE and cross-join the scalar value"},
            {"issue_type": "repeated_scalar_subquery_max", "line": 7,
             "description": "Scalar subquery MAX(total) WHERE status='completed' — same issue.",
             "severity": "high", "fix": "Include in the same pre-compute CTE"},
            {"issue_type": "missing_explicit_join", "line": 8,
             "description": "Rewrite with explicit INNER JOIN for clarity and safety.",
             "severity": "medium", "fix": "FROM users u INNER JOIN orders o ON u.id = o.customer_id"},
        ],
        "optimized_query": (
            "WITH global_stats AS (\n"
            "    SELECT\n"
            "        AVG(total)                                          AS global_avg,\n"
            "        MAX(total) FILTER (WHERE status = 'completed')      AS max_deal\n"
            "    FROM orders\n"
            ")\n"
            "SELECT\n"
            "    u.region,\n"
            "    u.plan,\n"
            "    COUNT(*)       AS total_orders,\n"
            "    SUM(o.total)   AS revenue,\n"
            "    gs.global_avg,\n"
            "    gs.max_deal\n"
            "FROM users u\n"
            "INNER JOIN orders o ON u.id = o.customer_id\n"
            "CROSS JOIN global_stats gs\n"
            "WHERE o.status IN ('completed', 'shipped')\n"
            "GROUP BY u.region, u.plan, gs.global_avg, gs.max_deal;"
        ),
        "summary": (
            "Comma-syntax implicit join is an anti-pattern that risks Cartesian products. "
            "Two scalar subqueries re-scan 500k orders per GROUP BY group. "
            "A CTE computes global stats exactly once; explicit INNER JOIN ensures correctness."
        ),
        "estimated_improvement": "8-15x faster — CTE eliminates repeated subquery scans",
        "approved": False,
    },

    "task_5_window_functions": {
        "suggestions": [
            {"issue_type": "no_pre_filter", "line": 11,
             "description": "No WHERE clause — all 5 window functions run over all 1M rows.",
             "severity": "critical", "fix": "Filter to relevant event_types in a CTE first"},
            {"issue_type": "global_rank_no_partition", "line": 8,
             "description": "RANK() OVER (ORDER BY occurred_at) sorts all 1M rows globally — most expensive op.",
             "severity": "critical", "fix": "Remove or replace with per-user ROW_NUMBER()"},
            {"issue_type": "redundant_window_functions", "line": 5,
             "description": "5 window functions with overlapping PARTITION BY — multiple passes.",
             "severity": "high", "fix": "Consolidate into fewer OVER() clauses where possible"},
            {"issue_type": "count_vs_conditional_sum", "line": 9,
             "description": "SUM(CASE WHEN event_type='purchase') can be COUNT(*) FILTER (WHERE ...).",
             "severity": "medium", "fix": "COUNT(*) FILTER (WHERE event_type = 'purchase') OVER (PARTITION BY user_id)"},
            {"issue_type": "select_all_unfiltered", "line": 1,
             "description": "Selecting all columns from 1M rows — project only needed columns.",
             "severity": "medium", "fix": "SELECT user_id, event_type, occurred_at + window columns"},
        ],
        "optimized_query": (
            "WITH base AS (\n"
            "    SELECT user_id, event_type, occurred_at\n"
            "    FROM events\n"
            "    WHERE event_type IN ('purchase', 'view', 'click')\n"
            ")\n"
            "SELECT\n"
            "    user_id,\n"
            "    event_type,\n"
            "    occurred_at,\n"
            "    COUNT(*) OVER (PARTITION BY user_id)                                    AS total_user_events,\n"
            "    COUNT(*) OVER (PARTITION BY user_id, event_type)                        AS type_count,\n"
            "    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY occurred_at DESC)      AS recency_rank,\n"
            "    COUNT(*) FILTER (WHERE event_type = 'purchase')\n"
            "        OVER (PARTITION BY user_id)                                          AS user_purchases\n"
            "FROM base;"
        ),
        "summary": (
            "Five window functions over 1M unfiltered rows causes 5 full sort/hash passes. "
            "The global RANK() sorts all 1M rows globally — the single most expensive operation. "
            "Pre-filtering to relevant event types in a CTE reduces the dataset to ~500k rows "
            "and removing the global RANK eliminates the costliest sort."
        ),
        "estimated_improvement": "5-12x faster — filter-first + remove global RANK()",
        "approved": False,
    },
}


def run_fallback_policy(env: SQLOptimEnv) -> Dict[str, Dict]:
    """Run deterministic fallback policy against all tasks."""
    results = {}
    for task_id in TASK_IDS:
        obs = env.reset(task_id=task_id)
        sol = FALLBACK_SOLUTIONS[task_id]
        action = Action(
            suggestions=sol["suggestions"],
            optimized_query=sol["optimized_query"],
            summary=sol["summary"],
            estimated_improvement=sol["estimated_improvement"],
            approved=sol["approved"],
        )
        result = env.step(action)
        exec_info = result.info.get("execution") or {}
        results[task_id] = {
            "task_name":    obs.task_name,
            "difficulty":   obs.difficulty,
            "score":        round(result.reward.score, 4),
            "speedup":      round(exec_info.get("speedup", 1.0), 2),
            "correct":      exec_info.get("results_match", False),
            "steps":        1,
            "policy":       "fallback",
        }
        print(
            f"  [Fallback] {obs.difficulty:12s} | "
            f"score={result.reward.score:.4f} | "
            f"speedup={exec_info.get('speedup', 1.0):.2f}x | "
            f"correct={exec_info.get('results_match', False)}",
            flush=True,
        )
    return results


def run_llm_policy(env: SQLOptimEnv) -> Optional[Dict[str, Dict]]:
    """Run LLM policy if HF_TOKEN is set."""
    if not HF_TOKEN:
        print("  [LLM] HF_TOKEN not set — skipping LLM baseline.", flush=True)
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("  [LLM] openai package not installed — skipping.", flush=True)
        return None

    from inference import SYSTEM_PROMPT, build_user_prompt, parse_action

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE)
    results = {}

    for task_id in TASK_IDS:
        obs = env.reset(task_id=task_id)
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs)},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            parsed = parse_action(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"  [LLM] Call failed for {task_id}: {e}", flush=True)
            parsed = FALLBACK_SOLUTIONS[task_id]

        action = Action(
            suggestions=parsed.get("suggestions", []),
            optimized_query=parsed.get("optimized_query", ""),
            summary=parsed.get("summary", ""),
            estimated_improvement=parsed.get("estimated_improvement", ""),
            approved=parsed.get("approved", False),
        )

        env.reset(task_id=task_id)
        result = env.step(action)
        exec_info = result.info.get("execution") or {}
        results[task_id] = {
            "task_name":    obs.task_name,
            "difficulty":   obs.difficulty,
            "score":        round(result.reward.score, 4),
            "speedup":      round(exec_info.get("speedup", 1.0), 2),
            "correct":      exec_info.get("results_match", False),
            "steps":        1,
            "policy":       f"llm:{MODEL_NAME}",
        }
        print(
            f"  [LLM] {obs.difficulty:12s} | "
            f"score={result.reward.score:.4f} | "
            f"speedup={exec_info.get('speedup', 1.0):.2f}x | "
            f"correct={exec_info.get('results_match', False)}",
            flush=True,
        )
    return results


def print_comparison_table(
    fallback: Dict[str, Dict],
    llm: Optional[Dict[str, Dict]],
):
    print("\n" + "=" * 80)
    print("  BASELINE RESULTS — SQL Query Optimization Environment")
    print("=" * 80)

    header = f"{'Task':<40} {'Diff':<12} {'F-Score':>8} {'F-Spdup':>8} {'F-Corr':>7}"
    if llm:
        header += f" {'L-Score':>8} {'L-Spdup':>8} {'L-Corr':>7} {'Delta':>7}"
    print(header)
    print("-" * 80)

    for task_id in TASK_IDS:
        fb = fallback[task_id]
        row = (
            f"{fb['task_name'][:38]:<40} "
            f"{fb['difficulty']:<12} "
            f"{fb['score']:>8.4f} "
            f"{fb['speedup']:>7.2f}x "
            f"{'YES' if fb['correct'] else 'NO':>7}"
        )
        if llm and task_id in llm:
            lm = llm[task_id]
            delta = lm["score"] - fb["score"]
            row += (
                f" {lm['score']:>8.4f} "
                f"{lm['speedup']:>7.2f}x "
                f"{'YES' if lm['correct'] else 'NO':>7} "
                f"{'+' if delta >= 0 else ''}{delta:>6.4f}"
            )
        print(row)

    print("=" * 80)
    fb_avg = sum(r["score"] for r in fallback.values()) / len(fallback)
    print(f"  Fallback avg score : {fb_avg:.4f}")
    if llm:
        lm_avg = sum(r["score"] for r in llm.values()) / len(llm)
        print(f"  LLM avg score      : {lm_avg:.4f}  (+{lm_avg - fb_avg:.4f} vs fallback)")
    print("=" * 80 + "\n")


def main():
    print("\n[SQLOptimEnv] Baseline Runner", flush=True)
    print("Initialising DuckDB environment (warm-up ~3s) ...\n", flush=True)
    env = SQLOptimEnv()

    print("[1/2] Running fallback (deterministic) policy ...", flush=True)
    fallback_results = run_fallback_policy(env)

    print("\n[2/2] Running LLM policy ...", flush=True)
    llm_results = run_llm_policy(env)

    print_comparison_table(fallback_results, llm_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fallback":  fallback_results,
        "llm":       llm_results,
    }
    out_path = "results/baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] Results written to {out_path}", flush=True)
    return output


if __name__ == "__main__":
    main()
