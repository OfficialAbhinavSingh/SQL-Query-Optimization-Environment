# Design: execution-grounded reward

## Goal

Train or evaluate LLMs on **SQL optimization** using feedback that reflects **what actually happens** when a rewritten query runs on realistic data—not lexical overlap with a rubric alone.

## Why this reward is hard to game

1. **Speedup (35%)** comes from median wall-clock over multiple DuckDB runs of both the original and the candidate rewrite. You cannot claim a 10× improvement without the engine measuring roughly that ratio on this dataset.
2. **Correctness (20%)** uses sorted row comparison for modest result sets, and a **order-independent checksum** (or count fallback) for large sets so parallel / non-deterministic ordering does not false-negative legitimate rewrites.
3. **Issue detection (25%)** still uses keyword overlap against declared ground-truth issue types. That piece *is* gameable in isolation—which is why it is capped and combined with execution signals. A model that only “talks” about fixes without a faster, correct query **cannot** max the score.

Together, “fast + wrong” loses the correctness mass; “verbose + slow” loses the speedup mass; “keywords only + empty SQL” loses both execution components.

## Observation loop

Each `step` returns an `Observation` that may include `last_execution` from the **previous** graded action (timing, speedup, `results_match`, verdict). The grader **always** re-executes when scoring a new action; `last_execution` is for agent iteration and demos, not a cached substitute for grading. Stripping `last_execution` from the prompt is an **observation-space** ablation for the LLM only; it does not change `grade()` (see [`scripts/ablation.py`](../scripts/ablation.py) for reward-component ablations).

## Edge cases and limitations

| Topic | Behavior |
|--------|----------|
| **Timeouts / huge latency** | Failed execution or extreme median times yield low or zero speedup credit; errors surface in feedback. |
| **Semantic rewrites that change row counts** | If the optimized query returns different rows, correctness is partial or zero even if the SQL is “clever.” Some tasks intentionally trade strict row identity for performance; the grader reflects that honestly. |
| **DuckDB version differences** | Executor prefers portable checksum patterns and falls back to count-only if needed. |
| **Single-agent design** | There is no second “oversight” LLM in the environment contract. The **database** is the ground-truth critic. Adding a critic model would be analysis-only unless the action space changes. |
| **Keyword detection** | Known limitation: suggestions should align with `tasks.py` ground-truth keywords for full detection credit. |

## Threat model (reward hacking)

- **Copying the original query as “optimized”** → speedup ≈ 1×, low speedup score; may still get issue/summary points.
- **Returning empty `optimized_query`** → no execution credit; very low total.
- **Wrong but fast query** → `results_match` false → at most partial correctness, capped total.

For systematic sensitivity analysis, run [`scripts/ablation.py`](../scripts/ablation.py) with component masks (see [`graders.py`](../graders.py) `GradeMask`).
