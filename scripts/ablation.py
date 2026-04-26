"""
Reward component ablation (no LLM, no API keys).

Runs the deterministic fallback action per task and recomputes the total
score with GradeMask variants to show how much each component contributes.

Usage:
  python scripts/ablation.py
  python scripts/ablation.py --quick    # single task (CI-friendly)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baseline_runner import FALLBACK_SOLUTIONS, TASK_IDS  # noqa: E402
from graders import GradeMask, grade  # noqa: E402
from models import Action  # noqa: E402
from tasks import TASKS  # noqa: E402


VARIANTS: dict[str, GradeMask] = {
    "full": GradeMask(),
    "no_execution_speedup": GradeMask(execution_speedup=False),
    "no_result_correctness": GradeMask(result_correctness=False),
    "no_duckdb_signal": GradeMask(
        execution_speedup=False, result_correctness=False
    ),
    "no_issue_detection": GradeMask(issue_detection=False),
    "no_approval": GradeMask(approval_correctness=False),
    "no_summary": GradeMask(summary_quality=False),
    "no_severity": GradeMask(severity_labels=False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only task_1 (faster for CI)",
    )
    args = ap.parse_args()
    task_ids = ["task_1_basic_antipatterns"] if args.quick else list(TASK_IDS)

    print("SQL-optim-env — reward component ablation (fallback actions)\n")

    for task_id in task_ids:
        td = TASKS[task_id]
        sol = FALLBACK_SOLUTIONS[task_id]
        action = Action(
            suggestions=sol["suggestions"],
            optimized_query=sol["optimized_query"],
            summary=sol["summary"],
            estimated_improvement=sol["estimated_improvement"],
            approved=sol["approved"],
        )
        full = grade(td, action, mask=None).score
        print(f"=== {task_id} ({td['difficulty']}) — full score {full:.4f} ===")
        for name, mask in VARIANTS.items():
            if name == "full":
                continue
            s = grade(td, action, mask=mask).score
            print(f"  {name:24s} score={s:.4f}  (Δ {s - full:+.4f})")
        print()

    acc: dict[str, list[float]] = defaultdict(list)
    for task_id in task_ids:
        td = TASKS[task_id]
        sol = FALLBACK_SOLUTIONS[task_id]
        action = Action(
            suggestions=sol["suggestions"],
            optimized_query=sol["optimized_query"],
            summary=sol["summary"],
            estimated_improvement=sol["estimated_improvement"],
            approved=sol["approved"],
        )
        for name, mask in VARIANTS.items():
            acc[name].append(grade(td, action, mask=mask).score)

    print("--- Mean score across all tasks ---")
    full_mean = sum(acc["full"]) / len(acc["full"])
    for name in VARIANTS:
        mean_v = sum(acc[name]) / len(acc[name])
        if name == "full":
            print(f"  {name:24s} {mean_v:.4f}")
        else:
            print(f"  {name:24s} {mean_v:.4f}  (Δ {mean_v - full_mean:+.4f} vs full)")


if __name__ == "__main__":
    main()
