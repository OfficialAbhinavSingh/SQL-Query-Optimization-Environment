"""
Before / after evaluation for README and judges.

"Before" = same structured suggestions as the fallback policy but an empty
optimized_query (no DuckDB comparison — analysis-only).

"After"  = full deterministic fallback with real optimized SQL.

No API keys required.

Usage:
  python training/eval_before_after.py --save-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baseline_runner import FALLBACK_SOLUTIONS, TASK_IDS  # noqa: E402
from graders import grade  # noqa: E402
from models import Action  # noqa: E402
from tasks import TASKS  # noqa: E402


def _before_action(task_id: str) -> Action:
    sol = FALLBACK_SOLUTIONS[task_id]
    return Action(
        suggestions=sol["suggestions"],
        optimized_query="",
        summary=sol["summary"],
        estimated_improvement=sol["estimated_improvement"],
        approved=sol["approved"],
    )


def _after_action(task_id: str) -> Action:
    sol = FALLBACK_SOLUTIONS[task_id]
    return Action(
        suggestions=sol["suggestions"],
        optimized_query=sol["optimized_query"],
        summary=sol["summary"],
        estimated_improvement=sol["estimated_improvement"],
        approved=sol["approved"],
    )


def run_eval() -> dict:
    rows = []
    for task_id in TASK_IDS:
        td = TASKS[task_id]
        b = grade(td, _before_action(task_id))
        a = grade(td, _after_action(task_id))
        rows.append(
            {
                "task_id": task_id,
                "task_name": td["task_name"],
                "difficulty": td["difficulty"],
                "before_score": b.score,
                "after_score": a.score,
                "delta": round(a.score - b.score, 4),
            }
        )
    return {"rows": rows}


def write_table(path: str, data: dict) -> None:
    lines = [
        "# Before / after — execution-grounded reward",
        "",
        "| Task | Difficulty | Before (no SQL) | After (fallback) | Δ |",
        "|------|------------|-----------------|------------------|---|",
    ]
    for r in data["rows"]:
        lines.append(
            f"| {r['task_name'][:40]} | {r['difficulty']} | "
            f"{r['before_score']:.4f} | {r['after_score']:.4f} | {r['delta']:+.4f} |"
        )
    b_avg = sum(r["before_score"] for r in data["rows"]) / len(data["rows"])
    a_avg = sum(r["after_score"] for r in data["rows"]) / len(data["rows"])
    lines += [
        "",
        f"**Mean before:** {b_avg:.4f}  ",
        f"**Mean after:** {a_avg:.4f}  ",
        f"**Mean Δ:** {a_avg - b_avg:+.4f}",
        "",
        "_Before = non-empty suggestions but `optimized_query` empty — no speedup/correctness signal._",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_chart(path: str, data: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping chart", flush=True)
        return

    labels = [r["task_id"].replace("task_", "") for r in data["rows"]]
    before = [r["before_score"] for r in data["rows"]]
    after = [r["after_score"] for r in data["rows"]]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - w / 2 for i in x], before, width=w, label="Before (no optimized SQL)")
    ax.bar([i + w / 2 for i in x], after, width=w, label="After (fallback + DuckDB)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Reward")
    ax.legend()
    ax.set_title("Reward spread: analysis-only vs execution-grounded")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Chart → {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--save-dir",
        default="results",
        help="Directory for before_after_table.md and JSON",
    )
    args = ap.parse_args()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    data = run_eval()
    json_path = os.path.join(save_dir, "before_after_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    md_path = os.path.join(save_dir, "before_after_table.md")
    write_table(md_path, data)
    png_path = os.path.join(save_dir, "before_after_chart.png")
    write_chart(png_path, data)

    print(f"[OK] {json_path}", flush=True)
    print(f"[OK] {md_path}", flush=True)


if __name__ == "__main__":
    main()
