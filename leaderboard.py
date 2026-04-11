"""
leaderboard.py — In-Memory Best-Score Tracker
Tracks every execution attempt across all tasks so the /leaderboard
endpoint can display real-time standings.
"""
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

_board: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


def record(
    task_id: str,
    speedup: float,
    score: float,
    results_match: bool,
    steps: int,
) -> None:
    _board[task_id].append(
        {
            "speedup":       round(speedup, 3),
            "score":         round(score, 4),
            "results_match": results_match,
            "steps":         steps,
            "ts":            datetime.now(timezone.utc).isoformat(),
        }
    )


def get_board() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for task_id, entries in _board.items():
        if not entries:
            continue
        best = max(entries, key=lambda e: e["score"])
        valid = [e for e in entries if e["results_match"]]
        fastest = max(valid, key=lambda e: e["speedup"]) if valid else None

        out[task_id] = {
            "best_score":       best["score"],
            "best_speedup":     fastest["speedup"] if fastest else 0.0,
            "total_attempts":   len(entries),
            "correct_attempts": len(valid),
            "success_rate":     round(len(valid) / len(entries), 3),
            "best_attempt_at":  best["ts"],
        }
    return out
