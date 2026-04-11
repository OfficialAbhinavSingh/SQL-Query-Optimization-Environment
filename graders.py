"""
graders.py — Execution-Grounded Reward Function
=================================================
What makes this environment unique: reward is computed from REAL
DuckDB execution results, not just keyword heuristics.

Scoring breakdown (sums to 1.0):
  Real Execution Speedup    35%   — actual timing ratio from DuckDB
  Result Correctness        20%   — both queries return identical data?
  Issue Detection           25%   — keyword match vs ground truth
  Approval Correctness       8%   — correctly flags query as bad?
  Summary Quality            7%   — is the written analysis thorough?
  Severity Labels            5%   — are severity values present?
"""

from typing import Any, Dict, List, Optional

from executor import get_executor
from models import Action, Reward


# ── Helpers ──────────────────────────────────────────────────────────────

def _kw_match(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _suggestions_text(action: Action) -> str:
    parts = [action.summary, action.optimized_query, action.estimated_improvement]
    for s in action.suggestions:
        parts += [
            str(s.get("issue_type", "")),
            str(s.get("description", "")),
            str(s.get("fix", "")),
            str(s.get("severity", "")),
        ]
    return " ".join(parts)


# ── Speedup → score mapping ───────────────────────────────────────────────

def _speedup_score(speedup: float, has_error: bool) -> float:
    """Map real speedup ratio to a score in [0.0, 0.35]."""
    if has_error:
        return 0.0
    if speedup >= 15.0:
        return 0.35
    if speedup >= 8.0:
        return 0.30
    if speedup >= 4.0:
        return 0.25
    if speedup >= 2.0:
        return 0.18
    if speedup >= 1.2:
        return 0.10
    if speedup >= 0.9:          # slightly slower — acceptable
        return 0.04
    return 0.0                  # regression


# ── Main grader ───────────────────────────────────────────────────────────

def grade(task_data: Dict[str, Any], action: Action) -> Reward:
    original_query: str = task_data["sql_query"]
    optimized_query: str = (action.optimized_query or "").strip()
    ground_truth: List[Dict[str, Any]] = task_data["ground_truth_issues"]
    full_text = _suggestions_text(action)

    # ── 1. Real Execution (0.0–0.35) ─────────────────────────────────
    exec_info: Dict[str, Any] = {}
    speedup_sc = 0.0
    correctness_sc = 0.0
    exec_feedback: List[str] = []

    if optimized_query:
        try:
            ex = get_executor()
            exec_info = ex.compare(original_query, optimized_query)
            speedup    = exec_info.get("speedup", 1.0)
            r_match    = exec_info.get("results_match", False)
            opt_err    = exec_info.get("optimized_error")

            # 1a. Speedup score
            speedup_sc = _speedup_score(speedup, bool(opt_err))

            # 1b. Correctness score (0.0-0.20)
            if opt_err:
                correctness_sc = 0.0
            elif r_match:
                correctness_sc = 0.20
            elif exec_info.get("optimized_rows", 0) > 0:
                # Query ran but different results -- partial credit
                correctness_sc = 0.05

            # Feedback lines
            exec_feedback = [
                "\n[DuckDB Execution Results]",
                f"   Original  : {exec_info['original_ms']:.1f} ms "
                f"({exec_info['original_rows']} rows)",
                f"   Optimized : {exec_info['optimized_ms']:.1f} ms "
                f"({exec_info['optimized_rows']} rows)",
                f"   Speedup   : {speedup:.2f}x",
                f"   Correct?  : {'YES' if r_match else 'NO -- results differ'}",
                f"   Verdict   : {exec_info.get('verdict', '')}",
            ]
            if opt_err:
                exec_feedback.append(f"   SQL Error : {opt_err[:200]}")

        except Exception as exc:
            exec_feedback = [f"\n[WARN] Execution engine error: {exc}"]

    # ── 2. Issue Detection (0.0–0.25) ────────────────────────────────
    detected = 0
    detection_fb: List[str] = ["\n[Issue Detection]"]
    for gt in ground_truth:
        found = _kw_match(full_text, gt["keywords"])
        if found:
            detected += 1
            detection_fb.append(f"   [FOUND] {gt['type']} (line ~{gt['line']})")
        else:
            detection_fb.append(f"   [MISS ] {gt['type']} (line ~{gt['line']})")
    detection_sc = (detected / len(ground_truth)) * 0.25 if ground_truth else 0.0

    # ── 3. Approval Correctness (0.0–0.08) ───────────────────────────
    expected_approved = task_data.get("approved_expected", False)
    approval_sc = 0.08 if action.approved == expected_approved else 0.0

    # ── 4. Summary Quality (0.0–0.07) ────────────────────────────────
    summary_sc = 0.0
    slen = len(action.summary)
    if slen > 50:
        summary_sc = 0.03
    if slen > 120:
        summary_sc = 0.07

    # ── 5. Severity Labels (0.0–0.05) ────────────────────────────────
    sev_kw = ["critical", "high", "medium", "low"]
    has_sev = any(
        _kw_match(str(s.get("severity", "")), sev_kw) for s in action.suggestions
    )
    severity_sc = 0.05 if has_sev else 0.0

    # ── Total ─────────────────────────────────────────────────────────
    total = min(
        max(speedup_sc + correctness_sc + detection_sc +
            approval_sc + summary_sc + severity_sc, 0.0),
        1.0,
    )
    total = round(total, 4)
    if total == 0.0 and action.suggestions:
        total = 0.02          # minimum signal for any submission

    breakdown = {
        "execution_speedup":    round(speedup_sc, 4),
        "result_correctness":   round(correctness_sc, 4),
        "issue_detection":      round(detection_sc, 4),
        "approval_correctness": round(approval_sc, 4),
        "summary_quality":      round(summary_sc, 4),
        "severity_labels":      round(severity_sc, 4),
    }

    feedback = "\n".join(
        exec_feedback
        + detection_fb
        + [
            f"\n   Suggestions submitted: {len(action.suggestions)} "
            f"(expected ~{len(ground_truth)})",
            f"   Approval: {'✅' if action.approved == expected_approved else '❌'} "
            f"(got {'approved' if action.approved else 'rejected'}, "
            f"expected {'approved' if expected_approved else 'rejected'})",
            f"\n🏆 Total score: {total:.4f}",
        ]
    )

    return Reward(score=total, breakdown=breakdown, feedback=feedback)
