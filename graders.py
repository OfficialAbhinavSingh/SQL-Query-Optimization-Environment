from typing import Dict, Any, List
from models import Action, Reward


def _keyword_match(text: str, keywords: List[str]) -> bool:
    """Check if any keyword appears in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _suggestions_text(action: Action) -> str:
    """Flatten all suggestion fields into one searchable string."""
    parts = [action.summary, action.optimized_query, action.estimated_improvement]
    for s in action.suggestions:
        parts.append(str(s.get("issue_type", "")))
        parts.append(str(s.get("description", "")))
        parts.append(str(s.get("fix", "")))
        parts.append(str(s.get("line", "")))
        parts.append(str(s.get("severity", "")))
    return " ".join(parts)


def grade(task_data: Dict[str, Any], action: Action) -> Reward:
    """
    Grade an agent's SQL optimization action against ground truth issues.

    Scoring breakdown:
      - Issue Detection:         60%  (did agent find the right problems?)
      - Optimized Query Quality: 15%  (did agent provide a meaningful rewrite?)
      - Approval Correctness:    10%  (correctly flagged as needing changes?)
      - Summary Quality:          8%  (is the summary thorough and informative?)
      - Improvement Estimate:     4%  (did agent quantify the expected gain?)
      - Severity Labels:          3%  (are severity levels present?)
    """
    ground_truth: List[Dict[str, Any]] = task_data["ground_truth_issues"]
    full_text = _suggestions_text(action)

    # ── 1. Issue Detection Score (0.0–0.60) ────────────────────────────
    detected = 0
    detection_feedback = []
    for gt_issue in ground_truth:
        found = _keyword_match(full_text, gt_issue["keywords"])
        if found:
            detected += 1
            detection_feedback.append(f"✅ Found: {gt_issue['type']} (line ~{gt_issue['line']})")
        else:
            detection_feedback.append(f"❌ Missed: {gt_issue['type']} (line ~{gt_issue['line']})")

    detection_score = (detected / len(ground_truth)) * 0.60

    # ── 2. Optimized Query Quality (0.0–0.15) ──────────────────────────
    query_score = 0.0
    oq = action.optimized_query.strip()
    if len(oq) > 50:
        query_score = 0.05
    if len(oq) > 150:
        query_score = 0.10
    # Bonus if the rewrite removes obvious anti-patterns found in original
    original_query = task_data["sql_query"].lower()
    if "select *" in original_query and "select *" not in oq.lower():
        query_score = min(query_score + 0.03, 0.15)
    if query_score < 0.15 and len(action.suggestions) > 0 and len(oq) > 100:
        query_score = min(query_score + 0.02, 0.15)
    query_score = min(query_score, 0.15)

    # ── 3. Approval Correctness (0.0–0.10) ─────────────────────────────
    expected_approved = task_data.get("approved_expected", False)
    approval_score = 0.10 if action.approved == expected_approved else 0.0

    # ── 4. Summary Quality (0.0–0.08) ──────────────────────────────────
    summary_score = 0.0
    if len(action.summary) > 40:
        summary_score = 0.04
    if len(action.summary) > 100:
        summary_score = 0.08

    # ── 5. Improvement Estimate Present (0.0–0.04) ─────────────────────
    improvement_keywords = ["x faster", "% less", "% faster", "% improvement", "times", "reduce", "improvement", "speedup"]
    has_estimate = _keyword_match(action.estimated_improvement, improvement_keywords) and len(action.estimated_improvement) > 5
    improvement_score = 0.04 if has_estimate else 0.0

    # ── 6. Severity Labels Present (0.0–0.03) ──────────────────────────
    severity_keywords = ["critical", "high", "medium", "low"]
    has_severity = any(
        _keyword_match(str(s.get("severity", "")), severity_keywords)
        for s in action.suggestions
    )
    severity_score = 0.03 if has_severity else 0.0

    # ── Final Score ─────────────────────────────────────────────────────
    total = (
        detection_score + query_score + approval_score +
        summary_score + improvement_score + severity_score
    )
    total = round(min(max(total, 0.0), 1.0), 4)

    # Minimum signal for any submission
    if total == 0.0 and len(action.suggestions) > 0:
        total = 0.02

    breakdown = {
        "issue_detection":       round(detection_score, 4),
        "optimized_query":       round(query_score, 4),
        "approval_correctness":  round(approval_score, 4),
        "summary_quality":       round(summary_score, 4),
        "improvement_estimate":  round(improvement_score, 4),
        "severity_labels":       round(severity_score, 4),
    }

    n_suggestions = len(action.suggestions)
    expected_n = len(ground_truth)

    feedback_lines = detection_feedback + [
        f"\nSuggestions submitted: {n_suggestions} (expected ~{expected_n})",
        f"Optimized query length: {len(oq)} chars",
        f"Approval correctness: {'✅' if action.approved == expected_approved else '❌'} "
        f"(you said {'approved' if action.approved else 'needs changes'}, "
        f"expected {'approved' if expected_approved else 'needs changes'})",
        f"Total score: {total:.4f}",
    ]

    return Reward(
        score=total,
        breakdown=breakdown,
        feedback="\n".join(feedback_lines)
    )
