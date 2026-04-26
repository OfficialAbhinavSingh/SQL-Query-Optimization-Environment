"""
Export a self-contained offline replay: JSON + HTML with embedded run data.

Uses the deterministic fallback one step per task (five scrubber steps).

Usage:
  python scripts/export_replay.py

Writes:
  runs/demo_fallback/replay.json
  runs/demo_fallback/replay.html
"""

from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baseline_runner import FALLBACK_SOLUTIONS, TASK_IDS  # noqa: E402
from env import SQLOptimEnv  # noqa: E402
from models import Action  # noqa: E402


def _build_payload() -> dict:
    env = SQLOptimEnv()
    steps: list[dict] = []
    for i, task_id in enumerate(TASK_IDS):
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
        ex = result.info.get("execution") or {}
        steps.append(
            {
                "index": i,
                "task_id": task_id,
                "task_name": obs.task_name,
                "difficulty": obs.difficulty,
                "reward": round(result.reward.score, 4),
                "breakdown": dict(result.reward.breakdown),
                "original_sql": obs.sql_query,
                "optimized_sql": action.optimized_query,
                "last_execution": ex,
            }
        )

    run_id = datetime.now(timezone.utc).strftime("demo_fallback_%Y%m%dT%H%M%SZ")
    return {
        "run_id": run_id,
        "environment": "sql-optim-env",
        "policy": "deterministic_fallback",
        "steps": steps,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SQL Optim Env — Episode replay</title>
  <style>
    :root {{
      --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --acc:#58a6ff; --bd:#30363d;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
    header {{ padding:16px 20px; border-bottom:1px solid var(--bd); display:flex; align-items:center; gap:16px; flex-wrap:wrap; }}
    h1 {{ font-size:1.1rem; margin:0; }}
    .controls {{ display:flex; align-items:center; gap:8px; margin-left:auto; }}
    button {{ background:#21262d; color:var(--fg); border:1px solid var(--bd); padding:6px 12px; border-radius:6px; cursor:pointer; }}
    button:hover {{ background:#30363d; }}
    .meta {{ font-size:0.8rem; color:var(--muted); }}
    main {{ padding:20px; max-width:1100px; margin:0 auto; }}
    pre {{ background:#161b22; border:1px solid var(--bd); border-radius:8px; padding:12px; overflow:auto; font-size:12px; line-height:1.45; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    @media (max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#1f3a5f; font-size:0.75rem; }}
    h2 {{ font-size:0.95rem; margin:16px 0 8px; color:var(--acc); }}
  </style>
</head>
<body>
  <header>
    <h1>SQL Query Optimization — replay</h1>
    <span class="pill" id="runLabel"></span>
    <div class="controls">
      <button type="button" id="prev">Prev</button>
      <button type="button" id="next">Next</button>
      <span class="meta" id="stepIdx"></span>
    </div>
  </header>
  <main>
    <p class="meta" id="taskTitle"></p>
    <h2>Reward</h2>
    <pre id="reward"></pre>
    <h2>Last execution (DuckDB)</h2>
    <pre id="exec"></pre>
    <div class="grid">
      <div>
        <h2>Original SQL</h2>
        <pre id="orig"></pre>
      </div>
      <div>
        <h2>Optimized SQL</h2>
        <pre id="opt"></pre>
      </div>
    </div>
  </main>
  <script>
  const DATA = JSON.parse(atob("{b64}"));
  const steps = DATA.steps || [];
  let cur = 0;
  function render() {{
    const s = steps[cur];
    if (!s) return;
    document.getElementById("runLabel").textContent = DATA.run_id || "run";
    document.getElementById("stepIdx").textContent = "Step " + (cur + 1) + " / " + steps.length;
    document.getElementById("taskTitle").textContent = s.task_name + " · " + s.difficulty + " · " + s.task_id;
    document.getElementById("reward").textContent = JSON.stringify({{ reward: s.reward, breakdown: s.breakdown }}, null, 2);
    document.getElementById("exec").textContent = JSON.stringify(s.last_execution || {{}}, null, 2);
    document.getElementById("orig").textContent = s.original_sql || "";
    document.getElementById("opt").textContent = s.optimized_sql || "";
  }}
  document.getElementById("prev").onclick = () => {{ cur = (cur - 1 + steps.length) % steps.length; render(); }};
  document.getElementById("next").onclick = () => {{ cur = (cur + 1) % steps.length; render(); }};
  render();
  </script>
</body>
</html>
"""


def main() -> None:
    out_dir = os.path.join(ROOT, "runs", "demo_fallback")
    os.makedirs(out_dir, exist_ok=True)
    payload = _build_payload()
    json_path = os.path.join(out_dir, "replay.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    html = HTML_TEMPLATE.format(b64=b64)
    html_path = os.path.join(out_dir, "replay.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
