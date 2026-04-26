# Where to look (judge / reviewer map)

Quick navigation for the **SQL Query Optimization Environment** (`sql-optim-env`).

| Layer | File | Role |
|--------|------|------|
| Task definitions | [`tasks.py`](tasks.py) | Five scenarios, SQL text, ground-truth issue keywords, `max_steps` |
| DuckDB engine | [`executor.py`](executor.py) | In-memory tables (users/orders/products/events), timing, checksum / row equality |
| Reward | [`graders.py`](graders.py) | Execution speedup + correctness + issue detection + structure; optional [`GradeMask`](graders.py) for ablations |
| Episode loop | [`env.py`](env.py) | `SQLOptimEnv.reset` / `step`, accumulates `last_execution` in observations |
| API | [`server/app.py`](server/app.py) | FastAPI OpenEnv endpoints + `/execute` + `/leaderboard` |
| Models | [`models.py`](models.py) | Pydantic `Observation`, `Action`, `Reward` |
| LLM driver | [`inference.py`](inference.py) | `[START]`/`[STEP]`/`[END]` stdout; HF Router client |
| Baselines | [`baseline_runner.py`](baseline_runner.py) | Deterministic fallback vs optional LLM; writes [`results/baseline_results.json`](results/baseline_results.json) |
| Training | [`train.py`](train.py) | GRPO-style loop on real env rewards |
| Design / results / training docs | [`docs/design.md`](docs/design.md), [`docs/results.md`](docs/results.md), [`docs/training.md`](docs/training.md) | Narrative for hackathon review |
| Replay artifact | [`runs/demo_fallback/replay.html`](runs/demo_fallback/replay.html) | Offline step scrubber (generate via `python scripts/export_replay.py`) |
| Ablation harness | [`scripts/ablation.py`](scripts/ablation.py) | Reward component sensitivity (no API keys) |
| Before/after table | [`training/eval_before_after.py`](training/eval_before_after.py) | “No real optimization” vs fallback policy → `results/before_after_*` |

OpenEnv manifest: [`openenv.yaml`](openenv.yaml).
