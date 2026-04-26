# Training (GRPO)

This environment’s training signal is the **same composite reward** as evaluation: DuckDB execution (speedup + correctness), issue keywords, and light structure checks. There is no separate “training reward” that could diverge from deployment.

## Scripts

| Entry | Purpose |
|-------|---------|
| [`train.py`](../train.py) | Custom GRPO-style loop: sample task → generate a **group** of completions → score each with `env.step` / `grade` → advantage normalize → policy update |
| [`train.py`](../train.py) `--use-trl` | Optional path using Hugging Face **TRL** `GRPOTrainer` (requires `trl`, proper KL handling) |
| [Kaggle notebook](https://www.kaggle.com/code/officialabhinavsingh/train-kaggle) | Full 100-episode run with plots (linked from README) |

## Default hyperparameters (`TrainConfig` in `train.py`)

| Field | Default | Notes |
|-------|---------|-------|
| `model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Small model for free-tier GPUs |
| `num_episodes` | 200 | Full runs; reduce for smoke tests |
| `group_size` | 4 | GRPO group size \(G\) |
| `max_new_tokens` | 1024 | JSON action payload |
| `temperature` | 0.8 | Sampling during rollout |
| `learning_rate` | 1e-5 | AdamW |
| `output_dir` | `./checkpoints` | Model + `training_history.json` + optional `training_curves.png` |

Override by editing `TrainConfig` in [`train.py`](../train.py) or extending the script (no CLI flags on the simple trainer today).

## Hardware

- **CUDA**: Recommended; `device_map="auto"` when available.
- **CPU**: Supported but slow; DuckDB warm-up + many forward passes dominate.

## Reproducibility

- **Tasks**: Fixed set in [`tasks.py`](../tasks.py); each episode samples uniformly unless you change `train.py`.
- **Randomness**: `random.choice` for task id; `model.generate` uses sampling — set seeds in PyTorch / CUDA / NumPy at the top of `train.py` if you need bitwise reproducibility for a paper run.

## Published artifact

Fine-tuned weights referenced in the README: [laterabhi/grpo-sql-optimizer](https://huggingface.co/laterabhi/grpo-sql-optimizer).

## Quick sanity check (no weight updates)

```bash
python training/eval_before_after.py --save-dir results
```

Shows how much reward comes from **actually running** optimized SQL vs analysis-only (see [results.md](results.md)).
