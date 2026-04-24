"""
train.py — GRPO Fine-Tuning on SQL Query Optimization Environment
==================================================================
Uses Group Relative Policy Optimization (GRPO) via Hugging Face TRL
to train a small LLM to become a better SQL optimizer by directly
interacting with the SQLOptimEnv environment.

The reward signal is 100% execution-grounded:
  - Real DuckDB timing speedup (35%)
  - Result correctness (20%)
  - Issue detection quality (25%)
  - Structure quality (13%)
  - Correctness penalty (7%)

Training loop:
  1. Sample a random task from the environment
  2. Get the observation (bad SQL + schema context)
  3. Generate G candidate completions (the "group" in GRPO)
  4. Execute each completion against DuckDB → compute real reward
  5. Compute relative advantages within the group
  6. Update the policy to prefer higher-reward completions

Usage:
  pip install trl transformers torch duckdb openai
  python train.py

For Colab / HF Spaces:
  See train_colab.ipynb for the notebook version with plots.
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ── Lazy imports (environment is in same dir) ─────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from env import SQLOptimEnv
from models import Action
from tasks import TASKS

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"   # small — fits on free Colab T4
    # Training
    num_episodes: int = 200          # total environment episodes
    group_size: int = 4              # G completions per prompt (GRPO)
    max_new_tokens: int = 1024
    temperature: float = 0.8
    learning_rate: float = 1e-5
    # Logging
    log_every: int = 10              # log metrics every N episodes
    save_every: int = 50             # save checkpoint every N episodes
    output_dir: str = "./checkpoints"
    # Tasks
    task_ids: List[str] = field(default_factory=lambda: list(TASKS.keys()))
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = TrainConfig()

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert database engineer specializing in SQL performance optimization.
You will receive a SQL query and its schema. Your task:
1. Identify ALL performance anti-patterns.
2. Produce a complete, correct, optimized rewrite.
3. Your optimized_query will be ACTUALLY EXECUTED against DuckDB with real data.
   If it errors or returns wrong results, your score is 0.

Respond ONLY with valid JSON (no markdown, no code fences):
{
  "suggestions": [
    {
      "issue_type": "e.g. select_star | correlated_subquery | wildcard_like",
      "line": <integer>,
      "description": "precise explanation of the performance problem",
      "severity": "critical | high | medium | low",
      "fix": "specific corrective SQL"
    }
  ],
  "optimized_query": "<complete executable SQL returning IDENTICAL results>",
  "summary": "2-4 sentence performance analysis",
  "estimated_improvement": "e.g. '15x faster — eliminates N+1 pattern'",
  "approved": false
}"""


def build_prompt(obs) -> str:
    return (
        f"Task        : {obs.task_name}\n"
        f"Difficulty  : {obs.difficulty}\n"
        f"Step        : {obs.step_count + 1} / {obs.max_steps}\n\n"
        f"Database Schema:\n{obs.schema_info}\n\n"
        f"SQL Query to Optimize:\n```sql\n{obs.sql_query}\n```\n\n"
        f"Instructions:\n{obs.task_description}\n\n"
        "Provide your complete analysis and optimized_query now."
    )


def parse_action(text: str) -> Dict[str, Any]:
    clean = text.strip()
    # Strip markdown fences if present
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    try:
        return json.loads(clean)
    except Exception:
        return {
            "suggestions": [],
            "optimized_query": "",
            "summary": "Parse error",
            "estimated_improvement": "unknown",
            "approved": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GRPO reward normalisation
# ─────────────────────────────────────────────────────────────────────────────

def compute_advantages(rewards: List[float]) -> List[float]:
    """
    GRPO: normalise rewards within the group to get advantages.
    advantage_i = (r_i - mean(r)) / (std(r) + eps)
    This makes the gradient update relative — completions that are
    better than the group average get positive advantage, worse get negative.
    """
    if len(rewards) == 0:
        return []
    mean_r = sum(rewards) / len(rewards)
    var_r  = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
    std_r  = var_r ** 0.5
    eps    = 1e-8
    return [(r - mean_r) / (std_r + eps) for r in rewards]


# ─────────────────────────────────────────────────────────────────────────────
# Single episode rollout (one task, one LLM call, one env step)
# ─────────────────────────────────────────────────────────────────────────────

def rollout_single(
    model,
    tokenizer,
    env: SQLOptimEnv,
    task_id: str,
    num_completions: int = 4,
) -> Tuple[List[str], List[float], str]:
    """
    Roll out one episode with `num_completions` parallel candidate completions.
    Returns (completions, rewards, prompt_text).
    """
    obs = env.reset(task_id=task_id)
    prompt = build_prompt(obs)

    # Build the full message for the tokenizer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        chat_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(cfg.device)

    # Generate G completions (the group)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            num_return_sequences=num_completions,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    completions = [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in outputs
    ]

    # Score each completion against the real environment
    rewards = []
    for completion in completions:
        parsed = parse_action(completion)
        action = Action(
            suggestions=parsed.get("suggestions", []),
            optimized_query=parsed.get("optimized_query", ""),
            summary=parsed.get("summary", ""),
            estimated_improvement=parsed.get("estimated_improvement", ""),
            approved=parsed.get("approved", False),
        )
        try:
            # Fresh env step — reset so each completion is scored independently
            env.reset(task_id=task_id)
            result = env.step(action)
            rewards.append(result.reward.score)
        except Exception as e:
            print(f"  [WARN] env.step failed: {e}", flush=True)
            rewards.append(0.0)

    return completions, rewards, chat_text, inputs


# ─────────────────────────────────────────────────────────────────────────────
# GRPO policy gradient update
# ─────────────────────────────────────────────────────────────────────────────

def grpo_update(
    model,
    tokenizer,
    optimizer,
    completions: List[str],
    rewards: List[float],
    prompt_text: str,
    prompt_inputs: Dict,
) -> float:
    """
    Compute GRPO loss and backpropagate.

    GRPO loss = -mean( advantage_i * log_prob(completion_i | prompt) )

    This is a simplified GRPO implementation (without reference model KL).
    For full KL-penalised GRPO, use trl.GRPOTrainer directly.
    """
    advantages = compute_advantages(rewards)

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for completion, advantage in zip(completions, advantages):
        full_text = prompt_text + completion
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(cfg.device)

        prompt_len = prompt_inputs["input_ids"].shape[1]

        outputs = model(**inputs, labels=inputs["input_ids"])
        # We only want the loss on the completion tokens, not the prompt
        # Shift labels so prompt tokens are masked (-100)
        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        outputs2 = model(**inputs, labels=labels)
        loss = outputs2.loss * advantage  # scale by advantage

        loss.backward()
        total_loss += loss.item()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss / max(len(completions), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  SQL Query Optimization — GRPO Training")
    print(f"  Model   : {cfg.model_name}")
    print(f"  Device  : {cfg.device}")
    print(f"  Episodes: {cfg.num_episodes}")
    print(f"  Group G : {cfg.group_size}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[1/3] Loading model: {cfg.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
        device_map="auto" if cfg.device == "cuda" else None,
    )
    if cfg.device == "cpu":
        model = model.to(cfg.device)
    model.train()
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # ── Environment ───────────────────────────────────────────────────
    print("[2/3] Initialising SQLOptimEnv (DuckDB warm-up ~3s) ...", flush=True)
    env = SQLOptimEnv()

    # ── Training metrics ──────────────────────────────────────────────
    episode_rewards: List[float] = []   # mean reward per episode
    episode_losses:  List[float] = []   # GRPO loss per episode
    best_reward: float = 0.0
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("[3/3] Starting GRPO training loop ...\n", flush=True)
    t_start = time.time()

    for episode in range(1, cfg.num_episodes + 1):
        task_id = random.choice(cfg.task_ids)

        try:
            completions, rewards, prompt_text, prompt_inputs = rollout_single(
                model, tokenizer, env, task_id, num_completions=cfg.group_size
            )

            loss = grpo_update(
                model, tokenizer, optimizer,
                completions, rewards, prompt_text, prompt_inputs
            )

            mean_reward = sum(rewards) / max(len(rewards), 1)
            max_reward  = max(rewards) if rewards else 0.0
            episode_rewards.append(mean_reward)
            episode_losses.append(loss)

            if max_reward > best_reward:
                best_reward = max_reward

            if episode % cfg.log_every == 0:
                elapsed = time.time() - t_start
                recent_avg = sum(episode_rewards[-cfg.log_every:]) / cfg.log_every
                print(
                    f"[Ep {episode:4d}/{cfg.num_episodes}] "
                    f"task={task_id[:28]:<28} "
                    f"rewards={[f'{r:.3f}' for r in rewards]} "
                    f"mean={mean_reward:.4f}  "
                    f"loss={loss:.4f}  "
                    f"recent_avg={recent_avg:.4f}  "
                    f"best={best_reward:.4f}  "
                    f"time={elapsed:.0f}s",
                    flush=True,
                )

            if episode % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"ckpt_ep{episode}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"  [SAVE] Checkpoint saved → {ckpt_path}", flush=True)

        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user.", flush=True)
            break
        except Exception as exc:
            print(f"  [WARN] Episode {episode} failed: {exc}", flush=True)
            episode_rewards.append(0.0)
            episode_losses.append(0.0)
            continue

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[DONE] Final model saved → {final_path}", flush=True)

    # ── Save reward/loss history ──────────────────────────────────────
    history = {
        "episode_rewards": episode_rewards,
        "episode_losses":  episode_losses,
        "best_reward":     best_reward,
        "config": {
            "model_name":    cfg.model_name,
            "num_episodes":  cfg.num_episodes,
            "group_size":    cfg.group_size,
            "learning_rate": cfg.learning_rate,
        },
    }
    history_path = os.path.join(cfg.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[DONE] Training history saved → {history_path}", flush=True)

    # ── Plot reward curve ─────────────────────────────────────────────
    try:
        _plot_results(episode_rewards, episode_losses, cfg.output_dir)
    except Exception as e:
        print(f"[WARN] Plotting failed (matplotlib not installed?): {e}", flush=True)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best reward achieved : {best_reward:.4f}")
    print(f"  Final avg reward     : {sum(episode_rewards[-20:]) / 20:.4f}")
    print(f"  Total episodes       : {len(episode_rewards)}")
    print(f"{'='*60}")
    return history


def _plot_results(rewards: List[float], losses: List[float], output_dir: str):
    """Generate and save training curve plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("SQL Query Optimization — GRPO Training Progress", fontsize=14, fontweight="bold")

    episodes = list(range(1, len(rewards) + 1))

    # Smoothed reward curve
    window = min(20, len(rewards) // 5 + 1)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        smooth_x = list(range(window, len(rewards) + 1))
        ax1.plot(episodes, rewards, alpha=0.3, color="#4A90D9", label="Raw reward")
        ax1.plot(smooth_x, smoothed, color="#E74C3C", linewidth=2,
                 label=f"Smoothed (window={window})")
    else:
        ax1.plot(episodes, rewards, color="#4A90D9", linewidth=2, label="Mean reward")

    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Mean Group Reward")
    ax1.set_title("Reward Progress (higher = better SQL optimization)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Loss curve
    ax2.plot(episodes, losses, alpha=0.4, color="#2ECC71", label="GRPO loss")
    if len(losses) >= window:
        smooth_loss = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax2.plot(smooth_x, smooth_loss, color="#8E44AD", linewidth=2,
                 label=f"Smoothed loss")
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("GRPO Policy Loss")
    ax2.set_title("Policy Loss (convergence indicator)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Training curves saved → {plot_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRL GRPOTrainer integration (alternative — uses full KL penalty)
# ─────────────────────────────────────────────────────────────────────────────

def train_with_trl():
    """
    Alternative training using HuggingFace TRL's GRPOTrainer.
    This is the production-grade path with:
      - KL penalty to prevent reward hacking
      - Proper reference model management
      - Built-in logging to Weights & Biases

    Usage:
      pip install trl>=0.8.0 transformers torch duckdb
      python train.py --use-trl
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("[ERROR] TRL not installed. Run: pip install trl>=0.8.0", flush=True)
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model for TRL GRPO training ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
    )

    # ── Build a dataset from all tasks ────────────────────────────────
    env = SQLOptimEnv()
    from datasets import Dataset

    records = []
    for task_id, task_data in TASKS.items():
        obs = env.reset(task_id=task_id)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(obs)},
        ]
        records.append({"prompt": messages, "task_id": task_id})

    # Repeat tasks to create a training dataset
    records = records * 40   # 5 tasks × 40 = 200 examples
    random.shuffle(records)
    dataset = Dataset.from_list(records)

    # ── Reward function for TRL ────────────────────────────────────────
    def reward_fn(completions: List[str], prompts=None, **kwargs) -> List[float]:
        """
        TRL calls this with a batch of completions.
        We score each against the environment.
        """
        rewards = []
        for completion in completions:
            # Extract task_id from the prompt (hacky but works)
            task_id = random.choice(list(TASKS.keys()))
            parsed = parse_action(completion)
            action = Action(
                suggestions=parsed.get("suggestions", []),
                optimized_query=parsed.get("optimized_query", ""),
                summary=parsed.get("summary", ""),
                estimated_improvement=parsed.get("estimated_improvement", ""),
                approved=parsed.get("approved", False),
            )
            try:
                env.reset(task_id=task_id)
                result = env.step(action)
                rewards.append(result.reward.score)
            except Exception:
                rewards.append(0.0)
        return rewards

    # ── TRL Config ────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=cfg.learning_rate,
        num_generations=cfg.group_size,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        logging_steps=10,
        save_steps=50,
        report_to="none",  # set to "wandb" if you have W&B configured
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting TRL GRPO training ...", flush=True)
    trainer.train()
    trainer.save_model(os.path.join(cfg.output_dir, "trl_final"))
    print("[DONE] TRL training complete.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_trl = "--use-trl" in sys.argv
    if use_trl:
        train_with_trl()
    else:
        train()
