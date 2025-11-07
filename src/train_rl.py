#!/usr/bin/env python3
"""
train_rl.py

Wrapper to accept optimized parameters (from optimized_params.json) and
launch a reinforcement learning (RL) training run using Stable-Baselines3 (PPO by default).

Usage:
    python train_rl.py --params optimized_params.json --timesteps 100000

Features:
- Loads optimized parameters from Optuna output
- Maps them to RL hyperparameters dynamically
- Supports both real PPO training (if stable-baselines3 is available) and mock dry-run
- Saves training metadata + results to rl_training_results.json
- Compatible with Binance Futures optimizer (param_optimizer.py)

Dependencies:
    pip install gymnasium stable-baselines3 torch numpy matplotlib pandas
"""

import argparse
import json
import sys
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, Optional

# --- Fix console encoding issues on Windows ---
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# === Try imports ===
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        gym = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

RESULTS_PATH = Path("rl_training_results.json")


# ===================================================================
# PARAM LOADING
# ===================================================================
def load_params_from_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params file not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept different shapes (legacy, direct dict, etc.)
    if isinstance(data, dict):
        if "BEST_PARAMS" in data:
            return data["BEST_PARAMS"]
        if "best_params" in data:
            return data["best_params"]
    return data


# ===================================================================
# PARAM → HYPERPARAM MAPPING
# ===================================================================
def map_params_to_hyperparams(params: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """Convert optimized trading parameters into RL hyperparameters."""
    h = {}
    seed = seed or int(time.time()) % (2**31 - 1)
    h["seed"] = seed

    # Learning rate: inverse of margin size (larger margin -> slower learning)
    margin_percent = float(params.get("margin_percent", params.get("MARGIN_PERCENT", 1.0)))
    lr = 3e-4 - (margin_percent / 5.0) * (2e-4)
    lr = max(1e-5, min(3e-4, lr))
    h["learning_rate"] = lr

    # Batch size: smaller for larger positions (simulates more stable updates)
    margin_usdt = float(params.get("margin_usdt", params.get("MARGIN_USDT", 3.0)))
    h["batch_size"] = 64 if margin_usdt <= 5 else 32

    # Steps per update: scale with ATR multiplier
    atr_mult = float(params.get("atr_mult_tp1", 1.0))
    h["n_steps"] = int(max(128, min(2048, 512 * (1 + (atr_mult - 1) / 2))))

    # Entropy coeff (for exploration)
    h["ent_coef"] = 0.0
    h["default_timesteps"] = 50_000
    return h


# ===================================================================
# UTILITIES
# ===================================================================
def save_results(out: Dict[str, Any], path: Path = RESULTS_PATH) -> None:
    """Safely save training results to disk."""
    out = dict(out)
    out["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved RL training results to: {path.resolve()}")


# ===================================================================
# TRAINING ROUTINES
# ===================================================================
def train_with_sb3(env_id: str, hyperparams: Dict[str, Any], timesteps: int, log_dir: Optional[str] = None) -> Dict[str, Any]:
    """Train a PPO agent using Stable-Baselines3."""
    assert SB3_AVAILABLE, "stable-baselines3 not installed."

    def make_env():
        try:
            return gym.make(env_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize environment '{env_id}': {e}")

    env = DummyVecEnv([make_env])
    seed = int(hyperparams.get("seed", 0))
    env.seed(seed)
    random.seed(seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        seed=seed,
        tensorboard_log=log_dir if log_dir else None,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    duration = time.time() - t0

    # Save model
    model_path = Path("models") / f"ppo_agent_{env_id}_{int(time.time())}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward.sum()) if hasattr(reward, "sum") else float(reward)
        steps += 1
        if done:
            obs = env.reset()

    metrics = {
        "algorithm": "PPO",
        "env_id": env_id,
        "timesteps": timesteps,
        "duration_seconds": round(duration, 2),
        "seed": seed,
        "learning_rate": hyperparams["learning_rate"],
        "n_steps": hyperparams["n_steps"],
        "batch_size": hyperparams["batch_size"],
        "rollout_eval_reward": round(total_reward / max(1, steps), 6),
        "model_path": str(model_path),
    }
    return metrics


def mock_train(hyperparams: Dict[str, Any], timesteps: int) -> Dict[str, Any]:
    """Simulated RL training (used if SB3 unavailable)."""
    print("stable-baselines3 not found — running mock training (dry run).")
    seed = int(hyperparams.get("seed", 0))
    random.seed(seed)
    time.sleep(1)
    duration = random.uniform(0.5, 2.0)
    synthetic_reward = (math.log1p(timesteps) * random.uniform(0.05, 0.2))
    return {
        "algorithm": "MOCK",
        "env_id": "mock-env",
        "timesteps": timesteps,
        "duration_seconds": round(duration, 2),
        "seed": seed,
        "learning_rate": hyperparams.get("learning_rate"),
        "n_steps": hyperparams.get("n_steps"),
        "batch_size": hyperparams.get("batch_size"),
        "rollout_eval_reward": round(synthetic_reward, 6),
    }


# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description="Train RL agent with optimized parameters")
    parser.add_argument("--params", required=True, help="Path to optimized_params.json or JSON string")
    parser.add_argument("--timesteps", type=int, default=None, help="Total timesteps for training")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym env ID or custom environment")
    parser.add_argument("--dry-run", action="store_true", help="Run without actual RL training")
    parser.add_argument("--log-dir", type=str, default="logs", help="Optional log directory for TensorBoard")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args(argv)

    try:
        params_path = Path(args.params)
        if params_path.exists():
            params = load_params_from_file(str(params_path))
        else:
            params = json.loads(args.params)
    except Exception as e:
        print(f"Failed to load parameters: {e}")
        sys.exit(1)

    hyperparams = map_params_to_hyperparams(params, seed=args.seed)
    timesteps = args.timesteps or hyperparams.get("default_timesteps", 50_000)

    print("=== RL TRAINING START ===")
    print(f"Env: {args.env} | Timesteps: {timesteps} | SB3: {SB3_AVAILABLE}")
    print(f"Hyperparams:\n{json.dumps(hyperparams, indent=2)}")

    out = {
        "params_used": params,
        "hyperparams": hyperparams,
        "timesteps": timesteps,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    try:
        if args.dry_run or not SB3_AVAILABLE:
            metrics = mock_train(hyperparams, timesteps)
        else:
            metrics = train_with_sb3(args.env, hyperparams, timesteps, log_dir=args.log_dir)
    except Exception as e:
        print(f"Training failed: {e}")
        out["error"] = str(e)
        save_results(out)
        sys.exit(2)

    out["metrics"] = metrics
    save_results(out)

    print("\n=== RL TRAINING COMPLETE ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as exc:
        print(f"Unhandled exception: {exc}")
        sys.exit(99)
