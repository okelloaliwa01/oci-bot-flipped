# -----------------------------
# File: src/micro_optimizer.py
# -----------------------------
"""
Micro-optimizer for volatility-linked execution parameters.

• Runs fast (< 5 seconds)
• Updates ATR-dependent exit logic
• Implements ATR Shock Detector (vectorized)
• Safe .env.systemd updater
• Compatible with SmartExitManager
• NEW: Background shock monitor that auto-triggers optimization
  — triggers only when a shock *starts* (False -> True)
  — persistent state + cooldown to avoid thrashing
"""

from __future__ import annotations
import os
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
import logging
from dotenv import load_dotenv
from typing import Optional

import pandas as pd
import numpy as np

# Local imports
from binance_client import BinanceClient
import config

# ======================================================================
# Logging
# ======================================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("micro_optimizer")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_DIR / "micro_optimizer.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

# ======================================================================
# Environment Path Detection
# ======================================================================
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
OCI_ENV_PATH = Path("/home/ubuntu/oci-bot-flipped/.env.systemd")

ENV_PATH = OCI_ENV_PATH if OCI_ENV_PATH.exists() else DEFAULT_ENV_PATH
logger.info(f"Micro optimizer using env: {ENV_PATH}")

def set_env_var(key: str, value: str):
    """Safe .env writer."""
    try:
        lines = []
        if ENV_PATH.exists():
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()

        updated = False
        out = []
        for L in lines:
            if L.startswith(f"{key}="):
                out.append(f"{key}={value}\n")
                updated = True
            else:
                out.append(L)

        if not updated:
            out.append(f"{key}={value}\n")

        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.writelines(out)

        logger.info(f"Updated {key}={value}")

    except Exception as e:
        logger.exception(f"Failed writing {key}: {e}")

# ======================================================================
# Vectorized ATR Computation
# ======================================================================
def compute_atr_vectorized(klines, period=14):
    """Compute ATR using vectorized pandas operations."""
    if len(klines) < period + 2:
        return None

    df = pd.DataFrame(klines)
    df["high"] = pd.to_numeric(df[2], errors="coerce")
    df["low"] = pd.to_numeric(df[3], errors="coerce")
    df["close"] = pd.to_numeric(df[4], errors="coerce")
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_series = tr.rolling(window=period, min_periods=1).mean()
    atr_series = atr_series.dropna()
    return float(atr_series.iloc[-1]) if not atr_series.empty else None

# ======================================================================
# Shock Detector
# ======================================================================
def atr_shock_detector(
    client: BinanceClient,
    symbol: str,
    threshold: Optional[float] = None,
    lookback_limit: int = 14000
) -> dict:
    """
    Detect volatility shocks:
    ATR(Today) / ATR(7d_avg) >= threshold
    """
    if threshold is None:
        threshold = float(os.getenv("MICRO_ATR_SHOCK_THRESHOLD", "1.7"))

    klines = client.get_klines(symbol, "5m", lookback_limit)
    if not klines or len(klines) < 2000:
        return {"ok": False, "reason": "not_enough_data"}

    df = pd.DataFrame(klines)
    df["high"] = pd.to_numeric(df[2], errors="coerce")
    df["low"] = pd.to_numeric(df[3], errors="coerce")
    df["close"] = pd.to_numeric(df[4], errors="coerce")
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_series = tr.rolling(window=14, min_periods=1).mean().dropna()
    if atr_series.empty:
        return {"ok": False, "reason": "no_atr_values"}

    atr_today = float(atr_series.iloc[-1])
    atr_7d_avg = float(atr_series.iloc[-2000:].mean() if len(atr_series) >= 2000 else atr_series.mean())
    ratio = atr_today / atr_7d_avg if atr_7d_avg > 0 else 0.0

    logger.info(f"ATR Shock Check → today={atr_today:.6f} avg={atr_7d_avg:.6f} ratio={ratio:.3f}")

    return {
        "ok": True,
        "atr_today": atr_today,
        "atr_7d_avg": atr_7d_avg,
        "ratio": ratio,
        "shock": ratio >= threshold,
        "threshold": threshold,
    }

# ======================================================================
# Micro Optimization Logic
# ======================================================================
def optimize_micro_params(atr_today: float) -> dict:
    """Deterministic micro-optimization based on current ATR."""
    search = {
        "ATR_MULT_SL": [1.2, 2.4],
        "ATR_MULT_TP1": [0.6, 1.6],
        "ATR_MULT_TP2": [1.4, 3.5],
        "TRAILING_START_ATR": [1.2, 2.0],
        "TRAILING_STEP_ATR": [0.15, 0.6],
        "BREAKEVEN_ATR": [0.8, 1.6],
        "BREAKEVEN_BUFFER_PTS": [0.02, 0.08],
    }

    vol_factor = max(1.0, min(atr_today * 100, 3.0))
    best = {
        "ATR_MULT_SL": round(search["ATR_MULT_SL"][0] * vol_factor / 1.3, 4),
        "ATR_MULT_TP1": round(search["ATR_MULT_TP1"][0] * vol_factor / 1.4, 4),
        "ATR_MULT_TP2": round(search["ATR_MULT_TP2"][0] * vol_factor / 1.1, 4),
        "TRAILING_START_ATR": round(search["TRAILING_START_ATR"][0] * vol_factor / 1.5, 4),
        "TRAILING_STEP_ATR": round(search["TRAILING_STEP_ATR"][0] * 1.05, 4),
        "BREAKEVEN_ATR": round(search["BREAKEVEN_ATR"][0] * vol_factor / 1.35, 4),
        "BREAKEVEN_BUFFER_PTS": round(search["BREAKEVEN_BUFFER_PTS"][0], 4),
    }

    for k, v in best.items():
        low, high = search[k]
        best[k] = max(low, min(v, high))

    return best

# ======================================================================
# Persistent state and lock
# ======================================================================
STATE_PATH = ENV_PATH.parent / "micro_optimizer_state.json"
LOCK_PATH = ENV_PATH.parent / "micro_optimizer.lock"

def _read_state():
    try:
        if STATE_PATH.exists():
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.exception("Failed to read state file")
    return {"prev_shock": False, "last_run_ts": 0}

def _write_state(state: dict):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        logger.exception("Failed to write state file")

def _acquire_lock():
    try:
        if LOCK_PATH.exists():
            try:
                pid = int(LOCK_PATH.read_text().strip())
                os.kill(pid, 0)
                return False
            except Exception:
                LOCK_PATH.unlink()
        LOCK_PATH.write_text(str(os.getpid()))
        return True
    except Exception:
        logger.exception("Failed to acquire lock")
        return False

def _release_lock():
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception:
        logger.exception("Failed to release lock")

# ======================================================================
# Run Micro Optimizer
# ======================================================================
def run_micro_optimizer():
    load_dotenv(override=True)
    symbol = getattr(config, "SYMBOL", "XRPUSDT")

    if not _acquire_lock():
        logger.info("Optimizer already running (lock prevented duplicate).")
        return

    try:
        client = BinanceClient()
        logger.info("=== Running Micro Optimizer ===")
        shock_info = atr_shock_detector(client, symbol)
        if not shock_info.get("ok", False):
            logger.warning("Shock detector failed — aborting micro optimization.")
            return

        logger.info(f"ATR Shock Mode: {shock_info['shock']}")
        params = optimize_micro_params(shock_info["atr_today"])

        for k, v in params.items():
            set_env_var(k, str(v))

        out = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "atr_today": shock_info["atr_today"],
            "atr_7d_avg": shock_info["atr_7d_avg"],
            "ratio": shock_info["ratio"],
            "shock_mode": shock_info["shock"],
            "optimized_params": params,
        }
        with open(ENV_PATH.parent / "optimized_micro_params.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        logger.info("Micro optimization complete.")
    except Exception as e:
        logger.exception(f"run_micro_optimizer error: {e}")
    finally:
        _release_lock()

# ======================================================================
# Background Monitor
# ======================================================================
def atr_shock_monitor_background(symbol: str, client: BinanceClient, check_interval_sec: int = 300, cooldown_sec: int = 1800):
    logger.info("Starting ATR Shock Background Monitor...")
    state = _read_state()
    prev_shock = bool(state.get("prev_shock", False))
    last_run_ts = float(state.get("last_run_ts", 0.0))

    while True:
        try:
            shock_info = atr_shock_detector(client, symbol)
            if not shock_info.get("ok", False):
                logger.warning("ATR monitor: insufficient data.")
            else:
                shock = bool(shock_info["shock"])
                now_ts = time.time()
                logger.info(f"[ATR MONITOR] Today={shock_info['atr_today']:.6f}, Avg={shock_info['atr_7d_avg']:.6f}, ratio={shock_info['ratio']:.3f}, shock={shock}")

                if shock and not prev_shock and now_ts - last_run_ts >= cooldown_sec:
                    logger.info("⚡ ATR SHOCK started → running micro optimizer...")
                    run_micro_optimizer()
                    last_run_ts = time.time()
                    prev_shock = True
                    state["prev_shock"] = True
                    state["last_run_ts"] = last_run_ts
                    _write_state(state)
                elif shock and prev_shock:
                    logger.debug("Shock ongoing — optimizer already applied for this shock window.")
                elif not shock and prev_shock:
                    logger.info("ATR shock ended — monitor will trigger on next shock start.")
                    prev_shock = False
                    state["prev_shock"] = False
                    _write_state(state)
        except Exception as e:
            logger.exception(f"ATR shock monitor error: {e}")

        time.sleep(check_interval_sec)

# ======================================================================
# CLI Dispatcher
# ======================================================================
if __name__ == "__main__":
    mode = os.getenv("MICRO_MODE", "single").lower()
    symbol = getattr(config, "SYMBOL", "XRPUSDT")
    client = BinanceClient()

    if mode == "monitor":
        check_interval_sec = int(os.getenv("MICRO_CHECK_INTERVAL_SEC", "300"))
        cooldown_sec = int(os.getenv("MICRO_COOLDOWN_SEC", "1800"))
        atr_shock_monitor_background(symbol, client, check_interval_sec, cooldown_sec)
    else:
        run_micro_optimizer()
