# src/param_optimizer.py
"""
Production-grade parameter optimizer for exit / sizing parameters.

Features:
- Cross-validation (rolling folds)
- Multi-objective scoring (Sharpe – Drawdown penalty)
- EMA, RSI, and MACD-based entry filtering
- Multi-core per-fold parallel optimization (pickle-safe)
- Auto-export to optimized_params.json
- Unified .env writer
- Structured logging
- Async RL sync (default) with logfile & PID
- Volatility-based adaptive timeframe weighting (writes back to config)
- New toggles: RSI thresholds, objective weighting
"""
from __future__ import annotations

import math
import random
import statistics
import json
import concurrent.futures
import re
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, cast
from datetime import datetime
import logging
import logging.handlers
import csv

# Local project imports (assumes module path resolves)
from binance_client import BinanceClient
import config
from guards.market_integrity import MarketIntegrityGuard

# ---------------------------
# Configuration (local overrides)
# ---------------------------
CONFIG: Dict[str, Any] = {
    "SYMBOL": getattr(config, "SYMBOL", "BTCUSDT"),
    "TIMEFRAME": getattr(config, "TIMEFRAME", "5m"),

    # --- Data & Entry ---
    "HISTORY_LIMIT": 15000,              # ~52 days of 5m data
    "ENTRY_LOOKBACK": 8,                 # very reactive breakout confirmation
    "ENTRY_THRESHOLD": 0.0007,           # ~0.07% breakout filter to avoid fakeouts
    "MIN_BAR_DISTANCE": 2,               # denser signals; 2-bar confirmation window

    # --- Momentum Filters ---
    "RSI_PERIOD": getattr(config, "RSI_PERIOD", 9),
    "RSI_OVERSOLD": float(os.getenv("RSI_OVERSOLD", "25")),   # tighter for high-frequency entries
    "RSI_OVERBOUGHT": float(os.getenv("RSI_OVERBOUGHT", "75")),
    "EMA_FAST": 13,
    "EMA_SLOW": 34,
    "MACD_FAST": 6,
    "MACD_SLOW": 19,
    "MACD_SIGNAL": 5,

    # --- Optimization ---
    "OPTUNA_TRIALS": int(os.getenv("OPTUNA_TRIALS", "500")),   # higher coverage for micro-movements
    "CROSSVAL_FOLDS": int(os.getenv("CROSSVAL_FOLDS", "3")),
    "SEED": int(os.getenv("OPT_SEED", "42")),

    # --- Leverage & Execution ---
    "LEVERAGE": getattr(config, "LEVERAGE", 60),               # aggressive but manageable with small margin
    # adjust MARGIN_USDT accordingly to reduce liquidation risk

    # --- RL Integration ---
    "RUN_RL_AFTER_OPTIMIZATION": getattr(config, "RUN_RL_AFTER_OPTIMIZATION", True),
    "RL_SYNC_MODE": os.getenv("RL_SYNC_MODE", "async"),
    "RL_LOG_DIR": os.getenv("RL_LOG_DIR", "rl_logs"),
    "RL_TRAIN_SCRIPT": os.getenv("RL_TRAIN_SCRIPT", "train_rl.py"),
    "RL_TIMESTEPS": int(os.getenv("RL_TIMESTEPS", "60000")),

    # --- Objective Weights ---
    "OBJ_WEIGHT_SHARPE": float(os.getenv("OBJ_WEIGHT_SHARPE", "0.85")),
    "OBJ_WEIGHT_DD": float(os.getenv("OBJ_WEIGHT_DD", "0.006")),

    # --- Volatility Adaptation ---
    "ATR_PERCENT_VOL_THRESHOLD": float(os.getenv("ATR_PERCENT_VOL_THRESHOLD", "1.1")),
}

random.seed(CONFIG["SEED"])


# ---------------------------
# Logging - structured
# ---------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("param_optimizer")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))

# Rotating file handler
fh = logging.handlers.RotatingFileHandler(LOG_DIR / "param_optimizer.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))

logger.addHandler(ch)
logger.addHandler(fh)

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class Params:
    use_percent_margin: bool
    margin_percent: float
    margin_usdt: float
    atr_mult_tp1: float
    atr_mult_tp2: float
    atr_mult_sl: float
    trailing_start_atr: float
    trailing_step_atr: float
    breakeven_atr: float
    breakeven_buffer_pts: float
    tp1_pct: float
    tp2_pct: float
    min_notional: float = 0.0

    def tp_levels_mults(self) -> List[float]:
        return [self.atr_mult_tp1, self.atr_mult_tp2]

@dataclass
class TradeResult:
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl_usdt: float
    pnl_pct: float
    duration_bars: int
    closed_by: str

# ---------------------------
# Indicators (improved & robust)
# ---------------------------
from typing import Optional

def compute_ema(values: List[float], period: int) -> List[Optional[float]]:
    n = len(values)
    emas: List[Optional[float]] = [None] * n
    if n < period:
        return emas
    k = 2.0 / (period + 1)
    # seed with simple mean
    seed = sum(values[:period]) / period
    emas[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = values[i] * k + prev * (1 - k)
        emas[i] = prev
    return emas

def compute_rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    n = len(values)
    if n < period + 1:
        return [None] * n
    deltas = [values[i] - values[i - 1] for i in range(1, n)]
    gains = [max(0.0, d) for d in deltas]
    losses = [abs(min(0.0, d)) for d in deltas]
    rsis: List[Optional[float]] = [None] * n
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
    rsis[period] = 100.0 - (100.0 / (1.0 + rs)) if avg_loss > 0 else 100.0
    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
        rsis[i] = 100.0 - (100.0 / (1.0 + rs)) if avg_loss > 0 else 100.0
    return rsis

def compute_macd(
    values: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    ema_fast = compute_ema(values, fast)
    ema_slow = compute_ema(values, slow)
    macd_line = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    
    # Extract valid MACD values
    valid = [v for v in macd_line if v is not None]
    
    if len(valid) < signal:
        signal_line_full: List[Optional[float]] = [None] * len(macd_line)
    else:
        sig_vals = compute_ema(valid, signal)
        pad = len(macd_line) - len(sig_vals)
        signal_line_full = cast(List[Optional[float]], [None] * pad + sig_vals)
    
    hist = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line_full)
    ]
    
    return macd_line, signal_line_full, hist

def compute_atr_from_klines(klines: List[List[Any]], length: int = 14) -> List[Optional[float]]:
    n = len(klines)
    if n < 2:
        return [None] * n
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    atrs: List[Optional[float]] = [None] * n
    if len(trs) < length:
        return atrs
    # ATR i corresponds to candle index i (trs index i-1)
    for i in range(length - 1, len(trs)):
        window = trs[i - (length - 1): i + 1]
        atrs[i + 1] = sum(window) / len(window)
    return atrs

# ---------------------------
# Entry, Simulation, Scoring
# ---------------------------
def find_entries(klines, lookback, threshold, ema_fast, ema_slow, rsi, macd_hist):
    highs = [float(k[2]) for k in klines]
    closes = [float(k[4]) for k in klines]
    entries = []
    last_entry_idx = -999
    rsi_oversold = CONFIG["RSI_OVERSOLD"]
    rsi_overbought = CONFIG["RSI_OVERBOUGHT"]
    for i in range(lookback, len(klines)):
        if i - last_entry_idx < CONFIG["MIN_BAR_DISTANCE"]:
            continue
        ef, es = ema_fast[i], ema_slow[i]
        if ef is None or es is None or ef <= es:
            continue
        rv = rsi[i] if i < len(rsi) else None
        mh = macd_hist[i] if i < len(macd_hist) else None
        # Require MACD histogram >0 and RSI not overbought (we prefer pullback entries)
        if rv is None or rv > rsi_overbought or mh is None or mh <= 0:
            continue
        if closes[i] > max(highs[i - lookback:i]) * (1 + threshold):
            entries.append(i)
            last_entry_idx = i
    return entries

def simulate_trade_from_index(klines, start_index, params: Params, atrs, leverage: int):
    entry_price = float(klines[start_index][4])
    atr = atrs[start_index] if start_index < len(atrs) else None
    if atr is None or atr <= 0:
        return None

    tp_levels = [entry_price + atr * mult for mult in params.tp_levels_mults()]
    sl_price = entry_price - atr * params.atr_mult_sl

    try:
        starting_balance = float(config.ACCOUNT_BALANCE)
    except Exception:
        starting_balance = 1000.0

    margin_usdt = (starting_balance * params.margin_percent / 100.0) if params.use_percent_margin else params.margin_usdt
    if margin_usdt <= 0:
        return None

    pos_value = margin_usdt * leverage
    qty = pos_value / entry_price
    trail_step = atr * params.trailing_step_atr
    trail_trigger = entry_price + atr * params.trailing_start_atr
    breakeven_trigger = entry_price + atr * params.breakeven_atr
    trailing_active = False
    current_sl = sl_price

    for t in range(start_index + 1, len(klines)):
        price = float(klines[t][4])
        if price >= tp_levels[1]:
            pnl = (tp_levels[1] - entry_price) * qty * leverage
            return TradeResult(int(klines[start_index][0]), entry_price, int(klines[t][0]), tp_levels[1], pnl, (pnl / margin_usdt) * 100, t - start_index, "TP2")
        if price >= tp_levels[0]:
            pnl = (tp_levels[0] - entry_price) * qty * leverage
            return TradeResult(int(klines[start_index][0]), entry_price, int(klines[t][0]), tp_levels[0], pnl, (pnl / margin_usdt) * 100, t - start_index, "TP1")
        if not trailing_active and price >= breakeven_trigger:
            current_sl = max(current_sl, entry_price + params.breakeven_buffer_pts)
        if not trailing_active and price >= trail_trigger:
            trailing_active = True
        if trailing_active:
            current_sl = max(current_sl, price - trail_step)
        if price <= current_sl:
            pnl = (price - entry_price) * qty * leverage
            return TradeResult(int(klines[start_index][0]), entry_price, int(klines[t][0]), price, pnl, (pnl / margin_usdt) * 100, t - start_index, "SL")

    last_price = float(klines[-1][4])
    pnl = (last_price - entry_price) * qty * leverage
    return TradeResult(int(klines[start_index][0]), entry_price, int(klines[-1][0]), last_price, pnl, (pnl / margin_usdt) * 100, len(klines) - 1 - start_index, "END")

# ---------------------------
# Scoring helpers
# ---------------------------
def sharpe_like(returns: List[float]) -> float:
    # safe Sharpe-like estimate: mean/std * sqrt(N), handle zero variance
    if not returns:
        return 0.0
    if len(returns) < 2:
        return returns[0] if returns else 0.0
    mean_r = statistics.mean(returns)
    std_r = statistics.pstdev(returns)
    if std_r == 0:
        return mean_r * math.sqrt(len(returns))
    return mean_r / std_r * math.sqrt(len(returns))

def score_trades(trades: List[TradeResult]) -> Dict[str, float]:
    if not trades:
        return {"total_return_usdt": 0.0, "max_drawdown": 0.0, "sharpe_like": 0.0, "win_rate": 0.0}
    pnls = [t.pnl_usdt for t in trades]
    wins = [p for p in pnls if p > 0]
    eq = [1000.0]
    s = 1000.0
    for p in pnls:
        s += p
        eq.append(s)
    peak, mdd = eq[0], 0.0
    for val in eq:
        peak = max(peak, val)
        mdd = max(mdd, peak - val)
    returns = [p / 1000.0 for p in pnls if p != 0.0]
    sharpe_val = sharpe_like(returns)
    win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0
    return {
        "total_return_usdt": s - 1000.0,
        "max_drawdown": mdd,
        "sharpe_like": sharpe_val,
        "win_rate": win_rate
    }

# ---------------------------
# .env writer / updater (unified, OCI-aware)
# ---------------------------


logger = logging.getLogger(__name__)

# Default local .env path
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

# Oracle Cloud specific .env path
OCI_ENV_PATH = Path("/home/ubuntu/oci_bot/.env.systemd")

# Determine which file to use
if OCI_ENV_PATH.exists():
    ENV_PATH = OCI_ENV_PATH
    logger.info(f"Detected Oracle Cloud environment. Using {ENV_PATH}")
else:
    # If running on OCI but file missing, create it automatically
    if os.path.isdir("/home/ubuntu/oci_bot"):
        try:
            OCI_ENV_PATH.touch(mode=0o644, exist_ok=True)
            ENV_PATH = OCI_ENV_PATH
            logger.info(f"Created new .env.systemd at {ENV_PATH}")
        except Exception as e:
            logger.warning(f"Failed to create .env.systemd: {e}")
            ENV_PATH = DEFAULT_ENV_PATH
    else:
        ENV_PATH = DEFAULT_ENV_PATH
        logger.info(f"Using local environment file: {ENV_PATH}")

# Log for debugging
logger.info(f"Active environment file path: {ENV_PATH}")



def _read_env_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def _write_env_lines(path: Path, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Wrote %d lines to %s", len(lines), path)

def set_env_var(key: str, value: Any, env_path: Path = ENV_PATH):
    lines = _read_env_lines(env_path)
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
    updated = False
    new_line = f"{key}={value}\n"
    for i, L in enumerate(lines):
        if pat.match(L):
            lines[i] = new_line
            updated = True
            break
    if not updated:
        lines.append(new_line)
    _write_env_lines(env_path, lines)

# ---------------------------
# Evaluate / wrapper
# ---------------------------
def evaluate_fold_wrapper(args):
    """
    Evaluate a data fold under given trade parameters + simulated market guard.
    Returns sharpe, drawdown, guard penalty, and trade list.
    """
    subset, params = args
    closes = [float(k[4]) for k in subset]
    ema_fast = compute_ema(closes, CONFIG["EMA_FAST"])
    ema_slow = compute_ema(closes, CONFIG["EMA_SLOW"])
    rsi = compute_rsi(closes, CONFIG["RSI_PERIOD"])
    _, _, macd_hist = compute_macd(closes, CONFIG["MACD_FAST"], CONFIG["MACD_SLOW"], CONFIG["MACD_SIGNAL"])
    atrs = compute_atr_from_klines(subset)

    trades: List[TradeResult] = []
    for idx in find_entries(subset, CONFIG["ENTRY_LOOKBACK"], CONFIG["ENTRY_THRESHOLD"],
                            ema_fast, ema_slow, rsi, macd_hist):
        tr = simulate_trade_from_index(subset, idx, params, atrs, CONFIG["LEVERAGE"])
        if tr is not None:
            trades.append(tr)

    metrics = score_trades(trades)

    # >>> PATCH: Guard simulation
    guard = MarketIntegrityGuard(
        spread_mult=float(os.getenv("SPREAD_MULT", 6.0)),
        imbalance_limit=float(os.getenv("IMBALANCE_LIMIT", 0.9)),
        book_event_limit=int(os.getenv("BOOK_EVENT_LIMIT", 200)),
        taker_imbal_limit=float(os.getenv("TAKER_IMBAL_LIMIT", 0.85)),
    )

    # Synthetic orderbook events to estimate guard aggressiveness
    guard_hits = 0
    for i in range(10, len(subset) - 10, 5):
        # Approximate bid/ask spread using close-to-close deltas
        spread = abs(closes[i] - closes[i - 1])
        fake_book = {
            "bids": [[closes[i] - spread / 2, 1.0]],
            "asks": [[closes[i] + spread / 2, 1.0]]
        }
        fake_trades = [{"qty": 1.0, "side": "buy" if closes[i] > closes[i - 1] else "sell"}]
        events_per_sec = random.uniform(50, 400)
        suspicious, reason = guard.check(fake_book, fake_trades, events_per_sec)
        if suspicious:
            guard_hits += 1

    guard_penalty = guard_hits / max(1, len(subset) // 5)
    # <<< PATCH END

    return {
        "sharpe": metrics["sharpe_like"],
        "dd": metrics["max_drawdown"],
        "guard_penalty": guard_penalty,
        "trades": trades,
    }


# ---------------------------
# Utility: plot equity curve
# ---------------------------
import matplotlib.pyplot as plt

def plot_equity_curve(trades: List[TradeResult], path: str):
    pnls = [t.pnl_usdt for t in trades]
    eq = [1000.0]
    for p in pnls:
        eq.append(eq[-1] + p)
    plt.figure(figsize=(10, 5))
    plt.plot(eq, label="Equity Curve", linewidth=2)
    dd = [max(eq[:i + 1]) - eq[i] for i in range(len(eq))]
    plt.fill_between(range(len(eq)), [e - d for e, d in zip(eq, dd)], eq, alpha=0.12, label="Drawdown")
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (USDT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved equity curve to %s", path)

def trades_to_serializable(trades: List[TradeResult]) -> List[Dict[str, Any]]:
    return [t.__dict__ for t in trades]

# ---------------------------
# RL launcher (async by default)
# ---------------------------
def launch_rl_async(train_script: str, params_file: str, timesteps: int, log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile = Path(log_dir) / f"train_rl_{ts}.log"
    cmd = ["python", str(train_script), "--params", str(params_file), "--timesteps", str(timesteps)]
    logger.info("Launching RL (async): %s -> log=%s", " ".join(cmd), logfile)
    try:
        lf = open(logfile, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        logger.info("Launched RL PID=%s (async).", proc.pid)
        return proc.pid, str(logfile)
    except Exception as e:
        logger.exception("Failed to launch RL async: %s", e)
        return None, None

def launch_rl_blocking(train_script: str, params_file: str, timesteps: int, log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile = Path(log_dir) / f"train_rl_{ts}.log"
    cmd = ["python", str(train_script), "--params", str(params_file), "--timesteps", str(timesteps)]
    logger.info("Launching RL (blocking): %s -> log=%s", " ".join(cmd), logfile)
    try:
        with open(logfile, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
        logger.info("RL finished with returncode=%s. log=%s", proc.returncode, logfile)
        return proc.returncode, str(logfile)
    except Exception as e:
        logger.exception("Failed to launch RL blocking: %s", e)
        return None, None

# ---------------------------
# Main optimization (+ adaptive weights)
# ---------------------------
def run_optuna_optimization(client: BinanceClient, n_trials: int = 50):
    import optuna  # type: ignore

    symbol = CONFIG["SYMBOL"]
    tf = CONFIG["TIMEFRAME"]
    limit = CONFIG["HISTORY_LIMIT"]
    logger.info("Downloading %d %s klines for %s", limit, tf, symbol)
    klines = client.get_klines(symbol, tf, limit)
    if not klines:
        raise RuntimeError("No klines returned")

    # === STEP 1: Compute ATR% on base timeframe for volatility ===
    atrs_full = compute_atr_from_klines(klines)
    last_atr = atrs_full[-1] if atrs_full else None
    last_close = float(klines[-1][4]) if klines else 0.0
    atr_percent = (last_atr / last_close * 100.0) if last_atr and last_close else 0.0
    logger.info("Base ATR%% = %.4f (atr=%.6f close=%.4f)", atr_percent, last_atr, last_close)

    # === STEP 2: Adaptive timeframe weights adjustment ===
    try:
        tfw = getattr(config, "TIMEFRAME_WEIGHTS", {"1h": 0.6, "4h": 0.4})
        threshold = CONFIG.get("ATR_PERCENT_VOL_THRESHOLD", 1.5)

        if atr_percent > threshold:
            tfw["1h"], tfw["4h"] = 0.7, 0.3
            logger.info("⚡ High volatility: adjusting timeframe weights -> 1h=0.7, 4h=0.3")
        else:
            tfw["1h"], tfw["4h"] = 0.5, 0.5
            logger.info("✅ Normal volatility: adjusting timeframe weights -> 1h=0.5, 4h=0.5")

        setattr(config, "TIMEFRAME_WEIGHTS", tfw)
        # Save to .env for persistence
        set_env_var("TIMEFRAME_WEIGHTS", json.dumps(tfw))
        logger.info("TIMEFRAME_WEIGHTS updated in config + .env")

    except Exception as e:
        logger.warning("Adaptive timeframe weights failed: %s", e)

    # === STEP 3: Prepare rolling cross-validation folds ===
    folds = max(1, CONFIG["CROSSVAL_FOLDS"])
    fold_size = max(1, len(klines) // folds)
    datasets = [klines[i * fold_size:(i + 1) * fold_size] for i in range(folds)]

    # === STEP 4: Trial CSV setup ===
    trials_csv = Path("optuna_trials.csv")
    if not trials_csv.exists():
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial", "objective", "use_percent_margin", "margin_percent", "margin_usdt",
                "atr_mult_tp1", "atr_mult_tp2", "atr_mult_sl",
                "trailing_start_atr", "trailing_step_atr", "breakeven_atr",
                "breakeven_buffer_pts", "tp1_pct", "tp2_pct"
            ])

    # === STEP 5: Define Optuna objective ===
    def objective(trial: Any) -> float:
        # --- Core trade parameters (unchanged) ---
        params = Params(
            use_percent_margin=trial.suggest_categorical("use_percent_margin", [True, False]),
            margin_percent=trial.suggest_float("margin_percent", 0.1, 5.0),
            margin_usdt=trial.suggest_float("margin_usdt", 1.0, 50.0),
            atr_mult_tp1=trial.suggest_float("atr_mult_tp1", 0.5, 3.0),
            atr_mult_tp2=trial.suggest_float("atr_mult_tp2", 1.0, 6.0),
            atr_mult_sl=trial.suggest_float("atr_mult_sl", 0.8, 3.0),
            trailing_start_atr=trial.suggest_float("trailing_start_atr", 0.5, 3.5),
            trailing_step_atr=trial.suggest_float("trailing_step_atr", 0.1, 1.0),
            breakeven_atr=trial.suggest_float("breakeven_atr", 0.5, 2.5),
            breakeven_buffer_pts=trial.suggest_float("breakeven_buffer_pts", 0.0, 1.0),
            tp1_pct=trial.suggest_float("tp1_pct", 0.2, 0.8),
            tp2_pct=trial.suggest_float("tp2_pct", 0.1, 0.8),
        )

        # --- Guard parameters (added) ---
        guard_params = {
            "SPREAD_MULT": trial.suggest_float("SPREAD_MULT", 2.0, 10.0),
            "IMBALANCE_LIMIT": trial.suggest_float("IMBALANCE_LIMIT", 0.6, 1.0),
            "BOOK_EVENT_LIMIT": trial.suggest_int("BOOK_EVENT_LIMIT", 50, 500),
            "TAKER_IMBAL_LIMIT": trial.suggest_float("TAKER_IMBAL_LIMIT", 0.5, 1.0),
        }
        for k, v in guard_params.items():
            os.environ[k] = str(v)

        # Normalize take-profit allocation
        ssum = params.tp1_pct + params.tp2_pct
        if ssum > 1.0:
            params.tp1_pct /= ssum
            params.tp2_pct /= ssum

        # Run evaluation across data folds
        args = [(d, params) for d in datasets]
        with concurrent.futures.ProcessPoolExecutor() as ex:
            results = list(ex.map(evaluate_fold_wrapper, args))

        sharpe_scores = [r["sharpe"] for r in results if r is not None]
        dd_scores = [r["dd"] for r in results if r is not None]
        guard_scores = [r.get("guard_penalty", 0.0) for r in results if r is not None]

        # Objective composition
        if not sharpe_scores:
            obj = -999.0
        else:
            avg_sharpe = statistics.mean(sharpe_scores)
            avg_dd = statistics.mean(dd_scores)
            avg_guard = statistics.mean(guard_scores)
            obj = (
                CONFIG["OBJ_WEIGHT_SHARPE"] * avg_sharpe
                - CONFIG["OBJ_WEIGHT_DD"] * avg_dd
                - 0.1 * avg_guard  # penalty for over-triggering guard
            )

        # Log each trial
        with open(trials_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial.number, obj,
                params.use_percent_margin, params.margin_percent, params.margin_usdt,
                params.atr_mult_tp1, params.atr_mult_tp2, params.atr_mult_sl,
                params.trailing_start_atr, params.trailing_step_atr,
                params.breakeven_atr, params.breakeven_buffer_pts,
                params.tp1_pct, params.tp2_pct,
                guard_params["SPREAD_MULT"], guard_params["IMBALANCE_LIMIT"],
                guard_params["BOOK_EVENT_LIMIT"], guard_params["TAKER_IMBAL_LIMIT"]
            ])

        return obj


    # === STEP 6: Run optimization ===
    study = optuna.create_study(direction="maximize")
    logger.info("Starting Optuna optimization (%d trials)...", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value
    logger.info("Optuna complete. Best objective=%.4f", best_value)
    logger.info("Best params: %s", json.dumps(best_params, indent=2))

    # === STEP 7: Export results and .env ===
    report_time = datetime.utcnow().isoformat()
    optimized_out = {
        "timestamp": report_time,
        "symbol": symbol,
        "timeframe": tf,
        "best_params": best_params,
        "objective_score": best_value,
        "atr_percent": atr_percent,
        "timeframe_weights": getattr(config, "TIMEFRAME_WEIGHTS", {})
    }
    with open("optimized_params.json", "w", encoding="utf-8") as f:
        json.dump(optimized_out, f, indent=2)
    logger.info("Wrote optimized_params.json")

    try:
        for key, val in {
            "USE_PERCENT_MARGIN": best_params.get("use_percent_margin", False),
            "MARGIN_PERCENT": f"{best_params.get('margin_percent', 1.0):.6f}",
            #"MARGIN_USDT": f"{best_params.get('margin_usdt', 3.0):.6f}",
            "TP_PERCENT": f"{best_params.get('tp1_pct', 0.5):.6f}",
            "SL_PERCENT": f"{best_params.get('tp2_pct', 0.18):.6f}",
            "ATR_MULT_SL": f"{best_params.get('atr_mult_sl', 1.5):.6f}",
            "TRAILING_START_ATR": f"{best_params.get('trailing_start_atr', 1.5):.6f}",
            "TRAILING_STEP_ATR": f"{best_params.get('trailing_step_atr', 0.5):.6f}",
            "BREAKEVEN_ATR": f"{best_params.get('breakeven_atr', 1.0):.6f}",
            "BREAKEVEN_BUFFER_PTS": f"{best_params.get('breakeven_buffer_pts', 0.5):.6f}",
            "TIMEFRAME_WEIGHTS": json.dumps(getattr(config, "TIMEFRAME_WEIGHTS", {})),
            "SPREAD_MULT": f"{best_params.get('SPREAD_MULT', 6.0):.3f}",
            "IMBALANCE_LIMIT": f"{best_params.get('IMBALANCE_LIMIT', 0.9):.3f}",
            "BOOK_EVENT_LIMIT": int(best_params.get('BOOK_EVENT_LIMIT', 200)),
            "TAKER_IMBAL_LIMIT": f"{best_params.get('TAKER_IMBAL_LIMIT', 0.85):.3f}",
        }.items():
            set_env_var(key, str(val))
        logger.info(".env updated with best params")
    except Exception as e:
        logger.exception("Failed to update .env: %s", e)
    logger.info(".env updated with best params")

    # === STEP 8: RL launch (async or blocking) ===
    if CONFIG.get("RUN_RL_AFTER_OPTIMIZATION", False):
        try:
            tmp_dir = Path(tempfile.gettempdir())
            tmp_params_file = tmp_dir / f"optimized_params_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

            # Save best parameters for RL training
            with open(tmp_params_file, "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2)

            # Launch RL either async or blocking
            if CONFIG.get("RL_SYNC_MODE", "async") == "async":
                pid, logfile = launch_rl_async(
                    CONFIG["RL_TRAIN_SCRIPT"],
                    str(tmp_params_file),  # ✅ convert Path → str
                    CONFIG["RL_TIMESTEPS"],
                    CONFIG["RL_LOG_DIR"]
                )
                logger.info("🤖 RL launched async PID=%s logfile=%s", pid, logfile)
            else:
                rc, logfile = launch_rl_blocking(
                    CONFIG["RL_TRAIN_SCRIPT"],
                    str(tmp_params_file),  # ✅ convert Path → str
                    CONFIG["RL_TIMESTEPS"],
                    CONFIG["RL_LOG_DIR"]
                )
                logger.info("🤖 RL finished rc=%s logfile=%s", rc, logfile)

        except Exception as e:
            logger.exception("RL sync failed: %s", e)


    # === STEP 9: Backtest summary ===
    try:
        params_obj = Params(**best_params)
    except Exception:
        # fallback: map keys manually and fill defaults if necessary
        params_obj = Params(
            use_percent_margin=best_params.get("use_percent_margin", False),
            margin_percent=best_params.get("margin_percent", 1.0),
            margin_usdt=best_params.get("margin_usdt", 3.0),
            atr_mult_tp1=best_params.get("atr_mult_tp1", best_params.get("atr_mult_tp1", 1.0)),
            atr_mult_tp2=best_params.get("atr_mult_tp2", best_params.get("atr_mult_tp2", 2.0)),
            atr_mult_sl=best_params.get("atr_mult_sl", 1.5),
            trailing_start_atr=best_params.get("trailing_start_atr", 1.5),
            trailing_step_atr=best_params.get("trailing_step_atr", 0.5),
            breakeven_atr=best_params.get("breakeven_atr", 1.0),
            breakeven_buffer_pts=best_params.get("breakeven_buffer_pts", 0.5),
            tp1_pct=best_params.get("tp1_pct", 0.5),
            tp2_pct=best_params.get("tp2_pct", 0.5),
            min_notional=best_params.get("min_notional", 0.0)
        )

    final_res = evaluate_fold_wrapper((klines, params_obj))
    final_trades = final_res.get("trades", [])
    backtest_out = {
        "timestamp": report_time,
        "symbol": symbol,
        "timeframe": tf,
        "num_trades": len(final_trades),
        "metrics": score_trades(final_trades),
    }
    with open("backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(backtest_out, f, indent=2)
    logger.info("Saved backtest_results.json")

    # Save equity curve if trades exist
    if final_trades:
        try:
            plot_equity_curve(final_trades, "optimization_report.png")
        except Exception as e:
            logger.warning("Failed to plot equity curve: %s", e)

    logger.info("=== Final Summary ===")
    logger.info("Trades: %d | Sharpe: %.3f | Drawdown: %.3f | Win%%: %.2f",
                len(final_trades),
                backtest_out["metrics"].get("sharpe_like", 0),
                backtest_out["metrics"].get("max_drawdown", 0),
                backtest_out["metrics"].get("win_rate", 0))
    logger.info("==============================")

    logger.info("Guard best parameters: %s", json.dumps({
        "SPREAD_MULT": best_params.get("SPREAD_MULT", 6.0),
        "IMBALANCE_LIMIT": best_params.get("IMBALANCE_LIMIT", 0.9),
        "BOOK_EVENT_LIMIT": best_params.get("BOOK_EVENT_LIMIT", 200),
        "TAKER_IMBAL_LIMIT": best_params.get("TAKER_IMBAL_LIMIT", 0.85),
    }, indent=2))


    return best_params

# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    client = BinanceClient()
    logger.info("Starting parameter optimization (async RL by default)...")
    # attempt to fetch live balance and persist to .env
    try:
        bal = client.get_futures_account_balance()
        if bal is not None:
            try:
                set_env_var("ACCOUNT_BALANCE", f"{float(bal):.8f}")
                config.ACCOUNT_BALANCE = float(bal)
                logger.info("Fetched & persisted ACCOUNT_BALANCE=%.8f", float(bal))
            except Exception:
                logger.warning("Could not persist ACCOUNT_BALANCE from client result.")
    except Exception as e:
        logger.warning("Balance fetch failed: %s", e)

    best = run_optuna_optimization(client, CONFIG["OPTUNA_TRIALS"])
    logger.info("Optimization finished. Best params summary exported.")
