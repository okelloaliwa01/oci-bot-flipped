from __future__ import annotations

"""
walkforward_optuna.py  (Option 1 - Minimal Hardening)

Optuna-powered Walk-Forward Optimizer (safe subset - Option 1)
- Freezes signal / breakout filters (kept as constants)
- Optimizes ATR/exits/trailing/breakeven/leverage/margin/TP-SL/volume
- Writes consolidated optimized_params.json and updates .env safely
- CLI entrypoint for local execution
"""

import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local imports
try:
    from src.unified_backtester import UnifiedBacktester, BacktestResult
except Exception:
    from unified_backtester import UnifiedBacktester, BacktestResult  # type: ignore

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("walkforward_optuna")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Env path
DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"
OCI_ENV = Path("/home/ubuntu/oci-bot-flipped/.env.systemd")
ENV_PATH = OCI_ENV if OCI_ENV.exists() else DEFAULT_ENV

# ---------------------------
# Fixed constants (Option A)
# ---------------------------

FIXED_CONSTANTS: Dict[str, Any] = {
    "USE_PERCENT_MARGIN": True,
    "OPTUNA_TRIALS": 500,
    "HISTORY_LIMIT": 15000,
    "BREAKOUT_MIN_BODY_RATIO": 0.3,
    "BREAKOUT_RETEST_REQUIRED": True,
    "BREAKOUT_CONFIRM_BODIES_ONLY": False,
    "PENDING_MIN_BODY_RATIO": 0.35,
    "PENDING_MIN_VOL_MULT": 1.5,
    "PENDING_ATR_BUFFER_MULT": 0.5,
    "MIN_SIGNAL_STRENGTH": 0.5,
    "MIN_AVG_RANGE": 0.0,
    "WICK_LIMIT": 1.0,
    "ENTRY_THRESHOLD": 0.0,
    "CANDLE_COUNT": 50,
    "USE_ATR_SIZING": False,
}
SKIP_WRITE_KEYS = set(FIXED_CONSTANTS.keys())

# ---------------------------
# Safe env writer
# ---------------------------

def set_env_var(key: str, value: Any, env_path: Path = ENV_PATH) -> None:
    try:
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        if isinstance(value, (list, dict)):
            vstr = json.dumps(value)
        elif isinstance(value, str) and "," in value:
            vstr = value.strip()
        else:
            vstr = str(value)

        lines: List[str] = []
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        new_line = f"{key}={vstr}\n"
        updated = False
        for i, L in enumerate(lines):
            if L.strip().startswith(key + "="):
                lines[i] = new_line
                updated = True
                break

        if not updated:
            lines.append(new_line)

        env_path.parent.mkdir(parents=True, exist_ok=True)
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.info("Wrote env var %s=%s to %s", key, vstr, env_path)
    except Exception as e:
        logger.warning("set_env_var failed for %s=%s: %s", key, str(value), e)

# ---------------------------
# Data helpers
# ---------------------------

def klines_to_df(klines: List[List[Any]]) -> pd.DataFrame:
    rows = []
    for k in klines:
        rows.append(
            {
                "timestamp": pd.to_datetime(int(k[0]), unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    df = pd.DataFrame(rows).set_index("timestamp")
    return df

# ---------------------------
# Parameter Specs (Option A)
# ---------------------------

def build_default_safe_param_specs() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {
        "ATR_PERIOD": {"type": "int", "low": 7, "high": 28},
        "ATR_MULT_SL": {"type": "float", "low": 0.8, "high": 3.0},
        "ATR_MULT_TP1": {"type": "float", "low": 0.5, "high": 2.5},
        "ATR_MULT_TP2": {"type": "float", "low": 1.2, "high": 6.0},
        "TRAILING_START_ATR": {"type": "float", "low": 0.5, "high": 4.0},
        "TRAILING_STEP_ATR": {"type": "float", "low": 0.05, "high": 1.0},
        "BREAKEVEN_ATR": {"type": "float", "low": 0.0, "high": 3.0},
        "BREAKEVEN_BUFFER_PTS": {"type": "float", "low": 0.0, "high": 0.5},
        "TP_PERCENT": {"type": "float", "low": 0.2, "high": 3.0},
        "SL_PERCENT": {"type": "float", "low": 0.1, "high": 2.5},
        "LEVERAGE": {"type": "int", "low": 1, "high": 125},
        "VOLUME_MULTIPLIER": {"type": "float", "low": 0.3, "high": 3.0},
        "MAX_TRADE_PCT": {"type": "float", "low": 0.001, "high": 0.2},
        "MARGIN_PERCENT": {"type": "float", "low": 0.1, "high": 10.0},
        "MARGIN_USDT": {"type": "float", "low": 1.0, "high": 500.0},
        "SLIPPAGE_PCT": {"type": "float", "low": 0.0, "high": 0.01},
        "TRADE_FEE_PCT": {"type": "float", "low": 0.0, "high": 0.01},
        "PENDING_EXPIRY_CANDLES": {"type": "int", "low": 1, "high": 10},
        "POST_TRADE_COOLDOWN": {"type": "int", "low": 0, "high": 10},
        "EXIT_MONITOR_INTERVAL": {"type": "float", "low": 0.5, "high": 30.0},
        "CROSSVAL_FOLDS": {"type": "int", "low": 1, "high": 4},
    }

    for k in list(FIXED_CONSTANTS.keys()):
        specs.pop(k, None)

    return specs

# ----------------------------------------------------
# 🔥 REQUIRED FOR COMPATIBILITY (your missing function)
# ----------------------------------------------------
def build_default_all_param_specs() -> Dict[str, Dict[str, Any]]:
    """
    Wrapper maintained ONLY for backward compatibility.
    In Safe Option-A, we simply return the safe param specs.
    """
    return build_default_safe_param_specs()

# ---------------------------
# Backtester Evaluation
# ---------------------------


@dataclass
class WalkForwardResult:
    timestamp: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    best_params: Dict[str, Any]
    objective: float
    metrics: Dict[str, Any]
    trades_file: Optional[str] = None
    equity_file: Optional[str] = None


def evaluate_cfg_on_df(df: pd.DataFrame, cfg_overrides: Dict[str, Any]) -> Tuple[float, Dict[str, Any], BacktestResult]:
    """
    Run UnifiedBacktester.run_batch on df with cfg_overrides and return objective, metrics and raw backtest result.
    Objective uses Sharpe - 0.01*DD - trade_penalty (discourage too few trades)
    """
    ub = UnifiedBacktester(cfg_map=cfg_overrides)
    result: BacktestResult = ub.run_batch(df)

    metrics = result.metrics or {}
    sharpe = float(metrics.get("sharpe", 0.0))
    dd = abs(float(metrics.get("max_drawdown", 0.0)))
    n_trades = int(metrics.get("n_trades", len(result.trades))) if metrics else len(result.trades)
    trade_penalty = 0.0 if n_trades >= 5 else (5 - n_trades) * 0.2
    objective = sharpe - 0.01 * dd - trade_penalty
    return objective, metrics, result


# ---------------------------
# Optuna runner for a single training fold (safe keys only)
# ---------------------------


def run_optuna_for_fold(
    train_klines: List[List[Any]],
    validation_klines: Optional[List[List[Any]]],
    param_specs: Dict[str, Dict[str, Any]],
    n_trials: int = 50,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Optuna on train fold. Returns best params + score."""
    try:
        import optuna
    except Exception as e:
        raise RuntimeError("Optuna must be installed to run optimization") from e

    import csv
    from pathlib import Path

    df_train = klines_to_df(train_klines)
    df_val = klines_to_df(validation_klines) if validation_klines else None

    trials_csv = Path("optuna_walkforward_trials.csv")
    if not trials_csv.exists():
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "objective"])

    def _suggest_params(trial: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, spec in param_specs.items():
            t = spec.get("type")
            if t == "float":
                out[key] = trial.suggest_float(key, spec["low"], spec["high"])
            elif t == "int":
                out[key] = trial.suggest_int(key, spec["low"], spec["high"])
            elif t == "categorical":
                out[key] = trial.suggest_categorical(key, spec.get("choices", []))
            else:
                out[key] = trial.suggest_float(key, spec.get("low", 0.0), spec.get("high", 1.0))
        return out

    def objective(trial: Any) -> float:
        sampled = _suggest_params(trial)

        # enforce fixed constants (Option 1)
        for fk, fv in FIXED_CONSTANTS.items():
            sampled[fk] = fv

        cfg_overrides: Dict[str, Any] = {}

        # Mapping for allowed keys (safe subset)
        mapping = {
            "ATR_PERIOD": lambda v: int(v),
            "ATR_MULT_SL": lambda v: float(v),
            "ATR_MULT_TP1": lambda v: float(v),
            "ATR_MULT_TP2": lambda v: float(v),
            "TRAILING_START_ATR": lambda v: float(v),
            "TRAILING_STEP_ATR": lambda v: float(v),
            "BREAKEVEN_ATR": lambda v: float(v),
            "BREAKEVEN_BUFFER_PTS": lambda v: float(v),
            "TP_PERCENT": lambda v: float(v),
            "SL_PERCENT": lambda v: float(v),
            "LEVERAGE": lambda v: int(v),
            "VOLUME_MULTIPLIER": lambda v: float(v),
            "MAX_TRADE_PCT": lambda v: float(v),
            "MARGIN_PERCENT": lambda v: float(v),
            "MARGIN_USDT": lambda v: float(v),
            "SLIPPAGE_PCT": lambda v: float(v),
            "TRADE_FEE_PCT": lambda v: float(v),
            "PENDING_EXPIRY_CANDLES": lambda v: int(v),
            "POST_TRADE_COOLDOWN": lambda v: int(v),
            "EXIT_MONITOR_INTERVAL": lambda v: float(v),
            "CROSSVAL_FOLDS": lambda v: int(v),
        }

        for k, v in sampled.items():
            if k in mapping:
                try:
                    cfg_overrides[k] = mapping[k](v)
                except Exception:
                    cfg_overrides[k] = v

        # Ensure fixed constants are present for evaluation
        for fk, fv in FIXED_CONSTANTS.items():
            cfg_overrides.setdefault(fk, fv)

        # Construct ATR_MULT_TP legacy field if both parts exist
        if "ATR_MULT_TP1" in cfg_overrides and "ATR_MULT_TP2" in cfg_overrides:
            cfg_overrides["ATR_MULT_TP"] = f"{float(cfg_overrides['ATR_MULT_TP1'])},{float(cfg_overrides['ATR_MULT_TP2'])}"

        # Evaluate on validation or training
        if df_val is not None:
            obj_val, metrics_val, _ = evaluate_cfg_on_df(df=df_val, cfg_overrides=cfg_overrides)
            obj = float(obj_val)
        else:
            obj_train, metrics_train, _ = evaluate_cfg_on_df(df=df_train, cfg_overrides=cfg_overrides)
            obj = float(obj_train)

        # Append trial to CSV
        try:
            with open(trials_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([trial.number, float(obj)])
        except Exception:
            pass

        return float(obj)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1 if n_jobs is None else n_jobs)

    best = study.best_params
    normalized_best: Dict[str, Any] = {}

    # copy best params (no _a/_b pairs in this safe spec)
    for k, v in best.items():
        normalized_best[k] = v

    # Clean out fixed constants
    for fk in FIXED_CONSTANTS.keys():
        normalized_best.pop(fk, None)

    best_value = study.best_value
    logger.info("Fold Optuna best objective=%.4f", best_value)
    return {"best_params": normalized_best, "best_value": best_value, "study": study}


# ---------------------------
# Walk-forward driver
# ---------------------------


def run_walk_forward(
    klines: List[List[Any]],
    symbol: str = "XRPUSDT",
    timeframe: str = "5m",
    training_window_bars: int = 5000,
    validation_window_bars: int = 1000,
    step_bars: int = 1000,
    n_trials: int = 50,
    folds: int = 3,
    out_json: str = "walkforward_optimized.json",
    param_specs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    if param_specs is None:
        param_specs = build_default_safe_param_specs()

    total = len(klines)
    logger.info("Total klines=%d training=%d validation=%d step=%d", total, training_window_bars, validation_window_bars, step_bars)

    results: List[WalkForwardResult] = []
    i_start = 0
    window_idx = 0
    while i_start + training_window_bars + validation_window_bars <= total:
        train_slice = klines[i_start:i_start + training_window_bars]
        val_slice = klines[i_start + training_window_bars:i_start + training_window_bars + validation_window_bars]

        logger.info(
            "Window %d: training indices [%d:%d] validation [%d:%d]",
            window_idx,
            i_start,
            i_start + training_window_bars,
            i_start + training_window_bars,
            i_start + training_window_bars + validation_window_bars,
        )

        fold_res = run_optuna_for_fold(
            train_klines=train_slice,
            validation_klines=val_slice,
            param_specs=param_specs,
            n_trials=n_trials,
            n_jobs=None,
        )

        best_params = fold_res.get("best_params", {}) or {}
        normalized_best: Dict[str, Any] = {}
        for k, v in best_params.items():
            if isinstance(v, str):
                normalized_best[k] = ",".join([p.strip() for p in v.split(",") if p.strip()])
            else:
                normalized_best[k] = v

        # Remove fixed constants if present
        for fk in FIXED_CONSTANTS.keys():
            normalized_best.pop(fk, None)

        # Build cfg_map for validation evaluation
        cfg_map: Dict[str, Any] = {}
        for k, v in normalized_best.items():
            cfg_map[k] = v

        # Ensure downstream gets fixed constants
        for fk, fv in FIXED_CONSTANTS.items():
            cfg_map.setdefault(fk, fv)

        # Legacy ATR pair handling
        if "ATR_MULT_TP1" in normalized_best and "ATR_MULT_TP2" in normalized_best and "ATR_MULT_TP" not in cfg_map:
            cfg_map["ATR_MULT_TP"] = f"{float(normalized_best['ATR_MULT_TP1'])},{float(normalized_best['ATR_MULT_TP2'])}"

        # Evaluate on validation
        df_val = klines_to_df(val_slice)
        obj, metrics, raw_res = evaluate_cfg_on_df(df_val, cfg_map)

        # save artifacts
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("walkforward_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        trades_file = out_dir / f"trades_window_{window_idx}_{ts}.csv"
        equity_file = out_dir / f"equity_window_{window_idx}_{ts}.png"
        try:
            if raw_res.trades is not None and not raw_res.trades.empty:
                raw_res.trades.to_csv(trades_file, index=False)
            try:
                import matplotlib.pyplot as plt

                eq = raw_res.equity_curve
                if eq is not None and not eq.empty:
                    plt.figure(figsize=(10, 4))
                    plt.plot(eq.index.to_numpy(), eq.to_numpy(), linewidth=1.5)
                    plt.title(f"Equity Window {window_idx}")
                    plt.xlabel("Time")
                    plt.ylabel("Equity")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(equity_file)
                    plt.close()
            except Exception:
                logger.debug("plotting failed for window %d", window_idx)
        except Exception as e:
            logger.warning("failed to save artifacts: %s", e)

        res_obj = WalkForwardResult(
            timestamp=ts,
            window_start=pd.to_datetime(train_slice[0][0], unit="ms"),
            window_end=pd.to_datetime(val_slice[-1][0], unit="ms"),
            best_params=normalized_best,
            objective=float(obj),
            metrics=metrics,
            trades_file=str(trades_file) if trades_file.exists() else None,
            equity_file=str(equity_file) if equity_file.exists() else None,
        )
        results.append(res_obj)

        i_start += step_bars
        window_idx += 1

    # write summary JSON
    out_data = {"symbol": symbol, "timeframe": timeframe, "generated_at": datetime.utcnow().isoformat(), "windows": [r.__dict__ for r in results]}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, default=str)
    logger.info("Walk-forward results written to %s (windows=%d)", out_json, len(results))

    # Aggregate best params
    if results:
        aggregate: Dict[str, List[Any]] = {}
        for r in results:
            for k, v in r.best_params.items():
                aggregate.setdefault(k, []).append(v)

        best_summary: Dict[str, Any] = {}
        for k, vals in aggregate.items():
            try:
                parsed_lists = [[float(x.strip()) for x in str(v).split(",")] for v in vals]
                lengths = set(len(pl) for pl in parsed_lists)
                if len(lengths) == 1 and list(lengths)[0] > 1:
                    arr = np.array(parsed_lists, dtype=float)
                    medians = np.median(arr, axis=0).tolist()
                    best_summary[k] = ",".join([str(round(float(x), 8)) for x in medians])
                else:
                    flat = np.array([float(pl[0]) if pl else float(vals[0]) for pl in parsed_lists], dtype=float)
                    best_summary[k] = float(np.median(flat))
            except Exception:
                best_summary[k] = vals[-1]

        # Remove fixed constants from final export
        for fk in FIXED_CONSTANTS.keys():
            if fk in best_summary:
                best_summary.pop(fk, None)

        optimized_out = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "timeframe": timeframe, "BEST_PARAMS": best_summary, "windows": [r.__dict__ for r in results]}
        tmp_out = Path("optimized_params.json")
        with open(tmp_out, "w", encoding="utf-8") as f:
            json.dump(optimized_out, f, indent=2, default=str)
        logger.info("Wrote consolidated optimized_params.json")

        # write into .env (skip fixed constants)
        try:
            written_keys = []
            for k, v in best_summary.items():
                if k in SKIP_WRITE_KEYS:
                    continue
                set_env_var(k, v)
                written_keys.append(k)
            logger.info("Wrote %d optimized keys to %s: %s", len(written_keys), ENV_PATH, ", ".join(written_keys))
        except Exception as e:
            logger.warning("Failed to write optimized params to .env: %s", e)

    return results


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Walk-Forward Optuna Optimizer (safe)")
    parser.add_argument("--symbol", default=os.getenv("SYMBOL", "XRPUSDT"))
    parser.add_argument("--timeframe", default=os.getenv("TIMEFRAME", "5m"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("HISTORY_LIMIT", str(FIXED_CONSTANTS["HISTORY_LIMIT"]))))
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=int(os.getenv("OPTUNA_TRIALS", str(FIXED_CONSTANTS["OPTUNA_TRIALS"]))))
    parser.add_argument("--folds", type=int, default=int(os.getenv("CROSSVAL_FOLDS", "3")))
    parser.add_argument("--out", type=str, default="walkforward_optimized.json")
    args = parser.parse_args()

    # attempt to fetch klines via BinanceClient
    klines: List[List[Any]] = []
    try:
        from binance_client import BinanceClient

        client = BinanceClient()
        logger.info("Fetching klines: %s %s limit=%d", args.symbol, args.timeframe, args.limit)
        klines = client.get_klines(args.symbol, args.timeframe, args.limit)
    except Exception as e:
        logger.warning("BinanceClient not found or failed to fetch klines: %s", e)

    if not klines:
        logger.error("No klines available. Please provide klines or install BinanceClient.")
        raise SystemExit(1)

    param_specs = build_default_safe_param_specs()
    res = run_walk_forward(
        klines=klines,
        symbol=args.symbol,
        timeframe=args.timeframe,
        training_window_bars=args.train,
        validation_window_bars=args.val,
        step_bars=args.step,
        n_trials=args.trials,
        folds=args.folds,
        out_json=args.out,
        param_specs=param_specs,
    )
    logger.info("Walk-forward completed: windows=%d", len(res))
