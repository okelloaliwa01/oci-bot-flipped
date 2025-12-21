"""
config.py
---------
Unified configuration loader for the Binance Trading System.
Supports .env, optimized_params.json, and runtime reload.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Any

# ================================================================
# 🔹 Environment variable helper functions
# ================================================================
def env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key, default)
    return v


def env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def env_list_float(key: str, default: Optional[List[float]] = None, sep: str = ",") -> List[float]:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default[:] if default is not None else []
    try:
        parts = [p.strip() for p in raw.split(sep) if p.strip()]
        return [float(p) for p in parts]
    except Exception:
        return default[:] if default is not None else []


def env_list_str(key: str, default: Optional[List[str]] = None, sep: str = ",") -> List[str]:
    raw = os.getenv(key)
    if raw is None:
        return default[:] if default is not None else []
    parts = [p.strip() for p in raw.split(sep) if p.strip()]
    return parts if parts else (default[:] if default is not None else [])

# ================================================================
# 🔹 Load .env and paths
# ================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")
OPT_PATH = os.path.join(BASE_DIR, "optimized_params.json")

if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# ================================================================
# 🔹 Core Trading Config
# ================================================================
SYMBOL: str = env_str("SYMBOL", "XRPUSDT") or "XRPUSDT"
TIMEFRAME: str = env_str("TIMEFRAME", "5m") or "5m"
LEVERAGE: int = env_int("LEVERAGE", 10)

# Margin & Position sizing
MARGIN_USDT: float = env_float("MARGIN_USDT", 3.0)
USE_PERCENT_MARGIN: bool = env_bool("USE_PERCENT_MARGIN", False)
MARGIN_PERCENT: float = env_float("MARGIN_PERCENT", 1.0)

TP_PERCENT: float = env_float("TP_PERCENT", 0.30)
SL_PERCENT: float = env_float("SL_PERCENT", 0.15)
VOLUME_MULTIPLIER: float = env_float("VOLUME_MULTIPLIER", 1.3)

DRY_RUN: bool = env_bool("DRY_RUN", True)
USE_TESTNET: bool = env_bool("USE_TESTNET", True)
CANDLE_COUNT: int = env_int("CANDLE_COUNT", 21)

LOG_FILE: str = env_str("LOG_FILE", "bot.log") or "bot.log"
ACCOUNT_BALANCE: float = env_float("ACCOUNT_BALANCE", 1000.0)

# ================================================================
# 🔹 API & Telegram
# ================================================================
TELEGRAM_BOT_TOKEN: str = env_str("TELEGRAM_BOT_TOKEN", "") or ""
TELEGRAM_CHAT_ID: str = env_str("TELEGRAM_CHAT_ID", "") or ""
API_KEY: Optional[str] = env_str("BINANCE_API_KEY", None)
API_SECRET: Optional[str] = env_str("BINANCE_API_SECRET", None)

# ================================================================
# 🔹 Multi-Timeframe Confirmation
# ================================================================
_mtf_raw: str = (env_str("MTF_CONFIRMATION", "") or "").strip()
MTF_CONFIRMATION: List[str] = [t.strip() for t in _mtf_raw.split(",") if t.strip()] if _mtf_raw else []
_mtf_req_raw: str = (env_str("MTF_REQUIRED_CONFIRM", "all") or "all").strip()
MTF_REQUIRED_CONFIRM: Any = int(_mtf_req_raw) if _mtf_req_raw.isdigit() else _mtf_req_raw

# ================================================================
# 🔹 Smart Exit (ATR, trailing, breakeven, partial TP)
# ================================================================
USE_SMART_EXIT: bool = env_bool("USE_SMART_EXIT", True)
ATR_PERIOD: int = env_int("ATR_PERIOD", 14)

ATR_MULT_TP: List[float] = env_list_float("ATR_MULT_TP", [1.0, 2.0, 3.0])
ATR_MULT_SL: float = env_float("ATR_MULT_SL", 1.5)

TRAILING_START_ATR: float = env_float("TRAILING_START_ATR", 1.5)
TRAILING_STEP_ATR: float = env_float("TRAILING_STEP_ATR", 0.5)
BREAKEVEN_ATR: float = env_float("BREAKEVEN_ATR", 1.0)
BREAKEVEN_BUFFER_PTS: float = env_float("BREAKEVEN_BUFFER_PTS", 0.5)
PARTIAL_TAKE_PROFIT_PCT: float = env_float("PARTIAL_TAKE_PROFIT_PCT", 0.5)

# Partial TP Handling
_atr_partial_raw = env_str("ATR_PARTIAL_TPS", "")
_tp_partial_sizes_raw = env_str("TP_PARTIAL_SIZES", "")

ATR_PARTIAL_TPS = [float(x) for x in _atr_partial_raw.split(",") if x.strip()] if _atr_partial_raw else [ATR_MULT_TP[0]]
TP_PARTIAL_SIZES = [float(x) for x in _tp_partial_sizes_raw.split(",") if x.strip()] if _tp_partial_sizes_raw else [PARTIAL_TAKE_PROFIT_PCT]

# normalize partial TP size count
if len(TP_PARTIAL_SIZES) == 1 and len(ATR_PARTIAL_TPS) > 1:
    first = TP_PARTIAL_SIZES[0]
    remaining = max(0.0, 1.0 - first)
    per = remaining / (len(ATR_PARTIAL_TPS) - 1)
    TP_PARTIAL_SIZES = [first] + [per] * (len(ATR_PARTIAL_TPS) - 1)
elif len(TP_PARTIAL_SIZES) != len(ATR_PARTIAL_TPS):
    TP_PARTIAL_SIZES = (TP_PARTIAL_SIZES + [0.0] * len(ATR_PARTIAL_TPS))[:len(ATR_PARTIAL_TPS)]

EXIT_MONITOR_INTERVAL: float = env_float("EXIT_MONITOR_INTERVAL", 3.0)
MAX_MARGIN_USDT: float = env_float("MAX_MARGIN_USDT", 100.0)

# ================================================================
# 🔹 Breakout Confirmation / Anti-Fakeout
# ================================================================
PENDING_EXPIRY_CANDLES: int = env_int("PENDING_EXPIRY_CANDLES", 3)
PENDING_MAX_DISTANCE: float = env_float("PENDING_MAX_DISTANCE", 0.0035)
POST_TRADE_COOLDOWN: int = env_int("POST_TRADE_COOLDOWN", 1)

PENDING_MIN_BODY_RATIO: float = env_float("PENDING_MIN_BODY_RATIO", 0.55)
PENDING_MIN_VOL_MULT: float = env_float("PENDING_MIN_VOL_MULT", 1.4)
PENDING_ATR_BUFFER_MULT: float = env_float("PENDING_ATR_BUFFER_MULT", 0.8)

BREAKOUT_CONFIRM_BODIES_ONLY: bool = env_bool("BREAKOUT_CONFIRM_BODIES_ONLY", True)
BREAKOUT_MIN_BODY_RATIO: float = env_float("BREAKOUT_MIN_BODY_RATIO", 0.4)
BREAKOUT_RETEST_REQUIRED: bool = env_bool("BREAKOUT_RETEST_REQUIRED", True)

# ================================================================
# 🔹 Auto-load optimized parameters
# ================================================================
def maybe_load_optimized_params() -> bool:
    """Auto-load optimized_params.json if newer than .env"""
    if not os.path.exists(OPT_PATH):
        return False

    env_mtime = os.path.getmtime(ENV_PATH) if os.path.exists(ENV_PATH) else 0
    opt_mtime = os.path.getmtime(OPT_PATH)
    if opt_mtime <= env_mtime:
        print("optimized_params.json exists but is older — skipping load.")
        return False

    try:
        with open(OPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("BEST_PARAMS", {})
        if not params:
            return False

        print(f"Auto-loading optimized parameters (updated {datetime.fromtimestamp(opt_mtime)})")

        globals().update({
            "USE_PERCENT_MARGIN": bool(params.get("use_percent_margin", USE_PERCENT_MARGIN)),
            "MARGIN_PERCENT": float(params.get("margin_percent", MARGIN_PERCENT)),
            "MARGIN_USDT": float(params.get("margin_usdt", MARGIN_USDT)),
            "ATR_MULT_TP": [float(params.get("atr_mult_tp1", ATR_MULT_TP[0])),
                            float(params.get("atr_mult_tp2", ATR_MULT_TP[1] if len(ATR_MULT_TP) > 1 else ATR_MULT_TP[0]))],
            "ATR_MULT_SL": float(params.get("atr_mult_sl", ATR_MULT_SL)),
            "TRAILING_START_ATR": float(params.get("trailing_start_atr", TRAILING_START_ATR)),
            "TRAILING_STEP_ATR": float(params.get("trailing_step_atr", TRAILING_STEP_ATR)),
            "BREAKEVEN_ATR": float(params.get("breakeven_atr", BREAKEVEN_ATR)),
            "BREAKEVEN_BUFFER_PTS": float(params.get("breakeven_buffer_pts", BREAKEVEN_BUFFER_PTS)),
        })
        print("✅ Optimized parameters loaded successfully.")
        return True
    except Exception as e:
        print(f"⚠️ Failed to load optimized_params.json: {e}")
        return False


maybe_load_optimized_params()

# ================================================================
# 🔹 Config Printing Helper
# ================================================================
def print_active_config() -> None:
    print("\n=== ACTIVE CONFIGURATION ===")
    for key in [
        "USE_TESTNET", "DRY_RUN", "SYMBOL", "TIMEFRAME", "LEVERAGE",
        "USE_PERCENT_MARGIN", "MARGIN_PERCENT", "MARGIN_USDT",
        "ATR_MULT_TP", "ATR_MULT_SL", "TRAILING_START_ATR",
        "TRAILING_STEP_ATR", "BREAKEVEN_ATR", "BREAKEVEN_BUFFER_PTS",
        "TP_PARTIAL_SIZES", "ATR_PARTIAL_TPS", "TP_PERCENT", "SL_PERCENT",
        "CANDLE_COUNT", "VOLUME_MULTIPLIER", "LOG_FILE"
    ]:
        print(f"{key}: {globals().get(key)}")
    print("=============================\n")

# ================================================================
# 🔹 Reload config at runtime
# ================================================================
def reload_config() -> None:
    """Reload .env and optimized_params.json dynamically."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    maybe_load_optimized_params()
    print("🔁 Config reloaded from .env and optimized_params.json")


# ================================================================
# 🔹 CLI Run
# ================================================================
if __name__ == "__main__":
    print_active_config()
    print(f"BINANCE_API_KEY loaded: {'Yes' if API_KEY else 'No'}")
    print(f"BINANCE_API_SECRET loaded: {'Yes' if API_SECRET else 'No'}")
