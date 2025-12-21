"""
src/config_utils.py
----------------
Utility for loading and reloading trading bot configuration dynamically.

Features:
✅ Dynamic reload of config.py without restarting the bot.
✅ Safe environment variable overrides.
✅ Default fallbacks for all parameters.
✅ Simple integration with src/bot.py.
"""

import importlib
import os
import sys
import logging

# --------------------------------------------
# Logger setup
# --------------------------------------------
logger = logging.getLogger("config_utils")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --------------------------------------------
# 🔄 Dynamic Config Reload
# --------------------------------------------
def reload_config():
    """Reload config.py dynamically at runtime."""
    try:
        import config
        importlib.reload(config)
        logger.info("✅ Config reloaded successfully.")
    except Exception as e:
        logger.error(f"⚠️ Failed to reload config: {e}")


# --------------------------------------------
# 🧩 Safe Value Getters
# --------------------------------------------
def _get_value(key, default=None):
    """Return environment override if exists, else use config.py or default."""
    try:
        import config
        return getattr(config, key, os.getenv(key, default))
    except ImportError:
        return os.getenv(key, default)


def get_int(key, default=0):
    val = _get_value(key, default)
    try:
        return int(str(val))
    except (TypeError, ValueError):
        logger.warning(f"⚠️ Invalid int for {key}={val!r}, using default {default}")
        return default


def get_float(key, default=0.0):
    val = _get_value(key, default)
    try:
        return float(str(val))
    except (TypeError, ValueError):
        logger.warning(f"⚠️ Invalid float for {key}={val!r}, using default {default}")
        return default


def get_bool(key, default=False):
    val = str(_get_value(key, default)).lower()
    return val in ("true", "1", "yes", "y", "t")


def get_str(key, default=""):
    val = _get_value(key, default)
    return str(val) if val is not None else default


def get_list(key, default=None):
    val = _get_value(key, default)
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return default or []


# --------------------------------------------
# ⚙️ Unified Config Loader
# --------------------------------------------
def load_config():
    """Load config parameters from config.py or environment."""
    cfg = {
        # Core
        "USE_TESTNET": get_bool("USE_TESTNET", True),
        "DRY_RUN": get_bool("DRY_RUN", True),
        "SYMBOL": get_str("SYMBOL", "BTCUSDT"),
        "TIMEFRAME": get_str("TIMEFRAME", "5m"),
        "LEVERAGE": get_int("LEVERAGE", 50),
        "MARGIN_USDT": get_float("MARGIN_USDT", 1.0),
        "TP_PERCENT": get_float("TP_PERCENT", 0.3),
        "SL_PERCENT": get_float("SL_PERCENT", 0.15),
        "VOLUME_MULTIPLIER": get_float("VOLUME_MULTIPLIER", 1.5),
        "CANDLE_COUNT": get_int("CANDLE_COUNT", 100),

        # MTF
        "MTF_CONFIRMATION": get_list("MTF_CONFIRMATION", []),
        "MTF_REQUIRED_CONFIRM": get_int("MTF_REQUIRED_CONFIRM", 1),

        # Smart Exit
        "USE_SMART_EXIT": get_bool("USE_SMART_EXIT", True),
        #"ATR_MULT_TP": get_float("ATR_MULT_TP", 2.0),
        "ATR_MULT_TP": get_list("ATR_MULT_TP", ["2.0", "3.0", "4.0"]),
        "ATR_MULT_SL": get_float("ATR_MULT_SL", 1.0),
        "TRAILING_START_ATR": get_float("TRAILING_START_ATR", 1.5),
        "TRAILING_STEP_ATR": get_float("TRAILING_STEP_ATR", 0.5),
        "BREAKEVEN_ATR": get_float("BREAKEVEN_ATR", 1.0),
        "BREAKEVEN_BUFFER_PTS": get_float("BREAKEVEN_BUFFER_PTS", 50),

        # Pending Breakout Config
        "PENDING_EXPIRY_CANDLES": get_int("PENDING_EXPIRY_CANDLES", 3),
        "PENDING_MAX_DISTANCE": get_float("PENDING_MAX_DISTANCE", 0.0035),
        "POST_TRADE_COOLDOWN": get_int("POST_TRADE_COOLDOWN", 1),
        "PENDING_MIN_BODY_RATIO": get_float("PENDING_MIN_BODY_RATIO", 0.5),
        "PENDING_MIN_VOL_MULT": get_float("PENDING_MIN_VOL_MULT", 1.2),
        "PENDING_ATR_BUFFER_MULT": get_float("PENDING_ATR_BUFFER_MULT", 0.1),
    }

    return cfg


# --------------------------------------------
# 🧠 Manual Test
# --------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print("=== CONFIG VALUES ===")
    for k, v in cfg.items():
        print(f"{k}: {v}")
