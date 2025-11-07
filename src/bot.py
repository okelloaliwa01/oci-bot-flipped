# src/bot.py
import os
import time
import signal
import datetime
import numpy as np
import pandas as pd
import traceback
import logging
from logging.handlers import RotatingFileHandler

from config_utils import (
    load_config,
    reload_config,
    get_int,
    get_float,
)
from binance_client import BinanceClient
from data_fetch import fetch_closed_candles
from breakout_logic import check_breakout
from execution import ExecutionManager
from logger import get_logger, get_startup_logger
from alerts import send_telegram
from mtf_filter import confirm_mtf_direction


# ============================================
# ⚙️ Load Config
# ============================================
CFG = load_config()

USE_TESTNET = CFG["USE_TESTNET"]
DRY_RUN = CFG["DRY_RUN"]
SYMBOL = CFG["SYMBOL"]
TIMEFRAME = CFG["TIMEFRAME"]
LEVERAGE = CFG["LEVERAGE"]
MARGIN_USDT = CFG["MARGIN_USDT"]
TP_PERCENT = CFG["TP_PERCENT"]
SL_PERCENT = CFG["SL_PERCENT"]
VOLUME_MULTIPLIER = CFG["VOLUME_MULTIPLIER"]
CANDLE_COUNT = CFG["CANDLE_COUNT"]

# MTF
MTF_CONFIRMATION = CFG["MTF_CONFIRMATION"]
MTF_REQUIRED_CONFIRM = CFG["MTF_REQUIRED_CONFIRM"]

# Smart Exit
USE_SMART_EXIT = CFG["USE_SMART_EXIT"]
ATR_MULT_TP = CFG["ATR_MULT_TP"]
ATR_MULT_SL = CFG["ATR_MULT_SL"]
TRAILING_START_ATR = CFG["TRAILING_START_ATR"]
TRAILING_STEP_ATR = CFG["TRAILING_STEP_ATR"]
BREAKEVEN_ATR = CFG["BREAKEVEN_ATR"]
BREAKEVEN_BUFFER_PTS = CFG["BREAKEVEN_BUFFER_PTS"]

# Pending Breakout Config
PENDING_EXPIRY_CANDLES = CFG["PENDING_EXPIRY_CANDLES"]
PENDING_MAX_DISTANCE = CFG["PENDING_MAX_DISTANCE"]
POST_TRADE_COOLDOWN = CFG["POST_TRADE_COOLDOWN"]
PENDING_MIN_BODY_RATIO = CFG["PENDING_MIN_BODY_RATIO"]
PENDING_MIN_VOL_MULT = CFG["PENDING_MIN_VOL_MULT"]
PENDING_ATR_BUFFER_MULT = CFG["PENDING_ATR_BUFFER_MULT"]

# ============================================
# 🪵 Logger Setup (use existing logger factory + extra debug.log)
# ============================================
logger = get_logger("bot")
startup_logger = get_startup_logger()

# ensure logger captures DEBUG locally
try:
    logger.setLevel(logging.DEBUG)
except Exception:
    pass

# Add separate rotating debug log file (debug.log) for all DEBUG-level messages
try:
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    os.makedirs(log_dir, exist_ok=True)
    debug_log_path = os.path.join(log_dir, "debug.log")
    debug_handler = RotatingFileHandler(debug_log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    debug_handler.setFormatter(debug_formatter)
    # attach to both bot logger and startup logger (they may be same or different)
    logger.addHandler(debug_handler)
    startup_logger.addHandler(debug_handler)
    logger.debug("✅ Debug file handler attached: %s", debug_log_path)
except Exception as e:
    # If logging setup fails, print to stdout (do not raise)
    print("⚠️ Failed to create debug log handler:", e)


# ============================================
# 🧠 Log Startup
# ============================================
def log_startup_parameters():
    session_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    startup_logger.info("=" * 60)
    startup_logger.info(f"Bot Startup - {session_time}")
    startup_logger.info("=" * 60)
    for k, v in CFG.items():
        startup_logger.info(f"{k}: {v}")


# ============================================
# ⏱️ Candle Sync Helper
# ============================================
def align_to_candle_close(tf: str) -> float:
    try:
        unit = tf[-1]
        val = int(tf[:-1])
        secs = val * 60 if unit == "m" else val * 3600 if unit == "h" else 60
        return secs - (time.time() % secs)
    except Exception:
        return 60.0


# ============================================
# 📊 ATR Calculator
# ============================================
def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


# ============================================
# 🛑 Graceful Shutdown
# ============================================
STOP = False


def _signal_handler(sig, frame):
    global STOP
    logger.info("🛑 Shutdown signal received — finishing current candle.")
    STOP = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================
# 🤖 Main Bot Loop
# ============================================
def run():
    log_startup_parameters()
    print("\n=== ACTIVE CONFIG ===")
    for k, v in CFG.items():
        print(f"{k}: {v}")
    print("=============================")

    client = BinanceClient(use_testnet=USE_TESTNET)
    exec_m = ExecutionManager(client)
    try:
        client.set_leverage(SYMBOL, LEVERAGE)
    except Exception as e:
        logger.warning("Failed to set leverage at startup: %s", e)
        logger.debug(traceback.format_exc())

    logger.info(f"🚀 Bot started (DRY_RUN={DRY_RUN}, TESTNET={USE_TESTNET})")

    try:
        exec_m.reconcile_open_positions()
        msg = "♻️ System restarted — open positions restored."
        logger.info(msg)
        send_telegram(msg)
    except Exception as e:
        logger.warning(f"⚠️ Reconciliation failed: {e}")
        logger.debug(traceback.format_exc())
        send_telegram(f"⚠️ Reconciliation skipped: {e}")

    pending_breakout = None
    cooldown_candles = 0

    while not STOP:
        try:
            sleep_time = align_to_candle_close(TIMEFRAME)
            logger.info(f"⏳ Sleeping {sleep_time:.1f}s until candle close")
            time.sleep(sleep_time + 1)

            # === Check open position
            try:
                has_pos = client.has_open_position(SYMBOL)
            except Exception as e:
                logger.warning(f"Position check failed: {e}")
                logger.debug(traceback.format_exc())
                has_pos = False

            # === Smart Exit Management ===
            if has_pos and USE_SMART_EXIT:
                try:
                    result = exec_m.manage_open_positions(symbol=SYMBOL)
                    if result and isinstance(result, dict):
                        logger.info(f"📊 SmartExit Update: {result}")
                except Exception as e:
                    logger.warning(f"SmartExit update failed: {e}")
                    logger.debug(traceback.format_exc())

            # === Reload Config If Idle ===
            if not has_pos:
                try:
                    reload_config()
                    logger.info("🔄 Parameters reloaded from config.")
                except Exception as e:
                    logger.warning("Failed to reload config: %s", e)
                    logger.debug(traceback.format_exc())

            # === Fetch Market Data ===
            df = fetch_closed_candles(SYMBOL, TIMEFRAME, CANDLE_COUNT, client=client)
            if df is None or df.shape[0] < 21 or df.isnull().any().any():
                logger.warning("⚠️ Candle data insufficient or invalid.")
                continue

            atr_series = df.get("atr", compute_atr(df))
            current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0
            last = df.iloc[-1]

            # ========================================
            # 🕵️ Deep Debug Logging (Context)
            # ========================================
            def log_ctx(ctx: dict, header="Context Dump"):
                try:
                    if not isinstance(ctx, dict):
                        logger.debug(f"🔍 {header}: (non-dict) {ctx}")
                        return
                    clean_ctx = {}
                    for k, v in ctx.items():
                        try:
                            # convert numpy / numpy types to python floats for readability
                            if hasattr(v, "item"):
                                clean_ctx[k] = v.item()
                            else:
                                clean_ctx[k] = float(v) if isinstance(v, (np.floating, float, int)) else v
                        except Exception:
                            clean_ctx[k] = v
                    logger.debug(f"🔍 {header}: {clean_ctx}")
                except Exception as err:
                    logger.warning(f"⚠️ Context logging error: {err}")
                    logger.debug(traceback.format_exc())

            # ========================================
            # 🔁 Pending Breakout Check
            # ========================================
            if pending_breakout:
                pending_breakout["candles_waited"] += 1
                level = pending_breakout.get("level")
                if not level:
                    pending_breakout = None
                    continue

                ptype = pending_breakout.get("type", "LONG")
                ctx = pending_breakout.get("ctx", {}) or {}
                log_ctx(ctx, f"Pending {ptype} Context")

                body = abs(last["close"] - last["open"])
                rng = max(last["high"] - last["low"], 1e-9)
                body_ratio = body / rng
                avg_vol = float(df["volume"].iloc[:-1].mean())
                atr_buf = PENDING_ATR_BUFFER_MULT * current_atr
                vol_req = avg_vol * PENDING_MIN_VOL_MULT

                close_ok = last["close"] > (level + atr_buf) if ptype == "LONG" else last["close"] < (level - atr_buf)
                vol_ok = last["volume"] >= vol_req
                body_ok = body_ratio >= PENDING_MIN_BODY_RATIO
                dist_pct = abs(last["close"] - level) / level if level else float("inf")

                logger.info(f"🧩 Pending {ptype} Check → close_ok={close_ok}, vol_ok={vol_ok}, body_ok={body_ok}, dist_pct={dist_pct:.4f}")

                if dist_pct > PENDING_MAX_DISTANCE:
                    send_telegram(f"❌ Pending breakout invalidated for {ptype} ({SYMBOL})")
                    pending_breakout = None
                    continue

                if close_ok and vol_ok and body_ok:
                    logger.info(f"✅ Breakout confirmed for {ptype} @ {level:.4f}")
                    send_telegram(f"✅ {SYMBOL} {ptype} breakout confirmed @ {level:.4f}")

                    mtf_ok = True
                    if MTF_CONFIRMATION:
                        try:
                            mtf_ok = confirm_mtf_direction(client, SYMBOL, ptype, MTF_CONFIRMATION)
                        except Exception as e:
                            logger.warning(f"MTF recheck failed: {e}")
                            logger.debug(traceback.format_exc())
                            mtf_ok = False

                    if not mtf_ok:
                        pending_breakout = None
                        continue

                    if not has_pos:
                        try:
                            logger.info(f"🚀 Attempting to place {ptype} order (pending confirmed).")
                            # Debug: log pre-order inputs
                            logger.debug(
                                "Order inputs -> symbol=%s, type=%s, margin_usdt=%s, tp_percent=%s, sl_percent=%s",
                                SYMBOL, ptype, MARGIN_USDT, TP_PERCENT, SL_PERCENT,
                            )
                            res = exec_m.open_position(SYMBOL, ptype, MARGIN_USDT, TP_PERCENT, SL_PERCENT)
                            logger.info(f"✅ Trade Result: {res}")
                            send_telegram(f"🚀 {ptype} entry executed: {res}")
                            cooldown_candles = POST_TRADE_COOLDOWN
                        except Exception as e:
                            logger.error(f"❌ Order placement failed: {e}")
                            logger.debug(traceback.format_exc())
                    pending_breakout = None
                    continue

                if pending_breakout["candles_waited"] >= PENDING_EXPIRY_CANDLES:
                    send_telegram(f"⏳ Pending breakout expired for {ptype}")
                    pending_breakout = None

            # === Cooldown ===
            if cooldown_candles > 0:
                cooldown_candles -= 1
                logger.info(f"⏸️ Cooldown active: {cooldown_candles} candles remaining.")
                continue

            # === Breakout Signal Detection ===
            try:
                signal_side, level, ctx = check_breakout(df.tail(21), volume_multiplier=VOLUME_MULTIPLIER)
            except Exception as e:
                logger.warning(f"check_breakout() failed: {e}")
                logger.debug(traceback.format_exc())
                continue

            log_ctx(ctx, "Signal Context")

            reason = ctx.get("reason", "N/A") if isinstance(ctx, dict) else "N/A"
            if level is not None and isinstance(level, (float, int, np.floating)):
                logger.info(f"📢 Signal={signal_side}, Level={float(level):.4f}, reason={reason}")
            else:
                logger.info(f"📢 Signal={signal_side}, Level={level}, reason={reason}")


            if not signal_side or level is None:
                logger.debug("No valid breakout signal found this candle.")
                continue

            # === PENDING_CONFIRM ===
            if signal_side == "PENDING_CONFIRM":
                reason = ctx.get("reason", "Unspecified") if isinstance(ctx, dict) else "Unspecified"
                logger.info(f"🕒 Pending confirmation (reason: {reason})")
                if not pending_breakout and not has_pos:
                    pending_breakout = {
                        "type": ctx.get("type", "LONG") if isinstance(ctx, dict) else "LONG",
                        "level": float(level),
                        "ctx": ctx,
                        "created_at_ts": datetime.datetime.utcnow(),
                        "candles_waited": 0,
                    }
                    send_telegram(f"⏳ {SYMBOL} {pending_breakout['type']} pending @ {level:.4f} — {reason}")
                    logger.info(f"Pending breakout stored: {pending_breakout}")
                continue

            # === DIRECT ENTRY (LONG / SHORT) ===
            if signal_side in ("LONG", "SHORT"):
                if has_pos:
                    logger.info(f"⚠️ Position already open, ignoring {signal_side} signal.")
                    continue

                mtf_ok = True
                if MTF_CONFIRMATION:
                    try:
                        mtf_ok = confirm_mtf_direction(client, SYMBOL, signal_side, MTF_CONFIRMATION)
                    except Exception as e:
                        logger.warning(f"MTF check failed: {e}")
                        logger.debug(traceback.format_exc())
                        mtf_ok = False
                if not mtf_ok:
                    logger.info("MTF filter rejected this signal.")
                    continue

                logger.info(f"🚀 Attempting to open {signal_side} position @ {level:.2f}")
                try:
                    # Debug: log pre-order inputs
                    logger.debug(
                        "Order inputs -> symbol=%s, side=%s, margin_usdt=%s, tp_percent=%s, sl_percent=%s",
                        SYMBOL, signal_side, MARGIN_USDT, TP_PERCENT, SL_PERCENT,
                    )
                    res = exec_m.open_position(SYMBOL, signal_side, MARGIN_USDT, TP_PERCENT, SL_PERCENT)
                    logger.info(f"✅ Trade opened successfully: {res}")
                    send_telegram(f"🚀 {SYMBOL} {signal_side} position opened: {res}")
                    cooldown_candles = POST_TRADE_COOLDOWN
                except Exception as e:
                    logger.error(f"❌ Trade open failed: {e}")
                    logger.debug(traceback.format_exc())

        except Exception as e:
            logger.exception(f"Unhandled runtime error: {e}")
            logger.debug(traceback.format_exc())
            send_telegram(f"⚠️ Runtime error: {e}")
            time.sleep(5)

    logger.info("✅ Bot stopped cleanly.")


# ============================================
# 🚀 Entry Point
# ============================================
if __name__ == "__main__":
    run()
