# -----------------------------
# File: src/smart_exit.py
# -----------------------------

import os
import math
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ----------------------------------------------------
# Logging Setup
# ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Debug logging to rotating files
_debug_log_dir = 'logs'
os.makedirs(_debug_log_dir, exist_ok=True)

_debug_log_path = os.path.join(
    _debug_log_dir,
    f"smartexit_debug_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
)

_debug_logger = logging.getLogger('smartexit_debug')
_debug_logger.setLevel(logging.DEBUG)

if not _debug_logger.handlers:
    fh = logging.FileHandler(_debug_log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    _debug_logger.addHandler(fh)
    _debug_logger.propagate = False


# ----------------------------------------------------
# Utility functions
# ----------------------------------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "null", "NaN"):
            return default
        return float(value)
    except Exception:
        return default


def parse_csv_floats(s: Optional[str], fallback: List[float]) -> List[float]:
    """Parse a CSV string of floats from env."""
    if not s:
        return fallback
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            logger.warning("Invalid float %r in CSV; using fallback", part)
            return fallback
    return out


# ----------------------------------------------------
# SmartExit Core Manager
# ----------------------------------------------------
class SmartExitManager:
    """
    Hardened SmartExit implementation fully aligned with hardened BinanceClient.

    Uses ONLY the unified, stable APIs provided by the hardened client:
      - get_klines()
      - get_symbol_info()
      - round_price()
      - round_qty()
      - get_open_orders()
      - cancel_order()
      - get_position()
      - futures_create_order()
      - update_stop_loss()
    """

    def __init__(self, client: Any):
        self.client = client

        # Load environment configuration
        load_dotenv(override=True)

        # Config settings
        self.use = os.getenv("USE_SMART_EXIT", "true").lower() == "true"
        self.atr_period = int(os.getenv("ATR_PERIOD", "14"))
        self.atr_mult_tp = parse_csv_floats(os.getenv("ATR_MULT_TP", "2.0,3.0"), [2.0, 3.0])
        self.atr_mult_sl = safe_float(os.getenv("ATR_MULT_SL", "1.9"), 1.9)

        self.trailing_start_atr = safe_float(os.getenv("TRAILING_START_ATR", "1.5"), 1.5)
        self.trailing_step_atr = safe_float(os.getenv("TRAILING_STEP_ATR", "0.25"), 0.25)

        self.breakeven_atr = safe_float(os.getenv("BREAKEVEN_ATR", "1.0"), 1.0)
        self.breakeven_buffer_pts = safe_float(os.getenv("BREAKEVEN_BUFFER_PTS", "0.03"), 0.03)

        # Partial TP configuration
        self.tp_partial_sizes = parse_csv_floats(os.getenv("TP_PARTIAL_SIZES", "0.5,0.5"), [0.5, 0.5])
        self.atr_partial_tps = parse_csv_floats(os.getenv("ATR_PARTIAL_TPS", "1.0,2.0"), [1.0, 2.0])

        # Dry-run override
        env_dry = os.getenv("DRY_RUN")
        self.dry_run = env_dry.lower() in ("1", "true", "yes") if env_dry else False

        logger.info("SmartExit initialized: use=%s dry_run=%s", self.use, self.dry_run)

    # ---------------------------------------------------------
    # ATR Calculation
    # ---------------------------------------------------------
    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> Optional[float]:
        try:
            if df.shape[0] < 2:
                return None

            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)

            prev_close = close.shift(1)

            tr = pd.concat(
                [
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

            atr = tr.rolling(window=window, min_periods=1).mean().iloc[-1]

            return float(atr) if not np.isnan(atr) else None
        except Exception:
            _debug_logger.exception("calculate_atr failure")
            return None

    # ---------------------------------------------------------
    # Klines fetch & DataFrame conversion
    # ---------------------------------------------------------
    def _fetch_klines(self, symbol: str, limit: int) -> Optional[List[Any]]:
        try:
            return self.client.get_klines(symbol=symbol, interval="5m", limit=limit)
        except Exception:
            _debug_logger.exception("_fetch_klines error")
            return None

    def _klines_to_df(self, klines: List[Any]) -> Optional[pd.DataFrame]:
        try:
            if not klines:
                return None

            # Standard futures klines
            df = pd.DataFrame(klines)
            df = df.iloc[:, :6]
            df.columns = ["open_time", "open", "high", "low", "close", "volume"]

            df[["open", "high", "low", "close", "volume"]] = df[
                ["open", "high", "low", "close", "volume"]
            ].apply(pd.to_numeric, errors="coerce")

            df.dropna(inplace=True)
            return df
        except Exception:
            _debug_logger.exception("_klines_to_df failure")
            return None

    # ---------------------------------------------------------
    # Clean unified position fetch
    # ---------------------------------------------------------
    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            pos = self.client.get_position(symbol)
            if not pos:
                return None

            qty = safe_float(pos.get("positionAmt") or pos.get("qty"))
            entry = safe_float(pos.get("entryPrice") or pos.get("avgPrice"))
            side = "LONG" if qty > 0 else "SHORT" if qty < 0 else None
            sl = safe_float(pos.get("stopPrice")) if pos.get("stopPrice") else None

            if side is None:
                return None

            return {"side": side, "qty": abs(qty), "entry": entry, "sl": sl}
        except Exception:
            _debug_logger.exception("_get_position failed")
            return None

    # ---------------------------------------------------------
    # Cancel all exit orders (TP/SL)
    # ---------------------------------------------------------
    def _cancel_all_exit_orders(self, symbol: str):
        """Cancel all TP/SL/reduceOnly orders for this symbol."""
        try:
            orders = self.client.get_open_orders(symbol)
        except Exception:
            _debug_logger.exception("get_open_orders failed")
            return []

        cancelled = []

        for o in orders:
            try:
                order_type = (o.get("type") or "").upper()
                reduce_only = bool(o.get("reduceOnly") or False)

                if "TAKE" in order_type or "STOP" in order_type or reduce_only:
                    oid = o.get("orderId")
                    try:
                        self.client.cancel_order(symbol=symbol, order_id=oid)
                        cancelled.append(oid)
                    except Exception:
                        _debug_logger.exception("cancel_order failed")
            except Exception:
                continue

        return cancelled

    # ---------------------------------------------------------
    # Trailing Stop / Breakeven logic
    # ---------------------------------------------------------
    def _handle_trailing_and_breakeven(
        self, symbol: str, position: Dict[str, Any], last_price: float, atr: float
    ) -> Optional[Dict[str, Any]]:
        side = position["side"]
        entry = position["entry"]
        qty = position["qty"]
        cur_sl = position.get("sl")

        profit = abs(last_price - entry)

        # Trailing SL
        if profit >= self.trailing_start_atr * atr:
            if side == "LONG":
                proposed_sl = last_price - self.trailing_step_atr * atr
            else:
                proposed_sl = last_price + self.trailing_step_atr * atr

            tick = self.client.round_price(symbol, 0.00000001)  # auto handled by client filters
            proposed_sl = self.client.round_price(symbol, proposed_sl)

            if cur_sl is None or (
                (side == "LONG" and proposed_sl > cur_sl)
                or (side == "SHORT" and proposed_sl < cur_sl)
            ):
                try:
                    res = self.client.update_stop_loss(
                        symbol, side, qty, proposed_sl
                    )
                    return {
                        "type": "trailing",
                        "symbol": symbol,
                        "new_sl": proposed_sl,
                        "result": res,
                    }
                except Exception:
                    _debug_logger.exception("Trailing SL update failed")

        # Breakeven SL
        if profit >= self.breakeven_atr * atr:
            if side == "LONG":
                proposed_sl = entry + self.breakeven_buffer_pts * atr
            else:
                proposed_sl = entry - self.breakeven_buffer_pts * atr

            proposed_sl = self.client.round_price(symbol, proposed_sl)

            if cur_sl is None or (
                (side == "LONG" and proposed_sl > cur_sl)
                or (side == "SHORT" and proposed_sl < cur_sl)
            ):
                try:
                    res = self.client.update_stop_loss(
                        symbol, side, qty, proposed_sl
                    )
                    return {
                        "type": "breakeven",
                        "symbol": symbol,
                        "new_sl": proposed_sl,
                        "result": res,
                    }
                except Exception:
                    _debug_logger.exception("Breakeven update failed")

        return None

    # ---------------------------------------------------------
    # Main periodic executor
    # ---------------------------------------------------------
    def manage_open_positions(self, symbol: str) -> Dict[str, Any]:
        try:
            load_dotenv(override=True)
            self.use = os.getenv("USE_SMART_EXIT", str(self.use)).lower() == "true"

            if not self.use:
                return {"type": "noop", "reason": "disabled"}

            klines = self._fetch_klines(symbol, self.atr_period + 5)
            if not klines:
                return {"type": "noop", "reason": "no_klines"}

            df = self._klines_to_df(klines)
            if df is None or df.empty:
                return {"type": "noop", "reason": "bad_klines"}

            atr = self.calculate_atr(df, self.atr_period)
            if not atr or atr <= 0:
                return {"type": "noop", "reason": "invalid_atr"}

            last_price = float(df["close"].iloc[-1])

            position = self._get_position(symbol)
            if not position:
                self._cancel_all_exit_orders(symbol)
                return {"type": "noop", "reason": "no_position"}

            # Trailing or breakeven
            result = self._handle_trailing_and_breakeven(symbol, position, last_price, atr)
            return result or {"type": "noop", "reason": "no_update"}

        except Exception as e:
            _debug_logger.exception("manage_open_positions fatal error")
            return {"type": "error", "error": str(e)}

    # ---------------------------------------------------------
    # Create TP/SL after opening position
    # ---------------------------------------------------------
    def create_exit_orders(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        atr_value: Optional[float] = None,
        tick_size: Optional[float] = None,
        step_size: Optional[float] = None,
        tp_levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Creates initial SL + TP orders immediately after opening a position.
        Hardened to use only unified client methods.
        """

        try:
            if not self.use:
                return {"status": "SKIPPED_SMART_EXIT_DISABLED"}

            is_long = side.upper() in ("LONG", "BUY")

            # Determine SL
            if atr_value:
                sl_raw = entry_price - self.atr_mult_sl * atr_value if is_long else entry_price + self.atr_mult_sl * atr_value
            else:
                sl_raw = entry_price * (0.997 if is_long else 1.003)

            sl_price = self.client.round_price(symbol, sl_raw)

            # Determine TP levels
            if atr_value and not tp_levels:
                tp_levels = [
                    self.client.round_price(symbol, entry_price + mult * atr_value if is_long else entry_price - mult * atr_value)
                    for mult in self.atr_mult_tp
                ]
            elif tp_levels:
                tp_levels = [self.client.round_price(symbol, p) for p in tp_levels]
            else:
                tp_levels = []

            # Dry-run
            if self.dry_run:
                return {
                    "status": "DRY_RUN",
                    "symbol": symbol,
                    "sl": sl_price,
                    "tp_levels": tp_levels,
                }

            # Cancel old orders
            self._cancel_all_exit_orders(symbol)

            # Create SL order
            stop_side = "SELL" if is_long else "BUY"
            sl_res = self.client.futures_create_order(
                symbol=symbol,
                side=stop_side,
                type="STOP_MARKET",
                stopPrice=str(sl_price),
                quantity=self.client.round_qty(symbol, qty),
                reduceOnly=True,
                workingType="CONTRACT_PRICE",
            )

            # Create TP orders
            tp_results = []
            if tp_levels:
                per_tp_qty = self.client.round_qty(symbol, qty / len(tp_levels))
                for tp in tp_levels:
                    tp_side = stop_side
                    res = self.client.futures_create_order(
                        symbol=symbol,
                        side=tp_side,
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=str(tp),
                        quantity=per_tp_qty,
                        reduceOnly=True,
                        workingType="CONTRACT_PRICE",
                    )
                    tp_results.append(res)

            return {
                "status": "OK",
                "symbol": symbol,
                "sl": sl_price,
                "tp_levels": tp_levels,
                "sl_result": sl_res,
                "tp_results": tp_results,
            }

        except Exception as e:
            _debug_logger.exception("create_exit_orders failure")
            return {"status": "ERROR", "error": str(e)}
