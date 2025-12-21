# src/smart_exit.py
"""
SmartExitManager — Step4 FULL MERGE

This file is the patched, recommended SmartExit implementation (Option A).
It preserves your existing trailing/breakeven logic and adds:
 - Emergency ATR spike kill
 - Spread-aware SL widening
 - Break-even tightening (ATR-buffered)
 - exit_meta per-symbol state
 - monitor_position_and_adjust() and monitor_all_positions()

Drop this file into src/smart_exit.py and restart your bot. Test on dry-run/testnet first.
"""

import os
import math
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING


import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ----------------------------------------------------
# Logging Setup
# ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_debug_log_dir = "logs"
os.makedirs(_debug_log_dir, exist_ok=True)
_debug_log_path = os.path.join(
    _debug_log_dir,
    f"smartexit_debug_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log",
)

_debug_logger = logging.getLogger("smartexit_debug")
_debug_logger.setLevel(logging.DEBUG)
if not _debug_logger.handlers:
    fh = logging.FileHandler(_debug_log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _debug_logger.addHandler(fh)
    _debug_logger.propagate = False

if TYPE_CHECKING:
    from execution import ExecutionManager
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
    if not s:
        return fallback
    out: List[float] = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            logger.warning("Invalid float %r in CSV; using fallback", p)
            return fallback
    return out


# ----------------------------------------------------
# SmartExit Core Manager (Unified client APIs)
# ----------------------------------------------------
class SmartExitManager:
    """
    Hardened SmartExitManager that expects a unified client API.

    Required (recommended) client methods (best-effort fallbacks are used):
      - get_klines(symbol, interval, limit) -> list
      - get_symbol_info(symbol) -> dict
      - round_price(symbol, price) -> float
      - round_qty(symbol, qty) -> float
      - get_open_orders(symbol) -> list
      - cancel_order(symbol, order_id) -> Any
      - get_position(symbol) -> dict | None
      - futures_create_order(**kwargs) -> dict
      - update_stop_loss(symbol, side, qty, new_price) -> Any
      - ticker_price(symbol) -> dict | float
    """

    def __init__(self, client: Any):
        self.client = client
        load_dotenv(override=True)

        # ---------------------------------------------------------
        # Back-reference to ExecutionManager (wired later in bot.py)
        # Helps SmartExit reach position cache + money manager.
        # ---------------------------------------------------------
        self.execution: Optional["ExecutionManager"] = None

        # ---------------------------------------------------------
        # Core config
        # ---------------------------------------------------------
        try:
            self.use: bool = os.getenv("USE_SMART_EXIT", "true").lower() == "true"
        except Exception:
            self.use = True

        try:
            self.atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
        except Exception:
            self.atr_period = 14

        self.atr_mult_tp: List[float] = parse_csv_floats(
            os.getenv("ATR_MULT_TP", "2.0,3.0"), [2.0, 3.0]
        )
        self.atr_mult_sl: float = safe_float(os.getenv("ATR_MULT_SL", "1.9"), 1.9)
        self.trailing_start_atr: float = safe_float(
            os.getenv("TRAILING_START_ATR", "1.5"), 1.5
        )
        self.trailing_step_atr: float = safe_float(
            os.getenv("TRAILING_STEP_ATR", "0.25"), 0.25
        )
        self.breakeven_atr: float = safe_float(
            os.getenv("BREAKEVEN_ATR", "1.0"), 1.0
        )
        self.breakeven_buffer_pts: float = safe_float(
            os.getenv("BREAKEVEN_BUFFER_PTS", "0.03"), 0.03
        )

        # ---------------------------------------------------------
        # Partial TP
        # ---------------------------------------------------------
        self.tp_partial_sizes: List[float] = parse_csv_floats(
            os.getenv("TP_PARTIAL_SIZES", "0.5,0.5"), [0.5, 0.5]
        )
        self.atr_partial_tps: List[float] = parse_csv_floats(
            os.getenv("ATR_PARTIAL_TPS", "1.0,2.0"), [1.0, 2.0]
        )

        env_dry = os.getenv("DRY_RUN")
        self.dry_run: bool = bool(env_dry and str(env_dry).lower() in ("1", "true", "yes"))

        # ---------------------------------------------------------
        # Step 4: Advanced metadata store (per-symbol)
        # Used by spread-widen monitor & ATR-shock kill-switch
        # ---------------------------------------------------------
        self.exit_meta: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "SmartExit initialized: use=%s dry_run=%s",
            self.use,
            self.dry_run,
        )

    # -------------------------
    # Helpers for symbol info
    # -------------------------
    def _get_symbol_info_filters(self, symbol: str) -> Dict[str, float]:
        """
        Returns a dict: {"tick": float, "step": float, "min_qty": float, "min_notional": float}
        Safe defaults if anything fails.
        """
        tick = 0.0
        step = 0.0
        min_qty = 0.0
        min_notional = 0.0
        try:
            if not hasattr(self.client, "get_symbol_info"):
                return {"tick": tick, "step": step, "min_qty": min_qty, "min_notional": min_notional}

            info = self.client.get_symbol_info(symbol)
            if not isinstance(info, dict):
                return {"tick": tick, "step": step, "min_qty": min_qty, "min_notional": min_notional}

            filters = info.get("filters")
            if isinstance(filters, list):
                for f in filters:
                    if not isinstance(f, dict):
                        continue
                    ftype = f.get("filterType")
                    if ftype == "PRICE_FILTER":
                        tick = safe_float(f.get("tickSize"), tick)
                    elif ftype == "LOT_SIZE":
                        step = safe_float(f.get("stepSize"), step)
                        try:
                            min_qty = float(f.get("minQty", min_qty or 0.0))
                        except Exception:
                            min_qty = min_qty or 0.0
                    elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                        try:
                            min_notional = float(f.get("minNotional", f.get("notional", min_notional or 0.0)))
                        except Exception:
                            min_notional = min_notional or 0.0
        except Exception:
            _debug_logger.exception("_get_symbol_info_filters failed")
        return {"tick": tick, "step": step, "min_qty": min_qty or 0.0, "min_notional": min_notional or 0.0}

    # -------------------------
    # ATR Calculation
    # -------------------------
    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> Optional[float]:
        try:
            if df is None or df.shape[0] < 2:
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

    # -------------------------
    # Klines fetch & conversion
    # -------------------------
    def _fetch_klines(self, symbol: str, limit: int) -> Optional[List[Any]]:
        """
        Returns list of klines or None.
        Expects client.get_klines(symbol, interval, limit) to exist and return list-like.
        """
        try:
            if not hasattr(self.client, "get_klines"):
                return None
            resp = self.client.get_klines(symbol=symbol, interval="5m", limit=limit)
            if isinstance(resp, list):
                return resp
            return None
        except Exception:
            _debug_logger.exception("_fetch_klines error")
            return None

    def _klines_to_df(self, klines: List[Any]) -> Optional[pd.DataFrame]:
        """
        Accepts klines in standard futures list form (each entry list-like with at least 6 columns)
        or list of dicts with keys 'open','high','low','close','volume'.
        """
        try:
            if not klines:
                return None

            # If dict-like rows
            if isinstance(klines[0], dict):
                df = pd.DataFrame(klines)
                # ensure columns exist
                for col in ("open", "high", "low", "close", "volume"):
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                # assume list-like rows; take first 6 columns
                df = pd.DataFrame(klines)
                if df.shape[1] < 6:
                    return None
                df = df.iloc[:, :6]
                df.columns = ["open_time", "open", "high", "low", "close", "volume"]

            # ensure numeric
            df[["open", "high", "low", "close", "volume"]] = df[
                ["open", "high", "low", "close", "volume"]
            ].apply(pd.to_numeric, errors="coerce")
            df.dropna(subset=["close"], inplace=True)
            if df.empty:
                return None
            return df
        except Exception:
            _debug_logger.exception("_klines_to_df failure")
            return None

    # -------------------------
    # Unified position fetch
    # -------------------------
    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Return dict {'side': 'LONG'|'SHORT', 'qty': float, 'entry': float, 'sl': Optional[float]}
        Always safe; returns None if no meaningful position.
        """
        try:
            if not hasattr(self.client, "get_position"):
                return None
            pos = self.client.get_position(symbol)
            # Expect dict-like or None
            if not pos or not isinstance(pos, dict):
                return None

            # normalize keys safely
            qty = safe_float(
                pos.get("positionAmt") if isinstance(pos.get("positionAmt"), (int, float, str)) else pos.get("qty")
            )
            entry = safe_float(
                pos.get("entryPrice") if isinstance(pos.get("entryPrice"), (int, float, str)) else pos.get("avgPrice")
            )
            side: Optional[str] = None
            if qty > 0:
                side = "LONG"
            elif qty < 0:
                side = "SHORT"

            if side is None or abs(qty) <= 0:
                return None

            sl_val = None
            for k in ("stopPrice", "stop_price", "stopLoss", "stop_loss"):
                if k in pos and pos.get(k) not in (None, "", 0, "0"):
                    sl_val = safe_float(pos.get(k))
                    break

            return {"side": side, "qty": abs(qty), "entry": entry, "sl": sl_val, "mark_price": safe_float(pos.get("markPrice") or pos.get("price") or 0.0)}
        except Exception:
            _debug_logger.exception("_get_position failed")
            return None

    # -------------------------
    # Cancel all exit orders
    # -------------------------
    def _cancel_all_exit_orders(self, symbol: str) -> List[Any]:
        """
        PATCHED VERSION (Fix repeated 400 errors)
        ----------------------------------------
        • Cancels only exit orders
        • Treats 400 BAD_REQUEST or UNKNOWN_ORDER as normal
        • Removes order IDs from exit_meta so they are never retried
        • Silent for missing orders (prevents error spam)
        """
        cancelled: List[Any] = []

        try:
            # Must have list-open-orders
            if not hasattr(self.client, "get_open_orders"):
                return cancelled

            orders = self.client.get_open_orders(symbol)
            if not isinstance(orders, list):
                return cancelled

            for o in orders:
                if not isinstance(o, dict):
                    continue

                order_type = str(o.get("type") or o.get("orderType") or "").upper()
                reduce_only = bool(
                    o.get("reduceOnly")
                    or o.get("reduce_only")
                    or o.get("reduce")
                )

                # Identify TP/SL orders ONLY
                is_exit_type = (
                    "STOP" in order_type
                    or "TAKE" in order_type
                    or "TP" in order_type
                )

                if not (is_exit_type or reduce_only):
                    continue

                order_id = (
                    o.get("orderId")
                    or o.get("order_id")
                    or o.get("id")
                    or o.get("clientOrderId")
                )

                if not order_id:
                    continue

                # ---- TRY CANCEL ----
                cancelled_ok = False
                try:
                    self.client.cancel_order(symbol=symbol, orderId=order_id)
                    cancelled_ok = True

                except Exception as e:
                    msg = str(e).lower()

                    # ---- EXPECTED ERROR (order gone) ----
                    if ("unknown order" in msg
                        or "order does not exist" in msg
                        or "bad request" in msg):
                        _debug_logger.debug(
                            f"[SmartExit] Order {order_id} already removed on exchange."
                        )
                        cancelled_ok = False
                    else:
                        # real issue
                        _debug_logger.exception(
                            f"[SmartExit] cancel_order error for {symbol} id={order_id}"
                        )

                # ---- REMOVE ORDER FROM LOCAL CACHE ----
                if symbol in self.exit_meta:
                    if "orders" in self.exit_meta[symbol]:
                        try:
                            self.exit_meta[symbol]["orders"] = [
                                oid for oid in self.exit_meta[symbol]["orders"]
                                if str(oid) != str(order_id)
                            ]
                        except Exception:
                            pass

                if cancelled_ok:
                    cancelled.append(order_id)

            # ---- FINAL SAFETY CLEANUP ----
            if symbol in self.exit_meta:
                # remove any empty containers
                if "orders" in self.exit_meta[symbol] and not self.exit_meta[symbol]["orders"]:
                    self.exit_meta[symbol].pop("orders", None)

        except Exception:
            _debug_logger.exception("_cancel_all_exit_orders patched fatal error")

        return cancelled


    # -------------------------------------------------------------
    # Public wrappers so ExecutionManager can call them safely
    # -------------------------------------------------------------
    def cancel_all_exit_orders(self, symbol: str):
        """
        Public wrapper around _cancel_all_exit_orders().
        Exists so external modules (ExecutionManager) do not need
        to access private methods directly.
        """
        try:
            return self._cancel_all_exit_orders(symbol)
        except Exception:
            _debug_logger.exception("cancel_all_exit_orders wrapper failed for %s", symbol)
            return []

    def cancel_all_reduce_only(self, symbol: str):
        """
        Alias for readability. Some code uses reduce-only semantics.
        """
        return self.cancel_all_exit_orders(symbol)

    def _refresh_exit_orders_after_update(self, symbol: str, res: Any) -> None:
        """
        • Cancels remaining old SL/TP orders
        • Stores ONLY the latest stop-loss order ID
        • Prevents infinite cancel loops / 400 errors
        """
        try:
            # 1) Cancel all old exit orders (safe patched version)
            self._cancel_all_exit_orders(symbol)

            # 2) Extract new order ID
            new_order_id = None
            if isinstance(res, dict):
                new_order_id = (
                    res.get("orderId")
                    or res.get("id")
                    or res.get("clientOrderId")
                )

            # 3) Update exit_meta
            if symbol not in self.exit_meta:
                self.exit_meta[symbol] = {}

            self.exit_meta[symbol]["orders"] = []
            if new_order_id:
                self.exit_meta[symbol]["orders"].append(str(new_order_id))

        except Exception:
            _debug_logger.exception("Failed refreshing exit_meta after SL update")


    # -------------------------
    # Trailing / Breakeven handling (existing logic preserved)
    # -------------------------
    def _handle_trailing_and_breakeven(
        self, symbol: str, position: Dict[str, Any], last_price: float, atr: float
    ) -> Optional[Dict[str, Any]]:
        """
        PATCHED VERSION
        -----------------------------------------
        Adds:
        • Purge old exit orders after each SL update
        • Refresh exit_meta and store only new orderId
        • Prevents repeated cancel loops / 400 errors
        • Keeps original trailing + breakeven logic
        """
        try:
            side = position.get("side")
            entry = position.get("entry", 0.0)
            qty = position.get("qty", 0.0)
            cur_sl = position.get("sl")

            if side not in ("LONG", "SHORT") or qty <= 0:
                return None

            profit = abs(last_price - entry)
            updates: List[Dict[str, Any]] = []

            # -----------------------------
            # Trailing Stop
            # -----------------------------
            if profit >= self.trailing_start_atr * atr:
                proposed_trailing_sl = (
                    last_price - self.trailing_step_atr * atr if side == "LONG"
                    else last_price + self.trailing_step_atr * atr
                )

                # price rounding
                if hasattr(self.client, "round_price"):
                    proposed_trailing_sl = self.client.round_price(symbol, proposed_trailing_sl)
                else:
                    proposed_trailing_sl = round(proposed_trailing_sl, 8)

                improved = (
                    cur_sl is None
                    or (side == "LONG" and proposed_trailing_sl > cur_sl)
                    or (side == "SHORT" and proposed_trailing_sl < cur_sl)
                )

                if improved:
                    try:
                        if hasattr(self.client, "update_stop_loss"):

                            res = self.client.update_stop_loss(symbol, side, qty, proposed_trailing_sl)

                            updates.append({
                                "type": "trailing",
                                "symbol": symbol,
                                "old_sl": cur_sl,
                                "new_sl": proposed_trailing_sl,
                                "profit": profit,
                                "atr": atr,
                                "result": res
                            })

                            cur_sl = proposed_trailing_sl

                            # 🔥 NEW: Clean old orders + store new order ID
                            self._refresh_exit_orders_after_update(symbol, res)

                    except Exception:
                        _debug_logger.exception("Trailing SL update failed")

            # -----------------------------
            # Breakeven Stop
            # -----------------------------
            if profit >= self.breakeven_atr * atr:
                proposed_breakeven_sl = (
                    entry + self.breakeven_buffer_pts * atr if side == "LONG"
                    else entry - self.breakeven_buffer_pts * atr
                )

                # rounding
                if hasattr(self.client, "round_price"):
                    proposed_breakeven_sl = self.client.round_price(symbol, proposed_breakeven_sl)
                else:
                    proposed_breakeven_sl = round(proposed_breakeven_sl, 8)

                improved = (
                    cur_sl is None
                    or (side == "LONG" and proposed_breakeven_sl > cur_sl)
                    or (side == "SHORT" and proposed_breakeven_sl < cur_sl)
                )

                if improved:
                    try:
                        if hasattr(self.client, "update_stop_loss"):

                            res = self.client.update_stop_loss(symbol, side, qty, proposed_breakeven_sl)

                            updates.append({
                                "type": "breakeven",
                                "symbol": symbol,
                                "old_sl": cur_sl,
                                "new_sl": proposed_breakeven_sl,
                                "profit": profit,
                                "atr": atr,
                                "result": res
                            })

                            cur_sl = proposed_breakeven_sl

                            # 🔥 NEW: Clean old orders + store new order ID
                            self._refresh_exit_orders_after_update(symbol, res)

                    except Exception:
                        _debug_logger.exception("Breakeven SL update failed")

            # -----------------------------
            # return most recent action
            # -----------------------------
            if updates:
                return updates[-1]

            return None

        except Exception:
            _debug_logger.exception("_handle_trailing_and_breakeven failure")
            return None



    # -------------------------
    # Main periodic executor
    # -------------------------
    def manage_open_positions(self, symbol: str) -> Dict[str, Any]:
        """
        Periodic executor to enforce trailing/breakeven SLs.
        Returns a detailed dict describing action or noop.
        """
        try:
            # refresh env config only if changed
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
                try:
                    self._cancel_all_exit_orders(symbol)
                except Exception:
                    _debug_logger.exception("cancel all exit orders when no position failed")
                return {"type": "noop", "reason": "no_position"}

            # handle trailing + breakeven
            result = self._handle_trailing_and_breakeven(symbol, position, last_price, atr)
            if result:
                logger.info(
                    "[SmartExit] %s: %s updated SL %s -> %s (profit=%.4f, ATR=%.4f)",
                    symbol,
                    result.get("type"),
                    result.get("old_sl"),
                    result.get("new_sl"),
                    result.get("profit"),
                    result.get("atr")
                )
                return result

            return {"type": "noop", "reason": "no_update", "profit": abs(last_price - position["entry"]), "atr": atr}

        except Exception as e:
            _debug_logger.exception("manage_open_positions fatal error")
            return {"type": "error", "error": str(e)}


    # -------------------------
    # Step4 Helpers & Monitor (NEW)
    # -------------------------
    def _get_mark_price(self, symbol: str) -> Optional[float]:
        try:
            if hasattr(self.client, "ticker_price"):
                resp = self.client.ticker_price(symbol)
                return safe_float(resp.get("price") if isinstance(resp, dict) else resp)
        except Exception:
            return None
        return None

    def _compute_spread(self, symbol: str) -> Optional[float]:
        """
        Best-effort compute top-of-book spread (ask - bid) in price units.
        """
        try:
            ob = None
            if hasattr(self.client, "get_orderbook"):
                try:
                    ob = self.client.get_orderbook(symbol)
                except Exception:
                    ob = None
            if not ob and hasattr(self.client, "get_order_book"):
                try:
                    ob = self.client.get_order_book(symbol)
                except Exception:
                    ob = None
            if not ob and hasattr(self.client, "get_orderbook_raw"):
                try:
                    ob = self.client.get_orderbook_raw(symbol)
                except Exception:
                    ob = None

            # fallback to client.get_orderbook or None
            if not ob and hasattr(self.client, "get_klines"):
                return None

            if not ob:
                return None

            bid = None
            ask = None

            bids = ob.get("bids")
            asks = ob.get("asks")
            if isinstance(bids, (list, tuple)) and len(bids) > 0:
                first = bids[0]
                bid = float(first[0]) if isinstance(first, (list, tuple)) else safe_float(first)
            if isinstance(asks, (list, tuple)) and len(asks) > 0:
                first = asks[0]
                ask = float(first[0]) if isinstance(first, (list, tuple)) else safe_float(first)

            if bid is None:
                b = ob.get("bidPrice") or ob.get("bestBid")
                if b:
                    bid = safe_float(b)
            if ask is None:
                a = ob.get("askPrice") or ob.get("bestAsk")
                if a:
                    ask = safe_float(a)

            if bid is None or ask is None:
                return None
            return max(0.0, ask - bid)
        except Exception:
            _debug_logger.exception("_compute_spread failed")
            return None

    def _get_atr_baseline(self, symbol: str, samples: int = 200, atr_len: int = 14) -> Optional[float]:
        """
        Best-effort baseline ATR: average ATR over `samples` bars using existing calculate_atr implementation.
        """
        try:
            klines = self._fetch_klines(symbol, samples)
            if not klines:
                return None
            df = self._klines_to_df(klines)
            if df is None:
                return None
            # compute ATR rolling and take mean of last N ATR values
            window = atr_len
            tr = pd.concat(
                [
                    (df["high"] - df["low"]).abs(),
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_series = tr.rolling(window=window, min_periods=1).mean()
            if atr_series.empty:
                return None
            # baseline as median of the ATR series to be robust
            baseline = float(atr_series.median())
            return baseline if baseline > 0 else None
        except Exception:
            _debug_logger.exception("_get_atr_baseline failed")
            return None

    def _initial_sl_price(self, entry_price: float, side: str, atr: float, sl_mult: float, tick: float) -> float:
        """
        Compute initial SL based on ATR buffer. For LONG: SL = entry - atr*sl_mult.
        For SHORT: SL = entry + atr*sl_mult.
        Round using tick if provided.
        """
        try:
            if atr is None or atr <= 0:
                # fallback small percent
                buffer = entry_price * 0.005
            else:
                buffer = atr * float(sl_mult)

            if str(side).upper() in ("LONG", "BUY"):
                raw = entry_price - buffer
            else:
                raw = entry_price + buffer

            if tick and tick > 0:
                # round to tick
                try:
                    mult = 1.0 / tick
                    return float(round(round(raw * mult) / mult, 8))
                except Exception:
                    return float(round(raw, 8))
            return float(round(raw, 8))
        except Exception:
            return float(entry_price)

    # -------------------------
    # Create TP/SL after opening position
    # -------------------------
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
        Create initial Stop Loss (SL) and Take Profit (TP) orders immediately after opening a position.
        The method now also records exit_meta for monitors.
        """
        try:
            if not self.use:
                return {"status": "SKIPPED_SMART_EXIT_DISABLED"}

            is_long = side.upper() in ("LONG", "BUY")

            # --- Symbol filters ---
            filters = self._get_symbol_info_filters(symbol)
            tick = float(tick_size or filters.get("tick", 0.0))
            step = float(step_size or filters.get("step", 0.0))
            min_qty = float(filters.get("min_qty", 0.0))
            min_notional = float(filters.get("min_notional", 0.0))

            if qty <= 0:
                _debug_logger.debug("create_exit_orders: qty <= 0 for %s, skipping", symbol)
                return {"status": "SKIPPED_ZERO_QTY"}

            # determine baseline ATR
            baseline_atr = atr_value if atr_value and atr_value > 0 else None
            if baseline_atr is None:
                baseline_atr = self._get_atr_baseline(symbol)

            sl_mult = float(os.getenv("ATR_MULT_SL", str(self.atr_mult_sl)))
            initial_sl = self._initial_sl_price(entry_price, side, baseline_atr or 0.0, sl_mult, tick)

            # --- Take Profit levels ---
            if atr_value and not tp_levels:
                tp_levels = [self.round_price(symbol, entry_price + mult * atr_value if is_long else entry_price - mult * atr_value)
                            for mult in self.atr_mult_tp]
            elif tp_levels:
                tp_levels = [self.round_price(symbol, float(p)) for p in tp_levels]
            else:
                tp_levels = []

            # --- DRY RUN ---
            if self.dry_run:
                _debug_logger.info("[DRY_RUN] create_exit_orders: %s SL=%s TP=%s", symbol, initial_sl, tp_levels)
                # record meta for monitoring in DRY_RUN to enable tests
                self.exit_meta[symbol] = {
                    "baseline_atr": baseline_atr,
                    "entry_price": float(entry_price),
                    "breakeven_moved": False,
                    "last_sl": float(initial_sl),
                    "sl_mult": float(sl_mult),
                    "tick": tick,
                    "last_checked": time.time(),
                }
                return {"status": "DRY_RUN", "symbol": symbol, "sl": initial_sl, "tp_levels": tp_levels}

            # --- Quantity validation ---
            rounded_qty = self.round_qty(symbol, qty)
            if min_qty > 0 and rounded_qty < min_qty:
                return {"status": "FAILED_VALIDATION", "error": f"rounded qty {rounded_qty} < min_qty {min_qty}"}

            # --- Optional min_notional check ---
            mark_price: Optional[float] = None
            try:
                if hasattr(self.client, "ticker_price"):
                    price_data = self.client.ticker_price(symbol)
                    mark_price = safe_float(price_data.get("price") if isinstance(price_data, dict) else price_data)
            except Exception:
                mark_price = None

            if min_notional > 0 and mark_price and mark_price > 0:
                notional = rounded_qty * mark_price
                if notional < min_notional:
                    return {"status": "FAILED_VALIDATION", "error": f"notional {notional:.8f} < min_notional {min_notional}"}

            # --- Cancel existing exit orders ---
            try:
                self._cancel_all_exit_orders(symbol)
            except Exception:
                _debug_logger.exception("Could not cancel existing exit orders for %s", symbol)

            # --- Place Stop Loss (STOP_MARKET) ---
            sl_res: Any = None
            try:
                sl_res = self.client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if is_long else "BUY",
                    type="STOP_MARKET",
                    stopPrice=str(initial_sl),
                    quantity=rounded_qty,
                    reduceOnly=True,
                    workingType="CONTRACT_PRICE",
                )
            except Exception as e:
                _debug_logger.exception("Failed SL creation for %s: %s", symbol, e)
                sl_res = {"error": str(e)}

            # --- Place Take Profit orders ---
            tp_results: List[Any] = []
            if tp_levels:
                per_tp_qty_raw = qty / len(tp_levels)
                per_tp_qty = self.round_qty(symbol, per_tp_qty_raw)
                if min_qty > 0 and per_tp_qty < min_qty:
                    per_tp_qty = self.round_qty(symbol, min_qty)

                for tp in tp_levels:
                    try:
                        res = self.client.futures_create_order(
                            symbol=symbol,
                            side="SELL" if is_long else "BUY",
                            type="TAKE_PROFIT_MARKET",
                            stopPrice=str(tp),
                            quantity=per_tp_qty,
                            reduceOnly=True,
                            workingType="CONTRACT_PRICE",
                        )
                        tp_results.append(res)
                    except Exception as e:
                        _debug_logger.exception("Failed TP creation for %s at %s: %s", symbol, tp, e)
                        tp_results.append({"error": str(e)})

            # store meta for monitoring: baseline_atr, entry price, breakeven_moved flag
            try:
                self.exit_meta[symbol] = {
                    "baseline_atr": baseline_atr,
                    "entry_price": float(entry_price),
                    "breakeven_moved": False,
                    "last_sl": float(initial_sl),
                    "sl_mult": float(sl_mult),
                    "tick": tick,
                    "last_checked": time.time(),
                }
            except Exception:
                _debug_logger.exception("exit_meta store failed for %s", symbol)

            logger.info("[SmartExit] Exit orders created for %s: SL=%s TP=%s", symbol, initial_sl, tp_levels)

            return {
                "status": "OK",
                "symbol": symbol,
                "sl": initial_sl,
                "tp_levels": tp_levels,
                "sl_result": sl_res,
                "tp_results": tp_results,
                "qty": rounded_qty,
                "entry": entry_price,
                "side": side.upper(),
            }

        except Exception:
            _debug_logger.exception("create_exit_orders failure")
            return {"status": "ERROR", "error": "create_exit_orders unexpected failure"}


    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Always returns a float. Never passes None to float()."""
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default


    def _safe_atr(self, atr_value: Any) -> float:
        """Convert ATR return (float | None) into float safely."""
        if atr_value is None:
            return 0.0
        try:
            return float(atr_value)
        except Exception:
            return 0.0


    # -------------------------
    # Step4 Monitor functions
    # -------------------------
    def monitor_position_and_adjust(self, symbol: str) -> None:
        """
        Fully patched SmartExit engine.
        - Persistent exit_meta storage
        - Correct throttle
        - Correct last_sl retention
        - Pylance clean
        - Prevents repeated breakeven resets
        """
        try:
            # ============================================================
            # SAFE, PERSISTENT META INITIALIZATION (major fix)
            # ============================================================
            meta = self.exit_meta.setdefault(symbol, {
                "created": time.time(),
                "last_sl": 0.0,
                "breakeven_moved": False,
                "baseline_atr": 0.0,
                "entry_price": 0.0,
                "last_update_ts": 0.0,
            })

            # ------------------------------------------------------------
            # Position
            # ------------------------------------------------------------
            pos = self._get_position(symbol)
            if not pos:
                return

            qty = self._safe_float(pos.get("qty"))
            if qty <= 0:
                return

            side = str(pos.get("side") or "").upper()
            if side not in ("LONG", "SHORT", "BUY", "SELL"):
                return

            # ------------------------------------------------------------
            # Throttle (correct because meta is persistent)
            # ------------------------------------------------------------
            last_ts = self._safe_float(meta.get("last_update_ts"), 0.0)
            if (time.time() - last_ts) < 0.3:
                return

            # ------------------------------------------------------------
            # Entry price (persistent fallback)
            # ------------------------------------------------------------
            entry_price = self._safe_float(meta.get("entry_price") or pos.get("entry"))
            if entry_price <= 0:
                entry_price = self._safe_float(pos.get("entry"))
                meta["entry_price"] = entry_price
                self.exit_meta[symbol] = meta  # persist

            # ------------------------------------------------------------
            # Last SL (persistent)
            # ------------------------------------------------------------
            last_sl = self._safe_float(meta.get("last_sl"), 0.0)

            # ------------------------------------------------------------
            # Baseline ATR (persistent)
            # ------------------------------------------------------------
            baseline_atr = self._safe_float(meta.get("baseline_atr"), 0.0)
            if baseline_atr <= 0:
                baseline_atr = self._safe_float(self._get_atr_baseline(symbol))
                meta["baseline_atr"] = baseline_atr
                self.exit_meta[symbol] = meta

            # ------------------------------------------------------------
            # Current ATR
            # ------------------------------------------------------------
            try:
                klines = self._fetch_klines(symbol, max(self.atr_period + 5, 30))
                df = self._klines_to_df(klines) if klines else None
                raw_atr = self.calculate_atr(df, self.atr_period) if df is not None else 0.0
                current_atr = self._safe_atr(raw_atr)
            except Exception:
                current_atr = 0.0

            # ============================================================
            # 1) EMERGENCY ATR SPIKE EXIT
            # ============================================================
            try:
                em_mult = self._safe_float(os.getenv("EMERGENCY_ATR_MULT", "6.0"))

                if baseline_atr > 0 and current_atr > baseline_atr * em_mult:
                    logger.warning(
                        f"[SmartExit] ATR spike on {symbol}: {current_atr:.4f} vs {baseline_atr:.4f}"
                    )

                    exe = getattr(self, "execution", None)
                    close_fn = getattr(exe, "close_position", None)

                    if callable(close_fn):
                        try:
                            close_fn(symbol, side)
                        except Exception:
                            logger.exception(f"Emergency close failed for {symbol}")

                    meta["last_update_ts"] = time.time()
                    self.exit_meta[symbol] = meta
                    return

            except Exception:
                _debug_logger.exception(f"Emergency ATR failure for {symbol}")

            # ============================================================
            # 2) SPREAD-BASED SL WIDENING
            # ============================================================
            try:
                spread = self._safe_float(self._compute_spread(symbol))
                sp_mult = self._safe_float(os.getenv("SPREAD_WIDEN_ATR_MULT", "0.5"))
                widen_cap = self._safe_float(os.getenv("SL_WIDEN_MAX_MULT", "3.0"))

                if baseline_atr > 0 and spread > baseline_atr * sp_mult:

                    widen_factor = min(widen_cap, 1.0 + (spread / (baseline_atr + 1e-9)))
                    sl_mult = self._safe_float(meta.get("sl_mult") or os.getenv("ATR_MULT_SL", self.atr_mult_sl))

                    if side in ("LONG", "BUY"):
                        desired_sl = entry_price - baseline_atr * sl_mult * widen_factor
                    else:
                        desired_sl = entry_price + baseline_atr * sl_mult * widen_factor

                    try:
                        desired_sl = (
                            self.client.round_price(symbol, desired_sl)
                            if hasattr(self.client, "round_price")
                            else round(desired_sl, 8)
                        )
                    except Exception:
                        desired_sl = round(desired_sl, 8)

                    improved = (
                        (side in ("LONG", "BUY") and desired_sl < last_sl)
                        or (side in ("SHORT", "SELL") and desired_sl > last_sl)
                        or last_sl == 0
                    )

                    if improved:
                        try:
                            self._cancel_all_exit_orders(symbol)

                            self.client.futures_create_order(
                                symbol=symbol,
                                side="SELL" if side in ("LONG", "BUY") else "BUY",
                                type="STOP_MARKET",
                                stopPrice=str(desired_sl),
                                quantity=qty,
                                reduceOnly=True,
                                workingType="CONTRACT_PRICE",
                            )

                            meta["last_sl"] = desired_sl
                            meta["last_update_ts"] = time.time()
                            self.exit_meta[symbol] = meta

                            logger.info(f"[SmartExit] Spread-widen SL for {symbol}: {desired_sl}")

                        except Exception:
                            _debug_logger.exception(f"Spread widen SL failed for {symbol}")

            except Exception:
                _debug_logger.exception(f"Spread adjust failure for {symbol}")

            # ============================================================
            # 3) BREAK-EVEN / TIGHTENING
            # ============================================================
            try:
                last_price = self._safe_float(pos.get("mark_price") or self._get_mark_price(symbol))
            except Exception:
                last_price = 0.0

            if last_price <= 0:
                return

            # Profit calculation
            if side in ("LONG", "BUY"):
                profit_pct = (last_price - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - last_price) / entry_price * 100

            be_trigger = self._safe_float(os.getenv("BREAK_EVEN_PROFIT_PCT", "0.6"))

            # move to BE exactly one time
            if profit_pct >= be_trigger and not meta.get("breakeven_moved", False):

                buffer_mult = self._safe_float(os.getenv("SL_TIGHTEN_BUFFER_PCT", "0.2"))
                min_buffer_atr = self._safe_float(os.getenv("SL_MIN_BUFFER_ATR", "0.5"))

                baseline = baseline_atr if baseline_atr > 0 else self._safe_float(self._get_atr_baseline(symbol))
                buffer = max(baseline * buffer_mult, min_buffer_atr)

                if side in ("LONG", "BUY"):
                    desired_sl = entry_price + buffer
                    improved = desired_sl > last_sl
                else:
                    desired_sl = entry_price - buffer
                    improved = (desired_sl < last_sl) or last_sl == 0

                try:
                    desired_sl = (
                        self.client.round_price(symbol, desired_sl)
                        if hasattr(self.client, "round_price")
                        else round(desired_sl, 8)
                    )
                except Exception:
                    desired_sl = round(desired_sl, 8)

                if improved:
                    try:
                        self._cancel_all_exit_orders(symbol)

                        self.client.futures_create_order(
                            symbol=symbol,
                            side="SELL" if side in ("LONG", "BUY") else "BUY",
                            type="STOP_MARKET",
                            stopPrice=str(desired_sl),
                            quantity=qty,
                            reduceOnly=True,
                            workingType="CONTRACT_PRICE",
                        )

                        meta["last_sl"] = desired_sl
                        meta["breakeven_moved"] = True
                        meta["last_update_ts"] = time.time()
                        self.exit_meta[symbol] = meta

                        logger.info(
                            f"[SmartExit] BE tighten for {symbol}: {desired_sl} ({profit_pct:.3f}%)"
                        )

                    except Exception:
                        _debug_logger.exception(f"Break-even tighten failed for {symbol}")

        except Exception:
            _debug_logger.exception(f"monitor_position_and_adjust crashed for {symbol}")





    def monitor_all_positions(self) -> None:
        """
        Global Step-4 SmartExit scanner.
        Runs monitor_position_and_adjust() safely for every open symbol.
        """

        try:
            now = time.time()
            last_ts = getattr(self, "_last_monitor_all_ts", 0)

            # 200ms global throttle
            if (now - last_ts) < 0.2:
                return
            self._last_monitor_all_ts = now

            # --- SAFE execution.open_positions detection ---
            symbols = None
            exe = getattr(self, "execution", None)

            if exe is not None:
                open_pos = getattr(exe, "open_positions", None)
                if isinstance(open_pos, dict):
                    symbols = list(open_pos.keys())

            # fallback to exit_meta
            if symbols is None:
                symbols = list(self.exit_meta.keys())

            # --- iterate safely ---
            for sym in symbols:
                try:
                    if sym not in self.exit_meta:
                        self.exit_meta[sym] = {
                            "created": now,
                            "baseline_atr": self._get_atr_baseline(sym) or 0.0
                        }

                    self.monitor_position_and_adjust(sym)

                except Exception:
                    _debug_logger.exception(f"monitor_all_positions inner failed for {sym}")

        except Exception:
            _debug_logger.exception("monitor_all_positions failed")




    # -------------------------
    # Create TP/SL old method preserved (alias)
    # -------------------------
    # create_exit_orders is already the primary method above.

    # -------------------------
    # Small wrappers that delegate to the client if present (preserved)
    # -------------------------
    def round_price(self, symbol: str, price: float) -> float:
        try:
            if hasattr(self.client, "round_price"):
                return float(self.client.round_price(symbol, price))
            # naive fallback
            return float(round(price, 8))
        except Exception:
            _debug_logger.exception("round_price failed")
            return float(round(price, 8))

    def round_qty(self, symbol: str, qty: float) -> float:
        try:
            if hasattr(self.client, "round_qty"):
                return float(self.client.round_qty(symbol, qty))
            # naive fallback
            return float(round(qty, 6))
        except Exception:
            _debug_logger.exception("round_qty failed")
            return float(round(qty, 6))
