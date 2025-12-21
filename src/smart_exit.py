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

    Required client methods:
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

        # config
        try:
            self.use: bool = os.getenv("USE_SMART_EXIT", "true").lower() == "true"
        except Exception:
            self.use = True

        try:
            self.atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
        except Exception:
            self.atr_period = 14

        self.atr_mult_tp: List[float] = parse_csv_floats(os.getenv("ATR_MULT_TP", "2.0,3.0"), [2.0, 3.0])
        self.atr_mult_sl: float = safe_float(os.getenv("ATR_MULT_SL", "1.9"), 1.9)
        self.trailing_start_atr: float = safe_float(os.getenv("TRAILING_START_ATR", "1.5"), 1.5)
        self.trailing_step_atr: float = safe_float(os.getenv("TRAILING_STEP_ATR", "0.25"), 0.25)
        self.breakeven_atr: float = safe_float(os.getenv("BREAKEVEN_ATR", "1.0"), 1.0)
        self.breakeven_buffer_pts: float = safe_float(os.getenv("BREAKEVEN_BUFFER_PTS", "0.03"), 0.03)

        # partial TP
        self.tp_partial_sizes: List[float] = parse_csv_floats(os.getenv("TP_PARTIAL_SIZES", "0.5,0.5"), [0.5, 0.5])
        self.atr_partial_tps: List[float] = parse_csv_floats(os.getenv("ATR_PARTIAL_TPS", "1.0,2.0"), [1.0, 2.0])

        env_dry = os.getenv("DRY_RUN")
        self.dry_run: bool = bool(env_dry and str(env_dry).lower() in ("1", "true", "yes"))

        logger.info("SmartExit initialized: use=%s dry_run=%s", self.use, self.dry_run)

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

            return {"side": side, "qty": abs(qty), "entry": entry, "sl": sl_val}
        except Exception:
            _debug_logger.exception("_get_position failed")
            return None

    # -------------------------
    # Cancel all exit orders
    # -------------------------
    def _cancel_all_exit_orders(self, symbol: str) -> List[Any]:
        """
        Cancel TP/SL/reduceOnly orders for the symbol.
        Expects: client.get_open_orders(symbol) -> list of dicts with keys 'type','reduceOnly','orderId'
                 client.cancel_order(symbol=symbol, order_id=orderId)
        """
        cancelled: List[Any] = []
        try:
            if not hasattr(self.client, "get_open_orders"):
                return cancelled

            orders = self.client.get_open_orders(symbol)
            if not isinstance(orders, list):
                return cancelled

            for o in orders:
                try:
                    if not isinstance(o, dict):
                        continue
                    order_type = str(o.get("type") or o.get("orderType") or "").upper()
                    reduce_only = bool(o.get("reduceOnly") or o.get("reduce_only") or False)
                    if "TAKE" in order_type or "STOP" in order_type or reduce_only:
                        oid = o.get("orderId") or o.get("order_id") or o.get("id") or o.get("clientOrderId")
                        if not oid:
                            # no id we can cancel individually; attempt symbol-wide cancel if available
                            try:
                                if hasattr(self.client, "futures_cancel_all_open_orders"):
                                    self.client.futures_cancel_all_open_orders(symbol=symbol)
                                    cancelled.append({"cancel_all": True, "symbol": symbol})
                            except Exception:
                                _debug_logger.exception("cancel-all variant failed for %s", symbol)
                            continue
                        # try unified cancel_order signature
                        try:
                            self.client.cancel_order(symbol=symbol, order_id=oid)
                            cancelled.append(oid)
                        except Exception:
                            # last resort: try passing orderId name
                            try:
                                self.client.cancel_order(symbol=symbol, orderId=oid)  # some clients accept this
                                cancelled.append(oid)
                            except Exception:
                                _debug_logger.exception("cancel_order failed for %s %s", symbol, oid)
                except Exception:
                    continue
        except Exception:
            _debug_logger.exception("_cancel_all_exit_orders failed")
        return cancelled

    # -------------------------
    # Trailing / Breakeven handling
    # -------------------------
    def _handle_trailing_and_breakeven(
        self, symbol: str, position: Dict[str, Any], last_price: float, atr: float
    ) -> Optional[Dict[str, Any]]:
        try:
            side = position.get("side")
            entry = position.get("entry", 0.0)
            qty = position.get("qty", 0.0)
            cur_sl = position.get("sl")

            if side not in ("LONG", "SHORT"):
                return None

            profit = abs(float(last_price) - float(entry))

            # trailing
            if profit >= (self.trailing_start_atr * atr):
                if side == "LONG":
                    proposed_sl_raw = float(last_price) - self.trailing_step_atr * atr
                else:
                    proposed_sl_raw = float(last_price) + self.trailing_step_atr * atr

                if hasattr(self.client, "round_price"):
                    proposed_sl = self.client.round_price(symbol, proposed_sl_raw)
                else:
                    proposed_sl = float(round(proposed_sl_raw, 8))

                improved = False
                if cur_sl is None:
                    improved = True
                else:
                    if side == "LONG" and proposed_sl > float(cur_sl):
                        improved = True
                    if side == "SHORT" and proposed_sl < float(cur_sl):
                        improved = True

                if improved:
                    try:
                        if hasattr(self.client, "update_stop_loss"):
                            res = self.client.update_stop_loss(symbol, side, qty, proposed_sl)
                            return {"type": "trailing", "symbol": symbol, "new_sl": proposed_sl, "result": res}
                    except Exception:
                        _debug_logger.exception("Trailing SL update failed")

            # breakeven
            if profit >= (self.breakeven_atr * atr):
                if side == "LONG":
                    proposed_sl_raw = float(entry) + self.breakeven_buffer_pts * atr
                else:
                    proposed_sl_raw = float(entry) - self.breakeven_buffer_pts * atr

                if hasattr(self.client, "round_price"):
                    proposed_sl = self.client.round_price(symbol, proposed_sl_raw)
                else:
                    proposed_sl = float(round(proposed_sl_raw, 8))

                improved = False
                if cur_sl is None:
                    improved = True
                else:
                    if side == "LONG" and proposed_sl > float(cur_sl):
                        improved = True
                    if side == "SHORT" and proposed_sl < float(cur_sl):
                        improved = True

                if improved:
                    try:
                        if hasattr(self.client, "update_stop_loss"):
                            res = self.client.update_stop_loss(symbol, side, qty, proposed_sl)
                            return {"type": "breakeven", "symbol": symbol, "new_sl": proposed_sl, "result": res}
                    except Exception:
                        _debug_logger.exception("Breakeven update failed")

            return None
        except Exception:
            _debug_logger.exception("_handle_trailing_and_breakeven failure")
            return None

    # -------------------------
    # Main periodic executor
    # -------------------------
    def manage_open_positions(self, symbol: str) -> Dict[str, Any]:
        """
        Called periodically to enforce trailing/breakeven SLs.
        Returns a dict describing action or noop.
        """
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
                # no position: cancel any leftover exit orders for safety
                try:
                    self._cancel_all_exit_orders(symbol)
                except Exception:
                    _debug_logger.exception("cancel all exit orders when no position failed")
                return {"type": "noop", "reason": "no_position"}

            result = self._handle_trailing_and_breakeven(symbol, position, last_price, atr)
            return result or {"type": "noop", "reason": "no_update"}
        except Exception as e:
            _debug_logger.exception("manage_open_positions fatal error")
            return {"type": "error", "error": str(e)}

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
        Create initial SL + TP orders immediately after opening a position,
        using unified client APIs and validating against exchange filters.
        """
        try:
            if not self.use:
                return {"status": "SKIPPED_SMART_EXIT_DISABLED"}

            is_long = side.upper() in ("LONG", "BUY")

            # symbol filters (caller-provided tick/step override if present)
            filters = self._get_symbol_info_filters(symbol)
            tick = float(tick_size or filters.get("tick", 0.0) or 0.0)
            step = float(step_size or filters.get("step", 0.0) or 0.0)
            min_qty = float(filters.get("min_qty", 0.0) or 0.0)
            min_notional = float(filters.get("min_notional", 0.0) or 0.0)

            # sanity qty
            if qty is None or qty <= 0:
                _debug_logger.debug("create_exit_orders: qty <= 0 for %s, skipping", symbol)
                return {"status": "SKIPPED_ZERO_QTY"}

            # SL calculation
            if atr_value:
                sl_raw = (entry_price - (self.atr_mult_sl * atr_value)) if is_long else (entry_price + (self.atr_mult_sl * atr_value))
            else:
                sl_raw = entry_price * (0.997 if is_long else 1.003)

            sl_price = self.round_price(symbol, sl_raw)

            # TP levels
            if atr_value and not tp_levels:
                tp_levels_calc: List[float] = []
                for mult in (self.atr_mult_tp or []):
                    p_raw = (entry_price + mult * atr_value) if is_long else (entry_price - mult * atr_value)
                    tp_levels_calc.append(self.round_price(symbol, p_raw))
                tp_levels = tp_levels_calc
            elif tp_levels:
                tp_levels = [self.round_price(symbol, float(p)) for p in tp_levels]
            else:
                tp_levels = []

            # DRY RUN
            if self.dry_run:
                return {"status": "DRY_RUN", "symbol": symbol, "sl": sl_price, "tp_levels": tp_levels}

            # validate min_qty
            rounded_qty = self.round_qty(symbol, qty)
            if min_qty > 0 and rounded_qty < min_qty:
                return {"status": "FAILED_VALIDATION", "error": f"rounded qty {rounded_qty} < min_qty {min_qty}"}

            # optional min_notional check with mark price
            mark_price: Optional[float] = None
            try:
                if hasattr(self.client, "ticker_price"):
                    price_data = self.client.ticker_price(symbol)
                    if isinstance(price_data, dict):
                        mark_price = safe_float(price_data.get("price"))
                    else:
                        mark_price = safe_float(price_data)
            except Exception:
                mark_price = None

            if min_notional > 0 and mark_price and mark_price > 0:
                notional = rounded_qty * mark_price
                if notional < min_notional:
                    return {"status": "FAILED_VALIDATION", "error": f"notional {notional:.8f} < min_notional {min_notional}"}

            # Cancel existing exit orders first
            try:
                self._cancel_all_exit_orders(symbol)
            except Exception:
                _debug_logger.exception("Could not cancel existing exit orders")

            # Place SL (STOP_MARKET)
            sl_res: Any = None
            try:
                sl_res = self.client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if is_long else "BUY",
                    type="STOP_MARKET",
                    stopPrice=str(sl_price),
                    quantity=rounded_qty,
                    reduceOnly=True,
                    workingType="CONTRACT_PRICE",
                )
            except Exception as e:
                _debug_logger.exception("Failed SL creation for %s: %s", symbol, e)
                sl_res = {"error": str(e)}

            # Place TP orders
            tp_results: List[Any] = []
            if tp_levels:
                per_tp_qty_raw = qty / len(tp_levels) if len(tp_levels) > 0 else qty
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

            return {
                "status": "OK",
                "symbol": symbol,
                "sl": sl_price,
                "tp_levels": tp_levels,
                "sl_result": sl_res,
                "tp_results": tp_results,
            }
        except Exception:
            _debug_logger.exception("create_exit_orders failure")
            return {"status": "ERROR", "error": "create_exit_orders unexpected failure"}

    # -------------------------
    # Small wrappers that delegate to the client if present
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
