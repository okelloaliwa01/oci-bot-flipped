# src/smart_exit.py
import os
import math
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from config import DRY_RUN

from config import (
    USE_SMART_EXIT,
    ATR_PERIOD,
    ATR_MULT_SL,
    ATR_MULT_TP,
    TRAILING_START_ATR,
    TRAILING_STEP_ATR,
    BREAKEVEN_ATR,
    BREAKEVEN_BUFFER_PTS,
    TP_PARTIAL_SIZES,
    ATR_PARTIAL_TPS,
)

# ---------- Logging ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# dedicated debug logger (per-session file)
_debug_log_dir = "logs"
os.makedirs(_debug_log_dir, exist_ok=True)
_session_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
_debug_log_path = os.path.join(_debug_log_dir, f"smartexit_debug_{_session_stamp}.log")

debug_logger = logging.getLogger("smartexit_debug")
debug_logger.setLevel(logging.DEBUG)
if not debug_logger.handlers:
    fh = logging.FileHandler(_debug_log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    debug_logger.addHandler(fh)
    debug_logger.propagate = False

logger.info(f"[SmartExit] Debug log: {_debug_log_path}")

# ============================================================
# Helpers
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "null", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default



def _parse_csv_floats(s: Optional[str], fallback: List[float]) -> List[float]:
    try:
        if s is None:
            return fallback
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        if not parts:
            return fallback
        return [safe_float(p) for p in parts]
    except Exception:
        return fallback


# ============================================================
# SmartExitManager
# ============================================================
class SmartExitManager:
    """
    SmartExit: create partial TPs and SL, update trailing/breakeven.
    Ensures cleanup of old SL/TP orders and prevents duplicates.
    """

    def __init__(self, binance_client):
        self.client = binance_client

        load_dotenv(override=True)
        self.use_smart_exit = os.getenv("USE_SMART_EXIT", str(USE_SMART_EXIT)).lower() == "true"
        self.atr_period = int(os.getenv("ATR_PERIOD", ATR_PERIOD))
        self.atr_mult_tp = _parse_csv_floats(os.getenv("ATR_MULT_TP", ",".join(map(str, ATR_MULT_TP))), list(ATR_MULT_TP))
        self.atr_mult_sl = safe_float(os.getenv("ATR_MULT_SL", str(ATR_MULT_SL)), ATR_MULT_SL)
        self.trailing_start_atr = safe_float(os.getenv("TRAILING_START_ATR", str(TRAILING_START_ATR)), TRAILING_START_ATR)
        self.trailing_step_atr = safe_float(os.getenv("TRAILING_STEP_ATR", str(TRAILING_STEP_ATR)), TRAILING_STEP_ATR)
        self.breakeven_atr = safe_float(os.getenv("BREAKEVEN_ATR", str(BREAKEVEN_ATR)), BREAKEVEN_ATR)
        self.breakeven_buffer_pts = safe_float(os.getenv("BREAKEVEN_BUFFER_PTS", str(BREAKEVEN_BUFFER_PTS)), BREAKEVEN_BUFFER_PTS)
        self.tp_partial_sizes = _parse_csv_floats(os.getenv("TP_PARTIAL_SIZES", ",".join(map(str, TP_PARTIAL_SIZES))), list(TP_PARTIAL_SIZES))
        self.atr_partial_tps = _parse_csv_floats(os.getenv("ATR_PARTIAL_TPS", ",".join(map(str, ATR_PARTIAL_TPS))), list(ATR_PARTIAL_TPS))

        logger.info(
            "[SmartExit] Init config: use=%s atr_period=%s atr_mult_tp=%s atr_mult_sl=%s trail_start=%s trail_step=%s breakeven_atr=%s be_buf=%s",
            self.use_smart_exit, self.atr_period, self.atr_mult_tp, self.atr_mult_sl,
            self.trailing_start_atr, self.trailing_step_atr, self.breakeven_atr, self.breakeven_buffer_pts
        )

    def round_to(self, value: Optional[float], step: Optional[float]) -> float:
        if value is None or not isinstance(value, (int, float)):
            return 0.0
        if not step or step <= 0:
            return float(value)
        decimals = max(0, int(abs(math.log10(step))))
        # ✅ floor-style rounding (expected 100.45 not 100.46)
        rounded = math.floor(value / step) * step
        return round(rounded, decimals)


    # -------------------------
    # Small utility: detect client's open order listing method
    # -------------------------
    def _list_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return list of currently open orders for a symbol (or all if symbol None).
        Tries multiple common SDK method names.
        """
        try:
            # prefer explicit futures endpoint names
            for name in ("futures_get_open_orders", "futures_open_orders", "get_open_orders", "get_open_orders_for_symbol", "get_open_orders_all"):
                if hasattr(self.client, name):
                    fn = getattr(self.client, name)
                    debug_logger.debug(f"[_list_open_orders] using client.{name} for symbol={symbol}")
                    try:
                        if symbol is None:
                            return fn()
                        # some signatures accept symbol kw
                        return fn(symbol=symbol)
                    except TypeError:
                        return fn(symbol)

            # fallback: generic method names
            if hasattr(self.client, "get_all_orders"):
                debug_logger.debug("[_list_open_orders] using client.get_all_orders")
                return self.client.get_all_orders(symbol=symbol) if symbol else self.client.get_all_orders()
        except Exception as e:
            debug_logger.exception(f"[_list_open_orders] error: {e}")
        return []

    # -------------------------
    # Cancel helpers
    # -------------------------
    def _cancel_order_by_id(self, symbol: str, order_id: Any) -> bool:
        """Cancel a single order by id with safe method detection."""
        try:
            for name in ("futures_cancel_order", "cancel_order", "futures_cancel", "cancel_open_order"):
                if hasattr(self.client, name):
                    fn = getattr(self.client, name)
                    debug_logger.debug(f"[_cancel_order_by_id] calling client.{name}({symbol}, {order_id})")
                    try:
                        # many SDKs accept keyword args
                        return bool(fn(symbol=symbol, orderId=order_id))
                    except TypeError:
                        return bool(fn(symbol, order_id))
            # last resort attempt: futures_cancel_all_open_orders (no id)
            if hasattr(self.client, "futures_cancel_all_open_orders"):
                debug_logger.debug("[_cancel_order_by_id] calling futures_cancel_all_open_orders (fallback)")
                self.client.futures_cancel_all_open_orders(symbol=symbol)
                return True
        except Exception as e:
            debug_logger.exception(f"[_cancel_order_by_id] cancel failed for {order_id}: {e}")
        return False

    def cancel_exit_orders_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Safely cancel all exit-type (TP/SL) orders for the given symbol.
        Returns a list of cancellation results with detailed logging.
        """
        results = []
        try:
            open_orders = self._list_open_orders(symbol)
            if not open_orders:
                logger.info(f"[SmartExit] No open orders found for {symbol}. Nothing to cancel.")
                debug_logger.debug(f"[cancel_exit_orders_for_symbol] open_orders=None for {symbol}")
                return []

            debug_logger.debug(f"[cancel_exit_orders_for_symbol] fetched {len(open_orders)} open orders for {symbol}")

            # Identify exit orders: reduceOnly True, or STOP/TAKE_PROFIT/limit with reduceOnly flag
            to_cancel = []
            for o in open_orders:
                try:
                    ot = str(o.get("type") or o.get("orderType") or "").upper()
                    ro = o.get("reduceOnly") or o.get("reduce_only") or o.get("reduce") or False
                    side = o.get("side") or o.get("orderSide") or ""
                    cid = o.get("clientOrderId") or o.get("orderId") or o.get("order_id")
                    if ot in ("STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "LIMIT") and (ro or ot.startswith("STOP") or ot.startswith("TAKE")):
                        to_cancel.append(o)
                        debug_logger.debug(f"[cancel_exit_orders_for_symbol] marking for cancel -> id={cid} type={ot} side={side} reduceOnly={ro}")
                except Exception as e_inner:
                    debug_logger.exception(f"[cancel_exit_orders_for_symbol] order parse error: {e_inner}")
                    continue

            if not to_cancel:
                logger.info(f"[SmartExit] No exit-type orders to cancel for {symbol}.")
                debug_logger.debug(f"[cancel_exit_orders_for_symbol] to_cancel list empty for {symbol}")
                return []

            logger.info(f"[SmartExit] Cancelling {len(to_cancel)} exit orders for {symbol}...")
            for o in to_cancel:
                oid = o.get("orderId") or o.get("clientOrderId") or o.get("order_id")
                if not oid:
                    debug_logger.debug(f"[cancel_exit_orders_for_symbol] Missing orderId, attempting cancel-all fallback for {symbol}")
                    try:
                        if hasattr(self.client, "futures_cancel_all_open_orders"):
                            self.client.futures_cancel_all_open_orders(symbol=symbol)
                            results.append({"symbol": symbol, "cancel_all": True})
                            logger.warning(f"[SmartExit] Fallback cancel-all used for {symbol}")
                        else:
                            logger.error(f"[SmartExit] No cancel-all method found on client for {symbol}")
                    except Exception as e_fallback:
                        debug_logger.exception(f"[cancel_exit_orders_for_symbol] cancel-all fallback failed: {e_fallback}")
                    continue

                ok = self._cancel_order_by_id(symbol, oid)
                results.append({"symbol": symbol, "order_id": oid, "cancelled": ok})
                if ok:
                    logger.info(f"[SmartExit] Cancelled exit order {oid} for {symbol}")
                else:
                    logger.warning(f"[SmartExit] Failed to cancel exit order {oid} for {symbol}")
                debug_logger.debug(f"[cancel_exit_orders_for_symbol] cancel result -> id={oid} ok={ok}")

        except Exception as e:
            logger.exception(f"[SmartExit] Fatal error while cancelling exit orders for {symbol}: {e}")
            debug_logger.exception(f"[cancel_exit_orders_for_symbol] fatal error: {e}")
        finally:
            debug_logger.debug(f"[cancel_exit_orders_for_symbol] completed for {symbol}, results={results}")
        return results


    # -------------------------
    # Helper: dedupe open orders to avoid duplicates
    # -------------------------
    def _find_similar_open_order(
        self, symbol: str, ord_type: str, price: Optional[float], qty: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Search open orders for one that matches type+price (within tiny tolerance) to prevent duplicates.
        """
        try:
            orders = self._list_open_orders(symbol)
            for o in orders or []:
                try:
                    ot = (o.get("type") or o.get("orderType") or "").upper()
                    op_raw = o.get("price") or o.get("priceStr") or o.get("stopPrice")
                    oq_raw = o.get("origQty") or o.get("quantity") or o.get("qty")

                    # skip if price is None
                    if ot == ord_type.upper():
                        if price is not None and op_raw is not None:
                            try:
                                op = float(op_raw)
                                if abs(op - float(price)) <= max(1e-8, 0.5 * (10 ** -6)):
                                    return o
                            except (TypeError, ValueError):
                                continue
                        else:
                            # same type, ignore price
                            return o
                except Exception:
                    continue
        except Exception:
            debug_logger.exception("[_find_similar_open_order] error")
        return None


    # ============================================================
    # Manage open positions (trailing / breakeven) - unchanged except we cancel stale orders when no position
    # ============================================================
    def manage_open_positions(self, symbol: str):
        try:
            load_dotenv(override=True)
            self.use_smart_exit = os.getenv("USE_SMART_EXIT", str(self.use_smart_exit)).lower() == "true"
            self.atr_period = int(os.getenv("ATR_PERIOD", str(self.atr_period)))
            self.atr_mult_tp = _parse_csv_floats(os.getenv("ATR_MULT_TP", ",".join(map(str, self.atr_mult_tp))), self.atr_mult_tp)
            self.atr_mult_sl = safe_float(os.getenv("ATR_MULT_SL", str(self.atr_mult_sl)), self.atr_mult_sl)
            self.trailing_start_atr = safe_float(os.getenv("TRAILING_START_ATR", str(self.trailing_start_atr)), self.trailing_start_atr)
            self.trailing_step_atr = safe_float(os.getenv("TRAILING_STEP_ATR", str(self.trailing_step_atr)), self.trailing_step_atr)
            self.breakeven_atr = safe_float(os.getenv("BREAKEVEN_ATR", str(self.breakeven_atr)), self.breakeven_atr)
            self.breakeven_buffer_pts = safe_float(os.getenv("BREAKEVEN_BUFFER_PTS", str(self.breakeven_buffer_pts)), self.breakeven_buffer_pts)
            self.tp_partial_sizes = _parse_csv_floats(os.getenv("TP_PARTIAL_SIZES", ",".join(map(str, self.tp_partial_sizes))), self.tp_partial_sizes)
            self.atr_partial_tps = _parse_csv_floats(os.getenv("ATR_PARTIAL_TPS", ",".join(map(str, self.atr_partial_tps))), self.atr_partial_tps)

            """if not self.use_smart_exit:
                debug_logger.debug("[manage_open_positions] smart exit disabled")
                return {"type": "noop", "symbol": symbol, "reason": "disabled"}"""

            if not self.use_smart_exit:
                logger.info(f"[SmartExit] Skipping manage_open_positions for {symbol} because smart exit disabled")
                return None

            
            klines = self._safe_get_klines(symbol, limit=self.atr_period + 5)
            if not klines:
                logger.warning(f"[SmartExit] No candle data for {symbol}.")
                return {"type": "noop", "symbol": symbol, "reason": "no_klines"}

            df = self._klines_to_df(klines)
            if df is None or df.empty:
                logger.warning(f"[SmartExit] Empty dataframe for {symbol}.")
                return {"type": "noop", "symbol": symbol, "reason": "empty_dataframe"}

            atr = self.calculate_atr(df)
            if atr is None or atr <= 0:
                logger.warning(f"[SmartExit] Invalid ATR for {symbol}. ATR={atr}")
                return {"type": "noop", "symbol": symbol, "reason": "invalid_atr"}

            last_price = safe_float(df["close"].iloc[-1])
            position = self._get_open_position(symbol)
            if not position:
                # No open position: cancel leftover exit orders to avoid hanging SL/TP after TP closes position
                debug_logger.debug(f"[manage_open_positions] No position for {symbol} - cleaning up exit orders")
                cancelled = self.cancel_exit_orders_for_symbol(symbol)
                debug_logger.debug(f"[manage_open_positions] cancelled leftover orders: {cancelled}")
                return {"type": "noop", "symbol": symbol, "reason": "no_position", "cancelled": cancelled}

            side = self._normalize_side(position.get("side"))
            entry = safe_float(position.get("entry"))
            cur_sl = safe_float(position.get("sl")) if position.get("sl") else None
            qty = safe_float(position.get("qty"))
            profit_distance = abs(last_price - entry)

            if qty <= 0 or entry <= 0 or not side:
                logger.debug(f"[SmartExit] Skipping invalid position data for {symbol}: {position}")
                return {"type": "noop", "symbol": symbol, "reason": "invalid_position"}

            logger.info(
                f"[SmartExit] Managing {symbol} {side}: entry={entry:.6f}, last={last_price:.6f}, curSL={cur_sl}, ATR={atr:.6f}"
            )

            # TRAILING STOP
            if profit_distance > self.trailing_start_atr * atr:
                proposed_sl = (
                    last_price - (self.trailing_step_atr * atr)
                    if side == "LONG"
                    else last_price + (self.trailing_step_atr * atr)
                )

                tick = 0.0
                try:
                    info = self.client.get_symbol_info(symbol)
                    for f in info.get("filters", []):
                        if f.get("filterType") == "PRICE_FILTER":
                            tick = float(f.get("tickSize", 0.0))
                            break
                except Exception:
                    tick = 0.0

                proposed_sl_rounded = self.round_to(proposed_sl, tick)
                if (side == "LONG" and (cur_sl is None or proposed_sl_rounded > cur_sl)) or (
                    side == "SHORT" and (cur_sl is None or proposed_sl_rounded < cur_sl)
                ):
                    logger.info(
                        f"[SmartExit] trailing stop triggered → new SL={proposed_sl_rounded:.8f} (prev={cur_sl})"
                    )
                    res = self._execute_stop_loss_update(symbol, side, qty, proposed_sl_rounded)
                    return {
                        "type": "trailing",
                        "symbol": symbol,
                        "side": side,
                        "new_sl": proposed_sl_rounded,
                        "result": res,
                    }

            # BREAKEVEN
            if profit_distance >= self.breakeven_atr * atr:
                proposed_sl = (
                    entry + (self.breakeven_buffer_pts * atr)
                    if side == "LONG"
                    else entry - (self.breakeven_buffer_pts * atr)
                )

                tick = 0.0
                try:
                    info = self.client.get_symbol_info(symbol)
                    for f in info.get("filters", []):
                        if f.get("filterType") == "PRICE_FILTER":
                            tick = float(f.get("tickSize", 0.0))
                            break
                except Exception:
                    tick = 0.0

                proposed_sl_rounded = self.round_to(proposed_sl, tick)
                if (side == "LONG" and (cur_sl is None or proposed_sl_rounded > cur_sl)) or (
                    side == "SHORT" and (cur_sl is None or proposed_sl_rounded < cur_sl)
                ):
                    logger.info(
                        f"[SmartExit] breakeven SL update → new SL={proposed_sl_rounded:.8f} (entry={entry:.6f}, ATR={atr:.6f})"
                    )
                    res = self._execute_stop_loss_update(symbol, side, qty, proposed_sl_rounded)
                    return {
                        "type": "breakeven",
                        "symbol": symbol,
                        "side": side,
                        "new_sl": proposed_sl_rounded,
                        "result": res,
                    }

            debug_logger.debug(f"[manage_open_positions] {symbol} no SL update this cycle")
            return {"type": "noop", "symbol": symbol, "reason": "no_update"}
        except Exception as e:
            logger.exception(f"[SmartExit] Error in manage_open_positions({symbol}): {e}")
            debug_logger.exception(f"[SmartExit] manage_open_positions error: {e}")
            return {"type": "error", "symbol": symbol, "error": str(e)}

    # ============================================================
    # Create Exit Orders (Partial TPs + SL) - patched to cancel and dedupe
    # ============================================================
    def create_exit_orders(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        atr_value: Optional[float] = None,
        atr: Optional[float] = None,
        tick_size: Optional[float] = None,
        step_size: Optional[float] = None,
        tp_levels: Optional[list] = None,  # Optional precomputed TP list
    ):
        import math
        import logging

        logger = logging.getLogger(__name__)

        try:
            # --- ATR validation ---
            atr = atr_value or atr
            if atr is None or atr <= 0:
                logger.warning(f"[SmartExit] Missing or invalid ATR value for {symbol}. atr={atr}")
                return None

            if qty <= 0:
                logger.warning(f"[SmartExit] Invalid quantity for {symbol}: {qty}")
                return None

            logger.info(
                f"[SmartExit] Creating exit orders for {symbol} | side={side} | entry={entry_price:.2f} | ATR={atr:.4f} | qty={qty}"
            )

            # --- Dry-run mode ---
            global_dry = globals().get("DRY_RUN", False)
            instance_dry = getattr(self, "dry_run", None)
            dry_mode = instance_dry if instance_dry is not None else global_dry

            # --- Symbol filters ---
            try:
                if not tick_size or not step_size:
                    info = self.client.get_symbol_info(symbol)
                    filters = info.get("filters", [])
                    tick_size = tick_size or next((float(f["tickSize"]) for f in filters if f["filterType"] == "PRICE_FILTER"), 0.1)
                    step_size = step_size or next((float(f["stepSize"]) for f in filters if f["filterType"] == "LOT_SIZE"), 0.001)
                    min_qty = next((float(f["minQty"]) for f in filters if f["filterType"] == "LOT_SIZE"), 0.001)
                    min_notional = next((float(f.get("notional", 5.0)) for f in filters if f["filterType"] == "MIN_NOTIONAL"), 5.0)
                else:
                    min_qty, min_notional = 0.001, 5.0
            except Exception as e:
                logger.warning(f"[SmartExit] Could not fetch symbol filters for {symbol}: {e}")
                tick_size, step_size, min_qty, min_notional = 0.1, 0.001, 0.001, 5.0

            # --- Precision helpers ---
            def snap_price(p):
                if not tick_size or tick_size <= 0:
                    return float(p)
                decimals = max(0, int(abs(math.log10(tick_size))))
                return round(round(p / tick_size) * tick_size, decimals)

            def snap_qty(q):
                q = max(q, min_qty)
                if not step_size or step_size <= 0:
                    return q
                decimals = max(0, int(abs(math.log10(step_size))))
                return round(round(q / step_size) * step_size, decimals)

            def valid_notional(price, quantity):
                if price * quantity < min_notional:
                    q2 = min_notional / price
                    logger.warning(
                        f"[SmartExit:FIX] Adjusting qty for minNotional ({price*quantity:.4f}<{min_notional}) -> {q2:.6f}"
                    )
                    return snap_qty(q2)
                return quantity

            # --- Cancel existing exits ---
            self.cancel_exit_orders_for_symbol(symbol)

            # --- Compute TP/SL levels ---
            TP_MULTS = getattr(self, "atr_mult_tp", [2.0, 3.0, 4.0])
            SL_MULT = getattr(self, "atr_mult_sl", 1.5)

            if tp_levels is not None:
                # ✅ SAFETY FIX: ensure tp_levels is iterable
                if isinstance(tp_levels, (int, float)):
                    tp_levels = [tp_levels]
                tp_levels = [snap_price(p) for p in tp_levels]
                if side.upper() in ("LONG", "BUY"):
                    sl_price = snap_price(entry_price - atr * SL_MULT)
                else:
                    sl_price = snap_price(entry_price + atr * SL_MULT)
            else:
                if side.upper() in ("LONG", "BUY"):
                    tp_levels = [snap_price(entry_price + atr * m) for m in TP_MULTS]
                    sl_price = snap_price(entry_price - atr * SL_MULT)
                else:
                    tp_levels = [snap_price(entry_price - atr * m) for m in TP_MULTS]
                    sl_price = snap_price(entry_price + atr * SL_MULT)

            # --- Defensive filtering ---
            tp_levels = [float(x) for x in tp_levels if x and x > 0]
            if not tp_levels:
                logger.warning(f"[SmartExit] No valid TP levels computed for {symbol}")
                return None

            # --- Partial qty allocation ---
            tp_count = len(tp_levels)
            partial_qty = max(qty / tp_count, min_qty)
            partial_qtys = [snap_qty(valid_notional(tp, partial_qty)) for tp in tp_levels]

            # --- Dry-run mode ---
            if dry_mode:
                for tp_price, tp_qty in zip(tp_levels, partial_qtys):
                    logger.info(f"[SmartExit:DRYRUN] TP -> {symbol} SELL {tp_qty} @ {tp_price}")
                logger.info(f"[SmartExit:DRYRUN] SL -> {symbol} STOP_MARKET @ {sl_price}")
                return {
                    "dry_run": True,
                    "symbol": symbol,
                    "side": side,
                    "tp_levels": tp_levels,
                    "partial_qtys": partial_qtys,
                    "sl_price": sl_price,
                }

            # --- Live TP orders ---
            tp_orders = []
            for tp_price, tp_qty in zip(tp_levels, partial_qtys):
                payload = dict(
                    symbol=symbol,
                    side="SELL" if side.upper() in ("LONG", "BUY") else "BUY",
                    type="LIMIT",
                    price=str(tp_price),
                    quantity=tp_qty,
                    reduceOnly=True,
                    timeInForce="GTC",
                )
                try:
                    order = self.client.futures_create_order(**payload)
                    tp_orders.append(order)
                    logger.info(f"[SmartExit] TP placed -> {symbol} {payload['side']} {tp_qty} @ {tp_price}")
                except Exception as e:
                    logger.error(f"[SmartExit] TP failed for {symbol} @ {tp_price}: {e}")

            # --- Live SL order ---
            try:
                payload = dict(
                    symbol=symbol,
                    side="SELL" if side.upper() in ("LONG", "BUY") else "BUY",
                    type="STOP_MARKET",
                    stopPrice=str(sl_price),
                    closePosition=True,
                    timeInForce="GTC",
                )
                self.client.futures_create_order(**payload)
                logger.info(f"[SmartExit] SL placed -> {symbol} @ {sl_price}")
            except Exception as e:
                logger.error(f"[SmartExit] SL failed for {symbol} @ {sl_price}: {e}")

            return {
                "dry_run": False,
                "symbol": symbol,
                "side": side,
                "tp_levels": tp_levels,
                "partial_qtys": partial_qtys,
                "sl_price": sl_price,
                "tp_orders": tp_orders,
            }

        except Exception as e:
            logger.exception(f"[SmartExit:FATAL] Unexpected error in create_exit_orders: {e}")
            return None





    # ============================================================
    # ATR + Kline helpers (unchanged)
    # ============================================================
    def calculate_atr(self, df: pd.DataFrame, window: int = ATR_PERIOD) -> Optional[float]:
        try:
            if df is None or df.shape[0] < 2:
                return None
            high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(window=window, min_periods=1).mean()
            atr = atr_series.iloc[-1]
            return float(atr) if not np.isnan(atr) else None
        except Exception:
            logger.exception("Error computing ATR")
            return None

    def _safe_get_klines(self, symbol: str, interval: str = "5m", limit: int = 50) -> Optional[List[Any]]:
        try:
            for name in ("get_klines", "get_candles", "get_recent_candles", "get_klines_raw"):
                if hasattr(self.client, name):
                    return getattr(self.client, name)(symbol=symbol, interval=interval, limit=limit)
            if hasattr(self.client, "client") and hasattr(self.client.client, "klines"):
                return self.client.client.klines(symbol=symbol, interval=interval, limit=limit)
        except Exception:
            logger.exception("Error fetching klines")
        return None

    def _klines_to_df(self, klines: Union[List[Any], None]) -> Optional[pd.DataFrame]:
        try:
            if not klines:
                return None
            if isinstance(klines[0], dict):
                df = pd.DataFrame(klines)
                for col in ["open", "high", "low", "close", "volume"]:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                df = pd.DataFrame(klines)
                df = df.iloc[:, :6]
                df.columns = ["open_time", "open", "high", "low", "close", "volume"]
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
            df.dropna(subset=["close"], inplace=True)
            return df
        except Exception:
            logger.exception("Error converting klines to DataFrame")
            return None

    # ============================================================
    # Position helpers & normalization (unchanged)
    # ============================================================
    def _get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(self.client, "get_position"):
                pos = self.client.get_position(symbol)
                if pos:
                    return self._normalize_client_position(pos)
            for name in ("get_open_positions", "get_positions", "get_all_positions"):
                if hasattr(self.client, name):
                    for p in getattr(self.client, name)() or []:
                        if p.get("symbol") == symbol:
                            return self._normalize_client_position(p)
            if hasattr(self.client, "futures_position_information"):
                info = self.client.futures_position_information(symbol=symbol)
                if info:
                    return self._normalize_client_position(info[0] if isinstance(info, list) else info)
        except Exception:
            logger.exception("Error getting open position")
        return None

    def _normalize_client_position(self, raw: dict) -> dict:
        try:
            side = self._normalize_side(raw.get("side") or raw.get("positionSide"))
            entry = safe_float(raw.get("entryPrice") or raw.get("avgPrice") or raw.get("price"))
            qty = safe_float(raw.get("quantity") or raw.get("positionAmt"))
            sl_raw = raw.get("stopLoss") or raw.get("stop_price") or raw.get("stopPrice")
            sl = safe_float(sl_raw) if sl_raw not in (None, "", "null") else None
            if not side:
                if qty > 0:
                    side = "LONG"
                elif qty < 0:
                    side = "SHORT"
            return {"side": side, "entry": entry, "sl": sl, "qty": abs(qty)}
        except Exception:
            logger.exception("Error normalizing position")
            return {"side": None, "entry": None, "sl": None, "qty": None}

    def _normalize_side(self, raw_side: Optional[str]) -> str:
        if not raw_side:
            return ""
        s = str(raw_side).strip().upper()
        if s in ("LONG", "BUY", "B"):
            return "LONG"
        if s in ("SHORT", "SELL", "S"):
            return "SHORT"
        return s

    # ============================================================
    # Stop-loss update helper (unchanged except debug)
    # ============================================================
    def _execute_stop_loss_update(self, symbol: str, side: str, qty: Optional[float], new_sl: float):
        try:
            # prefer dedicated update_stop_loss if available
            if hasattr(self.client, "update_stop_loss"):
                logger.info(f"[SmartExit] Updating SL via API -> {symbol} {new_sl:.8f}")
                debug_logger.debug(f"[_execute_stop_loss_update] using update_stop_loss: {symbol} {new_sl}")
                return self.client.update_stop_loss(symbol=symbol, side=side, quantity=qty, new_sl_price=new_sl)

            # fallback to cancel & place STOP_MARKET via futures_create_order
            if hasattr(self.client, "futures_create_order"):
                stop_side = "SELL" if side == "LONG" else "BUY"
                logger.warning(f"[SmartExit] Fallback STOP_MARKET creation for {symbol} at {new_sl:.8f}")
                debug_logger.debug(f"[_execute_stop_loss_update] placing STOP_MARKET {symbol} {stop_side} {new_sl}")
                # cancel older stop orders first to avoid duplicates
                try:
                    self.cancel_exit_orders_for_symbol(symbol)
                except Exception:
                    debug_logger.exception("[_execute_stop_loss_update] pre-cancel failed")
                return self.client.futures_create_order(
                    symbol=symbol,
                    side=stop_side,
                    type="STOP_MARKET",
                    stopPrice=str(new_sl),
                    quantity=qty,
                )

            # last resort: try generic place_order
            if hasattr(self.client, "place_order"):
                stop_side = "SELL" if side == "LONG" else "BUY"
                logger.warning(f"[SmartExit] Using place_order fallback for SL -> {symbol} {new_sl:.8f}")
                debug_logger.debug(f"[_execute_stop_loss_update] placing fallback place_order {symbol} {stop_side} {new_sl}")
                return self.client.place_order(
                    symbol=symbol,
                    side=stop_side,
                    type="STOP_MARKET",
                    stopPrice=str(new_sl),
                    quantity=qty,
                )

        except Exception as e:
            logger.error(f"[SmartExit] Failed SL update for {symbol}: {e}")
            debug_logger.exception(f"[_execute_stop_loss_update] error: {e}")
        return None