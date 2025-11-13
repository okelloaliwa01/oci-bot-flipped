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
    """
    Parse a comma-separated string of floats, returning a list of floats.
    Falls back to the provided default if parsing fails or input is empty.

    Examples:
        "1.2,3.4" -> [1.2, 3.4]
        ""        -> fallback
        None      -> fallback
    """
    if not s:
        return fallback

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return fallback

    result = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            # Could log a warning here to help detect bad .env entries
            logger.warning("Invalid float value in CSV string: %r. Using fallback.", p)
            return fallback

    return result



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

        # --- Always reload environment ---
        load_dotenv(override=True)

        # --- Configuration values ---
        self.use_smart_exit = os.getenv("USE_SMART_EXIT", "true").lower() == "true"
        self.atr_period = int(os.getenv("ATR_PERIOD", "14"))
        self.atr_mult_tp = _parse_csv_floats(
            os.getenv("ATR_MULT_TP", "2.461654,3.049415"),
            [2.461654, 3.049415]
        )
        self.atr_mult_sl = safe_float(os.getenv("ATR_MULT_SL", "1.897911"), 1.897911)
        self.trailing_start_atr = safe_float(os.getenv("TRAILING_START_ATR", "1.684771"), 1.684771)
        self.trailing_step_atr = safe_float(os.getenv("TRAILING_STEP_ATR", "0.279722"), 0.279722)
        self.breakeven_atr = safe_float(os.getenv("BREAKEVEN_ATR", "1.092215"), 1.092215)
        self.breakeven_buffer_pts = safe_float(os.getenv("BREAKEVEN_BUFFER_PTS", "0.034933"), 0.034933)
        self.tp_partial_sizes = _parse_csv_floats(
            os.getenv("TP_PARTIAL_SIZES", "0.5,0.5"),
            [0.5, 0.5]
        )

        # --- Critical missing attribute ---
        self.atr_partial_tps = _parse_csv_floats(
            os.getenv("ATR_PARTIAL_TPS", "1.0,2.0,3.0"),  # default values
            [1.0, 2.0, 3.0]
        )

        # --- Dry-run flag ---
        env_dry = os.getenv("DRY_RUN")
        if env_dry is not None:
            self.dry_run = env_dry.lower() in ("1", "true", "yes")
        else:
            self.dry_run = bool(globals().get("DRY_RUN", False))

        logger.info(
            "[SmartExit] Init config: use=%s dry_run=%s atr_period=%s atr_mult_tp=%s atr_mult_sl=%s "
            "trail_start=%s trail_step=%s breakeven_atr=%s be_buf=%s atr_partial_tps=%s",
            self.use_smart_exit,
            self.dry_run,
            self.atr_period,
            self.atr_mult_tp,
            self.atr_mult_sl,
            self.trailing_start_atr,
            self.trailing_step_atr,
            self.breakeven_atr,
            self.breakeven_buffer_pts,
            self.atr_partial_tps,
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
        import os
        import time

        logger = logging.getLogger(__name__)

        # --- helpers -------------------------------------------------------
        def _resolve_dry_run():
            inst_flag = getattr(self, "dry_run", None)
            env_flag = os.environ.get("DRY_RUN")
            if env_flag is not None:
                env_flag = env_flag.lower() in ("1", "true", "yes")
            global_flag = globals().get("DRY_RUN", False)
            resolved = inst_flag if inst_flag is not None else (env_flag if env_flag is not None else global_flag)
            self.dry_run = bool(resolved)
            return self.dry_run

        def _snap_price(p, tick):
            if not tick or tick <= 0:
                return float(p)
            # avoid float precision issues: snap using integer math
            decimals = max(0, int(round(-math.log10(tick))))
            snapped = round(round(p / tick) * tick, decimals)
            return float(snapped)

        def _snap_qty(q, step, min_qty):
            q = max(q, min_qty or 0)
            if not step or step <= 0:
                return float(q)
            decimals = max(0, int(round(-math.log10(step))))
            snapped = round(round(q / step) * step, decimals)
            # ensure at least min_qty
            if min_qty and snapped < min_qty:
                return float(min_qty)
            return float(snapped)

        def _position_confirmed(client, sym, side_expected, timeout=5.0, poll_interval=0.5):
            """
            Wait until the exchange reports an open position for the symbol matching side_expected.
            side_expected: "LONG" or "SHORT"
            """
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    # prefer get_position which returns dict for a symbol
                    pos = {}
                    if hasattr(client, "get_position"):
                        pos = client.get_position(sym) or {}
                    else:
                        # fallback to get_position_risk list
                        if hasattr(client, "get_position_risk"):
                            for p in client.get_position_risk(sym) or []:
                                if p.get("symbol") == sym:
                                    pos = p
                                    break
                    amt = 0.0
                    if isinstance(pos, dict):
                        amt = float(pos.get("positionAmt") or pos.get("position") or pos.get("quantity") or 0.0)
                    if side_expected.upper() in ("SHORT", "SELL") and amt < 0:
                        return True
                    if side_expected.upper() in ("LONG", "BUY") and amt > 0:
                        return True
                except Exception:
                    # ignore transient errors while polling
                    pass
                time.sleep(poll_interval)
            return False

        def _place_order_with_reduce_retry(payload, confirm_position_after=False):
            """
            Place an order with targeted retry behavior for ReduceOnly rejections.
            Returns order on success or raises last exception on failure.
            """
            # attempt direct call variants; treat reduce-only rejection specially
            MAX_RETRIES = 3
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    order = self.client.futures_create_order(**payload)
                    return order
                except Exception as e:
                    # try to detect Binance -2022 reduceOnly rejection
                    estr = str(e).lower()
                    if ("-2022" in estr) or ("reduceonly order is rejected" in estr) or ("reduceonly" in estr and "rejected" in estr):
                        logger.warning("[SmartExit] ReduceOnly rejected for %s @ %s (attempt %d/%d). Will retry after position confirmation.",
                                    payload.get("symbol"), payload.get("price"), attempt, MAX_RETRIES)
                        # If we haven't confirmed the position, wait a bit and re-confirm
                        if not _position_confirmed(self.client, payload.get("symbol"), side, timeout=2.5, poll_interval=0.5):
                            # small wait before retry
                            time.sleep(0.6)
                            continue
                        else:
                            # position is confirmed yet reduceonly still rejected -> don't retry infinitely
                            logger.warning("[SmartExit] Position reported but reduceOnly still rejected for %s; aborting this TP.", payload.get("symbol"))
                            raise
                    else:
                        # transient/other errors: backoff and retry a couple of times
                        if attempt < MAX_RETRIES:
                            backoff = 0.5 * attempt
                            logger.debug("[SmartExit] Order attempt %d failed, retrying in %.2fs: %s", attempt, backoff, e)
                            time.sleep(backoff)
                            continue
                        else:
                            logger.error("[SmartExit] Order attempt %d failed (final): %s", attempt, e)
                            raise
            raise RuntimeError("Unreachable _place_order_with_reduce_retry exit")

        # ------------------------------------------------------------------

        try:
            # Validate ATR / qty
            atr = atr_value or atr
            if atr is None or atr <= 0:
                logger.warning(f"[SmartExit] Missing or invalid ATR value for {symbol}. atr={atr}")
                return None

            if qty <= 0:
                logger.warning(f"[SmartExit] Invalid quantity for {symbol}: {qty}")
                return None

            dry = _resolve_dry_run()
            logger.info("[SmartExit] Creating exit orders for %s | side=%s | entry=%.2f | ATR=%.4f | qty=%.6f | dry_run=%s",
                        symbol, side, entry_price, atr, qty, dry)

            # --- DRY-RUN early return (simulate levels) -----------------------
            if dry:
                TP_MULTS = getattr(self, "atr_mult_tp", [2.0, 3.0, 4.0])
                SL_MULT = getattr(self, "atr_mult_sl", 1.5)
                if side.upper() in ("LONG", "BUY"):
                    tps = [entry_price + atr * m for m in TP_MULTS]
                    sl_price = entry_price - atr * SL_MULT
                else:
                    tps = [entry_price - atr * m for m in TP_MULTS]
                    sl_price = entry_price + atr * SL_MULT
                tps = [float(round(x, 6)) for x in tps]
                sl_price = float(round(sl_price, 6))
                logger.info("[SmartExit] DRY-RUN active; simulated TP levels: %s | SL: %s", tps, sl_price)
                return {"dry_run": True, "symbol": symbol, "side": side, "tp_levels": tps, "sl_price": sl_price}

            # --- Fetch symbol filters (price/qty precision + min_notional) -----
            try:
                info = self.client.get_symbol_info(symbol) or {}
                filters = info.get("filters", [])
                tick_size = tick_size or next((float(f["tickSize"]) for f in filters if f.get("filterType") == "PRICE_FILTER"), 0.1)
                step_size = step_size or next((float(f["stepSize"]) for f in filters if f.get("filterType") == "LOT_SIZE"), 0.0001)
                min_qty = next((float(f.get("minQty", 0.0)) for f in filters if f.get("filterType") == "LOT_SIZE"), 0.0001)
                min_notional = next((float(f.get("minNotional") or f.get("notional") or 0.0) for f in filters if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL")), 0.0)
                if not min_notional:
                    min_notional = 1.0  # sensible default if not provided
            except Exception as e:
                logger.warning("[SmartExit] Could not fetch symbol filters for %s: %s. Using defaults.", symbol, e)
                tick_size, step_size, min_qty, min_notional = 0.1, 0.0001, 0.0001, 1.0

            # --- Snap helpers bound to resolved precision ----------------------
            snap_price = lambda p: _snap_price(p, tick_size)
            snap_qty = lambda q: _snap_qty(q, step_size, min_qty)
            def enforce_min_notional(price, q):
                if price * q < min_notional:
                    needed_q = min_notional / price
                    logger.warning("[SmartExit:FIX] %s: notional (%.6f) < min_notional(%.2f). Adjusting qty -> %.6f",
                                symbol, price * q, min_notional, needed_q)
                    return snap_qty(needed_q)
                return q

            # --- Cancel any existing SmartExit orders for symbol ----------------
            try:
                if hasattr(self, "cancel_exit_orders_for_symbol"):
                    self.cancel_exit_orders_for_symbol(symbol)
            except Exception as e:
                logger.debug("[SmartExit] cancel_exit_orders_for_symbol failed: %s", e)

            # --- Compute TP and SL levels -------------------------------------
            TP_MULTS = getattr(self, "atr_mult_tp", [2.0, 3.0, 4.0])
            SL_MULT = getattr(self, "atr_mult_sl", 1.5)

            if tp_levels is not None:
                if isinstance(tp_levels, (int, float)):
                    tp_levels = [float(tp_levels)]
                tp_levels = [snap_price(float(p)) for p in tp_levels if p is not None]
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

            tp_levels = [float(x) for x in tp_levels if x and x > 0]
            if not tp_levels:
                logger.warning("[SmartExit] No valid TP levels computed for %s", symbol)
                return None

            # --- Compute per-TP partial quantities (respect min_qty & min_notional) ---
            tp_count = len(tp_levels)
            base_partial_qty = max(qty / tp_count, min_qty)
            partial_qtys = []
            for tp_price in tp_levels:
                q0 = base_partial_qty
                q1 = enforce_min_notional(tp_price, q0)
                q2 = snap_qty(q1)
                partial_qtys.append(q2)

            # --- Confirm position exists before placing reduceOnly TP orders ---
            # Determine expected side for position: SHORT => negative position (we sold), LONG => positive
            expected_side = "SHORT" if side.upper() in ("SHORT", "SELL") else "LONG"
            confirmed = _position_confirmed(self.client, symbol, expected_side, timeout=5.0, poll_interval=0.5)
            if not confirmed:
                logger.warning("[SmartExit] Position for %s not visible yet. Waiting briefly then trying to place reduceOnly TPs.", symbol)
                # extra short wait then re-check
                time.sleep(0.7)
                confirmed = _position_confirmed(self.client, symbol, expected_side, timeout=3.0, poll_interval=0.5)

            # --- Place TP orders (reduceOnly only if position confirmed) -------
            tp_orders = []
            placed_tp_errors = []
            for tp_price, tp_qty in zip(tp_levels, partial_qtys):
                side_tp = "SELL" if expected_side == "LONG" else "BUY"
                # only add reduceOnly if position confirmed; otherwise attempt without reduceOnly once (safer fallback)
                reduce_only_flag = bool(confirmed)
                payload = {
                    "symbol": symbol,
                    "side": side_tp,
                    "type": "LIMIT",
                    "price": str(tp_price),
                    "quantity": tp_qty,
                    "timeInForce": "GTC",
                    # do not include reduceOnly key when False for some SDK versions; include only when needed
                }
                if reduce_only_flag:
                    payload["reduceOnly"] = True

                try:
                    order = _place_order_with_reduce_retry(payload, confirm_position_after=confirmed)
                    tp_orders.append(order)
                    logger.info("[SmartExit] TP placed -> %s %s %s @ %s", symbol, payload["side"], tp_qty, tp_price)
                except Exception as e:
                    # If reduceOnly was True and refused, try once without reduceOnly (best-effort fallback)
                    estr = str(e).lower()
                    placed_tp_errors.append({"price": tp_price, "qty": tp_qty, "error": str(e)})
                    logger.error("[SmartExit] TP failed for %s @ %s: %s", symbol, tp_price, e)
                    if reduce_only_flag:
                        # fallback attempt without reduceOnly (only one attempt)
                        try:
                            payload.pop("reduceOnly", None)
                            order = self.client.futures_create_order(**payload)
                            tp_orders.append(order)
                            logger.info("[SmartExit] TP placed (without reduceOnly fallback) -> %s %s %s @ %s", symbol, payload["side"], tp_qty, tp_price)
                        except Exception as e2:
                            logger.error("[SmartExit] Fallback TP (no reduceOnly) also failed for %s @ %s: %s", symbol, tp_price, e2)
                            placed_tp_errors.append({"price": tp_price, "qty": tp_qty, "error_fallback": str(e2)})

            # --- Place SL (STOP_MARKET) using closePosition=True or closePosition flag per SDK
            try:
                sl_side = "SELL" if expected_side == "LONG" else "BUY"
                sl_payload = {
                    "symbol": symbol,
                    "side": sl_side,
                    "type": "STOP_MARKET",
                    "stopPrice": str(sl_price),
                    "closePosition": True,
                    "timeInForce": "GTC",
                }
                # some SDKs prefer reduceOnly vs closePosition; include both (exchange ignores unsupported)
                sl_payload["reduceOnly"] = True
                sl_order = self.client.futures_create_order(**sl_payload)
                logger.info("[SmartExit] SL placed -> %s @ %s", symbol, sl_price)
            except Exception as e:
                logger.error("[SmartExit] SL failed for %s @ %s: %s", symbol, sl_price, e)
                sl_order = None

            return {
                "dry_run": False,
                "symbol": symbol,
                "side": side,
                "tp_levels": tp_levels,
                "partial_qtys": partial_qtys,
                "sl_price": sl_price,
                "tp_orders": tp_orders,
                "sl_order": sl_order,
                "tp_errors": placed_tp_errors,
                "position_confirmed": bool(confirmed),
            }

        except Exception as e:
            logger.exception("[SmartExit:FATAL] Unexpected error in create_exit_orders: %s", e)
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