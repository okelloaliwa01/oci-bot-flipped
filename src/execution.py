# src/execution.py
"""
Modernized, cleaned, production-safe ExecutionManager implementation.
- Removed background retry thread (Option A).
- MarketIntegrityGuard suspicion -> immediate rejection (structured REJECTED_SUSPECT).
- Prevent duplicate OPEN positions for the same symbol.
- Do not cache a position if the exchange/guard returns a rejection or order failed.
- Keeps original APIs and behavior but extracts helpers, unifies qty logic,
  and removes noisy logging.
"""

import math
import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from config import (
    SYMBOL,
    LEVERAGE,
    MARGIN_USDT,
    USE_PERCENT_MARGIN,
    MARGIN_PERCENT,
    TP_PERCENT,
    SL_PERCENT,
    DRY_RUN,
    ACCOUNT_BALANCE,
    VOLUME_MULTIPLIER,
    USE_SMART_EXIT,
)
from smart_exit import SmartExitManager
from guards.market_integrity import MarketIntegrityGuard

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "null", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class ExecutionManager:
    """
    Handles trade lifecycle in a safe, testable way.

    - open_position(symbol, direction, tp_percent, sl_percent)
    - close_position(symbol, side)
    - cancel_all_orders(symbol=None)
    - manage_positions_live()
    - reconcile_open_positions()
    - sync_exchange_state()
    """

    def __init__(
        self,
        binance_client,
        dry_run: bool = False,
        retry_interval: float = 10.0,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ):
        self.client = binance_client
        self.dry_run = dry_run or bool(DRY_RUN)
        self.smart_exit = SmartExitManager(binance_client)
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.guard = MarketIntegrityGuard()
        # keep these values for compatibility/inspectability but we removed retry threads
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        logger.info(
            "ExecutionManager initialized (dry_run=%s, retry_interval=%s, max_retries=%s) - retry thread disabled",
            self.dry_run,
            self.retry_interval,
            self.max_retries,
        )

    # ---------------------------
    # Helper extraction (single source of truth)
    # ---------------------------
    def _extract_usdt_balance(self, balances_resp: Any) -> float:
        """Robustly extract USDT balance from various Binance responses."""
        try:
            if isinstance(balances_resp, (list, tuple)):
                for b in balances_resp:
                    if isinstance(b, dict):
                        asset = str(b.get("asset", b.get("currency", ""))).upper()
                        if asset == "USDT":
                            val = b.get("balance") or b.get("free") or b.get("walletBalance") or 0.0
                            return float(val)
                return 0.0

            if isinstance(balances_resp, dict):
                if "USDT" in balances_resp and isinstance(balances_resp["USDT"], (int, float, str)):
                    return float(balances_resp["USDT"])
                for key in ("balances", "assets", "accountBalances", "data"):
                    inner = balances_resp.get(key)
                    if isinstance(inner, (list, tuple)):
                        return self._extract_usdt_balance(inner)
                if str(balances_resp.get("asset", "")).upper() == "USDT":
                    val = balances_resp.get("balance") or balances_resp.get("free") or 0.0
                    return float(val)
                return 0.0

            if isinstance(balances_resp, (int, float, str)):
                try:
                    return float(balances_resp)
                except Exception:
                    return 0.0
            return 0.0
        except Exception:
            return 0.0

    def _safe_parse_tp_mults(self, raw) -> List[float]:
        """Safely parse ATR_MULT_TP values from any format into a list of floats."""
        defaults = [2.0, 3.0, 4.0]
        try:
            if raw is None:
                return defaults
            if isinstance(raw, (list, tuple)):
                return [float(v) for v in raw if v is not None]
            if isinstance(raw, (int, float)):
                return [float(raw)]
            if isinstance(raw, str):
                clean = raw.replace("[", "").replace("]", "").strip()
                parts = [p.strip() for p in clean.split(",") if p.strip()]
                return [float(p) for p in parts] if parts else defaults
        except Exception:
            pass
        return defaults

    def _calc_atr(self, symbol: str, interval: str = "15m", samples: int = 30, atr_length: int = 14) -> Optional[float]:
        """Attempt to compute a simple ATR from klines; fallback to None."""
        try:
            klines = self.client.get_klines(symbol, interval, samples) or []
            if len(klines) <= atr_length:
                return None
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            trs = []
            for i in range(1, len(highs)):
                h = highs[i]
                l = lows[i]
                prev_c = closes[i - 1]
                trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
            if not trs:
                return None
            return float(sum(trs[-atr_length:]) / len(trs[-atr_length:]))
        except Exception:
            return None

    def _get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """Return a dict with useful symbol filters (tickSize, stepSize, minQty, minNotional)."""
        tick = 0.0
        step = 0.0
        min_qty = None
        min_notional = None
        try:
            info = self.client.get_symbol_info(symbol) or {}
            for f in info.get("filters", []):
                t = f.get("filterType")
                if t == "PRICE_FILTER":
                    tick = float(f.get("tickSize", tick))
                elif t == "LOT_SIZE":
                    step = float(f.get("stepSize", step))
                    try:
                        min_qty = float(f.get("minQty", min_qty or 0))
                    except Exception:
                        pass
                elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                    try:
                        min_notional = float(f.get("minNotional", f.get("notional", min_notional or 0)))
                    except Exception:
                        pass
        except Exception:
            # Non-fatal
            pass
        return {"tick": tick, "step": step, "min_qty": min_qty, "min_notional": min_notional}

    def _round_to(self, value: float, step: float) -> float:
        try:
            if not step or step <= 0:
                return value
            return math.floor(value / step) * step
        except Exception:
            return value

    # ---------------------------
    # Helper: is there an open position on the exchange?
    # ---------------------------
    def _is_open_on_exchange(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Try multiple methods to determine whether the exchange currently reports
        an open position for `symbol`. Returns normalized dict {'side','qty','entry','sl'} or None.
        """
        try:
            sym = symbol.upper()
            # 1) Preferred: smart_exit helper (already normalizes)
            try:
                if hasattr(self.smart_exit, "_get_position"):
                    pos = self.smart_exit._get_position(sym)
                    if pos:
                        return pos
            except Exception:
                pass

            # 2) Direct client.get_position (many clients expose this)
            try:
                if hasattr(self.client, "get_position"):
                    p = self.client.get_position(sym)
                    if isinstance(p, dict) and p:
                        # normalize similar to SmartExit expectations
                        qty = safe_float(p.get("positionAmt") or p.get("position") or p.get("qty") or p.get("quantity") or 0.0)
                        entry = safe_float(p.get("entryPrice") or p.get("avgPrice") or p.get("price") or 0.0)
                        side = None
                        if qty > 0:
                            side = "LONG"
                        elif qty < 0:
                            side = "SHORT"
                        if side is None or abs(qty) <= 0:
                            return None
                        sl_val = None
                        for k in ("stopPrice", "stop_price", "stopLoss", "stop_loss"):
                            if k in p and p.get(k) not in (None, "", 0, "0"):
                                sl_val = safe_float(p.get(k))
                                break
                        return {"side": side, "qty": abs(qty), "entry": entry, "sl": sl_val}
            except Exception:
                pass

            # 3) Position risk or all positions (list) fallback
            try:
                if hasattr(self.client, "get_position_risk"):
                    pr = self.client.get_position_risk(symbol)
                    if isinstance(pr, list) and pr:
                        for item in pr:
                            if isinstance(item, dict) and (item.get("symbol") == sym or item.get("symbol") == symbol):
                                qty = safe_float(item.get("positionAmt") or item.get("position") or 0.0)
                                entry = safe_float(item.get("entryPrice") or item.get("avgPrice") or 0.0)
                                if abs(qty) > 0:
                                    side = "LONG" if qty > 0 else "SHORT"
                                    sl_val = None
                                    for k in ("stopPrice", "stop_price", "stopLoss", "stop_loss"):
                                        if k in item and item.get(k) not in (None, "", 0, "0"):
                                            sl_val = safe_float(item.get(k))
                                            break
                                    return {"side": side, "qty": abs(qty), "entry": entry, "sl": sl_val}
            except Exception:
                pass

            # 4) REST fallback: get_all_positions / futures_position_information
            try:
                candidates = []
                if hasattr(self.client, "get_all_positions"):
                    candidates = self.client.get_all_positions() or []
                elif hasattr(self.client, "futures_position_information"):
                    candidates = self.client.futures_position_information() or []
                for item in (candidates or []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("symbol") == sym or item.get("symbol") == symbol:
                        qty = safe_float(item.get("positionAmt") or item.get("position") or item.get("quantity") or 0.0)
                        entry = safe_float(item.get("entryPrice") or item.get("avgPrice") or 0.0)
                        if abs(qty) > 0:
                            side = "LONG" if qty > 0 else "SHORT"
                            sl_val = None
                            for k in ("stopPrice", "stop_price", "stopLoss", "stop_loss"):
                                if k in item and item.get(k) not in (None, "", 0, "0"):
                                    sl_val = safe_float(item.get(k))
                                    break
                            return {"side": side, "qty": abs(qty), "entry": entry, "sl": sl_val}
            except Exception:
                pass

        except Exception:
            logger.debug("_is_open_on_exchange unexpected error", exc_info=True)
        return None


    # ---------------------------
    # Post-exit cleanup (cancel reduce-only / exit orders)
    # ---------------------------
    def _post_exit_cleanup(self, symbol: str) -> None:
        """
        Ensure any TP/SL/reduce-only orders for `symbol` are cancelled.
        Idempotent and safe to call even if SmartExit or client cancel APIs are missing.
        """
        try:
            # 1) Prefer SmartExit public API if available
            try:
                if hasattr(self.smart_exit, "cancel_all_exit_orders"):
                    cancelled = self.smart_exit.cancel_all_exit_orders(symbol)
                    logger.info("SmartExit cancelled exit orders for %s: count=%s", symbol, len(cancelled) if isinstance(cancelled, list) else "unknown")
                elif hasattr(self.smart_exit, "_cancel_all_exit_orders"):
                    cancelled = self.smart_exit._cancel_all_exit_orders(symbol)
                    logger.info("SmartExit (private) cancelled exit orders for %s: count=%s", symbol, len(cancelled) if isinstance(cancelled, list) else "unknown")
            except Exception as e:
                logger.warning("SmartExit cancel attempt failed for %s: %s", symbol, e)

            # 2) As a fallback, attempt exchange-wide cancel for symbol
            try:
                if hasattr(self.client, "futures_cancel_all_open_orders"):
                    self.client.futures_cancel_all_open_orders(symbol=symbol)
                    logger.info("Exchange: futures_cancel_all_open_orders called for %s", symbol)
                elif hasattr(self.client, "cancel_all_open_orders"):
                    self.client.cancel_all_open_orders(symbol=symbol)
                    logger.info("Exchange: cancel_all_open_orders called for %s", symbol)
            except Exception as e:
                logger.debug("Exchange-level cancel_all_open_orders failed for %s: %s", symbol, e)

            # 3) Finally, attempt to cancel any open orders returned by get_open_orders individually
            try:
                if hasattr(self.client, "get_open_orders"):
                    open_orders = self.client.get_open_orders(symbol) or []
                elif hasattr(self.client, "futures_get_open_orders"):
                    open_orders = self.client.futures_get_open_orders(symbol) or []
                else:
                    open_orders = []
                for o in open_orders:
                    try:
                        oid = o.get("orderId") or o.get("order_id") or o.get("id") or o.get("clientOrderId")
                        if not oid:
                            continue
                        # Try common cancel signatures
                        try:
                            self.client.cancel_order(symbol=symbol, order_id=oid)
                        except Exception:
                            try:
                                self.client.cancel_order(symbol=symbol, orderId=oid)
                            except Exception:
                                # best effort; ignore
                                continue
                    except Exception:
                        continue
            except Exception:
                logger.debug("Individual open-orders cancellation pass failed for %s", symbol)

            # 4) Normalize local cache if no live position exists
            try:
                exch_pos = self._is_open_on_exchange(symbol)
                if not exch_pos:
                    # no live position — ensure cached entry removed
                    if symbol in self.open_positions:
                        self.open_positions.pop(symbol, None)
                        logger.info("Post-exit cleanup: removed %s from local cache", symbol)
                else:
                    logger.debug("Post-exit cleanup: position still present on exchange for %s, skipping cache removal", symbol)
            except Exception:
                logger.debug("Post-exit cleanup: exchange position check failed for %s", symbol)

        except Exception as e:
            logger.exception("Post-exit cleanup unexpected failure for %s: %s", symbol, e)


    # ---------------------------
    # Unified quantity calculation
    # ---------------------------
    def _calc_quantity(
        self,
        symbol: str,
        margin_usdt: Optional[float] = None,
        mode: str = "percent",
        rl_qty: Optional[float] = None,
        fixed_qty: Optional[float] = None,
        leverage: Optional[int] = None,
    ) -> float:
        try:
            import os
            import math

            try:
                max_trade_pct = float(os.getenv("MAX_TRADE_PCT", "3.0")) / 100.0
            except Exception:
                max_trade_pct = 0.03

            if leverage is None:
                leverage = int(getattr(self, "leverage", LEVERAGE or 40))

            # RL override
            if mode == "rl" and rl_qty is not None and rl_qty > 0:
                return max(round(float(rl_qty), 6), 0.001)

            # Fixed override
            if mode == "fixed" and fixed_qty is not None and fixed_qty > 0:
                return max(round(float(fixed_qty), 6), 0.001)

            if margin_usdt is None:
                if USE_PERCENT_MARGIN:
                    margin_usdt = (MARGIN_PERCENT / 100.0) * float(ACCOUNT_BALANCE or 0)
                else:
                    margin_usdt = float(MARGIN_USDT or 0)

            real_usdt = None
            try:
                balances = self.client.futures_account_balance()
                real_usdt = self._extract_usdt_balance(balances)
                if real_usdt > 0:
                    margin_usdt = min(margin_usdt, real_usdt * 0.9)
            except Exception:
                pass

            target_notional = (margin_usdt * (leverage or 1))

            if real_usdt is not None:
                max_notional = real_usdt * max_trade_pct * (leverage or 1)
                target_notional = min(target_notional, max_notional)

            try:
                price_data = self.client.ticker_price(symbol=symbol)
                mark_price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            except Exception:
                return 0.001

            if mark_price <= 0:
                return 0.001

            qty = target_notional / mark_price
            qty *= float(VOLUME_MULTIPLIER or 1.0)

            try:
                filters = self._get_symbol_filters(symbol)
                min_qty = float(filters.get("min_qty", 0.0))
                min_notional = float(filters.get("min_notional", 0.0))
                step = float(filters.get("step", 0.0))
            except Exception:
                min_qty = 0.0
                min_notional = 0.0
                step = 0.0

            if min_qty > 0:
                qty = max(qty, min_qty)

            if min_notional > 0:
                min_qty_needed = (min_notional * 1.10) / mark_price
                qty = max(qty, min_qty_needed)

            if step > 0:
                qty = math.floor(qty / step) * step

            if min_notional > 0:
                if qty * mark_price < min_notional:
                    qty = math.ceil(((min_notional * 1.10) / mark_price) / step) * step

            qty = round(qty, 6)
            return max(qty, 0.001)

        except Exception as e:
            logger.exception("_calc_quantity error: %s", e)
            return 0.001

    # ---------------------------
    # Public API: open_position
    # ---------------------------
    def open_position(
        self,
        symbol: str,
        direction: str,
        tp_percent: float = TP_PERCENT,
        sl_percent: float = SL_PERCENT,
        size_mode: str = "percent",
        rl_qty: Optional[float] = None,
        fixed_qty: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Safe, hardened position opener.
        - No stale cache entries
        - MarketIntegrityGuard enforced
        - SmartExit protected
        - Exchange reconciliation after order
        """

        load_dotenv(override=True)
        # normalize symbol consistently for cache/exchange checks
        symbol = symbol.upper()
        direction = direction.upper()
        is_long = direction in ("LONG", "BUY")

        try:
            # -----------------------------------------------------
            # 🛑 1. Duplicate-open protection (verify with exchange before honoring cache)
            # -----------------------------------------------------
            existing = self.open_positions.get(symbol)
            if existing and existing.get("status") == "OPEN":
                # double-check exchange state in case cache is stale
                try:
                    exch_pos = self._is_open_on_exchange(symbol)
                    if exch_pos:
                        logger.warning("Attempt to open %s for %s but an OPEN position already exists (confirmed by exchange)", direction, symbol)
                        return {
                            "symbol": symbol,
                            "status": "FAILED_ALREADY_OPEN",
                            "reason": "already_open",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    else:
                        # stale cache: remove and continue
                        logger.info("Found stale OPEN cache for %s — clearing and continuing", symbol)
                        self.open_positions.pop(symbol, None)
                except Exception:
                    # if exchange check fails, be conservative and allow opening (rather than block)
                    logger.debug("Exchange check failed while validating existing cache for %s — clearing local cache and continuing", symbol)
                    self.open_positions.pop(symbol, None)

            # -----------------------------------------------------
            # 2. Sync server time — best effort
            # -----------------------------------------------------
            try:
                server_time = getattr(self.client, "futures_time", lambda: None)()
                if isinstance(server_time, dict) and "serverTime" in server_time:
                    offset = server_time["serverTime"] - int(time.time() * 1000)
                    setattr(self.client, "time_offset", offset)
            except Exception:
                logger.debug("Server time sync failed")

            # -----------------------------------------------------
            # 3. Price fetch
            # -----------------------------------------------------
            try:
                price_data = self.client.ticker_price(symbol)
                price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            except Exception:
                logger.warning("Failed to fetch ticker price for %s", symbol)
                return None

            if not price or price <= 0:
                logger.warning("Invalid price for %s: %s", symbol, price)
                return None

            # -----------------------------------------------------
            # 4. Symbol filters
            # -----------------------------------------------------
            filters = self._get_symbol_filters(symbol)
            tick = filters.get("tick", 0.0)
            step = filters.get("step", 0.0)
            min_qty = filters.get("min_qty")

            # -----------------------------------------------------
            # 5. ATR-based TP/SL
            # -----------------------------------------------------
            atr = self._calc_atr(symbol)
            if atr and atr > 0:
                env_tp = os.environ.get("ATR_MULT_TP")
                tp_mults = self._safe_parse_tp_mults(env_tp)
                raw_sl = os.environ.get("ATR_MULT_SL")
                sl_mult = float(raw_sl) if raw_sl else 1.5

                if is_long:
                    tp_levels = [self._round_to(price + atr * m, tick) for m in tp_mults]
                    sl_price = self._round_to(price - atr * sl_mult, tick)
                else:
                    tp_levels = [self._round_to(price - atr * m, tick) for m in tp_mults]
                    sl_price = self._round_to(price + atr * sl_mult, tick)
            else:
                # % fallback
                if is_long:
                    tp_levels = [self._round_to(price * (1 + tp_percent / 100.0), tick)]
                    sl_price = self._round_to(price * (1 - sl_percent / 100.0), tick)
                else:
                    tp_levels = [self._round_to(price * (1 - tp_percent / 100.0), tick)]
                    sl_price = self._round_to(price * (1 + sl_percent / 100.0), tick)

            # -----------------------------------------------------
            # 6. Quantity calculation
            # -----------------------------------------------------
            qty = self._calc_quantity(
                symbol=symbol,
                margin_usdt=None,
                mode=size_mode,
                rl_qty=rl_qty,
                fixed_qty=fixed_qty,
                leverage=None,
            )

            qty = self._round_to(qty, step) if step and step > 0 else round(qty, 6)

            if min_qty and qty < float(min_qty):
                logger.error("Qty validation failed for %s: qty %s < min_qty %s", symbol, qty, min_qty)
                return {"symbol": symbol, "status": "FAILED_VALIDATION", "error": f"qty<{min_qty}"}

            # -----------------------------------------------------
            # 7. DRY-RUN mode
            # -----------------------------------------------------
            if self.dry_run:
                pos = {
                    "symbol": symbol,
                    "side": direction,
                    "entry": price,
                    "qty": qty,
                    "tp_levels": tp_levels,
                    "sl": sl_price,
                    "atr": atr,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "DRY_RUN",
                }
                self.open_positions[symbol] = pos
                logger.info("DRY-RUN: %s %s qty=%s", direction, symbol, qty)
                return pos

            # -----------------------------------------------------
            # 8. Market Integrity Guard
            # -----------------------------------------------------
            orderbook = self._fetch_orderbook(symbol)
            trades = self._fetch_recent_trades(symbol)
            suspect, reason = self.guard.check(orderbook, trades, events_per_sec=0.0)

            if suspect:
                logger.warning("MarketIntegrityGuard rejected %s: %s", symbol, reason)
                return {
                    "symbol": symbol,
                    "status": "REJECTED_SUSPECT",
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # -----------------------------------------------------
            # 9. Pre-order notional validation
            # -----------------------------------------------------
            try:
                live_price = float(self.client.ticker_price(symbol)["price"])
            except Exception:
                live_price = price

            min_notional = float(filters.get("min_notional") or 0.0)
            notional = float(qty) * float(live_price)

            if min_notional and notional < min_notional:
                logger.error("ORDER BLOCKED: notional %.8f < min_notional %.8f", notional, min_notional)
                return {"symbol": symbol, "status": "FAILED_VALIDATION", "error": f"notional {notional:.8f} < {min_notional}"}

            # -----------------------------------------------------
            # 10. MARKET ORDER → attempt execution
            # -----------------------------------------------------
            try:
                order_result = self.client.futures_create_order(
                    symbol=symbol,
                    side="BUY" if is_long else "SELL",
                    type="MARKET",
                    quantity=qty,
                )
                logger.info("Market order placed for %s %s qty=%s", symbol, direction, qty)
            except Exception as e:
                logger.exception("Market order failed for %s: %s", symbol, e)
                return {"symbol": symbol, "status": "PENDING_ORDER_FAILED", "error": str(e)}

            # -----------------------------------------------------
            # 🧹 11. POST-ORDER SAFETY — reject if order_result is bad
            # -----------------------------------------------------
            if isinstance(order_result, dict):
                bad_status = order_result.get("status")
                code = order_result.get("code")

                if bad_status in ("REJECTED_SUSPECT", "FAILED_VALIDATION", "PENDING_ORDER_FAILED", "FAILED_UNHANDLED"):
                    return order_result

                if isinstance(code, (int, float)) and int(code) < 0:
                    return {
                        "symbol": symbol,
                        "status": "PENDING_ORDER_FAILED",
                        "error": str(order_result),
                    }

            # -----------------------------------------------------
            # 🧠 12. SmartExit (with safety)
            # -----------------------------------------------------
            smartexit_err = None
            if USE_SMART_EXIT:
                try:
                    time.sleep(0.6)
                    self.smart_exit.create_exit_orders(
                        symbol=symbol,
                        side=direction,
                        entry_price=price,
                        qty=qty,
                        atr_value=atr,
                        tick_size=tick,
                        step_size=step,
                        tp_levels=tp_levels,
                    )
                except Exception as e:
                    smartexit_err = str(e)
                    logger.exception("SmartExit.create_exit_orders failed for %s: %s", symbol, e)

            # -----------------------------------------------------
            # 🔥 13. FINAL SAFETY — RECONCILE WITH LIVE EXCHANGE
            # -----------------------------------------------------
            time.sleep(0.5)

            pos = None
            try:
                pos = self.smart_exit._get_position(symbol)
            except Exception:
                pos = None

            if not pos:
                # Order placed but no actual position detected → treat as FAIL
                logger.error("Order executed but no exchange position found — clearing cache")
                self.open_positions.pop(symbol, None)
                return {"symbol": symbol, "status": "FAILED_NO_POSITION_AFTER_ORDER"}

            # -----------------------------------------------------
            # 💾 14. Cache only REAL exchange-confirmed position
            # -----------------------------------------------------
            tracked = {
                "symbol": symbol,
                "side": pos["side"],
                "entry": pos["entry"],
                "qty": pos["qty"],
                "tp_levels": tp_levels,
                "sl": sl_price,
                "atr": atr,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "OPEN",
                "order_result": order_result,
                "smartexit_error": smartexit_err,
            }

            self.open_positions[symbol] = tracked
            return tracked

        except Exception as e:
            logger.exception("Unhandled open_position error for %s: %s", symbol, e)
            return {"symbol": symbol, "status": "FAILED_UNHANDLED", "error": str(e)}


    # ---------------------------
    # Reusable network helpers
    # ---------------------------
    def _fetch_orderbook(self, symbol: str, limit: int = 50) -> Dict[str, List[Any]]:
        try:
            for name in ("depth", "get_order_book", "get_depth", "order_book"):
                if hasattr(self.client, name):
                    try:
                        resp = getattr(self.client, name)(symbol=symbol, limit=limit)
                        if isinstance(resp, dict):
                            bids = resp.get("bids") or []
                            asks = resp.get("asks") or []
                            return {"bids": list(bids), "asks": list(asks)}
                    except Exception:
                        continue
            import requests
            url = "https://fapi.binance.com/fapi/v1/depth"
            r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=3)
            r.raise_for_status()
            j = r.json()
            return {"bids": j.get("bids") or [], "asks": j.get("asks") or []}
        except Exception as e:
            logger.debug("_fetch_orderbook error: %s", e)
            return {"bids": [], "asks": []}

    def _fetch_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            for name in ("trades", "recent_trades", "get_recent_trades", "agg_trades"):
                if hasattr(self.client, name):
                    try:
                        resp = getattr(self.client, name)(symbol=symbol, limit=limit)
                        if isinstance(resp, list):
                            out: List[Dict[str, Any]] = []
                            for t in resp:
                                price = safe_float(t.get("price") or t.get("p"), 0.0)
                                qty = safe_float(t.get("qty") or t.get("q"), 0.0)
                                if "isBuyerMaker" in t:
                                    side = "sell" if t["isBuyerMaker"] else "buy"
                                else:
                                    side = str(t.get("side") or t.get("S") or "").lower()
                                out.append({"price": price, "qty": qty, "side": side})
                            return out
                    except Exception:
                        continue
            import requests
            url = "https://fapi.binance.com/fapi/v1/trades"
            r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=3)
            r.raise_for_status()
            data = r.json()
            out: List[Dict[str, Any]] = []
            for t in data:
                price = safe_float(t.get("price"), 0.0)
                qty = safe_float(t.get("qty"), 0.0)
                side = "buy" if t.get("isBuyerMaker") is False else "sell"
                out.append({"price": price, "qty": qty, "side": side})
            return out
        except Exception as e:
            logger.debug("_fetch_recent_trades error: %s", e)
            return []

    # ---------------------------
    # Close position
    # ---------------------------
    def close_position(self, symbol: str, side: str) -> None:
        import json
        import math
        from typing import Any

        def _safe_float(x: Any, default: float = 0.0) -> float:
            try:
                if x is None:
                    return default
                return float(x)
            except (ValueError, TypeError):
                return default

        if symbol not in self.open_positions:
            logger.warning("No open position cached for %s", symbol)
            return

        pos = self.open_positions[symbol]
        qty = _safe_float(pos.get("qty"), 0.0)
        if qty <= 0:
            logger.warning("Invalid qty=%.8f for %s — cannot close", qty, symbol)
            return

        opposite = "SELL" if side.upper() in ("LONG", "BUY") else "BUY"

        # Cancel SmartExit-related orders first
        try:
            if USE_SMART_EXIT and hasattr(self.smart_exit, "_cancel_all_exit_orders"):
                logger.info("Cancelling SmartExit TP/SL for %s before closing", symbol)
                self.smart_exit._cancel_all_exit_orders(symbol)
            else:
                self.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning("SmartExit cleanup failed for %s: %s", symbol, e)

        # Fetch filters safely
        filters = self._get_symbol_filters(symbol) or {}
        step_size = safe_float(filters.get("step"))
        min_qty = safe_float(filters.get("min_qty"))
        min_notional = safe_float(filters.get("min_notional"))

        def _round_down_step(q: float, step: float) -> float:
            if step > 0:
                return math.floor(q / step) * step
            return round(q, 6)

        rounded_qty = _round_down_step(qty, step_size)
        if rounded_qty <= 0 and qty > 0:
            rounded_qty = round(qty, 6)

        # Fetch mark price safely
        mark_price_raw = None
        try:
            price_data = self.client.ticker_price(symbol)
            mark_price_raw = price_data.get("price") if isinstance(price_data, dict) else price_data
        except Exception:
            logger.debug("Could not fetch mark price; proceeding without minNotional check")
        mark_price = safe_float(mark_price_raw)

        # Ensure rounded_qty meets minNotional
        if min_notional > 0 and mark_price > 0:
            notional = rounded_qty * mark_price
            if notional < min_notional:
                min_qty_needed = min_notional / mark_price if mark_price > 0 else 0.0
                if step_size > 0:
                    min_qty_needed = math.ceil(min_qty_needed / step_size) * step_size
                else:
                    min_qty_needed = round(min_qty_needed, 6)
                desired_qty = min(min_qty_needed, qty)
                desired_qty = _round_down_step(desired_qty, step_size) if step_size > 0 else round(desired_qty, 6)
                desired_notional = desired_qty * mark_price
                if desired_qty > 0 and desired_notional >= min_notional:
                    logger.info(
                        "Adjusting close qty to meet minNotional: %.8f -> %.8f (min_notional %.8f, price %.8f)",
                        rounded_qty, desired_qty, min_notional, mark_price
                    )
                    rounded_qty = desired_qty
                else:
                    logger.warning(
                        "Available qty %.8f cannot meet minNotional %.8f at price %.8f. Best-effort close %.8f",
                        qty, min_notional, mark_price, rounded_qty
                    )

        # Validate min_qty
        if min_qty > 0 and rounded_qty < min_qty:
            if qty >= min_qty:
                desired = _round_down_step(min_qty, step_size) if step_size > 0 else round(min_qty, 6)
                if desired <= qty:
                    logger.info("Raising close qty to min_qty: %.8f -> %.8f", rounded_qty, desired)
                    rounded_qty = desired
                else:
                    logger.warning("Cannot raise to min_qty %.8f; available qty %.8f smaller", min_qty, qty)
            else:
                logger.warning(
                    "Rounded qty %.8f below min_qty %.8f and available qty %.8f < min_qty — aborting close",
                    rounded_qty, min_qty, qty
                )
                return

        if rounded_qty <= 0:
            logger.warning("Rounded qty invalid (<=0) for %s", symbol)
            return

        # Final safety
        rounded_qty = min(rounded_qty, qty)

        if self.dry_run:
            logger.info("[DRY RUN] Would close %s %s qty=%.8f", symbol, opposite, rounded_qty)
            self.open_positions.pop(symbol, None)
            return

        # Detect dual-side position mode safely
        position_side_flag = None
        try:
            acc_info = None
            if callable(getattr(self.client, "futures_account", None)):
                acc_info = self.client.futures_account()
            elif callable(getattr(self.client, "get_futures_account", None)):
                acc_info = self.client.get_futures_account()
            if isinstance(acc_info, dict) and acc_info.get("dualSidePosition"):
                orig_side = str(pos.get("side", "")).upper()
                if orig_side in ("LONG", "BUY"):
                    position_side_flag = "LONG"
                elif orig_side in ("SHORT", "SELL"):
                    position_side_flag = "SHORT"
        except Exception:
            pass

        # Build market close order kwargs
        order_kwargs = {"symbol": symbol, "side": opposite, "type": "MARKET", "quantity": rounded_qty}
        if position_side_flag:
            order_kwargs["positionSide"] = position_side_flag
        order_kwargs["reduceOnly"] = True

        try:
            logger.debug("Placing CLOSE order: %s", {k: v for k, v in order_kwargs.items() if k != "quantity"})
            order = self.client.futures_create_order(**order_kwargs)
            order_id = None
            if isinstance(order, dict):
                order_id = order.get("orderId") or order.get("order_id") or order.get("id") or order.get("clientOrderId")
            logger.info("Closed %s %s | qty=%.8f order_id=%s", symbol, opposite, rounded_qty, order_id or "?")
        except Exception as e:
            err_text = str(e)
            try:
                resp = getattr(e, "response", None)
                if resp and hasattr(resp, "text"):
                    txt = resp.text
                    try:
                        j = json.loads(txt)
                        err_text = f"{j.get('code')}: {j.get('msg')}"
                    except Exception:
                        err_text = txt
                elif hasattr(e, "args") and e.args and isinstance(e.args[0], str):
                    try:
                        j = json.loads(e.args[0])
                        err_text = f"{j.get('code')}: {j.get('msg')}"
                    except Exception:
                        err_text = e.args[0]
            except Exception:
                pass
            logger.error("Error closing %s: %s", symbol, err_text)

        # Cleanup
                # Final cleanup: ensure any reduce-only/exit orders are cancelled and cache is normalized
        try:
            # best-effort post-exit cleanup (idempotent)
            self._post_exit_cleanup(symbol)
        except Exception as e:
            logger.warning("Post-exit cleanup failed for %s: %s", symbol, e)

        # Ensure local cache is removed
        self.open_positions.pop(symbol, None)
        logger.info("%s removed from local cache", symbol)


    # ---------------------------
    # Cancel orders
    # ---------------------------
    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        if self.dry_run:
            if symbol:
                logger.info("[DRY RUN] Would cancel all orders for %s", symbol)
            else:
                logger.info("[DRY RUN] Would cancel all open orders (global)")
            return

        try:
            symbols_to_cancel = []
            if symbol:
                symbols_to_cancel = [symbol]
            else:
                open_orders = []
                if hasattr(self.client, "futures_get_open_orders"):
                    open_orders = self.client.futures_get_open_orders() or []
                elif hasattr(self.client, "get_open_orders"):
                    open_orders = self.client.get_open_orders() or []
                for o in open_orders:
                    sym = o.get("symbol")
                    if sym and sym not in symbols_to_cancel:
                        symbols_to_cancel.append(sym)

            if not symbols_to_cancel:
                logger.info("No open orders found to cancel.")
                return

            for sym in symbols_to_cancel:
                try:
                    if hasattr(self.client, "futures_cancel_all_open_orders"):
                        self.client.futures_cancel_all_open_orders(symbol=sym)
                    elif hasattr(self.client, "cancel_all_open_orders"):
                        self.client.cancel_all_open_orders(symbol=sym)
                    else:
                        logger.warning("No cancel_all_orders() endpoint available on client.")
                        continue
                    logger.info("All open orders cancelled for %s", sym)
                except Exception as e:
                    logger.error("Failed to cancel orders for %s: %s", sym, e)
        except Exception as e:
            logger.exception("cancel_all_orders failed: %s", e)

    # ---------------------------
    # SmartExit management
    # ---------------------------
    def manage_positions_live(self) -> None:
        if not USE_SMART_EXIT:
            return
        for symbol in list(self.open_positions.keys()):
            try:
                self.smart_exit.manage_open_positions(symbol)
            except Exception as e:
                logger.warning("Smart exit management failed for %s: %s", symbol, e)

    # ---------------------------
    # Reconcile & manage
    # ---------------------------
    def reconcile_open_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.open_positions

    def manage_open_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Manage open positions using SmartExit. Returns a dictionary mapping
        symbols to SmartExit results.
        """
        if not USE_SMART_EXIT:
            return {}

        targets = [symbol] if symbol else list(self.open_positions.keys())
        results: Dict[str, Any] = {}
        failed: List[str] = []

        _get_position = getattr(self.smart_exit, "_get_position", lambda s: None)

        for sym in targets:
            try:
                prev = _get_position(sym) or {}
                prev_sl = prev.get("sl")
                
                res = self.smart_exit.manage_open_positions(sym)
                post = _get_position(sym) or {}
                new_sl = post.get("sl")

                # If SmartExit reports no live position but we still have cached OPEN, perform cleanup
                if not post and sym in self.open_positions:
                    try:
                        logger.info("Detected external close for %s — running post-exit cleanup and clearing cache", sym)
                        self._post_exit_cleanup(sym)
                    except Exception as e:
                        logger.warning("manage_open_positions post-exit cleanup failed for %s: %s", sym, e)
                    # remove cached entry to stay consistent with exchange
                    self.open_positions.pop(sym, None)


                if res and isinstance(res, dict):
                    rtype = res.get("type")
                    if rtype in ("trailing", "breakeven") and new_sl != prev_sl:
                        logger.info("[SmartExit] %s SL change: %s -> %s", sym, prev_sl, new_sl)
                    # Optional: log all SL changes
                    elif new_sl != prev_sl:
                        logger.debug("[SmartExit] %s SL changed (non-trailing/breakeven): %s -> %s", sym, prev_sl, new_sl)

                results[sym] = res

            except Exception as e:
                logger.exception("manage_open_positions failed for %s: %s", sym, e)
                failed.append(sym)

        if failed:
            logger.warning("manage_open_positions failed for symbols: %s", ", ".join(failed))

        return results


    # ---------------------------
    # Cache utilities
    # ---------------------------
    def remove_cached(self, symbol: Optional[str] = None) -> None:
        if symbol:
            self.open_positions.pop(symbol, None)
            logger.info("Removed cached position for %s", symbol)
        else:
            self.open_positions.clear()
            logger.info("Cleared all cached positions")

    # ---------------------------
    # Sync exchange state
    # ---------------------------
    def sync_exchange_state(self) -> Dict[str, Any]:
        try:
            if self.dry_run:
                return self.open_positions

            # ---------------------------
            # 1. Fetch live positions
            # ---------------------------
            live_positions_raw = []
            try:
                if hasattr(self.client, "futures_position_information"):
                    res = self.client.futures_position_information()
                    if isinstance(res, list):
                        live_positions_raw = res
                if not live_positions_raw and hasattr(self.client, "get_all_positions"):
                    live_positions_raw = getattr(self.client, "get_all_positions")() or []
            except Exception as e:
                logger.debug("Could not fetch live positions: %s", e)

            # ---------------------------
            # 2. Normalize positions
            # ---------------------------
            live_positions = {}
            for p in live_positions_raw or []:
                try:
                    amt = safe_float(p.get("positionAmt") or p.get("position") or p.get("quantity") or 0.0)
                    if abs(amt) > 0:
                        sym = str(p.get("symbol") or "").upper()
                        if sym:
                            live_positions[sym] = p
                            # Optionally update local cache with latest qty/sl
                            if sym in self.open_positions:
                                self.open_positions[sym].update({"qty": amt, "entry": safe_float(p.get("entryPrice") or 0.0)})
                except Exception:
                    continue

            # ---------------------------
            # 3. Fetch open orders
            # ---------------------------
            open_orders = []
            try:
                if hasattr(self.client, "futures_get_open_orders"):
                    open_orders = getattr(self.client, "futures_get_open_orders")() or []
                elif hasattr(self.client, "get_open_orders"):
                    open_orders = getattr(self.client, "get_open_orders")() or []
            except Exception as e:
                logger.debug("Could not fetch open orders: %s", e)

            order_symbols = {str(o.get("symbol") or "").upper() for o in open_orders if isinstance(o, dict)}

            # ---------------------------
            # 4. Clean stale cache
            # ---------------------------
            stale_cached = set(self.open_positions.keys()) - set(live_positions.keys())
            for s in stale_cached:
                logger.info("Cleaning stale cache for %s", s)
                try:
                    self._post_exit_cleanup(s)
                except Exception as e:
                    logger.warning("Post-exit cleanup failed while cleaning stale cache for %s: %s", s, e)
                self.remove_cached(s)


            # ---------------------------
            # 5. Detect orphaned orders
            # ---------------------------
            orphaned = order_symbols - set(live_positions.keys())
            for s in orphaned:
                logger.warning("Orphaned orders detected for %s", s)

            logger.info(
                "sync_exchange_state complete: %s live positions, %s cached",
                len(live_positions),
                len(self.open_positions),
            )
            return live_positions

        except Exception as e:
            logger.exception("sync_exchange_state failed: %s", e)
            return {}