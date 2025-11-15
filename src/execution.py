# src/execution.py
"""
Modernized, cleaned, production-safe ExecutionManager implementation.
- Keeps original APIs and behavior but extracts helpers, unifies qty logic,
  and removes noisy logging (clean logs).
- Default sizing: percentage-of-balance with leverage. RL sizing optional.
- No emoji logging; uses standard logging.
"""

import math
import os
import threading
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

    Design goals:
    - Extract helpers to avoid duplication
    - Unified quantity calculation (_calc_quantity)
    - DRY-RUN safe
    - Clean logging (no emojis)
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
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._stop_retry = False

        # start background retry thread
        self._retry_thread = threading.Thread(target=self._pending_retry_loop, daemon=True)
        self._retry_thread.start()
        logger.info(
            "ExecutionManager initialized (dry_run=%s, retry_interval=%s, max_retries=%s)",
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

        # Debug line to see what is passed in
        logger.debug("Parsing ATR_MULT_TP, received raw value: %s", raw)

        try:
            if raw is None:
                return defaults
            if isinstance(raw, (list, tuple)):
                return [float(v) for v in raw if v is not None]
            if isinstance(raw, (int, float)):
                return [float(raw)]
            if isinstance(raw, str):
                # Remove brackets and whitespace
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
            # mean of last atr_length TRs
            return float(sum(trs[-atr_length:]) / len(trs[-atr_length:]))
        except Exception:
            return None

    def _get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """Return a dict with useful symbol filters (tickSize, stepSize, minQty, minNotional).
        Non-fatal: returns defaults if fetch fails.
        """
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
            # Non-fatal: return defaults
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
        """
        Unified quantity calculation with:
        - MAX_TRADE_PCT safety cap
        - minQty safeguard
        - minNotional safeguard (with buffer)
        - stepSize alignment
        - returns an always-valid quantity

        returns: float qty (base asset)
        """
        try:
            import os
            import math

            # -------------------------
            # Load MAX_TRADE_PCT
            # -------------------------
            try:
                max_trade_pct = float(os.getenv("MAX_TRADE_PCT", "3.0")) / 100.0
            except Exception:
                max_trade_pct = 0.03  # fallback 3%

            # -------------------------
            # Load leverage
            # -------------------------
            if leverage is None:
                leverage = int(getattr(self, "leverage", LEVERAGE or 40))

            # -------------------------
            # RL override
            # -------------------------
            if mode == "rl" and rl_qty is not None and rl_qty > 0:
                return max(round(float(rl_qty), 6), 0.001)

            # -------------------------
            # Fixed override
            # -------------------------
            if mode == "fixed" and fixed_qty is not None and fixed_qty > 0:
                return max(round(float(fixed_qty), 6), 0.001)

            # -------------------------
            # Percent-of-balance mode
            # -------------------------
            if margin_usdt is None:
                if USE_PERCENT_MARGIN:
                    margin_usdt = (MARGIN_PERCENT / 100.0) * float(ACCOUNT_BALANCE or 0)
                else:
                    margin_usdt = float(MARGIN_USDT or 0)

            # Attempt real USDT balance
            real_usdt = None
            try:
                balances = self.client.futures_account_balance()
                real_usdt = self._extract_usdt_balance(balances)
                if real_usdt > 0:
                    margin_usdt = min(margin_usdt, real_usdt * 0.9)
            except Exception:
                pass

            # -------------------------
            # Compute notional (with leverage)
            # -------------------------
            target_notional = (margin_usdt * (leverage or 1))

            # Hard cap by MAX_TRADE_PCT
            if real_usdt is not None:
                max_notional = real_usdt * max_trade_pct * (leverage or 1)
                target_notional = min(target_notional, max_notional)

            # -------------------------
            # Get mark price
            # -------------------------
            try:
                price_data = self.client.ticker_price(symbol=symbol)
                mark_price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            except Exception:
                return 0.001

            if mark_price <= 0:
                return 0.001

            # -------------------------
            # Base quantity
            # -------------------------
            qty = target_notional / mark_price

            # Apply multiplier
            qty *= float(VOLUME_MULTIPLIER or 1.0)

            # -------------------------
            # Load symbol filters
            # -------------------------
            try:
                filters = self._get_symbol_filters(symbol)
                min_qty = float(filters.get("min_qty", 0.0))
                min_notional = float(filters.get("min_notional", 0.0))
                step = float(filters.get("step", 0.0))
            except Exception:
                min_qty = 0.0
                min_notional = 0.0
                step = 0.0

            # -------------------------
            # Enforce minQty
            # -------------------------
            if min_qty > 0:
                qty = max(qty, min_qty)

            # -------------------------
            # FIRST minNotional check (raw)
            # add 10% safety buffer to avoid Binance testnet errors
            # -------------------------
            if min_notional > 0:
                min_qty_needed = (min_notional * 1.10) / mark_price
                qty = max(qty, min_qty_needed)

            # -------------------------
            # Align to stepSize
            # -------------------------
            if step > 0:
                qty = math.floor(qty / step) * step

            # -------------------------
            # SECOND minNotional safety check (after step rounding)
            # -------------------------
            if min_notional > 0:
                if qty * mark_price < min_notional:
                    # Increase until minNotional met
                    qty = math.ceil(((min_notional * 1.10) / mark_price) / step) * step

            # -------------------------
            # Final rounding
            # -------------------------
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
        Open a futures position in a safe, traceable manner.

        - size_mode: 'percent' (default), 'rl', or 'fixed'
        - rl_qty: base-asset qty if RL mode
        - fixed_qty: base-asset qty if fixed mode

        Returns a dict describing the opened/tracked position or None on failure.
        """
        load_dotenv(override=True)
        direction = direction.upper()
        is_long = direction in ("LONG", "BUY")

        try:
            # ---------------------------
            # Server time sync
            # ---------------------------
            try:
                server_time = self.client.futures_time()
                if isinstance(server_time, dict) and "serverTime" in server_time:
                    offset = server_time["serverTime"] - int(time.time() * 1000)
                    try:
                        self.client.time_offset = offset
                    except Exception:
                        pass
            except Exception:
                logger.debug("Server time sync failed")

            # ---------------------------
            # Fetch current price
            # ---------------------------
            price = None
            try:
                price_data = self.client.ticker_price(symbol)
                price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            except Exception:
                logger.warning("Failed to fetch ticker price for %s", symbol)
                return None

            if not price or price <= 0:
                logger.warning("Invalid price for %s: %s", symbol, price)
                return None

            # ---------------------------
            # Symbol filters
            # ---------------------------
            filters = self._get_symbol_filters(symbol)
            tick = filters.get("tick", 0.0)
            step = filters.get("step", 0.0)
            min_qty = filters.get("min_qty")
            # min_notional handled at final validation

            # ---------------------------
            # ATR-based TP/SL
            # ---------------------------
            atr = self._calc_atr(symbol)
            if atr and atr > 0:
                import os
                env_tp = os.environ.get("ATR_MULT_TP")
                tp_mults = self._safe_parse_tp_mults(env_tp)

                raw_sl = os.environ.get("ATR_MULT_SL")
                try:
                    sl_mult = float(raw_sl) if raw_sl else 1.5
                except Exception:
                    sl_mult = 1.5

                if is_long:
                    tp_levels = [self._round_to(price + atr * m, tick) for m in tp_mults]
                    sl_price = self._round_to(price - atr * sl_mult, tick)
                else:
                    tp_levels = [self._round_to(price - atr * m, tick) for m in tp_mults]
                    sl_price = self._round_to(price + atr * sl_mult, tick)
            else:
                if is_long:
                    tp_levels = [self._round_to(price * (1 + tp_percent / 100.0), tick)]
                    sl_price = self._round_to(price * (1 - sl_percent / 100.0), tick)
                else:
                    tp_levels = [self._round_to(price * (1 - tp_percent / 100.0), tick)]
                    sl_price = self._round_to(price * (1 + sl_percent / 100.0), tick)

            # ---------------------------
            # Quantity calculation
            # ---------------------------
            qty = self._calc_quantity(
                symbol=symbol,
                margin_usdt=None,
                mode=size_mode,
                rl_qty=rl_qty,
                fixed_qty=fixed_qty,
                leverage=None,
            )
            qty = self._round_to(qty, step) if step and step > 0 else round(qty, 6)

            # Validate min_qty only
            if min_qty and qty < float(min_qty):
                failure = {"symbol": symbol, "status": "FAILED_VALIDATION", "error": f"qty<{min_qty}"}
                self.open_positions[symbol] = failure
                logger.error("Qty validation failed for %s: %s < %s", symbol, qty, min_qty)
                return failure

            # ---------------------------
            # DRY-RUN handling
            # ---------------------------
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

            # ---------------------------
            # Market integrity check
            # ---------------------------
            orderbook = self._fetch_orderbook(symbol)
            trades = self._fetch_recent_trades(symbol)
            suspect, reason = self.guard.check(orderbook, trades, events_per_sec=0.0)
            if suspect:
                logger.warning("MarketIntegrityGuard flagged %s: %s", symbol, reason)
                pending = {
                    "symbol": symbol,
                    "status": "PENDING_SUSPECT",
                    "suspect_reason": reason,
                    "entry": price,
                    "qty": qty,
                    "tp_levels": tp_levels,
                    "sl": sl_price,
                    "atr": atr,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                self.open_positions[symbol] = pending
                return pending

            # ---------------------------
            # Set leverage (best-effort)
            # ---------------------------
            try:
                if hasattr(self.client, "futures_change_leverage"):
                    self.client.futures_change_leverage(symbol=symbol, leverage=int(LEVERAGE or 10))
            except Exception:
                logger.debug("Leverage set failed for %s", symbol)

            # -------------------------------------------------------
            # 🔥 PRE-ORDER VALIDATION (minNotional re-check)
            # -------------------------------------------------------
            try:
                price_data = self.client.ticker_price(symbol=symbol)
                live_price = float(price_data["price"])
            except Exception:
                live_price = price

            min_notional = float(filters.get("min_notional", 0.0))
            notional = float(qty) * float(live_price)

            logger.warning(
                f"❗Pre-Order Validation: qty={qty}, price={live_price}, "
                f"notional={notional}, minNotional={min_notional}"
            )

            if notional < min_notional:
                logger.error(
                    f"❌ ORDER BLOCKED — Notional {notional} < minNotional {min_notional}"
                )
                failure = {
                    "symbol": symbol,
                    "status": "FAILED_VALIDATION",
                    "error": f"notional {notional:.4f} < minNotional {min_notional}",
                }
                self.open_positions[symbol] = failure
                return failure
            # -------------------------------------------------------

            # ---------------------------
            # Place market order
            # ---------------------------
            try:
                order_result = self.client.futures_create_order(
                    symbol=symbol,
                    side="BUY" if is_long else "SELL",
                    type="MARKET",
                    quantity=qty,
                )
                logger.info("Opened %s %s @%s qty=%s", direction, symbol, price, qty)
            except Exception as e:
                logger.exception("Market order failed for %s: %s", symbol, e)
                return {"symbol": symbol, "status": "PENDING_ORDER_FAILED", "error": str(e)}

            # ---------------------------
            # SmartExit integration
            # ---------------------------
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

            # ---------------------------
            # Cache and return
            # ---------------------------
            tracked = {
                "symbol": symbol,
                "side": direction,
                "entry": price,
                "qty": qty,
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
    # Pending retry loop
    # ---------------------------
    def _pending_retry_loop(self) -> None:
        import time
        from datetime import datetime

        while not getattr(self, "_stop_retry", False):
            try:
                time.sleep(getattr(self, "retry_interval", 5))

                pending = [
                    (symbol, pos)
                    for symbol, pos in self.open_positions.items()
                    if pos.get("status") == "PENDING_SUSPECT"
                ]
                pending.sort(key=lambda x: x[1].get("last_retry_ts", datetime.min))

                for symbol, pos in pending:
                    if getattr(self, "_stop_retry", False):
                        break

                    retries = pos.get("retry_count", 0)
                    backoff = pos.get("retry_backoff", getattr(self, "retry_interval", 5))

                    if retries >= getattr(self, "max_retries", 5):
                        pos["status"] = "FAILED_MAX_RETRIES"
                        self.open_positions[symbol] = pos
                        logger.error("Max retries reached for %s", symbol)
                        continue

                    logger.info("Retrying %s (attempt %s)", symbol, retries + 1)

                    # Use configured TP/SL from config when retrying
                    self.open_position(symbol, pos.get("side", "LONG"), TP_PERCENT, SL_PERCENT)

                    # Refresh updated position
                    pos = self.open_positions.get(symbol, pos)

                    pos["retry_count"] = retries + 1
                    pos["retry_backoff"] = backoff * getattr(self, "retry_backoff", 2.0)
                    pos["last_retry_ts"] = datetime.utcnow()
                    self.open_positions[symbol] = pos

                    time.sleep(pos["retry_backoff"])

            except Exception as e:
                logger.exception("_pending_retry_loop error: %s", e)

    def stop_retry_thread(self) -> None:
        self._stop_retry = True
        try:
            if hasattr(self, "_retry_thread") and self._retry_thread.is_alive():
                self._retry_thread.join(timeout=5)
        except Exception:
            pass

    def __del__(self):
        try:
            self.stop_retry_thread()
        except Exception:
            pass

    # ---------------------------
    # Close position
    # ---------------------------
    def close_position(self, symbol: str, side: str) -> None:
        import json
        import math
        from datetime import datetime
        from typing import Any

        # Helper to safely convert any value to float
        def safe_float(x: Any, default: float = 0.0) -> float:
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
        qty = safe_float(pos.get("qty"), 0.0)
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

    def manage_open_positions(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not USE_SMART_EXIT:
            return None
        targets = [symbol] if symbol else list(self.open_positions.keys())
        last_result = None
        for sym in targets:
            try:
                prev = self.smart_exit._get_position(sym) or {}
                prev_sl = prev.get("sl")
                res = self.smart_exit.manage_open_positions(sym)
                post = self.smart_exit._get_position(sym) or {}
                new_sl = post.get("sl")
                if res and isinstance(res, dict):
                    rtype = res.get("type")
                    if rtype in ("trailing", "breakeven") and new_sl != prev_sl:
                        logger.info("[SmartExit] %s SL change: %s -> %s", sym, prev_sl, new_sl)
                last_result = res
            except Exception as e:
                logger.exception("manage_open_positions failed for %s: %s", sym, e)
        return last_result

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

            live_positions = {}
            for p in (live_positions_raw or []):
                try:
                    amt = safe_float(p.get("positionAmt") or p.get("position") or p.get("quantity") or 0.0, 0.0)
                    if abs(amt) > 0:
                        sym = p.get("symbol") or (p.get("symbol".upper()) if isinstance(p, dict) else None)
                        if sym:
                            live_positions[sym] = p
                except Exception:
                    continue

            open_orders = []
            try:
                if hasattr(self.client, "futures_get_open_orders"):
                    open_orders = getattr(self.client, "futures_get_open_orders")() or []
                elif hasattr(self.client, "get_open_orders"):
                    open_orders = getattr(self.client, "get_open_orders")() or []
            except Exception as e:
                logger.debug("Could not fetch open orders: %s", e)

            order_symbols = {o.get("symbol") for o in (open_orders or []) if isinstance(o, dict)}
            stale_cached = set(self.open_positions.keys()) - set(live_positions.keys())
            for s in stale_cached:
                logger.info("Cleaning stale cache for %s", s)
                self.remove_cached(s)

            orphaned = order_symbols - set(live_positions.keys())
            for s in orphaned:
                logger.warning("Orphaned orders detected for %s", s)

            logger.info("sync_exchange_state complete: %s live positions, %s cached", len(live_positions), len(self.open_positions))
            return live_positions
        except Exception as e:
            logger.exception("sync_exchange_state failed: %s", e)
            return {}