# ==========================
# File: src/execution.py (fully patched, production-stable)
# ==========================
import os
import math
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import statistics

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
    """Safely convert any value to float."""
    try:
        if value in (None, "", "null", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class ExecutionManager:
    """
    Handles trade lifecycle:
      - open_position() with MarketIntegrityGuard and SmartExit setup
      - background pending retry loop with ATR-based backoff
      - manage_open_positions() wrapper for SmartExit
      - reconcile_open_positions() to expose cached state
      - sync_exchange_state() to reconcile local cache with exchange
      - remove_cached() to clear local cache entries
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
        self.dry_run = dry_run or DRY_RUN
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
            "ExecutionManager initialized (dry_run=%s, retry_interval=%ss, max_retries=%s)",
            self.dry_run,
            self.retry_interval,
            self.max_retries,
        )
    # Inside ExecutionManager
    def _rest_signed_post(self, endpoint: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Dict[str, Any]:
        """Proxy to client's _rest_signed_post"""
        return self.client._rest_signed_post(endpoint, payload=payload, timeout=timeout)

    # ---------------------------
    # Quantity calculation
    # ---------------------------
    def _calc_quantity(self, price: float, margin_usdt: Optional[float] = None) -> float:
        try:
            if margin_usdt is None:
                margin_usdt = (
                    (MARGIN_PERCENT / 100.0) * ACCOUNT_BALANCE
                    if USE_PERCENT_MARGIN
                    else MARGIN_USDT
                )
            margin_usdt = min(margin_usdt, ACCOUNT_BALANCE)
            if price <= 0:
                logger.warning("Invalid price for quantity calc: %s", price)
                return 0.0
            qty = (margin_usdt * LEVERAGE) / price
            qty *= VOLUME_MULTIPLIER or 1.0
            qty = round(qty, 3)
            return max(qty, 0.001)
        except Exception as e:
            logger.error("Error in _calc_quantity: %s", e)
            return 0.0
        
    def _round_step_size(self, quantity: float, step_size: float) -> float:
        """Round quantity down to the nearest step size."""
        if step_size and step_size > 0:
            return math.floor(quantity / step_size) * step_size
        return float(f"{quantity:.6f}")


    # ---------------------------
    # Rounding helper
    # ---------------------------
    @staticmethod
    def _round_to(value: float, step: float) -> float:
        try:
            if not step or step <= 0:
                return value
            return math.floor(value / step) * step
        except Exception:
            return value
        
        # ---------------------------
    # Symbol/order validation + rounding
    # ---------------------------
    def _validate_and_round_order(self, symbol: str, price: float, qty: float, tick_size: float, step_size: float):
        """
        Validate and round qty/price against symbol filters.
        Returns (rounded_price, rounded_qty) or raises ValueError if invalid.
        """
        # safe defaults
        if tick_size is None:
            tick_size = 0.0
        if step_size is None:
            step_size = 0.0

        # Round qty/price to allowed grid
        try:
            rounded_qty = qty
            rounded_price = price

            if step_size and step_size > 0:
                # use floor rounding to avoid exceeding precision limits
                rounded_qty = self._round_to(qty, step_size)
            else:
                # fallback: round to 6 decimals (BTC typically needs 3-6)
                rounded_qty = float(f"{qty:.6f}")

            if tick_size and tick_size > 0:
                rounded_price = self._round_to(price, tick_size)
            else:
                rounded_price = float(f"{price:.2f}")

            # fetch symbol filters if available to validate minQty/minPrice
            try:
                info = self.client.get_symbol_info(symbol)
                if isinstance(info, dict):
                    for f in info.get("filters", []):
                        if f.get("filterType") == "LOT_SIZE":
                            min_qty = float(f.get("minQty", 0))
                            if rounded_qty < min_qty:
                                raise ValueError(f"Quantity {rounded_qty} < minQty {min_qty}")
                        if f.get("filterType") == "MIN_NOTIONAL":
                            min_notional = float(f.get("notional", f.get("minNotional", 0)))
                            if min_notional > 0 and (rounded_qty * rounded_price) < min_notional:
                                raise ValueError(f"Order notional {(rounded_qty * rounded_price):.8f} < min_notional {min_notional}")
                        if f.get("filterType") == "PRICE_FILTER":
                            min_price = float(f.get("minPrice", 0))
                            if rounded_price < min_price:
                                raise ValueError(f"Price {rounded_price} < minPrice {min_price}")
            except Exception:
                # don't fail hard on symbol info fetch; we already rounded
                pass

            # guard rails: enforce minimum practical qty
            if rounded_qty <= 0:
                raise ValueError("Rounded quantity is zero or negative")

            return rounded_price, rounded_qty

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"_validate_and_round_order failed: {e}")


    # ---------------------------
    # Orderbook & recent trades helpers
    # ---------------------------
    def _fetch_orderbook(self, symbol: str, limit: int = 50) -> dict[str, list]:
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
            bids = j.get("bids") or []
            asks = j.get("asks") or []
            return {"bids": list(bids), "asks": list(asks)}
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
    # Open position (multi-TP + ATR retry)
    # ---------------------------
    def open_position(
        self,
        symbol: str,
        direction: str,
        margin_usdt: float,
        tp_percent: float,
        sl_percent: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Hybrid patched position opener:
        - Uses Binance SDK futures_create_order (avoids _rest_signed_post issues)
        - Fixes -1102 timestamp error by syncing server time
        - Supports ATR fallback TP/SL, SmartExit, dry-run, logging
        - Includes MarketIntegrityGuard and quantity validation
        """
        import os
        import statistics
        import logging
        from datetime import datetime
        from dotenv import load_dotenv

        logger = logging.getLogger(__name__)

        try:
            # --- Load environment ---
            try:
                load_dotenv(override=True)
            except Exception:
                pass

            # --- 1️⃣ Sync server time to avoid -1102 ---
            try:
                server_time = self.client.futures_time()
                if isinstance(server_time, dict) and "serverTime" in server_time:
                    offset = server_time["serverTime"] - int(time.time() * 1000)
                    self.client.time_offset = offset
            except Exception as e:
                logger.debug("Server time sync failed: %s", e)

            # --- 2️⃣ ATR multipliers ---
            tp_mults = [float(x) for x in os.getenv("ATR_MULT_TP", "2.0,3.0,4.0").split(",") if x.strip()]
            sl_mult = float(os.getenv("ATR_MULT_SL", "1.5"))

            # --- 3️⃣ Current price ---
            price_data = self.client.ticker_price(symbol)
            price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            if not price or price <= 0:
                logger.warning(f"⚠️ Invalid price for {symbol}: {price}")
                return None

            # --- 4️⃣ Symbol filters ---
            tick, step, min_qty, min_notional = 0.0, 0.0, None, None
            try:
                info = self.client.get_symbol_info(symbol) or {}
                for f in info.get("filters", []):
                    t = f.get("filterType")
                    if t == "PRICE_FILTER":
                        tick = float(f.get("tickSize", tick))
                    elif t == "LOT_SIZE":
                        step = float(f.get("stepSize", step))
                        if f.get("minQty") is not None:
                            min_qty = float(f.get("minQty"))
                    elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                        min_notional = float(f.get("minNotional", f.get("notional", min_notional or 0)))
            except Exception as e:
                logger.debug("Filter fetch failed for %s: %s", symbol, e)

            # --- 5️⃣ ATR fallback ---
            atr = None
            try:
                klines = self.client.get_klines(symbol, "15m", 30) or []
                if len(klines) > 14:
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    closes = [float(k[4]) for k in klines]
                    trs = [
                        max(h - l, abs(h - closes[i - 1]), abs(l - closes[i - 1]))
                        for i, (h, l) in enumerate(zip(highs, lows)) if i > 0
                    ]
                    atr = statistics.fmean(trs[-14:])
            except Exception as e:
                logger.debug("ATR fetch failed: %s", e)

            # --- 6️⃣ Compute TP & SL ---
            is_long = direction.upper() in ("LONG", "BUY")
            if atr and atr > 0:
                tp_levels = [price + atr * m if is_long else price - atr * m for m in tp_mults]
                sl = price - atr * sl_mult if is_long else price + atr * sl_mult
            else:
                tp_levels = [price * (1 + tp_percent / 100)] if is_long else [price * (1 - tp_percent / 100)]
                sl = price * (1 - sl_percent / 100) if is_long else price * (1 + sl_percent / 100)
            tp_levels = [self._round_to(tp, tick) for tp in tp_levels]
            sl = self._round_to(sl, tick)

            # --- 7️⃣ Quantity ---
            qty = self._round_to(self._calc_quantity(price, margin_usdt), step)
            if qty <= 0:
                logger.warning("Invalid qty computed for %s", symbol)
                return None

            # --- 8️⃣ Dry-run ---
            if self.dry_run:
                pos = {
                    "symbol": symbol, "side": direction,
                    "entry": price, "qty": qty,
                    "tp_levels": tp_levels, "sl": sl, "atr": atr,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "DRY_RUN"
                }
                self.open_positions[symbol] = pos
                logger.info("🧪 DRY-RUN: %s %s", direction, symbol)
                return pos

            # --- 9️⃣ MarketIntegrityGuard ---
            orderbook = self._fetch_orderbook(symbol)
            trades = self._fetch_recent_trades(symbol)
            suspect, reason = self.guard.check(orderbook, trades, events_per_sec=0.0)
            if suspect:
                logger.warning("⚠️ MarketIntegrityGuard flagged %s: %s", symbol, reason)
                pending = {
                    "symbol": symbol,
                    "status": "PENDING_SUSPECT",
                    "suspect_reason": reason,
                    "entry": price,
                    "qty": qty,
                    "tp_levels": tp_levels,
                    "sl": sl,
                    "atr": atr,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                self.open_positions[symbol] = pending
                return pending

            # --- 🔟 Leverage & hedge detection ---
            try:
                if hasattr(self.client, "futures_change_leverage"):
                    self.client.futures_change_leverage(symbol=symbol, leverage=int(os.getenv("LEVERAGE", "10")))
            except Exception as le:
                logger.debug("Leverage set failed: %s", le)

            # --- 1️⃣1️⃣ Validation ---
            if min_qty is not None and qty < float(min_qty):
                failure = {"symbol": symbol, "status": "FAILED_VALIDATION", "error": f"qty<{min_qty}"}
                self.open_positions[symbol] = failure
                logger.error("Quantity validation failed for %s: qty=%s minQty=%s", symbol, qty, min_qty)
                return failure
            if min_notional is not None and qty * price < float(min_notional):
                failure = {"symbol": symbol, "status": "FAILED_VALIDATION", "error": "below minNotional"}
                self.open_positions[symbol] = failure
                logger.error("Notional validation failed for %s: notional=%.8f minNotional=%s", symbol, qty * price, min_notional)
                return failure

            # --- 1️⃣2️⃣ Place Market Order (SDK call, safe) ---
            try:
                order_result = self.client.futures_create_order(
                    symbol=symbol,
                    side="BUY" if is_long else "SELL",
                    type="MARKET",
                    quantity=qty,
                )
                logger.info("✅ Opened %s %s @%.2f qty=%.8f", direction, symbol, price, qty)
            except Exception as e:
                logger.exception("❌ Market order failed for %s: %s", symbol, e)
                pending = {
                    "symbol": symbol,
                    "side": direction,
                    "entry": price,
                    "qty": qty,
                    "tp_levels": tp_levels,
                    "sl": sl,
                    "atr": atr,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "PENDING_ORDER_FAILED",
                    "error": str(e),
                }
                self.open_positions[symbol] = pending
                return pending

            # --- 1️⃣3️⃣ SmartExit integration ---
            smartexit_err = None
            if USE_SMART_EXIT:
                try:
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
                    logger.exception("SmartExit.create_exit_orders failed for %s", symbol)

            # --- 1️⃣4️⃣ Track opened position ---
            tracked = {
                "symbol": symbol,
                "side": direction,
                "entry": price,
                "qty": qty,
                "tp_levels": tp_levels,
                "sl": sl,
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
            failure = {
                "symbol": symbol,
                "status": "FAILED_UNHANDLED",
                "error": str(e),
                "error_detail": repr(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.open_positions[symbol] = failure
            return failure

    # ---------------------------
    # Pending retry loop
    # ---------------------------
    def _pending_retry_loop(self) -> None:
        while not self._stop_retry:
            try:
                time.sleep(self.retry_interval)
                pending = [(s, p) for s, p in self.open_positions.items() if p.get("status") == "PENDING_SUSPECT"]
                pending.sort(key=lambda x: x[1].get("last_retry_ts", datetime.min))
                for symbol, p in pending:
                    if self._stop_retry:
                        break
                    retries = p.get("retry_count", 0)
                    backoff = p.get("retry_backoff", self.retry_interval)
                    if retries >= self.max_retries:
                        p["status"] = "FAILED_MAX_RETRIES"
                        self.open_positions[symbol] = p
                        logger.error("❌ Max retries reached for %s", symbol)
                        continue

                    margin_usdt = (p.get("qty") or 0) * (p.get("entry") or 0)
                    if margin_usdt <= 0:
                        margin_usdt = MARGIN_USDT

                    logger.info("🔄 Retrying %s (attempt %s)", symbol, retries + 1)
                    # call open_position again which will re-check guard & either place order or update pending entry
                    self.open_position(symbol, p.get("side", "LONG"), margin_usdt, TP_PERCENT, SL_PERCENT)

                    # update retry counters/backoff
                    p = self.open_positions.get(symbol, p)  # refresh in case open_position updated it
                    p["retry_count"] = p.get("retry_count", retries) + 1
                    p["retry_backoff"] = p.get("retry_backoff", backoff) * self.retry_backoff
                    p["last_retry_ts"] = datetime.utcnow()
                    self.open_positions[symbol] = p
                    time.sleep(p["retry_backoff"])
            except Exception as e:
                logger.exception("_pending_retry_loop error: %s", e)

    # ---------------------------
    # Stop retry thread
    # ---------------------------
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
    # Close position (with SmartExit cleanup)
    # ---------------------------
    def close_position(self, symbol: str, side: str) -> None:
        """
        Close an existing futures position safely.

        Features:
        - Cancels linked SmartExit orders (TP/SL) before closing
        - Hedge/one-way detection (auto-sets positionSide if dualSidePosition=True)
        - Validates and rounds qty using symbol filters
        - Enhanced error decoding (JSON-aware)
        - DRY-RUN safe
        """
        import math, json
        from datetime import datetime

        logger = logging.getLogger(__name__)

        if symbol not in self.open_positions:
            logger.warning("No open position cached for %s", symbol)
            return

        pos = self.open_positions[symbol]
        qty = float(pos.get("qty", 0))
        if qty <= 0:
            logger.warning("Invalid qty=%.8f for %s — cannot close", qty, symbol)
            return

        # Opposite side for closing
        opposite = "SELL" if side.upper() in ("LONG", "BUY") else "BUY"

        # ---------------------------
        # 1️⃣ Cancel SmartExit-related orders
        # ---------------------------
        try:
            if USE_SMART_EXIT and hasattr(self.smart_exit, "cancel_exit_orders"):
                logger.info("🧹 Cancelling SmartExit TP/SL for %s before closing", symbol)
                self.smart_exit.cancel_exit_orders_for_symbol(symbol)

            else:
                # fallback: clean all open orders for the symbol
                self.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning("SmartExit cleanup failed for %s: %s", symbol, e)

        # ---------------------------
        # 2️⃣ Fetch symbol filters for rounding
        # ---------------------------
        tick_size = step_size = 0.0
        min_qty = None
        try:
            info = self.client.get_symbol_info(symbol)
            if isinstance(info, dict):
                for f in info.get("filters", []):
                    if f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", step_size))
                        min_qty = float(f.get("minQty", min_qty or 0)) if f.get("minQty") else min_qty
        except Exception as e:
            logger.debug("Failed to fetch symbol filters for %s during close: %s", symbol, e)

        def _round_step_size_local(q: float, step: float) -> float:
            try:
                if step and step > 0:
                    return math.floor(q / step) * step
            except Exception:
                pass
            return float(f"{q:.6f}")

        rounded_qty = _round_step_size_local(qty, step_size)
        if min_qty and rounded_qty < min_qty:
            logger.warning("Qty %.8f < minQty %.8f for %s", rounded_qty, min_qty, symbol)
            return
        if rounded_qty <= 0:
            logger.warning("Rounded qty invalid (<=0) for %s", symbol)
            return

        # ---------------------------
        # 3️⃣ DRY-RUN Mode
        # ---------------------------
        if self.dry_run:
            logger.info("[DRY RUN] Would close %s %s qty=%.8f", symbol, opposite, rounded_qty)
            self.open_positions.pop(symbol, None)
            return

        # ---------------------------
        # 4️⃣ Detect hedge mode (dualSidePosition)
        # ---------------------------
        position_side_flag = None
        try:
            acc_info = None
            if hasattr(self.client, "futures_account"):
                acc_info = self.client.futures_account()
            elif hasattr(self.client, "get_futures_account"):
                acc_info = self.client.get_futures_account()
            if isinstance(acc_info, dict) and acc_info.get("dualSidePosition"):
                orig_side = pos.get("side", "").upper()
                if orig_side in ("LONG", "BUY"):
                    position_side_flag = "LONG"
                elif orig_side in ("SHORT", "SELL"):
                    position_side_flag = "SHORT"
        except Exception:
            pass

        # ---------------------------
        # 5️⃣ Place close order
        # ---------------------------
        try:
            order_kwargs = {
                "symbol": symbol,
                "side": opposite,
                "type": "MARKET",
                "quantity": rounded_qty,
            }
            if position_side_flag:
                order_kwargs["positionSide"] = position_side_flag

            logger.debug(
                "Placing CLOSE order: %s %s qty=%.8f kwargs=%s",
                opposite, symbol, rounded_qty,
                {k: v for k, v in order_kwargs.items() if k != 'quantity'}
            )

            order = self.client.futures_create_order(**order_kwargs)

            order_id = None
            if isinstance(order, dict):
                order_id = (
                    order.get("orderId")
                    or order.get("order_id")
                    or order.get("id")
                    or order.get("clientOrderId")
                )

            logger.info("✅ Closed %s %s | qty=%.8f order_id=%s", symbol, opposite, rounded_qty, order_id or "?")
        except Exception as e:
            err_text = str(e)
            try:
                resp = getattr(e, "response", None)
                if resp is not None and hasattr(resp, "text"):
                    txt = resp.text
                    try:
                        j = json.loads(txt)
                        err_text = f"{j.get('code')}: {j.get('msg')}"
                    except Exception:
                        err_text = txt
                elif hasattr(e, "args") and len(e.args) > 0:
                    arg0 = e.args[0]
                    if isinstance(arg0, str):
                        try:
                            j = json.loads(arg0)
                            err_text = f"{j.get('code')}: {j.get('msg')}"
                        except Exception:
                            err_text = arg0
            except Exception:
                pass
            logger.error("❌ Error closing %s: %s", symbol, err_text)

        # ---------------------------
        # 6️⃣ Final cache cleanup
        # ---------------------------
        self.open_positions.pop(symbol, None)
        logger.info("🧾 %s removed from local cache", symbol)


    # ---------------------------
    # Cancel all open orders (used by close_position)
    # ---------------------------
    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        """
        Cancels all open futures orders for a given symbol (or all if None).
        - DRY_RUN safe
        - SmartExit-compatible
        - Hedge-mode tolerant
        """
        import json

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
                    logger.info("🧹 All open orders cancelled for %s", sym)
                except Exception as e:
                    err_text = str(e)
                    try:
                        resp = getattr(e, "response", None)
                        if resp is not None and hasattr(resp, "text"):
                            txt = resp.text
                            try:
                                j = json.loads(txt)
                                err_text = f"{j.get('code')}: {j.get('msg')}"
                            except Exception:
                                err_text = txt
                    except Exception:
                        pass
                    logger.error("❌ Failed to cancel orders for %s: %s", sym, err_text)
        except Exception as e:
            logger.exception("cancel_all_orders failed: %s", e)


    # ---------------------------
    # Live SmartExit management
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
    # Reconcile cached positions
    # ---------------------------
    def reconcile_open_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.open_positions

    # ---------------------------
    # Wrapper: manage_open_positions
    # ---------------------------
    def manage_open_positions(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not USE_SMART_EXIT:
            return None
        targets = [symbol] if symbol else list(self.open_positions.keys())
        last_result = None
        for sym in targets:
            try:
                prev = self.smart_exit._get_open_position(sym) or {}
                prev_sl = prev.get("sl")
                res = self.smart_exit.manage_open_positions(sym)
                post = self.smart_exit._get_open_position(sym) or {}
                new_sl = post.get("sl")
                if res and isinstance(res, dict):
                    rtype = res.get("type")
                    if rtype in ("trailing", "breakeven") and new_sl != prev_sl:
                        logger.info("🔄 [SmartExit] %s SL change: %s -> %s", sym, prev_sl, new_sl)
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
    # Sync exchange
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
                logger.info("🧹 Cleaning stale cache for %s", s)
                self.remove_cached(s)

            orphaned = order_symbols - set(live_positions.keys())
            for s in orphaned:
                logger.warning("⚠️ Orphaned orders detected for %s", s)

            logger.info("sync_exchange_state complete: %s live positions, %s cached", len(live_positions), len(self.open_positions))
            return live_positions
        except Exception as e:
            logger.exception("sync_exchange_state failed: %s", e)
            return {}
