# ==========================
# File: src/execution.py  (fully patched, production-stable)
# ==========================
import os
import math
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
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

# ============================================================
# Helper: Safe float conversion
# ============================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to float."""
    try:
        if value in (None, "", "null", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# Execution Manager
# ============================================================
class ExecutionManager:
    """
    Handles full trade execution lifecycle:
    - Opens new positions
    - Delegates ATR-based exit management to SmartExitManager
    - Supports DRY_RUN simulation
    - Integrates MarketIntegrityGuard for robust entry
    - Automated retry of pending positions with backoff
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
        self.dry_run = dry_run
        self.smart_exit = SmartExitManager(binance_client)
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.guard = MarketIntegrityGuard()
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._stop_retry = False
        self._retry_thread = threading.Thread(
            target=self._pending_retry_loop, daemon=True
        )
        self._retry_thread.start()
        logger.info(
            f"ExecutionManager initialized (dry_run={self.dry_run}) "
            f"retry_interval={self.retry_interval}s, max_retries={self.max_retries}"
        )

    # ------------------------------------------------------------
    # 🔹 Quantity Calculation
    # ------------------------------------------------------------
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
                logger.warning(f"⚠️ Invalid price for quantity calc: {price}")
                return 0.0
            qty = (margin_usdt * LEVERAGE) / price
            qty *= VOLUME_MULTIPLIER or 1.0
            qty = round(qty, 3)
            return max(qty, 0.001)
        except Exception as e:
            logger.error(f"❌ Error in _calc_quantity: {e}")
            return 0.0

    # ------------------------------------------------------------
    # 🔹 Helper: Binance rounding
    # ------------------------------------------------------------
    @staticmethod
    def _round_to(value: float, step: float) -> float:
        try:
            if step <= 0:
                return value
            return math.floor(value / step) * step
        except Exception:
            return value

    # ------------------------------------------------------------
    # 🔹 Orderbook / trades fetch helpers
    # ------------------------------------------------------------
    def _fetch_orderbook(self, symbol: str, limit: int = 50) -> dict:
        try:
            for m in ("depth", "get_order_book", "get_depth", "order_book"):
                if hasattr(self.client, m):
                    try:
                        resp = getattr(self.client, m)(symbol=symbol, limit=limit)
                        if isinstance(resp, dict) and resp.get("bids") and resp.get("asks"):
                            return {"bids": resp.get("bids"), "asks": resp.get("asks")}
                    except Exception:
                        continue
            import requests

            url = "https://fapi.binance.com/fapi/v1/depth"
            r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=3)
            r.raise_for_status()
            j = r.json()
            return {"bids": j.get("bids", []), "asks": j.get("asks", [])}
        except Exception as e:
            logger.debug("_fetch_orderbook error: %s", e)
            return {"bids": [], "asks": []}

    def _fetch_recent_trades(self, symbol: str, limit: int = 100) -> list:
        try:
            for m in ("trades", "recent_trades", "get_recent_trades", "agg_trades"):
                if hasattr(self.client, m):
                    try:
                        resp = getattr(self.client, m)(symbol=symbol, limit=limit)
                        if isinstance(resp, list):
                            out = []
                            for t in resp:
                                price = safe_float(t.get("price") or t.get("p"), 0.0)
                                qty = safe_float(t.get("qty") or t.get("q"), 0.0)
                                side = (
                                    "sell"
                                    if t.get("isBuyerMaker")
                                    else "buy"
                                    if "isBuyerMaker" in t
                                    else str(t.get("side") or t.get("S") or "").lower()
                                )
                                out.append({"price": price, "qty": qty, "side": side})
                            return out
                    except Exception:
                        continue
            import requests

            url = "https://fapi.binance.com/fapi/v1/trades"
            r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=3)
            r.raise_for_status()
            j = r.json()
            out = []
            for t in j:
                price = safe_float(t.get("price"), 0.0)
                qty = safe_float(t.get("qty"), 0.0)
                side = "buy" if t.get("isBuyerMaker") is False else "sell"
                out.append({"price": price, "qty": qty, "side": side})
            return out
        except Exception as e:
            logger.debug("_fetch_recent_trades error: %s", e)
            return []

    # ------------------------------------------------------------
    # 🔹 Open Position
    # ------------------------------------------------------------
    def open_position(self, symbol: str, direction: str, margin_usdt: float, tp_percent: float, sl_percent: float):
        """Open a futures position with optional SmartExit and MarketIntegrityGuard."""
        try:
            load_dotenv(override=True)

            # 1️⃣ Current price
            price_data = self.client.ticker_price(symbol)
            price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            if not price or price <= 0:
                logger.warning(f"⚠️ Invalid price for {symbol}: {price}")
                return None

            # 2️⃣ Symbol precision filters
            tick_size = step_size = 0.0
            try:
                info = self.client.get_symbol_info(symbol)
                for f in info.get("filters", []):
                    if f.get("filterType") == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", tick_size))
                    elif f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", step_size))
            except Exception as e:
                logger.warning(f"Failed to fetch symbol filters for {symbol}: {e}")

            # 3️⃣ ATR fallback (smoothed)
            atr = None
            try:
                klines = self.client.get_klines(symbol, "15m", 15)
                if klines and len(klines) > 2:
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    closes = [float(k[4]) for k in klines]
                    trs = [
                        max(h - l, abs(h - closes[i - 1]), abs(l - closes[i - 1]))
                        for i, (h, l) in enumerate(zip(highs, lows))
                        if i > 0
                    ]
                    atr = statistics.fmean(trs[-14:])
            except Exception as e:
                logger.debug(f"ATR calculation failed for {symbol}: {e}")

            # 4️⃣ TP/SL calculation
            if atr and atr > 0:
                if direction.upper() == "LONG":
                    tp_levels = [price + atr * 2.5]
                    sl = price - atr * 1.8
                else:
                    tp_levels = [price - atr * 2.5]
                    sl = price + atr * 1.8
            else:
                if direction.upper() == "LONG":
                    tp_levels = [price * (1 + tp_percent / 100)]
                    sl = price * (1 - sl_percent / 100)
                else:
                    tp_levels = [price * (1 - tp_percent / 100)]
                    sl = price * (1 + sl_percent / 100)

            tp_levels = [self._round_to(tp, tick_size) for tp in tp_levels]
            sl = self._round_to(sl, tick_size)

            # 5️⃣ Quantity
            qty = self._calc_quantity(price, margin_usdt)
            qty = self._round_to(qty, step_size)
            if qty <= 0:
                logger.warning(f"⚠️ Invalid quantity for {symbol}: {qty}")
                return None

            position_type = direction.upper()
            logger.info(f"📈 Preparing {position_type} {symbol} | entry={price:.2f} qty={qty}")

            # 6️⃣ Dry run
            if self.dry_run:
                logger.info(f"[DRY RUN] Would open {position_type} {symbol} | entry={price:.2f} qty={qty}")
                return {
                    "symbol": symbol,
                    "side": position_type,
                    "entry": price,
                    "qty": qty,
                    "tp_levels": tp_levels,
                    "sl": sl,
                    "atr": atr,
                    "timestamp": datetime.utcnow().isoformat(),
                    "dry_run": True,
                }

            # 7️⃣ Market Integrity Guard
            orderbook = self._fetch_orderbook(symbol, limit=50)
            recent_trades = self._fetch_recent_trades(symbol, limit=100)
            suspect, reason = self.guard.check(orderbook, recent_trades, events_per_sec=0.0)
            if suspect:
                pending = self.open_positions.get(symbol, {})
                retries = pending.get("retry_count", 0)
                backoff = pending.get("retry_backoff", self.retry_interval)
                if retries >= self.max_retries:
                    logger.error(f"❌ Max retries reached for {symbol}, skipping further attempts.")
                    pending["status"] = "FAILED_MAX_RETRIES"
                    self.open_positions[symbol] = pending
                    return pending

                logger.warning(f"❗ MarketIntegrityGuard flagged {symbol}: {reason} — PENDING entry (retry {retries+1})")
                pending.update({
                    "symbol": symbol,
                    "side": position_type,
                    "entry": price,
                    "qty": qty,
                    "tp_levels": tp_levels,
                    "sl": sl,
                    "atr": atr,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "PENDING_SUSPECT",
                    "suspect_reason": reason,
                    "retry_count": retries,
                    "retry_backoff": backoff
                })
                self.open_positions[symbol] = pending
                return pending

            # 8️⃣ Execute live market order
            self.client.futures_create_order(
                symbol=symbol,
                side="BUY" if position_type == "LONG" else "SELL",
                type="MARKET",
                quantity=qty,
            )
            logger.info(f"✅ Opened {position_type} {symbol} | entry={price:.2f} qty={qty}")

            # 9️⃣ Smart Exit Setup
            if USE_SMART_EXIT:
                try:
                    self.smart_exit.create_exit_orders(
                        symbol=symbol,
                        side=position_type,
                        entry_price=price,
                        qty=qty,
                        atr_value=atr,
                        tick_size=tick_size,
                        step_size=step_size,
                    )
                except Exception as e:
                    logger.error(f"⚠️ SmartExit.create_exit_orders failed for {symbol}: {e}")

            # 10️⃣ Track position
            self.open_positions[symbol] = {
                "side": position_type,
                "entry": price,
                "qty": qty,
                "tp_levels": tp_levels,
                "sl": sl,
                "atr": atr,
                "timestamp": datetime.utcnow().isoformat(),
            }
            return self.open_positions[symbol]

        except Exception as e:
            logger.exception(f"Unhandled error in open_position() for {symbol}: {e}")
            return None

    # ------------------------------------------------------------
    # 🔹 Pending Retry Loop
    # ------------------------------------------------------------
    def _pending_retry_loop(self):
        while not self._stop_retry:
            time.sleep(self.retry_interval)

            pending_symbols = [
                (s, p) for s, p in self.open_positions.items()
                if p.get("status") == "PENDING_SUSPECT"
            ]
            pending_symbols.sort(key=lambda x: x[1].get("last_retry_ts", datetime.min), reverse=True)

            for symbol, pending in pending_symbols:
                retries = pending.get("retry_count", 0)
                backoff = pending.get("retry_backoff", self.retry_interval)

                if retries >= self.max_retries:
                    logger.error(f"❌ Max retries reached for {symbol}, marking as permanently invalid.")
                    pending["status"] = "FAILED_MAX_RETRIES"
                    self.open_positions[symbol] = pending
                    continue

                margin_usdt = (pending.get("qty") or 0) * (pending.get("entry") or 0)
                if margin_usdt <= 0:
                    margin_usdt = MARGIN_USDT

                logger.info(f"🔄 Retrying pending position for {symbol} (attempt {retries+1})")
                self.open_position(
                    symbol,
                    pending.get("side", "LONG"),
                    margin_usdt=margin_usdt,
                    tp_percent=TP_PERCENT,
                    sl_percent=SL_PERCENT,
                )

                pending["retry_count"] = retries + 1
                pending["retry_backoff"] = backoff * self.retry_backoff
                pending["last_retry_ts"] = datetime.utcnow()
                self.open_positions[symbol] = pending

                time.sleep(pending["retry_backoff"])

    # ------------------------------------------------------------
    # 🔹 Stop Retry Thread
    # ------------------------------------------------------------
    def stop_retry_thread(self):
        """Gracefully stop the background retry thread."""
        self._stop_retry = True
        if self._retry_thread.is_alive():
            self._retry_thread.join(timeout=5)

    def __del__(self):
        self.stop_retry_thread()

    # ------------------------------------------------------------
    # 🔹 Manage Active Positions
    # ------------------------------------------------------------
    def manage_positions_live(self):
        if not USE_SMART_EXIT:
            return
        for symbol in list(self.open_positions.keys()):
            try:
                result = self.smart_exit.manage_open_positions(symbol=symbol)
                if result and isinstance(result, dict):
                    rtype = result.get("type")
                    if rtype == "trailing":
                        logger.info(f"📈 Trailing stop updated for {symbol}: {result}")
                    elif rtype == "breakeven":
                        logger.info(f"⚖️ Breakeven update for {symbol}: {result}")
                    elif rtype == "noop":
                        logger.debug(f"No SL change for {symbol}.")
            except Exception as e:
                logger.warning(f"⚠️ Smart exit management failed for {symbol}: {e}")

    # ------------------------------------------------------------
    # 🔹 Close Position
    # ------------------------------------------------------------
    def close_position(self, symbol: str, side: str):
        if symbol not in self.open_positions:
            logger.warning(f"No open position for {symbol}")
            return
        pos = self.open_positions[symbol]
        opposite = "SELL" if side == "BUY" else "BUY"
        if DRY_RUN:
            logger.info(f"[DRY RUN] Would close {symbol} ({opposite}) qty={pos['qty']}")
        else:
            try:
                self.client.futures_create_order(
                    symbol=symbol,
                    side=opposite,
                    type="MARKET",
                    quantity=pos["qty"],
                )
                logger.info(f"✅ Closed {symbol} position at market.")
            except Exception as e:
                logger.error(f"❌ Error closing {symbol}: {e}")
        self.open_positions.pop(symbol, None)

    # ------------------------------------------------------------
    # 🔹 Compatibility Helpers
    # ------------------------------------------------------------
    def reconcile_open_positions(self):
        return self.open_positions

    def manage_open_positions(self, symbol: Optional[str] = None):
        if not USE_SMART_EXIT:
            return
        targets = [symbol] if symbol else list(self.open_positions.keys())
        for sym in targets:
            try:
                prev_pos = self.smart_exit._get_open_position(sym) or {}
                prev_sl = prev_pos.get("sl")
                result = self.smart_exit.manage_open_positions(sym)
                post_pos = self.smart_exit._get_open_position(sym) or {}
                new_sl = post_pos.get("sl")
                if result and isinstance(result, dict):
                    rtype = result.get("type")
                    if rtype in ("trailing", "breakeven") and new_sl != prev_sl:
                        logger.info(f"🔄 [SmartExit] {rtype.upper()} SL change {sym}: {prev_sl or 'None'} → {new_sl}")
                    elif rtype == "error":
                        logger.warning(f"[SmartExit] {sym} error: {result.get('error')}")
                    elif rtype == "noop":
                        logger.debug(f"[SmartExit] {sym}: No update required.")
            except Exception as e:
                logger.exception(f"⚠️ manage_open_positions() failed for {sym}: {e}")
