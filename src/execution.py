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
    Handles trade lifecycle:
      - open_position() with MarketIntegrityGuard and SmartExit setup
      - background pending retry loop for flagged entries
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

        # start background pending retry thread
        self._retry_thread = threading.Thread(target=self._pending_retry_loop, daemon=True)
        self._retry_thread.start()

        logger.info(
            "ExecutionManager initialized (dry_run=%s, retry_interval=%ss, max_retries=%s)",
            self.dry_run,
            self.retry_interval,
            self.max_retries,
        )

    # ------------------------------------------------------------
    # Quantity calculation
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
                logger.warning("Invalid price for quantity calc: %s", price)
                return 0.0
            qty = (margin_usdt * LEVERAGE) / price
            qty *= VOLUME_MULTIPLIER or 1.0
            qty = round(qty, 3)
            return max(qty, 0.001)
        except Exception as e:
            logger.error("Error in _calc_quantity: %s", e)
            return 0.0

    # ------------------------------------------------------------
    # Binance rounding helper
    # ------------------------------------------------------------
    @staticmethod
    def _round_to(value: float, step: float) -> float:
        try:
            if not step or step <= 0:
                return value
            return math.floor(value / step) * step
        except Exception:
            return value

    # ------------------------------------------------------------
    # Orderbook/trades helpers (multiple API shapes supported)
    # ------------------------------------------------------------
    def _fetch_orderbook(self, symbol: str, limit: int = 50) -> dict[str, list]:
        """Try several client methods then fallback to public REST endpoint."""
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

            # fallback to REST
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


    from typing import List, Dict, Any

    def _fetch_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Try several client methods then fallback to public REST endpoint."""
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

            # Fallback to REST endpoint
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


    # ------------------------------------------------------------
    # Public: open_position
    # ------------------------------------------------------------
    def open_position(self, symbol: str, direction: str, margin_usdt: float, tp_percent: float, sl_percent: float) -> Optional[Dict[str, Any]]:
        """
        Open a futures position.
        Returns a dict describing tracked position on success or pending/None on failure.
        """
        try:
            load_dotenv(override=True)

            # current price
            price_data = self.client.ticker_price(symbol)
            price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            if not price or price <= 0:
                logger.warning("Invalid price for %s: %s", symbol, price)
                return None

            # symbol filters
            tick_size = step_size = 0.0
            try:
                info = self.client.get_symbol_info(symbol)
                for f in info.get("filters", []):
                    if f.get("filterType") == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", tick_size))
                    elif f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", step_size))
            except Exception as e:
                logger.debug("Failed to fetch symbol filters for %s: %s", symbol, e)

            # ATR fallback
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
            except Exception:
                logger.debug("ATR fallback failed for %s", symbol)

            # TP/SL compute
            if atr and atr > 0:
                if direction.upper() == "LONG":
                    tp_levels = [price + atr * 2.5]
                    sl = price - atr * 1.8
                else:
                    tp_levels = [price - atr * 2.5]
                    sl = price + atr * 1.8
            else:
                if direction.upper() == "LONG":
                    tp_levels = [price * (1 + tp_percent / 100.0)]
                    sl = price * (1 - sl_percent / 100.0)
                else:
                    tp_levels = [price * (1 - tp_percent / 100.0)]
                    sl = price * (1 + sl_percent / 100.0)

            tp_levels = [self._round_to(tp, tick_size) for tp in tp_levels]
            sl = self._round_to(sl, tick_size)

            # quantity
            qty = self._calc_quantity(price, margin_usdt)
            qty = self._round_to(qty, step_size)
            if qty <= 0:
                logger.warning("Invalid qty calculated for %s: %s", symbol, qty)
                return None

            position_type = direction.upper()
            logger.info("Preparing %s %s | entry=%s qty=%s", position_type, symbol, price, qty)

            # dry run path
            if self.dry_run:
                logger.info("[DRY RUN] Would open %s %s | entry=%s qty=%s", position_type, symbol, price, qty)
                tracked = {
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
                self.open_positions[symbol] = tracked
                return tracked

            # MarketIntegrityGuard check
            orderbook = self._fetch_orderbook(symbol, limit=50)
            recent_trades = self._fetch_recent_trades(symbol, limit=100)
            suspect, reason = self.guard.check(orderbook, recent_trades, events_per_sec=0.0)
            if suspect:
                pending = self.open_positions.get(symbol, {})
                retries = pending.get("retry_count", 0)
                backoff = pending.get("retry_backoff", self.retry_interval)
                if retries >= self.max_retries:
                    logger.error("Max retries reached for %s; marking as FAILED_MAX_RETRIES", symbol)
                    pending["status"] = "FAILED_MAX_RETRIES"
                    self.open_positions[symbol] = pending
                    return pending

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
                    "retry_backoff": backoff,
                })
                logger.warning("❗ MarketIntegrityGuard flagged %s: %s — PENDING entry (retry %s)", symbol, reason, retries + 1)
                self.open_positions[symbol] = pending
                return pending

            # execute market order
            try:
                self.client.futures_create_order(
                    symbol=symbol,
                    side="BUY" if position_type == "LONG" else "SELL",
                    type="MARKET",
                    quantity=qty,
                )
                logger.info("Opened %s %s | entry=%s qty=%s", position_type, symbol, price, qty)
            except Exception as e:
                logger.error("Market order failed for %s: %s", symbol, e)
                # keep as pending failure if desired or return None
                return None

            # SmartExit create TPs/SL
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
                    logger.error("SmartExit.create_exit_orders failed for %s: %s", symbol, e)

            # track position locally
            tracked = {
                "side": position_type,
                "entry": price,
                "qty": qty,
                "tp_levels": tp_levels,
                "sl": sl,
                "atr": atr,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "OPEN",
            }
            self.open_positions[symbol] = tracked
            return tracked

        except Exception as e:
            logger.exception("Unhandled error in open_position(%s): %s", symbol, e)
            return None

    # ------------------------------------------------------------
    # Pending retry loop (background)
    # ------------------------------------------------------------
    def _pending_retry_loop(self) -> None:
        while not self._stop_retry:
            try:
                time.sleep(self.retry_interval)
                pending = [(s, p) for s, p in self.open_positions.items() if p.get("status") == "PENDING_SUSPECT"]
                # sort by last retry timestamp (oldest first)
                pending.sort(key=lambda x: x[1].get("last_retry_ts", datetime.min))
                for symbol, p in pending:
                    if self._stop_retry:
                        break
                    retries = p.get("retry_count", 0)
                    backoff = p.get("retry_backoff", self.retry_interval)
                    if retries >= self.max_retries:
                        logger.error("❌ Max retries reached for %s, marking as permanently invalid.", symbol)
                        p["status"] = "FAILED_MAX_RETRIES"
                        self.open_positions[symbol] = p
                        continue

                    margin_usdt = (p.get("qty") or 0) * (p.get("entry") or 0)
                    if margin_usdt <= 0:
                        margin_usdt = MARGIN_USDT

                    logger.info("🔄 Retrying pending position for %s (attempt %s)", symbol, retries + 1)
                    # attempt open again
                    self.open_position(symbol, p.get("side", "LONG"), margin_usdt, TP_PERCENT, SL_PERCENT)

                    # update retry counters/backoff
                    p["retry_count"] = retries + 1
                    p["retry_backoff"] = backoff * self.retry_backoff
                    p["last_retry_ts"] = datetime.utcnow()
                    self.open_positions[symbol] = p

                    # wait before next pending retry
                    time.sleep(p["retry_backoff"])
            except Exception as e:
                logger.exception("_pending_retry_loop error: %s", e)

    def stop_retry_thread(self) -> None:
        """Signal background retry thread to stop and join."""
        self._stop_retry = True
        try:
            if hasattr(self, "_retry_thread") and self._retry_thread.is_alive():
                self._retry_thread.join(timeout=5)
        except Exception:
            pass

    def __del__(self):
        # best-effort cleanup
        try:
            self.stop_retry_thread()
        except Exception:
            pass

    # ------------------------------------------------------------
    # Manage positions live via SmartExit (public wrapper)
    # ------------------------------------------------------------
    def manage_positions_live(self) -> None:
        """Call SmartExit management for all cached open positions."""
        if not USE_SMART_EXIT:
            logger.debug("SmartExit disabled; skipping manage_positions_live()")
            return
        for symbol in list(self.open_positions.keys()):
            try:
                self.smart_exit.manage_open_positions(symbol)
            except Exception as e:
                logger.warning("Smart exit management failed for %s: %s", symbol, e)

    # ------------------------------------------------------------
    # Close position helper
    # ------------------------------------------------------------
    def close_position(self, symbol: str, side: str) -> None:
        """Close a tracked/open position with a market order and remove from cache."""
        if symbol not in self.open_positions:
            logger.warning("No open position cached for %s", symbol)
            return
        pos = self.open_positions[symbol]
        opposite = "SELL" if side == "BUY" else "BUY"
        if self.dry_run:
            logger.info("[DRY RUN] Would close %s (%s) qty=%s", symbol, opposite, pos.get("qty"))
        else:
            try:
                self.client.futures_create_order(
                    symbol=symbol,
                    side=opposite,
                    type="MARKET",
                    quantity=pos.get("qty"),
                )
                logger.info("Closed %s at market", symbol)
            except Exception as e:
                logger.error("Error closing %s: %s", symbol, e)
        # remove cache
        self.open_positions.pop(symbol, None)

    # ------------------------------------------------------------
    # Public compatibility: reconcile_open_positions
    # ------------------------------------------------------------
    def reconcile_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the local cached open positions dict.
        Kept for compatibility with existing bot.py usage.
        """
        logger.debug("reconcile_open_positions called, returning %s cached positions", len(self.open_positions))
        return self.open_positions

    # ------------------------------------------------------------
    # Public: manage_open_positions wrapper (keeps signature used by bot.py)
    # ------------------------------------------------------------
    def manage_open_positions(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Wrapper used by bot.py. If symbol is passed, manage that position via SmartExit.
        Returns the SmartExit result dict (if any) or None.
        """
        if not USE_SMART_EXIT:
            logger.debug("SmartExit disabled; manage_open_positions noop")
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
                    elif rtype == "error":
                        logger.warning("[SmartExit] %s error: %s", sym, res.get("error"))
                last_result = res
            except Exception as e:
                logger.exception("manage_open_positions failed for %s: %s", sym, e)
        return last_result

    # ------------------------------------------------------------
    # Cache & exchange sync utilities
    # ------------------------------------------------------------
    def remove_cached(self, symbol: Optional[str] = None) -> None:
        """Remove a cached position or clear all cached positions."""
        if symbol:
            if symbol in self.open_positions:
                logger.info("Removed cached position for %s", symbol)
                self.open_positions.pop(symbol, None)
            else:
                logger.debug("remove_cached called for %s but no cached entry", symbol)
        else:
            logger.info("Clearing all cached positions")
            self.open_positions.clear()

    def sync_exchange_state(self) -> Dict[str, Any]:
        """
        Reconcile local cache with exchange:
          - Fetch live positions and open orders from exchange
          - Remove stale cached entries (no live position)
          - Report orphaned orders (orders without live positions)
        Returns dict of live positions keyed by symbol (best-effort).
        """
        try:
            if self.dry_run:
                logger.info("[DRY RUN] sync_exchange_state skipped")
                return self.open_positions

            # try several client method names
            live_positions_raw = []
            try:
                # many clients expose futures_position_information()
                if hasattr(self.client, "futures_position_information"):
                    res = self.client.futures_position_information()
                    if isinstance(res, list):
                        live_positions_raw = res
                # fallback: get_all_positions
                if not live_positions_raw and hasattr(self.client, "get_all_positions"):
                    live_positions_raw = getattr(self.client, "get_all_positions")() or []
            except Exception as e:
                logger.debug("Could not fetch live positions via primary methods: %s", e)

            live_positions = {}
            for p in (live_positions_raw or []):
                try:
                    amt = safe_float(p.get("positionAmt") or p.get("position") or p.get("quantity") or 0.0, 0.0)
                    if abs(amt) > 0:
                        sym = p.get("symbol") or p.get("symbol".upper()) if isinstance(p, dict) else None
                        if sym:
                            live_positions[sym] = p
                except Exception:
                    continue

            # open orders
            open_orders = []
            try:
                if hasattr(self.client, "futures_get_open_orders"):
                    open_orders = getattr(self.client, "futures_get_open_orders")() or []
                elif hasattr(self.client, "get_open_orders"):
                    open_orders = getattr(self.client, "get_open_orders")() or []
            except Exception as e:
                logger.debug("Could not fetch open orders: %s", e)

            order_symbols = {o.get("symbol") for o in (open_orders or []) if isinstance(o, dict)}

            # remove stale cached positions (cache contains symbol but exchange has none)
            stale_cached = set(self.open_positions.keys()) - set(live_positions.keys())
            for s in stale_cached:
                logger.info("🧹 Cleaning stale cache for %s (no live position)", s)
                self.remove_cached(s)

            # detect orphaned orders (orders without live positions)
            orphaned = order_symbols - set(live_positions.keys())
            for s in orphaned:
                logger.warning("⚠️ Orphaned orders detected for %s (orders present but no live position)", s)

            logger.info("sync_exchange_state complete: %s live positions, %s cached", len(live_positions), len(self.open_positions))
            return live_positions
        except Exception as e:
            logger.exception("sync_exchange_state failed: %s", e)
            return {}

