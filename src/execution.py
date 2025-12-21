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
from money_manager import MoneyManager

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
        # wire SmartExit back to ExecutionManager
        try:
            self.smart_exit.execution = self
        except Exception:
            logger.warning("SmartExit wiring failed (execution reference)")

        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.guard = MarketIntegrityGuard()

        # initialize money manager (best-effort)
        try:
            self.money = MoneyManager(self.client)
            logger.info("MoneyManager initialized (state=%s)", getattr(self.money, "state_path", "unknown"))
        except Exception:
            self.money = None
            logger.warning("MoneyManager initialization failed; continuing without MM protections")



        # keep these values for compatibility/inspectability but we removed retry threads
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._last_trade_close_time = {}


        logger.info(
            "ExecutionManager initialized (dry_run=%s, retry_interval=%s, max_retries=%s) - retry thread disabled",
            self.dry_run,
            self.retry_interval,
            self.max_retries,
        )

    def _dynamic_margin_percent(self, balance: float) -> float:
        """
        Return MARGIN_PERCENT based on tiers:
        env format: "100:1.0,300:1.5,1000:2.0"
        Meaning:
            balance < 100 → 1.0%
            balance < 300 → 1.5%
            balance < 1000 → 2.0%
            otherwise → last tier value
        """

        raw = os.getenv("DYNAMIC_MARGIN_TIERS", "")
        if not raw:
            return float(os.getenv("MARGIN_PERCENT", "1.0"))

        tiers = []
        try:
            for p in raw.split(","):
                th, v = p.split(":")
                tiers.append((float(th), float(v)))
            tiers.sort(key=lambda x: x[0])
        except Exception:
            return float(os.getenv("MARGIN_PERCENT", "1.0"))

        for threshold, val in tiers:
            if balance < threshold:
                return val

        return tiers[-1][1]


    def _cooldown_active(self) -> bool:
        """Return True if cooldown blocks new trades."""
        money = getattr(self, "money", None)
        if money is None:
            return False

        cons = int(money.state.get("consecutive_losses", 0))

        if cons >= 2:
            cooldown_candles = int(os.getenv("COOLDOWN_AFTER_2_LOSSES", "10"))
        elif cons == 1:
            cooldown_candles = int(os.getenv("COOLDOWN_AFTER_LOSS", "3"))
        else:
            return False

        last = getattr(self, "_last_trade_close_time", {}).get("ts")
        if last is None:
            return False

        now = datetime.utcnow().timestamp()
        elapsed_seconds = now - last

        candle_sec = 5 * 60
        needed = cooldown_candles * candle_sec

        return elapsed_seconds < needed

    def _in_news_blackout(self) -> bool:
        """
        Check NEWS_BLACKOUT_UTC env var like "12:20-12:40,13:20-13:40" (UTC ranges).
        Returns True if current UTC time falls into any range.
        """
        raw = os.getenv("NEWS_BLACKOUT_UTC", "")
        if not raw:
            return False
        try:
            now = datetime.utcnow()
            minutes_now = now.hour * 60 + now.minute
            ranges = [r.strip() for r in raw.split(",") if r.strip()]
            for r in ranges:
                try:
                    start_s, end_s = r.split("-")
                    sh, sm = [int(x) for x in start_s.split(":")]
                    eh, em = [int(x) for x in end_s.split(":")]
                    start_min = sh * 60 + sm
                    end_min = eh * 60 + em
                    if start_min <= minutes_now <= end_min:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _atr_volatility_block(self, symbol: str) -> tuple[bool, Optional[str]]:
        """
        Return (blocked:bool, reason:str|None) if ATR volatility is too high.
        Uses env ATR_VOL_MULT and ATR_VOL_LOOKBACK_SAMPLES.
        """
        try:
            mult = float(os.getenv("ATR_VOL_MULT", "3.0"))
            lookback = int(os.getenv("ATR_VOL_LOOKBACK_SAMPLES", "200"))

            # current ATR (short window)
            cur_atr = self._calc_atr(symbol, interval="5m", samples=30, atr_length=14) or 0.0

            # long-window ATR average
            avg_atr = self._calc_atr(symbol, interval="5m", samples=lookback, atr_length=14) or 0.0

            if avg_atr > 0 and cur_atr > (avg_atr * mult):
                return True, f"atr_volatility_high (cur={cur_atr:.6f} avg={avg_atr:.6f} mult={mult})"

        except Exception:
            return False, None

        return False, None
 


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
            margin_usdt: Optional[float],
            mode: str,
            rl_qty: Optional[float],
            fixed_qty: Optional[float],
            leverage: Optional[int],
        ) -> float:

        try:
            price_data = self.client.ticker_price(symbol)
            price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
        except Exception:
            return 0.0

        # Direct modes first
        if rl_qty is not None:
            return float(rl_qty)
        if fixed_qty is not None:
            return float(fixed_qty)

        # Fetch account balance (Pylance-safe)
        money = getattr(self, "money", None)
        bal = None

        if money is not None:
            bal = money.get_account_balance()

        if bal is None:
            try:
                bal = float(self.client.balance("USDT"))
            except Exception:
                bal = 100.0  # safe fallback

        # Dynamic MARGIN%
        margin_pct = self._dynamic_margin_percent(bal)

        # ATR sizing?
        if os.getenv("USE_ATR_SIZING", "false").lower() == "true":
            atr = self._calc_atr(symbol)
            if atr and atr > 0:
                risk_pct = float(os.getenv("RISK_PER_TRADE_PCT", "1.5"))
                risk_usdt = (risk_pct / 100.0) * bal

                raw_sl = os.getenv("ATR_MULT_SL")
                sl_mult = float(raw_sl) if raw_sl else 1.5
                stop_distance = atr * sl_mult

                if stop_distance * price > 0:
                    qty = risk_usdt / stop_distance
                    return max(qty, 0.0)

        # Fallback: percent margin mode
        if mode == "percent":
            usdt_to_use = bal * (margin_pct / 100.0)
            qty = usdt_to_use / price
            return max(qty, 0.0)

        return 0.0



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

        import math
        import time
        import os
        from datetime import datetime

        # ------------------------------------------------------------------
        # Safe load_dotenv import + fallback (Pylance-safe)
        # ------------------------------------------------------------------
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            def load_dotenv(*args, **kwargs) -> bool:
                return False

        load_dotenv(override=True)

        symbol = symbol.upper()
        direction = direction.upper()
        is_long = direction in ("LONG", "BUY")
        entry_price: float = 0.0  # <-- Pylance-safe initialization

        try:
            # -----------------------------------------------------
            # 1. Duplicate-open protection (cache + exchange)
            # -----------------------------------------------------
            existing = self.open_positions.get(symbol)
            if existing and existing.get("status") == "OPEN":
                try:
                    exch_pos = self._is_open_on_exchange(symbol)
                    if exch_pos:
                        logger.warning(
                            "Attempt to open %s for %s but an OPEN position already exists (exchange confirmed)",
                            direction, symbol,
                        )
                        return {
                            "symbol": symbol,
                            "status": "FAILED_ALREADY_OPEN",
                            "reason": "already_open",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    else:
                        logger.info("Stale OPEN cache for %s — clearing", symbol)
                        self.open_positions.pop(symbol, None)
                except Exception:
                    logger.debug("Exchange check failed; clearing cache", exc_info=True)
                    self.open_positions.pop(symbol, None)

            # -----------------------------------------------------
            # 1.5 MoneyManager allow?
            # -----------------------------------------------------
            try:
                money = getattr(self, "money", None)
                if money is not None and hasattr(money, "can_trade"):
                    allowed, reason = money.can_trade()
                    if not allowed:
                        logger.warning("Open blocked by MoneyManager: %s", reason)
                        return {
                            "symbol": symbol,
                            "status": "BLOCKED_MM",
                            "reason": reason,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
            except Exception:
                logger.debug("MoneyManager check failed", exc_info=True)

            # -----------------------------------------------------
            # 1.6 Cooldown throttle
            # -----------------------------------------------------
            try:
                if self._cooldown_active():
                    logger.warning("Open blocked: cooldown after losses")
                    return {
                        "symbol": symbol,
                        "status": "BLOCKED_COOLDOWN",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception:
                logger.debug("Cooldown check failed", exc_info=True)

            # -----------------------------------------------------
            # 1.7 News/session blackout
            # -----------------------------------------------------
            try:
                if self._in_news_blackout():
                    logger.warning("Open blocked: NEWS_BLACKOUT_UTC window")
                    return {
                        "symbol": symbol,
                        "status": "BLOCKED_NEWS",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception:
                logger.debug("News blackout check failed", exc_info=True)

            # -----------------------------------------------------
            # 1.8 ATR-based volatility block
            # -----------------------------------------------------
            try:
                blocked, reason = self._atr_volatility_block(symbol)
                if blocked:
                    logger.warning("Open blocked by ATR volatility: %s", reason)
                    return {
                        "symbol": symbol,
                        "status": "BLOCKED_ATR_VOL",
                        "reason": reason,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception:
                logger.debug("ATR volatility check failed", exc_info=True)

            # -----------------------------------------------------
            # 2. Time sync
            # -----------------------------------------------------
            try:
                fut_time_fn = getattr(self.client, "futures_time", None)
                if callable(fut_time_fn):
                    server_time = fut_time_fn()
                    if isinstance(server_time, dict) and "serverTime" in server_time:
                        offset = server_time["serverTime"] - int(time.time() * 1000)
                        setattr(self.client, "time_offset", offset)
            except Exception:
                logger.debug("Server time sync failed")

            # -----------------------------------------------------
            # 3. Get price
            # -----------------------------------------------------
            try:
                price_data = self.client.ticker_price(symbol)
                price = (
                    float(price_data["price"])
                    if isinstance(price_data, dict)
                    else float(price_data)
                )
            except Exception:
                logger.warning("Failed to fetch ticker price for %s", symbol)
                return None

            if price <= 0:
                logger.warning("Invalid price for %s: %s", symbol, price)
                return None

            # -----------------------------------------------------
            # 4. Symbol filters
            # -----------------------------------------------------
            filters = self._get_symbol_filters(symbol)
            tick = float(filters.get("tick", 0.0))
            step = float(filters.get("step", 0.0))
            min_qty = filters.get("min_qty")

            # -----------------------------------------------------
            # 5. ATR-based TP/SL
            # -----------------------------------------------------
            atr = self._calc_atr(symbol)
            sl_mult: float = float(os.getenv("ATR_MULT_SL", "1.5"))

            if atr and atr > 0:
                tp_env = os.getenv("ATR_MULT_TP")
                tp_mults = self._safe_parse_tp_mults(tp_env)

                if is_long:
                    tp_levels = [self._round_to(price + atr * m, tick) for m in tp_mults]
                    sl_price = self._round_to(price - atr * sl_mult, tick)
                else:
                    tp_levels = [self._round_to(price - atr * m, tick) for m in tp_mults]
                    sl_price = self._round_to(price + atr * sl_mult, tick)
            else:
                # fallback % mode
                if is_long:
                    tp_levels = [self._round_to(price * (1 + tp_percent / 100), tick)]
                    sl_price = self._round_to(price * (1 - sl_percent / 100), tick)
                else:
                    tp_levels = [self._round_to(price * (1 - tp_percent / 100), tick)]
                    sl_price = self._round_to(price * (1 + sl_percent / 100), tick)

            # -----------------------------------------------------
            # 6. Quantity
            # -----------------------------------------------------
            qty = self._calc_quantity(
                symbol=symbol,
                margin_usdt=None,
                mode=size_mode,
                rl_qty=rl_qty,
                fixed_qty=fixed_qty,
                leverage=None,
            )

            # ensure qty is always a float
            try:
                qty = float(qty)
            except Exception:
                return {
                    "symbol": symbol,
                    "status": "FAILED_VALIDATION",
                    "error": "qty_not_numeric",
                }

            qty = self._round_to(qty, step) if step > 0 else round(qty, 6)

            # safely parse min_qty
            try:
                min_qty_f = float(min_qty) if min_qty is not None else None
            except Exception:
                min_qty_f = None

            if min_qty_f is not None and qty < min_qty_f:
                return {
                    "symbol": symbol,
                    "status": "FAILED_VALIDATION",
                    "error": f"qty<{min_qty_f}",
                }


            # -----------------------------------------------------
            # 6.1 NOTIONAL CAP (fully Pylance-safe)
            # -----------------------------------------------------
            bal: Optional[float] = None

            try:
                max_notional_pct = float(os.getenv("MAX_NOTIONAL_PCT", "20.0"))
            except Exception:
                max_notional_pct = 20.0

            # safe float converter
            def _to_float(v: Any) -> Optional[float]:
                try:
                    return float(v)
                except Exception:
                    return None


            # ---- MoneyManager balance ----
            money = getattr(self, "money", None)
            if money is not None:
                for m in ("get_balance_safe", "get_account_balance", "get_balance"):
                    fn = getattr(money, m, None)
                    if callable(fn):
                        try:
                            f = _to_float(fn())
                            if f is not None:
                                bal = f
                                break
                        except Exception:
                            continue

            # ---- fallback: exchange balance ----
            if bal is None:
                try:
                    acc_fn = getattr(self.client, "futures_account_balance", None)
                    if callable(acc_fn):
                        resp = acc_fn()

                        if isinstance(resp, list):
                            for e in resp:
                                if isinstance(e, dict) and e.get("asset") in ("USDT", "USD"):
                                    b = _to_float(e.get("balance") or e.get("free"))
                                    if b is not None:
                                        bal = b
                                        break

                        elif isinstance(resp, dict):
                            for k in ("totalWalletBalance", "availableBalance"):
                                b = _to_float(resp.get(k))
                                if b is not None:
                                    bal = b
                                    break
                except Exception:
                    bal = None


            # ---- apply notional limit ----
            if bal is not None:
                max_notional = (max_notional_pct / 100.0) * bal
                intended = qty * price

                if intended > max_notional:
                    new_qty = max_notional / price if price > 0 else 0.0

                    if step > 0:
                        new_qty = math.floor(new_qty / step) * step

                    new_qty = max(round(new_qty, 6), 0.0)

                    if new_qty <= 0:
                        return {
                            "symbol": symbol,
                            "status": "FAILED_VALIDATION",
                            "error": "notional_cap_reduced_to_zero",
                        }

                    qty = new_qty


            # -----------------------------------------------------
            # 6.2 AUTO-MIN-NOTIONAL BOOSTER (Pylance-clean)
            # -----------------------------------------------------
            try:

                # Pylance-safe float converter
                def safe_float(v: Any) -> float:
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                # -------------------------
                # Live price (type-safe)
                # -------------------------
                try:
                    lp = self.client.ticker_price(symbol)

                    if isinstance(lp, dict):
                        live_price = safe_float(lp.get("price"))
                    else:
                        live_price = safe_float(lp)

                except Exception:
                    live_price = safe_float(price)

                # -------------------------
                # min_notional (type-safe)
                # -------------------------
                min_notional_f = safe_float(filters.get("min_notional"))

                if min_notional_f > 0 and (qty * live_price) < min_notional_f:

                    # required qty
                    required_qty = (
                        min_notional_f / live_price
                        if live_price > 0 else qty
                    )

                    # round to step
                    if step > 0:
                        required_qty = math.ceil(required_qty / step) * step

                    # re-check risk cap
                    if bal is not None and max_notional_pct > 0:
                        max_notional = (max_notional_pct / 100.0) * bal
                        max_qty_allowed = (
                            max_notional / live_price
                            if live_price > 0 else required_qty
                        )

                        if step > 0:
                            max_qty_allowed = math.floor(max_qty_allowed / step) * step

                        if required_qty > max_qty_allowed:
                            return {
                                "symbol": symbol,
                                "status": "FAILED_VALIDATION",
                                "error": "min_notional_required_but_exceeds_risk_limits",
                            }

                    # enforce min_qty_f safely
                    if min_qty_f is not None and required_qty < min_qty_f:
                        required_qty = min_qty_f

                    # final apply
                    if required_qty > 0:
                        qty = required_qty

            except Exception:
                logger.debug("Auto-min-notional fallback failed", exc_info=True)



            # -----------------------------------------------------
            # 7. Dry-run mode
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
            guard = getattr(self, "guard", None)
            if guard is not None:
                try:
                    orderbook = self._fetch_orderbook(symbol)
                    trades = self._fetch_recent_trades(symbol)
                    suspect, reason = guard.check(orderbook, trades, events_per_sec=0.0)
                    if suspect:
                        return {
                            "symbol": symbol,
                            "status": "REJECTED_SUSPECT",
                            "reason": reason,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                except Exception:
                    logger.debug("MarketIntegrityGuard check failed", exc_info=True)

            # -----------------------------------------------------
            # 9. minNotional check (final)
            # -----------------------------------------------------
            try:
                live_price = float(self.client.ticker_price(symbol)["price"])
            except Exception:
                live_price = price

            min_notional = float(filters.get("min_notional") or 0.0)
            if qty * live_price < min_notional:
                return {
                    "symbol": symbol,
                    "status": "FAILED_VALIDATION",
                    "error": f"notional {qty * live_price:.8f} < {min_notional}",
                }

            # -----------------------------------------------------
            # 10. MARKET ORDER
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
                logger.exception("Market order failed: %s", e)
                return {
                    "symbol": symbol,
                    "status": "PENDING_ORDER_FAILED",
                    "error": str(e),
                }

            # -----------------------------------------------------
            # 11. Sync exit_meta (Pylance-clean + SmartExit-safe)
            # -----------------------------------------------------
            try:
                smart_exit = getattr(self, "smart_exit", None)
                bot_ref = getattr(self, "bot", None)

                # ----------------------------------------
                # Determine entry_price (fills > mark > ticker)
                # ----------------------------------------
                entry_price = 0.0

                # From fills
                try:
                    if isinstance(order_result, dict):
                        fills = order_result.get("fills")
                        if isinstance(fills, list) and fills:
                            total_q = 0.0
                            total_pxq = 0.0
                            for f in fills:
                                q_raw = f.get("qty") or f.get("commissionQty") or 0
                                p_raw = f.get("price") or 0
                                try:
                                    q = float(q_raw)
                                    p = float(p_raw)
                                except Exception:
                                    continue
                                if q > 0 and p > 0:
                                    total_q += q
                                    total_pxq += q * p
                            if total_q > 0:
                                entry_price = total_pxq / total_q
                except Exception:
                    entry_price = 0.0

                # From mark price if fills failed
                if entry_price <= 0:
                    try:
                        mp = self.client.futures_mark_price(symbol=symbol)
                        entry_price = float(mp.get("markPrice") or price)
                    except Exception:
                        entry_price = price

                # -------------------------------------------------------
                # Baseline ATR — MUST come from SmartExit, not Execution
                # -------------------------------------------------------
                baseline_atr = 0.0
                try:
                    se_atr_source = smart_exit or bot_ref
                    if se_atr_source and hasattr(se_atr_source, "_get_atr_baseline"):
                        raw_atr = se_atr_source._get_atr_baseline(symbol)
                        baseline_atr = float(raw_atr) if raw_atr else 0.0
                except Exception:
                    baseline_atr = 0.0

                # -------------------------------------------------------
                # Write exit_meta ONLY on the SmartExit holder
                # -------------------------------------------------------
                holder = None
                if smart_exit is not None and hasattr(smart_exit, "exit_meta"):
                    holder = smart_exit
                elif bot_ref is not None and hasattr(bot_ref, "exit_meta"):
                    holder = bot_ref

                if holder:
                    meta = {
                        "entry_price": float(entry_price),
                        "qty": float(qty),
                        "side": "LONG" if is_long else "SHORT",
                        "last_sl": float(sl_price),
                        "baseline_atr": float(baseline_atr),
                        "breakeven_moved": False,
                        "last_update_ts": time.time(),
                        "created": time.time(),
                        "sl_mult": float(sl_mult),
                        "tick": float(tick),
                        "tp_levels": [float(x) for x in tp_levels],
                    }

                    holder.exit_meta[symbol] = meta
                    logger.debug(f"[Execution] exit_meta initialized for {symbol}: {meta}")

            except Exception:
                logger.exception("Failed syncing exit_meta")

            # -----------------------------------------------------
            # 12. SmartExit: create exit orders
            # -----------------------------------------------------
            smartexit_err: Optional[str] = None
            smart_exit = getattr(self, "smart_exit", None)

            # Ensure entry_price always exists (Pylance-safe)
            safe_entry_price = entry_price if (isinstance(entry_price, (int, float)) and entry_price > 0) else price

            if USE_SMART_EXIT and smart_exit is not None:
                try:
                    time.sleep(0.6)
                    smart_exit.create_exit_orders(
                        symbol=symbol,
                        side=direction,
                        entry_price=safe_entry_price,
                        qty=qty,
                        atr_value=atr,
                        tick_size=tick,
                        step_size=step,
                        tp_levels=tp_levels,
                    )
                except Exception as e:
                    smartexit_err = str(e)
                    logger.exception("SmartExit.create_exit_orders failed: %s", e)

            # -----------------------------------------------------
            # 13. Reconcile
            # -----------------------------------------------------
            time.sleep(0.5)
            pos = None
            if smart_exit is not None:
                try:
                    pos = smart_exit._get_position(symbol)
                except Exception:
                    pos = None

            if not pos:
                logger.error("Order executed but no position detected — clearing cache")
                self.open_positions.pop(symbol, None)
                return {"symbol": symbol, "status": "FAILED_NO_POSITION_AFTER_ORDER"}

            # -----------------------------------------------------
            # 14. Store open position
            # -----------------------------------------------------
            tracked = {
                "symbol": symbol,
                "side": pos.get("side"),
                "entry": pos.get("entry"),
                "qty": pos.get("qty"),
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
            logger.exception("Unhandled open_position error: %s", e)
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
        from datetime import datetime

        # ---------------------------------------------------------
        # Local helper safe float
        # ---------------------------------------------------------
        def _safe_float(x: Any, default: float = 0.0) -> float:
            try:
                if x is None:
                    return default
                return float(x)
            except (ValueError, TypeError):
                return default

        # ---------------------------------------------------------
        # Local helper: safe PnL recording
        # ---------------------------------------------------------
        def _record_mm_pnl(pnl: float) -> None:
            try:
                mm = getattr(self, "money", None)
                if mm is not None and hasattr(mm, "record_closed_trade"):
                    mm.record_closed_trade(float(pnl))
            except Exception:
                pass

        # Ensure cooldown dictionary exists
        if not hasattr(self, "_last_trade_close_time") or not isinstance(self._last_trade_close_time, dict):
            self._last_trade_close_time = {"ts": 0.0}

        if symbol not in self.open_positions:
            logger.warning("No open position cached for %s", symbol)
            return

        pos = self.open_positions[symbol]
        qty = _safe_float(pos.get("qty"), 0.0)
        if qty <= 0:
            logger.warning("Invalid qty=%.8f for %s — cannot close", qty, symbol)
            return

        opposite = "SELL" if side.upper() in ("LONG", "BUY") else "BUY"

        # ---------------------------------------------------------
        # Cancel SmartExit orders first
        # ---------------------------------------------------------
        try:
            if USE_SMART_EXIT and getattr(self, "smart_exit", None) is not None:
                if hasattr(self.smart_exit, "_cancel_all_exit_orders"):
                    logger.info("Cancelling SmartExit TP/SL for %s before closing", symbol)
                    self.smart_exit._cancel_all_exit_orders(symbol)
                else:
                    self.cancel_all_orders(symbol)
            else:
                self.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning("SmartExit cleanup failed for %s: %s", symbol, e)

        # ---------------------------------------------------------
        # Fetch filters
        # ---------------------------------------------------------
        filters = self._get_symbol_filters(symbol) or {}
        step_size = _safe_float(filters.get("step"))
        min_qty = _safe_float(filters.get("min_qty"))
        min_notional = _safe_float(filters.get("min_notional"))

        def _round_down_step(q: float, step: float) -> float:
            if step > 0:
                return math.floor(q / step) * step
            return round(q, 6)

        rounded_qty = _round_down_step(qty, step_size)
        if rounded_qty <= 0 and qty > 0:
            rounded_qty = round(qty, 6)

        # ---------------------------------------------------------
        # Fetch mark price
        # ---------------------------------------------------------
        mark_price_raw = None
        try:
            price_data = self.client.ticker_price(symbol)
            mark_price_raw = price_data.get("price") if isinstance(price_data, dict) else price_data
        except Exception:
            logger.debug("Could not fetch mark price; proceeding without minNotional check")

        mark_price = _safe_float(mark_price_raw)

        # ---------------------------------------------------------
        # Ensure meets minNotional
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Validate min_qty
        # ---------------------------------------------------------
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

        rounded_qty = min(rounded_qty, qty)

        # ---------------------------------------------------------
        # DRY RUN
        # ---------------------------------------------------------
        if self.dry_run:
            logger.info("[DRY RUN] Would close %s %s qty=%.8f", symbol, opposite, rounded_qty)

            _record_mm_pnl(0.0)

            self.open_positions.pop(symbol, None)
            self._last_trade_close_time["ts"] = datetime.utcnow().timestamp()
            return

        # ---------------------------------------------------------
        # dualSidePosition detection
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Place close order
        # ---------------------------------------------------------
        order_kwargs = {
            "symbol": symbol,
            "side": opposite,
            "type": "MARKET",
            "quantity": rounded_qty,
            "reduceOnly": True,
        }
        if position_side_flag:
            order_kwargs["positionSide"] = position_side_flag

        try:
            order = self.client.futures_create_order(**order_kwargs)
            logger.info("Closed %s %s | qty=%.8f", symbol, opposite, rounded_qty)
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
            except Exception:
                pass

            logger.error("Error closing %s: %s", symbol, err_text)

        # ---------------------------------------------------------
        # MoneyManager PnL
        # ---------------------------------------------------------
        try:
            entry = _safe_float(pos.get("entry"), 0.0)
            qty_f = _safe_float(pos.get("qty"), 0.0)
            side_raw = str(pos.get("side", "")).upper()

            cp_raw = None
            try:
                cp = self.client.ticker_price(symbol)
                cp_raw = cp["price"] if isinstance(cp, dict) else cp
            except Exception:
                pass

            close_price = _safe_float(cp_raw, entry)

            pnl = 0.0
            if entry > 0 and close_price > 0 and qty_f > 0:
                if side_raw in ("LONG", "BUY"):
                    pnl = (close_price - entry) * qty_f
                elif side_raw in ("SHORT", "SELL"):
                    pnl = (entry - close_price) * qty_f

            _record_mm_pnl(pnl)
            logger.info("MoneyManager: PNL=%.4f for %s", pnl, symbol)

        except Exception:
            logger.debug("MoneyManager PNL record failed for %s", symbol, exc_info=True)

        # ---------------------------------------------------------
        # Final cleanup
        # ---------------------------------------------------------
        try:
            self._post_exit_cleanup(symbol)
        except Exception as e:
            logger.warning("Post-exit cleanup failed for %s: %s", symbol, e)

        self.open_positions.pop(symbol, None)
        logger.info("%s removed from local cache", symbol)

        self._last_trade_close_time["ts"] = datetime.utcnow().timestamp()





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
        """
        Patched Version
        ------------------------------------------
        Periodic live SmartExit controller.

        Fixes:
            • prevents double SL updates in same loop
            • prevents repeated cancel loops
            • ensures Step-4 engine runs only after trailing/breakeven
            • backoff for Binance rate limits
            • avoids recursive failures
        """
        if not USE_SMART_EXIT:
            return

        # Ensure SmartExit exists
        if not hasattr(self, "smart_exit") or self.smart_exit is None:
            logger.warning("manage_positions_live called without SmartExit instance")
            return

        for symbol in list(self.open_positions.keys()):
            try:
                pos = self.open_positions.get(symbol)
                if not pos or pos.get("qty", 0) <= 0:
                    continue

                # Prevent SL update spam: if smart_exit already updated this symbol in the last cycle
                last_update_ts = self.smart_exit.exit_meta.get(symbol, {}).get("last_update_ts", 0)
                now_ts = time.time()

                already_updated = (now_ts - last_update_ts) < 0.3  # 300ms guard
                if already_updated:
                    logger.debug(f"Skipping {symbol}: SL update already applied recently.")
                    continue

                # ===================================================
                # 1) Classic trailing/breakeven engine
                # ===================================================
                try:
                    result = self.smart_exit.manage_open_positions(symbol)

                    # If trailing/breakeven updated something
                    if result:
                        # mark update timestamp so Step-4 engine doesn't double adjust
                        self.smart_exit.exit_meta.setdefault(symbol, {})["last_update_ts"] = time.time()

                except Exception as e1:
                    logger.warning(f"SmartExit trailing/breakeven failed for {symbol}: {e1}")

                # ===================================================
                # 2) Step-4 Advanced Engine
                #    (Runs only if Step-1 did NOT already update SL)
                # ===================================================
                try:
                    # Check again in case Step-1 just updated
                    last_update_ts = self.smart_exit.exit_meta.get(symbol, {}).get("last_update_ts", 0)
                    if (time.time() - last_update_ts) >= 0.3:

                        if hasattr(self.smart_exit, "monitor_position_and_adjust"):
                            self.smart_exit.monitor_position_and_adjust(symbol)

                except Exception as e2:
                    logger.warning(f"SmartExit Step-4 advanced SL engine failed for {symbol}: {e2}")

            except Exception as outer:
                # Hard fail-safe
                logger.warning(f"Smart exit management failed for {symbol}: {outer}")



    def enhanced_monitor(self):
        """
        Runs all Step-4 SmartExit monitors for all open symbols.
        Includes:
        - trailing/breakeven SL (original engine)
        - emergency ATR kill
        - spread SL widening
        - SL choke protection
        """
        if not USE_SMART_EXIT:
            return {}

        results = {}
        for sym in list(self.open_positions.keys()):
            try:
                r1 = self.smart_exit.manage_open_positions(sym)

                r2 = None
                if hasattr(self.smart_exit, "monitor_position_and_adjust"):
                    r2 = self.smart_exit.monitor_position_and_adjust(sym)

                results[sym] = {"base": r1, "advanced": r2}

            except Exception:
                logger.exception("enhanced_monitor failed for %s", sym)

        return results



    # ---------------------------
    # Reconcile & manage
    # ---------------------------
    def reconcile_open_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.open_positions

    def manage_open_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified SmartExit + Execution reconciliation layer.

        Responsibilities:
        -------------------------------------------------------
        ✓ Run SmartExit.manage_open_positions() safely
        ✓ Detect external position closes (exchange → cache cleanup)
        ✓ Sync bot.exit_meta entries (SmartExit Step-4 compatibility)
        ✓ Track SL changes, breakeven moves, or invalid positions
        ✓ Harden against:
            - stale cache,
            - missing exit_meta,
            - SmartExit failures,
            - exchange/calc desync.
        -------------------------------------------------------
        """

        if not USE_SMART_EXIT:
            return {}

        results: Dict[str, Any] = {}
        failed: List[str] = []

        smart_exit = getattr(self, "smart_exit", None)
        if smart_exit is None:
            logger.warning("manage_open_positions called but smart_exit is None")
            return {}

        # Safe accessor method (may not exist on all clients)
        _get_position = getattr(smart_exit, "_get_position", lambda s: None)

        # Determine target symbols
        targets = [symbol] if symbol else list(self.open_positions.keys())

        for sym in targets:
            try:
                # ------------------------------------------------------------
                # 1) PRE-SCAN: get latest exchange position snapshot
                # ------------------------------------------------------------
                prev = _get_position(sym) or {}
                prev_sl = prev.get("sl")

                # Ensure exit_meta exists
                bot = getattr(self, "bot", None)
                if bot is not None and hasattr(bot, "exit_meta"):
                    if sym not in bot.exit_meta:
                        bot.exit_meta[sym] = {
                            "created": time.time(),
                            "baseline_atr": 0.0,
                            "entry_price": prev.get("entry", 0.0),
                            "qty": prev.get("qty", 0.0),
                            "side": prev.get("side", "LONG"),
                            "breakeven_moved": False,
                            "last_update_ts": time.time(),
                        }
                        logger.debug(f"[Execution] exit_meta auto-created for {sym}")

                # ------------------------------------------------------------
                # 2) CALL SmartExit
                # ------------------------------------------------------------
                try:
                    res = smart_exit.manage_open_positions(sym)
                except Exception as se:
                    logger.exception("SmartExit.manage_open_positions failed for %s: %s", sym, se)
                    failed.append(sym)
                    continue

                # ------------------------------------------------------------
                # 3) POST-SCAN: retrieve updated exchange state
                # ------------------------------------------------------------
                post = _get_position(sym) or {}
                new_sl = post.get("sl")

                # Log SL movement
                if new_sl is not None and prev_sl != new_sl:
                    logger.info("[SmartExit] %s SL changed: %s → %s", sym, prev_sl, new_sl)

                # ------------------------------------------------------------
                # 4) If SmartExit reports *no live position* but cache says OPEN → cleanup
                # ------------------------------------------------------------
                if not post and sym in self.open_positions:
                    try:
                        logger.info("External close detected for %s — cleaning up...", sym)
                        self._post_exit_cleanup(sym)
                    except Exception as ce:
                        logger.warning("post-exit cleanup failed for %s: %s", sym, ce)
                    self.open_positions.pop(sym, None)

                    # also remove matching exit_meta
                    if bot is not None and hasattr(bot, "exit_meta"):
                        bot.exit_meta.pop(sym, None)

                    results[sym] = {"status": "CLOSED_EXTERNALLY"}
                    continue

                # ------------------------------------------------------------
                # 5) Return SmartExit output for visibility
                # ------------------------------------------------------------
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