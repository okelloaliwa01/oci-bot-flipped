# src/execution.py
import os
import math
import logging
from datetime import datetime
from typing import Optional, Dict, Any
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
    """

    def __init__(self, binance_client, dry_run: bool = False):
        self.client = binance_client
        self.dry_run = dry_run
        self.smart_exit = SmartExitManager(binance_client)
        self.open_positions: Dict[str, Dict[str, Any]] = {}

        logger.info(f"ExecutionManager initialized (dry_run={self.dry_run})")

    # ------------------------------------------------------------
    # 🔹 Quantity Calculation
    # ------------------------------------------------------------
    def _calc_quantity(self, price: float, margin_usdt: Optional[float] = None) -> float:
        """Calculate position size based on margin or % balance."""
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
        """Round a value down to Binance tick/step size."""
        try:
            if step <= 0:
                return value
            return math.floor(value / step) * step
        except Exception:
            return value

    # ------------------------------------------------------------
    # 🔹 Open Position
    # ------------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        direction: str,
        margin_usdt: float,
        tp_percent: float,
        sl_percent: float,
    ):
        """Open a futures position with optional SmartExit integration."""
        try:
            load_dotenv(override=True)

            # 1️⃣ Get current price
            price_data = self.client.ticker_price(symbol)
            price = float(price_data["price"]) if isinstance(price_data, dict) else float(price_data)
            if not price or price <= 0:
                logger.warning(f"⚠️ Invalid price for {symbol}: {price}")
                return None

            # 2️⃣ Fetch precision filters
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

            # 3️⃣ Estimate ATR (basic fallback)
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
                    atr = sum(trs[-14:]) / min(len(trs), 14)
            except Exception as e:
                logger.debug(f"ATR calculation failed for {symbol}: {e}")

            # 4️⃣ Fallback TP/SL (percentage if ATR unavailable)
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

            # 6️⃣ Dry Run
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

            # 7️⃣ Live Market Order
            try:
                self.client.futures_create_order(
                    symbol=symbol,
                    side="BUY" if position_type == "LONG" else "SELL",
                    type="MARKET",
                    quantity=qty,
                )
                logger.info(f"✅ Opened {position_type} {symbol} | entry={price:.2f} qty={qty}")
            except Exception as e:
                logger.error(f"❌ Failed to open market order for {symbol}: {e}")
                return None

            # 8️⃣ Smart Exit Setup
            if USE_SMART_EXIT:
                try:
                    logger.info(f"🚀 SmartExitManager creating partial TP/SL for {symbol}")
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

            # 9️⃣ Track Position
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
    # 🔹 Manage Active Positions (Smart Exit)
    # ------------------------------------------------------------
    def manage_positions_live(self):
        """Run SmartExit trailing/breakeven management for all open positions."""
        if not USE_SMART_EXIT:
            logger.debug("SmartExit disabled; skipping management.")
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
        """Immediately close an active position."""
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
        logger.debug("♻️ reconcile_open_positions() returning open positions.")
        return self.open_positions

    def manage_open_positions(self, symbol: Optional[str] = None):
        """Detailed wrapper for SmartExitManager.manage_open_positions()."""
        if not USE_SMART_EXIT:
            logger.debug("SmartExit disabled; skipping manage_open_positions() wrapper.")
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
                        logger.info(
                            f"🔄 [SmartExit] {rtype.upper()} SL change {sym}: {prev_sl or 'None'} → {new_sl}"
                        )
                    elif rtype == "error":
                        logger.warning(f"[SmartExit] {sym} error: {result.get('error')}")
                    elif rtype == "noop":
                        logger.debug(f"[SmartExit] {sym}: No update required.")

            except Exception as e:
                logger.exception(f"⚠️ manage_open_positions() failed for {sym}: {e}")
