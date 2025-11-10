# ==========================
# File: src/guards/market_integrity.py
# ==========================
from collections import deque
import logging
import numpy as np
import os

logger = logging.getLogger("market_integrity")


class MarketIntegrityGuard:
    """Tiered (🟢🟡🔴) Market Integrity Guard to detect spoofing / churn / imbalances.

    Returns:
        (suspicious: bool, reason: str | None)
    """

    def __init__(self,
                 spread_window_len: int = 200,
                 depth_window: int = 10,
                 spread_mult: float = 3.0,
                 imbalance_limit: float = 0.75,
                 book_event_limit: int = 50,
                 taker_imbal_limit: float = 0.7,
                 event_rate_window_len: int = 50):

        # Rolling buffers
        self.spread_window = deque(maxlen=spread_window_len)
        self.event_rate_window = deque(maxlen=event_rate_window_len)

        # Tunable parameters via env overrides
        self.depth_window = int(os.getenv("DEPTH_WINDOW", depth_window))
        self.SPREAD_MULT = float(os.getenv("SPREAD_MULT", spread_mult))
        self.IMBALANCE_LIMIT = float(os.getenv("IMBALANCE_LIMIT", imbalance_limit))
        self.BOOK_EVENT_LIMIT = float(os.getenv("BOOK_EVENT_LIMIT", book_event_limit))
        self.TAKER_IMBAL_LIMIT = float(os.getenv("TAKER_IMBAL_LIMIT", taker_imbal_limit))

        # Tiered guard mode
        # 1 = full (block severe, retry moderate)
        # 0 = log only (never block)
        self.block_mode = bool(int(os.getenv("GUARD_BLOCK_MODE", "1")))

        # Define tier thresholds
        self.SEVERE_MULT = 0.95   # 🔴 block
        self.MODERATE_MULT = 0.85 # 🟡 retry / warn

        logger.info(
            f"MarketIntegrityGuard initialized: "
            f"SPREAD_MULT={self.SPREAD_MULT}, IMBALANCE_LIMIT={self.IMBALANCE_LIMIT}, "
            f"BOOK_EVENT_LIMIT={self.BOOK_EVENT_LIMIT}, TAKER_IMBAL_LIMIT={self.TAKER_IMBAL_LIMIT}, "
            f"BLOCK_MODE={self.block_mode}"
        )

    def clear_state(self):
        """Reset rolling windows and stats."""
        self.spread_window.clear()
        self.event_rate_window.clear()
        logger.info("MarketIntegrityGuard state cleared.")

    def _tiered_flag(self, level: str, reason: str, details: str = ""):
        """Internal helper to log uniformly based on tier."""
        tag = {"SEVERE": "🚫 BLOCK", "MODERATE": "⚠️ RETRY", "INFO": "ℹ️ LOG"}[level]
        logger_method = {
            "SEVERE": logger.error,
            "MODERATE": logger.warning,
            "INFO": logger.info,
        }[level]
        logger_method(f"[Integrity {tag}] {reason} {details}")

        # Return blocking intent depending on mode
        if self.block_mode and level == "SEVERE":
            return True, f"{reason}|BLOCK"
        elif level == "MODERATE":
            return False, f"{reason}|RETRY"
        else:
            return False, f"{reason}|LOG"

    def check(self, orderbook: dict, recent_trades: list, events_per_sec: float = 0.0):
        """
        Run a deterministic integrity evaluation on the given orderbook + trades.
        Returns (suspicious: bool, reason: str | None).
        """
        try:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if not bids or not asks:
                return True, "empty_orderbook|BLOCK"

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            self.spread_window.append(spread)

            # Spread spike detection
            if len(self.spread_window) >= 10:
                mean_spread = float(np.mean(self.spread_window))
                if mean_spread > 0 and spread > mean_spread * self.SPREAD_MULT:
                    return self._tiered_flag(
                        "SEVERE", "spread_spike", f"(spread={spread:.4f}, mean={mean_spread:.4f})"
                    )

            # Spread volatility instability
            if len(self.spread_window) >= 20:
                mean_spread = np.mean(self.spread_window)
                std_spread = np.std(self.spread_window)
                if mean_spread > 0 and std_spread > mean_spread * 0.5:
                    return self._tiered_flag(
                        "MODERATE", "spread_volatility",
                        f"(std={std_spread:.4f}, mean={mean_spread:.4f})"
                    )

            # Depth imbalance detection
            bid_vol = sum(float(q) for _, q in bids[: self.depth_window])
            ask_vol = sum(float(q) for _, q in asks[: self.depth_window])
            total_depth = bid_vol + ask_vol
            if total_depth > 0:
                imbalance = (bid_vol - ask_vol) / total_depth
                abs_imb = abs(imbalance)
                side = "bid" if imbalance > 0 else "ask"

                if abs_imb > self.SEVERE_MULT:
                    return self._tiered_flag("SEVERE", f"depth_imbalance_{side}_{imbalance:.3f}")
                elif abs_imb > self.MODERATE_MULT:
                    return self._tiered_flag("MODERATE", f"depth_imbalance_{side}_{imbalance:.3f}")

            # Rolling churn / quote stuffing
            if events_per_sec > 0:
                self.event_rate_window.append(events_per_sec)
                if len(self.event_rate_window) >= 10:
                    avg_rate = np.mean(self.event_rate_window)
                    if avg_rate > self.BOOK_EVENT_LIMIT * 1.5:
                        return self._tiered_flag("SEVERE", f"persistent_churn_{avg_rate:.1f}")
                    elif avg_rate > self.BOOK_EVENT_LIMIT:
                        return self._tiered_flag("MODERATE", f"moderate_churn_{avg_rate:.1f}")
                elif events_per_sec > self.BOOK_EVENT_LIMIT * 1.5:
                    return self._tiered_flag("SEVERE", f"instant_churn_{events_per_sec:.1f}")

            # Taker imbalance
            buys = sells = 0.0
            for t in recent_trades:
                try:
                    qty = float(t.get("qty") or t.get("quantity") or 0)
                except Exception:
                    qty = 0.0

                side = t.get("side") or t.get("makerSide") or t.get("isBuyerMaker")
                if isinstance(side, bool):
                    if side:  # True → buyer was maker → taker was sell
                        sells += qty
                    else:
                        buys += qty
                else:
                    s = str(side).lower()
                    if s in ("buy", "b", "buyer", "taker_buy"):
                        buys += qty
                    elif s in ("sell", "s", "seller", "taker_sell"):
                        sells += qty

            total_t = buys + sells
            if total_t > 0:
                ratio = buys / total_t
                if ratio > self.SEVERE_MULT or ratio < (1 - self.SEVERE_MULT):
                    return self._tiered_flag("SEVERE", f"taker_skew_{ratio:.3f}")
                elif ratio > self.MODERATE_MULT or ratio < (1 - self.MODERATE_MULT):
                    return self._tiered_flag("MODERATE", f"taker_skew_{ratio:.3f}")

            # ✅ Nothing abnormal
            return False, None

        except Exception as e:
            logger.exception("MarketIntegrityGuard.check error: %s", e)
            # Fail-safe: do not block on internal error
            return False, None
