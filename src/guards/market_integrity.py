# ==========================
# File: src/guards/market_integrity.py
# ==========================
from collections import deque
import logging
import numpy as np
import os

logger = logging.getLogger("market_integrity")


class MarketIntegrityGuard:
    """Simple, standalone market integrity guard to detect spoofing / quote stuffing / churn.

    Methods return (suspicious: bool, reason: Optional[str]).
    Lightweight, deterministic, and tunable via .env or constructor arguments.
    """

    def __init__(self,
                 spread_window_len: int = 200,
                 depth_window: int = 10,
                 spread_mult: float = 3.0,
                 imbalance_limit: float = 0.75,
                 book_event_limit: int = 50,
                 taker_imbal_limit: float = 0.7,
                 event_rate_window_len: int = 50):
        # Rolling windows
        self.spread_window = deque(maxlen=spread_window_len)
        self.event_rate_window = deque(maxlen=event_rate_window_len)

        # Tunable parameters (load from env if available)
        self.depth_window = depth_window
        self.SPREAD_MULT = float(os.getenv("SPREAD_MULT", spread_mult))
        self.IMBALANCE_LIMIT = float(os.getenv("IMBALANCE_LIMIT", imbalance_limit))
        self.BOOK_EVENT_LIMIT = float(os.getenv("BOOK_EVENT_LIMIT", book_event_limit))
        self.TAKER_IMBAL_LIMIT = float(os.getenv("TAKER_IMBAL_LIMIT", taker_imbal_limit))

    def clear_state(self):
        """Reset rolling windows and temporary statistics."""
        self.spread_window.clear()
        self.event_rate_window.clear()
        logger.info("MarketIntegrityGuard state cleared.")

    def check(self, orderbook: dict, recent_trades: list, events_per_sec: float = 0.0):
        """Run a set of deterministic checks on provided market microstructure data.

        orderbook: {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}
        recent_trades: [{'price': float, 'qty': float, 'side': 'buy'|'sell', ...}, ...]
        events_per_sec: observed orderbook update rate per second (optional)
        """
        try:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if not bids or not asks:
                return True, "empty_orderbook"

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            self.spread_window.append(spread)

            # Spread spike detection
            if len(self.spread_window) >= 10:
                mean_spread = float(np.mean(self.spread_window))
                if mean_spread > 0 and spread > mean_spread * self.SPREAD_MULT:
                    reason = f"spread_spike_{spread:.6f}"
                    logger.warning(f"[Integrity Alert] {reason} | mean={mean_spread:.6f}")
                    return True, reason

            # Spread volatility detection (stability check)
            if len(self.spread_window) >= 20:
                mean_spread = np.mean(self.spread_window)
                std_spread = np.std(self.spread_window)
                if mean_spread > 0 and std_spread > mean_spread * 0.5:
                    reason = f"spread_volatility_{std_spread:.6f}"
                    logger.warning(f"[Integrity Alert] {reason} | mean={mean_spread:.6f}")
                    return True, reason

            # Depth imbalance
            bid_vol = sum(float(q) for _, q in bids[: self.depth_window])
            ask_vol = sum(float(q) for _, q in asks[: self.depth_window])
            total_depth = bid_vol + ask_vol
            if total_depth > 0:
                imbalance = (bid_vol - ask_vol) / total_depth
                if abs(imbalance) > self.IMBALANCE_LIMIT:
                    side = "bid" if imbalance > 0 else "ask"
                    reason = f"depth_imbalance_{side}_{imbalance:.3f}"
                    logger.warning(f"[Integrity Alert] {reason} | bid_vol={bid_vol:.2f} ask_vol={ask_vol:.2f}")
                    return True, reason

            # Rolling churn / quote stuffing
            if events_per_sec > 0:
                self.event_rate_window.append(events_per_sec)
                if len(self.event_rate_window) >= 10:
                    avg_rate = np.mean(self.event_rate_window)
                    if avg_rate > self.BOOK_EVENT_LIMIT:
                        reason = f"persistent_churn_{avg_rate:.1f}"
                        logger.warning(f"[Integrity Alert] {reason}")
                        return True, reason
                elif events_per_sec > self.BOOK_EVENT_LIMIT * 1.5:
                    reason = f"instant_churn_{events_per_sec:.1f}"
                    logger.warning(f"[Integrity Alert] {reason}")
                    return True, reason

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
                if ratio > self.TAKER_IMBAL_LIMIT or ratio < (1 - self.TAKER_IMBAL_LIMIT):
                    reason = f"taker_skew_{ratio:.3f}"
                    logger.warning(f"[Integrity Alert] {reason} | buys={buys:.2f} sells={sells:.2f}")
                    return True, reason

            return False, None

        except Exception as e:
            logger.exception("MarketIntegrityGuard.check error: %s", e)
            # Be conservative and non-blocking on error
            return False, None
