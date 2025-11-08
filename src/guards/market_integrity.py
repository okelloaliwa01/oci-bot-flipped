# ==========================
# File: src/guards/market_integrity.py
# ==========================
from collections import deque
import logging
import numpy as np

logger = logging.getLogger("market_integrity")


class MarketIntegrityGuard:
    """Simple, standalone market integrity guard to detect spoofing / quote stuffing / churn.

    Methods return (suspicious: bool, reason: Optional[str]).
    This is intentionally lightweight and deterministic (no ML) so you can tune thresholds.
    """

    def __init__(self,
                 spread_window_len: int = 200,
                 depth_window: int = 10,
                 spread_mult: float = 3.0,
                 imbalance_limit: float = 0.75,
                 book_event_limit: int = 50,
                 taker_imbal_limit: float = 0.7):
        self.spread_window = deque(maxlen=spread_window_len)
        self.depth_window = depth_window
        self.SPREAD_MULT = spread_mult
        self.IMBALANCE_LIMIT = imbalance_limit
        self.BOOK_EVENT_LIMIT = book_event_limit
        self.TAKER_IMBAL_LIMIT = taker_imbal_limit

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

            # Spread spike
            if len(self.spread_window) >= 10:
                mean_spread = float(np.mean(list(self.spread_window)))
                if mean_spread > 0 and spread > mean_spread * self.SPREAD_MULT:
                    return True, f"spread_spike_{spread:.6f}"

            # Depth imbalance on top-N levels
            bid_vol = sum(float(q) for _, q in bids[: self.depth_window])
            ask_vol = sum(float(q) for _, q in asks[: self.depth_window])
            total = bid_vol + ask_vol
            if total > 0:
                imbalance = (bid_vol - ask_vol) / total
                if abs(imbalance) > self.IMBALANCE_LIMIT:
                    side = "bid" if imbalance > 0 else "ask"
                    return True, f"depth_imbalance_{side}_{imbalance:.3f}"

            # High churn (quote stuffing)
            if events_per_sec and events_per_sec > self.BOOK_EVENT_LIMIT:
                return True, f"high_churn_{events_per_sec:.1f}"

            # Taker imbalance
            buys = 0.0
            sells = 0.0
            for t in recent_trades:
                try:
                    qty = float(t.get("qty") or t.get("quantity") or 0)
                except Exception:
                    qty = 0.0
                side = (t.get("side") or t.get("makerSide") or t.get("isBuyerMaker"))
                # normalize a few common trade shapes
                if isinstance(side, bool):
                    # some APIs have isBuyerMaker: True -> buyer was maker => taker was sell
                    if side:
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
                    return True, f"taker_skew_{ratio:.3f}"

            return False, None
        except Exception as e:
            logger.exception("MarketIntegrityGuard.check error: %s", e)
            # on error, be conservative and do not block by default
            return False, None


