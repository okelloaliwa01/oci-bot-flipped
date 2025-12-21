# src/mtf_filter.py

from __future__ import annotations

from typing import List, Optional, Any
from data_fetch import fetch_closed_candles
from breakout_logic import check_breakout
from logger import get_logger, log_mtf_disagreement
from config import MTF_REQUIRED_CONFIRM

logger = get_logger("mtf_filter")


def confirm_mtf_direction(
    client: Any,
    symbol: str,
    signal_side: str,
    timeframes: List[str] | None
) -> bool:
    """
    Confirm signal direction using higher timeframes.
    
    MTF_REQUIRED_CONFIRM options:
    - "all": all timeframes must agree
    - int (e.g., 2): at least N must agree
    
    Neutral MTF (no breakout) does NOT count as confirmation or disagreement.
    """
    
    if not timeframes:
        return True  # MTF disabled

    confirmations = 0
    total = len(timeframes)

    for tf in timeframes:
        try:
            df = fetch_closed_candles(symbol, tf, 50, client=client)

            # check_breakout returns: (signal, level, context)
            tf_signal, _, _ = check_breakout(df, 1.0)

            # Handle cases
            if tf_signal is None:
                logger.info(f"[MTF] {tf}: NEUTRAL — no breakout")
                continue

            if tf_signal == signal_side:
                confirmations += 1
            else:
                log_mtf_disagreement(tf, tf_signal, signal_side)

        except Exception as e:
            logger.warning(f"[MTF] Failed check for {tf}: {e}")

    # ------------------------------
    # Decision logic
    # ------------------------------
    if MTF_REQUIRED_CONFIRM == "all":
        return confirmations == total

    if isinstance(MTF_REQUIRED_CONFIRM, int):
        return confirmations >= MTF_REQUIRED_CONFIRM

    # Default = strict
    return confirmations == total
