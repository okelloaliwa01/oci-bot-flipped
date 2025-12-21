#src/mtf_filter

from data_fetch import fetch_closed_candles
from breakout_logic import check_breakout
from logger import get_logger, log_mtf_disagreement
from config import MTF_REQUIRED_CONFIRM

logger = get_logger('mtf_filter')

def confirm_mtf_direction(client, symbol, signal_side, timeframes):
    """
    Confirm signal direction using higher timeframes.
    - 'all' = all timeframes must agree
    - int (e.g., 2) = at least that many must agree
    """
    if not timeframes:
        return True  # MTF disabled

    confirmations = 0
    total = len(timeframes)

    for tf in timeframes:
        try:
            df = fetch_closed_candles(symbol, tf, 50, client=client)
            tf_signal, _ = check_breakout(df, 1.0)

            if tf_signal is None:
                logger.info(f"MTF {tf} is NEUTRAL — no breakout")
                # ✅ neutral is neither confirmation nor disagreement
            elif tf_signal == signal_side:
                confirmations += 1
            else:
                log_mtf_disagreement(tf, tf_signal, signal_side)

        except Exception as e:
            logger.warning(f"MTF check failed for {tf}: {e}")

    # Decision logic
    if MTF_REQUIRED_CONFIRM == "all":
        return confirmations == total
    elif isinstance(MTF_REQUIRED_CONFIRM, int):
        return confirmations >= MTF_REQUIRED_CONFIRM
    else:
        # default to strict
        return confirmations == total
