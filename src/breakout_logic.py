import pandas as pd
import numpy as np
from config import (
    PENDING_MIN_BODY_RATIO,
    PENDING_MIN_VOL_MULT,
    PENDING_ATR_BUFFER_MULT,
    BREAKOUT_MIN_BODY_RATIO,
    BREAKOUT_RETEST_REQUIRED
)

# ----------------------------------------------------------------------
# Utility EMA calculator (fallback if ta-lib not installed)
# ----------------------------------------------------------------------
def ema(series: pd.Series, window: int):
    """Compute EMA using pandas' built-in ewm to avoid extra dependencies."""
    return series.ewm(span=window, adjust=False).mean()


# ----------------------------------------------------------------------
# Internal helper utilities (clean + testable)
# ----------------------------------------------------------------------
def candle_body_ratio(open_price: float, close_price: float, high: float, low: float) -> float:
    """Return candle body size as ratio of full range."""
    candle_range = high - low
    if candle_range <= 0:
        return 0.0
    return abs(close_price - open_price) / candle_range


def average_volume(prev: pd.DataFrame) -> float:
    """Returns average volume of previous candles."""
    return prev["volume"].mean() if "volume" in prev and not prev.empty else 0


def compute_atr_like(prev: pd.DataFrame, window: int) -> float:
    """Compute simplified ATR-like average true range."""
    high_low = prev['high'] - prev['low']
    return (
        high_low.rolling(window=window).mean().iloc[-1]
        if len(high_low) >= window else high_low.mean()
    )


def debug_log(msg: str):
    """Unified debug logging."""
    print(f"[DEBUG BREAKOUT] {msg}")


# ----------------------------------------------------------------------
# Breakout Detection Logic (with anti-fakeout integration)
# ----------------------------------------------------------------------
def check_breakout(
    df_last21: pd.DataFrame,
    volume_multiplier: float = 1.5,
    min_body_ratio: float = 0.6,
    atr_window: int = 14,
    atr_multiplier: float = 0.8,
    confirm_retest: bool = True
):
    """
    Robust breakout detection integrating dynamic anti-fakeout filters.

    Returns:
        (signal, level, context)
        signal ∈ {"LONG", "SHORT", "PENDING_CONFIRM", None}
    """

    # Validate input
    if df_last21 is None or df_last21.shape[0] < 21:
        return None, None, {}

    breakout = df_last21.iloc[-1]
    prev = df_last21.iloc[:-1].copy()

    close_price = breakout["close"]
    open_price = breakout["open"]

    # ------------------------------------------------------------------
    # Key levels
    # ------------------------------------------------------------------
    resistance = prev['high'].max()
    support = prev['low'].min()

    # ------------------------------------------------------------------
    # Volume confirmation
    # ------------------------------------------------------------------
    avg_vol = average_volume(prev)
    vol_ok = close_price >= 0 and breakout['volume'] >= volume_multiplier * avg_vol if avg_vol > 0 else False

    # ------------------------------------------------------------------
    # Candle body strength
    # ------------------------------------------------------------------
    body_ratio = candle_body_ratio(open_price, close_price, breakout['high'], breakout['low'])

    # ------------------------------------------------------------------
    # ATR-like volatility environment filter
    # ------------------------------------------------------------------
    atr = compute_atr_like(prev, atr_window)
    avg_range = (prev['high'] - prev['low']).mean()
    vol_env_ok = avg_range >= atr_multiplier * atr if atr > 0 else True

    # ------------------------------------------------------------------
    # Trend filter using EMA cross
    # ------------------------------------------------------------------
    ema_fast = ema(df_last21['close'], 20).iloc[-1]
    ema_slow = ema(df_last21['close'], 50).iloc[-1]
    trend_long = ema_fast > ema_slow
    trend_short = ema_fast < ema_slow

    # ------------------------------------------------------------------
    # Breakout strength filters
    # ------------------------------------------------------------------
    strong_long = (open_price > resistance * 0.995) and (close_price > resistance)
    strong_short = (open_price < support * 1.005) and (close_price < support)

    # Context container
    ctx = {
        "trend_long": trend_long,
        "trend_short": trend_short,
        "vol_ok": vol_ok,
        "body_ratio": body_ratio,
        "vol_env_ok": vol_env_ok,
        "resistance": resistance,
        "support": support,
        "atr": atr,
        "type": None
    }

    # ------------------------------------------------------------------
    # Final breakout conditions
    # ------------------------------------------------------------------
    long_cond = strong_long and vol_ok and (body_ratio >= min_body_ratio) and vol_env_ok and trend_long
    short_cond = strong_short and vol_ok and (body_ratio >= min_body_ratio) and vol_env_ok and trend_short

    # ------------------------------------------------------------------
    # Anti-Fakeout Guards
    # ------------------------------------------------------------------
    atr_buffer = PENDING_ATR_BUFFER_MULT * atr if atr > 0 else 0

    fakeout_long = (
        (close_price - resistance) < atr_buffer or
        body_ratio < PENDING_MIN_BODY_RATIO or
        breakout['volume'] < PENDING_MIN_VOL_MULT * avg_vol
    )

    fakeout_short = (
        (support - close_price) < atr_buffer or
        body_ratio < PENDING_MIN_BODY_RATIO or
        breakout['volume'] < PENDING_MIN_VOL_MULT * avg_vol
    )

    # ------------------------------------------------------------------
    # Retest or Pending Confirmation Logic
    # ------------------------------------------------------------------
    if confirm_retest or BREAKOUT_RETEST_REQUIRED:

        # LONG breakout
        if close_price > resistance and vol_ok and body_ratio >= BREAKOUT_MIN_BODY_RATIO and vol_env_ok:
            ctx["type"] = "LONG"

            signal = "PENDING_CONFIRM" if fakeout_long else "LONG"
            ctx["fakeout_flag"] = fakeout_long
            ctx["reason"] = "Weak candle or low volume after breakout" if fakeout_long else "Strong breakout"

            debug_log(
                f"signal={signal} | close={close_price:.2f}, open={open_price:.2f}, "
                f"res={resistance:.2f}, sup={support:.2f}, ema_fast={ema_fast:.2f}, ema_slow={ema_slow:.2f}, "
                f"trend_long={trend_long}, vol_ok={vol_ok}, body_ratio={body_ratio:.3f}"
            )
            return signal, resistance, ctx

        # SHORT breakout
        if close_price < support and vol_ok and body_ratio >= BREAKOUT_MIN_BODY_RATIO and vol_env_ok:
            ctx["type"] = "SHORT"

            signal = "PENDING_CONFIRM" if fakeout_short else "SHORT"
            ctx["fakeout_flag"] = fakeout_short
            ctx["reason"] = "Weak candle or low volume after breakdown" if fakeout_short else "Strong breakdown"

            debug_log(
                f"signal={signal} | close={close_price:.2f}, open={open_price:.2f}, "
                f"res={resistance:.2f}, sup={support:.2f}, ema_fast={ema_fast:.2f}, ema_slow={ema_slow:.2f}, "
                f"trend_short={trend_short}, vol_ok={vol_ok}, body_ratio={body_ratio:.3f}"
            )
            return signal, support, ctx

    # ------------------------------------------------------------------
    # Direct breakout confirmation (no retest needed)
    # ------------------------------------------------------------------
    if long_cond and not fakeout_long:
        ctx["type"] = "LONG"
        signal, level = "LONG", resistance

    elif short_cond and not fakeout_short:
        ctx["type"] = "SHORT"
        signal, level = "SHORT", support

    else:
        signal, level = None, None

    # Log final decision
    debug_log(
        f"signal={signal} | close={close_price:.2f}, open={open_price:.2f}, "
        f"res={resistance:.2f}, sup={support:.2f}, ema_fast={ema_fast:.2f}, ema_slow={ema_slow:.2f}, "
        f"trend_long={trend_long}, trend_short={trend_short}, vol_ok={vol_ok}, body_ratio={body_ratio:.3f}"
    )

    return (signal, level, ctx) if signal else (None, None, {})
