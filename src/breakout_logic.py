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
# ✅ Utility EMA calculator (fallback if ta-lib not installed)
# ----------------------------------------------------------------------
def ema(series: pd.Series, window: int):
    """Compute EMA using pandas built-in ewm to avoid extra dependencies."""
    return series.ewm(span=window, adjust=False).mean()


# ----------------------------------------------------------------------
# ✅ Breakout Detection Logic (with anti-fakeout integration)
# ----------------------------------------------------------------------
def check_breakout(df_last21: pd.DataFrame,
                   volume_multiplier: float = 1.5,
                   min_body_ratio: float = 0.6,
                   atr_window: int = 14,
                   atr_multiplier: float = 0.8,
                   confirm_retest: bool = True):
    """
    Robust breakout detection that integrates dynamic anti-fakeout filters.
    Inputs:
      df_last21: DataFrame of 21 candles (oldest -> newest). Columns required:
                 ['open', 'high', 'low', 'close', 'volume']
    Returns:
      (signal, level, context)
      signal ∈ {"LONG", "SHORT", "PENDING_CONFIRM", None}
      level: breakout level (resistance/support)
      context: dict with metadata (trend, vol_ok, body_ratio, atr, etc.)
    """

    if df_last21 is None or df_last21.shape[0] < 21:
        return None, None, {}

    breakout = df_last21.iloc[-1]
    prev = df_last21.iloc[:-1].copy()

    # ------------------------------------------------------------------
    # Key levels (support & resistance)
    # ------------------------------------------------------------------
    resistance = prev['high'].max()
    support = prev['low'].min()

    # ------------------------------------------------------------------
    # Volume confirmation
    # ------------------------------------------------------------------
    avg_vol = prev['volume'].mean()
    vol_ok = breakout['volume'] >= volume_multiplier * avg_vol if avg_vol > 0 else False

    # ------------------------------------------------------------------
    # Candle body strength
    # ------------------------------------------------------------------
    body = abs(breakout['close'] - breakout['open'])
    candle_range = breakout['high'] - breakout['low']
    body_ratio = (body / candle_range) if candle_range > 0 else 0

    # ------------------------------------------------------------------
    # ATR-like volatility filter
    # ------------------------------------------------------------------
    high_low = prev['high'] - prev['low']
    if len(high_low) >= atr_window:
        atr = high_low.rolling(window=atr_window).mean().iloc[-1]
    else:
        atr = high_low.mean()
    avg_range = high_low.mean()
    vol_env_ok = True
    if not np.isnan(atr) and atr > 0:
        vol_env_ok = avg_range >= atr_multiplier * atr

    # ------------------------------------------------------------------
    # Trend filter using EMA cross
    # ------------------------------------------------------------------
    ema_fast = ema(df_last21['close'], 20).iloc[-1]
    ema_slow = ema(df_last21['close'], 50).iloc[-1]
    trend_long = ema_fast > ema_slow
    trend_short = ema_fast < ema_slow

    close = breakout['close']
    open_ = breakout['open']

    # ------------------------------------------------------------------
    # Define breakout strength (reduce wick-based false breakouts)
    # ------------------------------------------------------------------
    strong_long = (open_ > resistance * 0.995) and (close > resistance)
    strong_short = (open_ < support * 1.005) and (close < support)

    # ------------------------------------------------------------------
    # Context container
    # ------------------------------------------------------------------
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
    # Long / Short breakout conditions
    # ------------------------------------------------------------------
    long_cond = strong_long and vol_ok and (body_ratio >= min_body_ratio) and vol_env_ok and trend_long
    short_cond = strong_short and vol_ok and (body_ratio >= min_body_ratio) and vol_env_ok and trend_short

    # ------------------------------------------------------------------
    # 🧠 Anti-Fakeout Dynamic Filters
    # ------------------------------------------------------------------
    # Apply stricter thresholds for pending signals or when volatility is narrow.
    atr_buffer = PENDING_ATR_BUFFER_MULT * atr if atr > 0 else 0
    fakeout_guard_long = (
        (close - resistance) < atr_buffer or
        body_ratio < PENDING_MIN_BODY_RATIO or
        breakout['volume'] < PENDING_MIN_VOL_MULT * avg_vol
    )
    fakeout_guard_short = (
        (support - close) < atr_buffer or
        body_ratio < PENDING_MIN_BODY_RATIO or
        breakout['volume'] < PENDING_MIN_VOL_MULT * avg_vol
    )

    # ------------------------------------------------------------------
    # Retest / Pending confirmation logic
    # ------------------------------------------------------------------
    if confirm_retest or BREAKOUT_RETEST_REQUIRED:
        # LONG pending confirm
        if close > resistance and vol_ok and body_ratio >= BREAKOUT_MIN_BODY_RATIO and vol_env_ok:
            ctx["type"] = "LONG"
            # If fails anti-fakeout guard => pending confirm
            if fakeout_guard_long:
                ctx["fakeout_flag"] = True
                ctx["reason"] = "Weak candle or low volume after breakout"
                signal, level = "PENDING_CONFIRM", resistance
            else:
                signal, level = "LONG", resistance

            # 🧾 Debug trace
            print(
                f"[DEBUG BREAKOUT] signal={signal} | close={close:.2f}, open={open_:.2f}, "
                f"res={resistance:.2f}, sup={support:.2f}, ema_fast={ema_fast:.2f}, ema_slow={ema_slow:.2f}, "
                f"trend_long={trend_long}, trend_short={trend_short}, vol_ok={vol_ok}, body_ratio={body_ratio:.3f}"
            )
            return signal, level, ctx

        # SHORT pending confirm
        if close < support and vol_ok and body_ratio >= BREAKOUT_MIN_BODY_RATIO and vol_env_ok:
            ctx["type"] = "SHORT"
            if fakeout_guard_short:
                ctx["fakeout_flag"] = True
                ctx["reason"] = "Weak candle or low volume after breakdown"
                signal, level = "PENDING_CONFIRM", support
            else:
                signal, level = "SHORT", support

            # 🧾 Debug trace
            print(
                f"[DEBUG BREAKOUT] signal={signal} | close={close:.2f}, open={open_:.2f}, "
                f"res={resistance:.2f}, sup={support:.2f}, ema_fast={ema_fast:.2f}, ema_slow={ema_slow:.2f}, "
                f"trend_long={trend_long}, trend_short={trend_short}, vol_ok={vol_ok}, body_ratio={body_ratio:.3f}"
            )
            return signal, level, ctx

    # ------------------------------------------------------------------
    # Direct breakout confirmation (no retest required)
    # ------------------------------------------------------------------
    if long_cond and not fakeout_guard_long:
        ctx["type"] = "LONG"
        signal, level = "LONG", resistance
    elif short_cond and not fakeout_guard_short:
        ctx["type"] = "SHORT"
        signal, level = "SHORT", support
    else:
        signal, level = None, None

    # 🧾 Debug trace (always log decision)
    print(
        f"[DEBUG BREAKOUT] signal={signal} | close={close:.2f}, open={open_:.2f}, "
        f"res={resistance:.2f}, sup={support:.2f}, ema_fast={ema_fast:.2f}, ema_slow={ema_slow:.2f}, "
        f"trend_long={trend_long}, trend_short={trend_short}, vol_ok={vol_ok}, body_ratio={body_ratio:.3f}"
    )

    # ------------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------------
    if signal:
        return signal, level, ctx
    return None, None, {}
