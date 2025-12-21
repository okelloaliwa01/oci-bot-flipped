# src/breakout_logic.py
"""
Cleaned and hardened breakout detection logic with automatic column normalization.
- Ensures open/high/low/close/volume are present
- Renames columns from common variants (Open/o/etc.)
- Forces numeric conversion for all price/volume fields
- Prevents "Operator '*' not supported" errors
- Safer ATR (true range)
- Robust NaN handling
- Trend filter, fakeout detection, and optional retest behavior

Usage: import check_breakout from this module and pass a DataFrame slice of recent candles
"""
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np

# Config defaults (override by providing a config module in your project)
try:
    from config import (
        PENDING_MIN_BODY_RATIO,
        PENDING_MIN_VOL_MULT,
        PENDING_ATR_BUFFER_MULT,
        BREAKOUT_MIN_BODY_RATIO,
        BREAKOUT_RETEST_REQUIRED,
    )
except Exception:
    PENDING_MIN_BODY_RATIO = 0.4
    PENDING_MIN_VOL_MULT = 1.0
    PENDING_ATR_BUFFER_MULT = 0.5
    BREAKOUT_MIN_BODY_RATIO = 0.6
    BREAKOUT_RETEST_REQUIRED = True

# -------------------------
# Column Normalization
# -------------------------
STANDARD_COLS = {
    'Open': 'open', 'open': 'open', 'o': 'open',
    'High': 'high', 'high': 'high', 'h': 'high',
    'Low': 'low', 'low': 'low', 'l': 'low',
    'Close': 'close', 'close': 'close', 'c': 'close',
    'Volume': 'volume', 'volume': 'volume', 'v': 'volume'
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with normalized column names and numeric conversions.
    Missing numeric values are coerced to NaN.
    """
    df = df.copy()
    # Rename columns to standard names when possible
    rename_map = {k: v for k, v in STANDARD_COLS.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # If columns exist but with upper/lower variants not exact, try a case-insensitive mapping
    # (handles 'OPEN', 'High', etc.)
    for col in list(df.columns):
        col_lower = col.lower()
        if col_lower in STANDARD_COLS and col not in STANDARD_COLS.values():
            df = df.rename(columns={col: STANDARD_COLS[col_lower]})

    # Force numeric conversion on price/volume columns if present
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# -------------------------
# Helpers
# -------------------------

def ema(series: pd.Series, window: int) -> pd.Series:
    """Safe EMA wrapper returning a Series the same length as input."""
    return series.ewm(span=window, adjust=False, min_periods=1).mean()


def candle_body_ratio(open_price: float, close_price: float, high: float, low: float) -> float:
    """Return candle body size as ratio of full range (0..1)."""
    try:
        candle_range = float(high) - float(low)
        if candle_range <= 0 or not np.isfinite(candle_range):
            return 0.0
        return float(abs(float(close_price) - float(open_price))) / candle_range
    except Exception:
        return 0.0


def average_volume(prev: pd.DataFrame) -> float:
    """Returns average volume of previous candles (ignores NaNs)."""
    if prev is None or prev.empty or 'volume' not in prev:
        return 0.0
    return float(prev['volume'].dropna().mean() or 0.0)


def compute_atr_like(prev: pd.DataFrame, window: int = 14) -> float:
    """Compute ATR-like using True Range and simple rolling mean. Returns last ATR value or 0."""
    if prev is None or prev.empty:
        return 0.0
    high = prev['high']
    low = prev['low']
    close = prev['close']
    prev_close = close.shift(1)

    tr1 = (high - low)
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()
    if tr.empty:
        return 0.0
    if len(tr) >= window:
        return float(tr.rolling(window).mean().iloc[-1])
    return float(tr.mean())


def debug_log(msg: str):
    # Replace with logging in production
    print(f"[DEBUG BREAKOUT] {msg}")

# -------------------------
# Breakout Detection
# -------------------------

def check_breakout(
    df_last21: pd.DataFrame,
    volume_multiplier: float = 1.5,
    min_body_ratio: float = 0.6,
    atr_window: int = 14,
    atr_multiplier: float = 0.8,
    confirm_retest: bool = True,
) -> Tuple[Optional[str], Optional[float], Dict]:
    """Detect breakout signals from a slice of recent candles.

    Returns (signal, level, context) where signal is one of: "LONG", "SHORT",
    "PENDING_CONFIRM", or None. Level is the breakout level (resistance/support).
    """

    # Basic validation
    if df_last21 is None or not isinstance(df_last21, pd.DataFrame):
        return None, None, {}
    if df_last21.shape[0] < 5:
        return None, None, {}

    # Normalize columns and enforce numeric types
    df = normalize_dataframe(df_last21).reset_index(drop=True)

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return None, None, {"error": "missing_columns", "available": list(df.columns)}

    # Drop rows with NaN in essential prices (but keep original index mapping)
    if df[['open', 'high', 'low', 'close']].isnull().any(axis=1).any():
        # If too many NaNs, abort
        if df[['open', 'high', 'low', 'close']].isnull().sum().sum() > 5:
            return None, None, {"error": "too_many_nans"}
        
        if df[['open', 'high', 'low', 'close']].isnull().any(axis=1).any():
            # If too many NaNs, abort
            if df[['open', 'high', 'low', 'close']].isnull().sum().sum() > 5:
                return None, None, {"error": "too_many_nans"}
            df = df.ffill().bfill()


    breakout = df.iloc[-1]
    prev = df.iloc[:-1]

    try:
        close_price = float(breakout['close'])
        open_price = float(breakout['open'])
    except Exception:
        return None, None, {"error": "bad_prices"}

    # Levels
    resistance = float(prev['high'].max())
    support = float(prev['low'].min())

    # Volume confirmation
    avg_vol = average_volume(prev)
    vol_ok = False
    try:
        vol_ok = (avg_vol > 0) and (float(breakout['volume']) >= float(volume_multiplier * avg_vol))
    except Exception:
        vol_ok = False

    # Candle body strength
    body_ratio = candle_body_ratio(open_price, close_price, float(breakout['high']), float(breakout['low']))

    # ATR-like volatility environment filter
    atr = compute_atr_like(prev, atr_window)
    avg_range = float((prev['high'] - prev['low']).dropna().mean()) if not prev.empty else 0.0
    vol_env_ok = True if atr == 0 else (avg_range >= (atr_multiplier * atr))

    # Trend filter (EMA cross)
    ema_fast = float(ema(df['close'], 20).iloc[-1])
    ema_slow = float(ema(df['close'], 50).iloc[-1])
    trend_long = ema_fast > ema_slow
    trend_short = ema_fast < ema_slow

    # Breakout strength
    strong_long = (open_price > resistance * 0.995) and (close_price > resistance)
    strong_short = (open_price < support * 1.005) and (close_price < support)

    ctx: Dict = {
        "trend_long": trend_long,
        "trend_short": trend_short,
        "vol_ok": vol_ok,
        "body_ratio": body_ratio,
        "vol_env_ok": vol_env_ok,
        "resistance": resistance,
        "support": support,
        "atr": atr,
    }

    long_cond = strong_long and vol_ok and (body_ratio >= min_body_ratio) and vol_env_ok and trend_long
    short_cond = strong_short and vol_ok and (body_ratio >= min_body_ratio) and vol_env_ok and trend_short

    # Anti-fakeout guards
    atr_buffer = PENDING_ATR_BUFFER_MULT * atr if atr > 0 else 0.0

    fakeout_long = (
        (close_price - resistance) < atr_buffer or
        (body_ratio < PENDING_MIN_BODY_RATIO) or
        (float(breakout['volume']) < PENDING_MIN_VOL_MULT * avg_vol if avg_vol > 0 else True)
    )

    fakeout_short = (
        (support - close_price) < atr_buffer or
        (body_ratio < PENDING_MIN_BODY_RATIO) or
        (float(breakout['volume']) < PENDING_MIN_VOL_MULT * avg_vol if avg_vol > 0 else True)
    )

    # Retest / pending confirmation logic
    if confirm_retest or BREAKOUT_RETEST_REQUIRED:
        if (close_price > resistance) and vol_ok and (body_ratio >= BREAKOUT_MIN_BODY_RATIO) and vol_env_ok:
            ctx["type"] = "LONG"
            signal = "PENDING_CONFIRM" if fakeout_long else "LONG"
            ctx["fakeout_flag"] = bool(fakeout_long)
            ctx["reason"] = "Weak candle or low volume after breakout" if fakeout_long else "Strong breakout"

            debug_log(f"signal={signal} close={close_price} open={open_price} res={resistance} sup={support} "
                      f"ema_fast={ema_fast} ema_slow={ema_slow} trend_long={trend_long} vol_ok={vol_ok} "
                      f"body_ratio={body_ratio:.3f} atr={atr}")
            return signal, resistance, ctx

        if (close_price < support) and vol_ok and (body_ratio >= BREAKOUT_MIN_BODY_RATIO) and vol_env_ok:
            ctx["type"] = "SHORT"
            signal = "PENDING_CONFIRM" if fakeout_short else "SHORT"
            ctx["fakeout_flag"] = bool(fakeout_short)
            ctx["reason"] = "Weak candle or low volume after breakdown" if fakeout_short else "Strong breakdown"

            debug_log(f"signal={signal} close={close_price} open={open_price} res={resistance} sup={support} "
                      f"ema_fast={ema_fast} ema_slow={ema_slow} trend_short={trend_short} vol_ok={vol_ok} "
                      f"body_ratio={body_ratio:.3f} atr={atr}")
            return signal, support, ctx

    # Direct confirmation
    if long_cond and not fakeout_long:
        ctx["type"] = "LONG"
        return "LONG", resistance, ctx
    if short_cond and not fakeout_short:
        ctx["type"] = "SHORT"
        return "SHORT", support, ctx

    # No signal
    debug_log(f"signal=None close={close_price} open={open_price} res={resistance} sup={support} "
              f"ema_fast={ema_fast} ema_slow={ema_slow} trend_long={trend_long} trend_short={trend_short} "
              f"vol_ok={vol_ok} body_ratio={body_ratio:.3f} atr={atr}")
    return None, None, {}


# -------------------------
# Internal Test Harness
# -------------------------
if __name__ == "__main__":
    # Build synthetic dataset of n candles (numerics guaranteed)
    n = 21
    base = 100.0
    rng = np.random.RandomState(42)

    highs = base + np.cumsum(rng.normal(0, 0.2, n))
    lows = highs - (0.5 + rng.normal(0, 0.05, n))
    opens = lows + (highs - lows) * 0.2
    closes = opens + (highs - lows) * 0.6
    volumes = np.abs(rng.normal(100, 10, n))

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })

    # Normalize (ensures numeric types even if upstream returns strings)
    df = normalize_dataframe(df)

    # Force last candle to be a clear breakout above previous highs
    last_open = float(df['high'][:-1].max()) * 1.001
    last_close = float(df['high'][:-1].max()) * 1.01
    df.loc[n-1, 'open'] = last_open
    df.loc[n-1, 'close'] = last_close
    df.loc[n-1, 'high'] = last_close
    df.loc[n-1, 'low'] = last_open * 0.999
    df.loc[n-1, 'volume'] = float(df['volume'][:-1].mean()) * 2.0

    sig, lvl, ctx = check_breakout(df)
    print('RESULT:', sig, lvl)
    print('CTX:', ctx)
