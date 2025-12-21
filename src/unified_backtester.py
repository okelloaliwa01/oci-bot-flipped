"""
src/unified_backtester.py

Unified Backtester (batch + online) with ATR SmartExit simulation and plotting.

This file is the patched, full version that:
 - contains ExitSimulator and ExecutionSimulator
 - runs batch/online backtests
 - automatically calls plotting helpers from src.backtest_plots (Matplotlib)
   after a run when enabled in cfg_map (SAVE_PLOTS or AUTO_PLOT)

This variant adds strength-aware behavior:
 - signals with strength_score < MIN_SIGNAL_STRENGTH are filtered out
 - ExitSimulator.plan_exit accepts a `strength` argument and tightens SL/widens TP
   based on configurable factors (STRENGTH_SL_TIGHTEN_FACTOR, STRENGTH_TP_WIDEN_FACTOR)
 - position sizing (_size_from_risk) multiplies qty by a strength-based factor
   (between STRENGTH_SIZE_MIN and 1.0)

Plotting behavior is controlled via cfg_map keys:
 - SAVE_PLOTS (bool-like): if True, save PNGs to disk (default: False)
 - AUTO_PLOT (bool-like): if True, attempt to display/show plots (default: False)
 - PLOTS_DIR (str): directory to save plots (default: "./plots")

The plotting helpers are imported from src.backtest_plots. If that module
is missing or the functions are not present, plotting is skipped gracefully.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
import logging
import math
import os
import pandas as pd
import numpy as np
from datetime import datetime

# try to import local modules; fallback to root imports for tests
try:
    from src.config_utils import load_config
    from src.signal_engine import SignalEngine, Signal
    from src.breakout_scanner import find_breakouts, tag_dataframe_signals
    from src.breakout_logic import BreakoutConfig, DEFAULT_CONFIG, check_breakout
    # optionally use live SmartExitManager (requires a client)
    from src.smart_exit import SmartExitManager  # type: ignore
    # plotting helpers
    from src.backtest_plots import plot_equity_curve, plot_drawdowns, plot_trade_prices
except Exception:
    try:
        from config_utils import load_config  # type: ignore
        from signal_engine import SignalEngine, Signal  # type: ignore
        from breakout_scanner import find_breakouts, tag_dataframe_signals  # type: ignore
        from breakout_logic import BreakoutConfig, DEFAULT_CONFIG, check_breakout  # type: ignore
    except Exception:
        # allow tests that don't need those modules
        load_config = lambda: {}
        SignalEngine = None  # type: ignore
        Signal = None  # type: ignore
        find_breakouts = None
        tag_dataframe_signals = None
        BreakoutConfig = None
        DEFAULT_CONFIG = None
        check_breakout = None
    try:
        from smart_exit import SmartExitManager  # type: ignore
    except Exception:
        SmartExitManager = None  # type: ignore
    # plotting helpers fallback
    try:
        from backtest_plots import plot_equity_curve, plot_drawdowns, plot_trade_prices
    except Exception:
        plot_equity_curve = None  # type: ignore
        plot_drawdowns = None  # type: ignore
        plot_trade_prices = None  # type: ignore

logger = logging.getLogger("unified_backtester")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -------------------------
# Data containers
# -------------------------
@dataclass
class TradeRecord:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    net_pnl: float
    reason: str
    details: Dict[str, Any]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    returns: pd.Series
    initial_balance: float
    final_balance: float
    metrics: Dict[str, Any]

    def save_plots(self, df: Optional[pd.DataFrame] = None, plots_dir: Optional[str] = None, show: bool = False) -> None:
        """Save or display plots using the plotting helpers if available.

        df (optional): original OHLCV dataframe used for the backtest (helps plot trades on price chart).
        plots_dir: directory to save plots; if None uses './plots'.
        show: if True, call plotting functions with interactive display (if supported).
        """
        if plots_dir is None:
            plots_dir = "./plots"
        try:
            os.makedirs(plots_dir, exist_ok=True)
        except Exception:
            logger.debug("Could not create plots dir %s", plots_dir)
            plots_dir = "."

        # equity
        try:
            if plot_equity_curve is not None:
                fpath = os.path.join(plots_dir, "equity_curve.png")
                plot_equity_curve(self.equity_curve, save_path=fpath, show=show)
                logger.info("Saved equity curve to %s", fpath)
        except Exception as e:
            logger.debug("Failed to plot equity curve: %s", e)

        # drawdowns
        try:
            if plot_drawdowns is not None:
                fpath = os.path.join(plots_dir, "drawdowns.png")
                plot_drawdowns(self.equity_curve, save_path=fpath, show=show)
                logger.info("Saved drawdowns to %s", fpath)
        except Exception as e:
            logger.debug("Failed to plot drawdowns: %s", e)

        # price chart with trades (if df provided)
        try:
            if df is not None and plot_trade_prices is not None and not self.trades.empty:
                fpath = os.path.join(plots_dir, "trade_price_chart.png")
                plot_trade_prices(df.copy(), self.trades.copy(), save_path=fpath, show=show)
                logger.info("Saved trade price chart to %s", fpath)
        except Exception as e:
            logger.debug("Failed to plot trade price chart: %s", e)


# -------------------------
# Internal Exit Simulator
# -------------------------
class ExitSimulator:
    """
    Simulates ATR-based exits offline:
    - ATR SL (atr_mult_sl)
    - Partial TPs (atr_mult_tp + partial sizes)
    - Trailing start/step, breakeven
    - Slippage and fee model

    Strength-aware adjustments:
    - plan_exit accepts `strength` in [0..1] and will tighten SL and widen TP's
      according to cfg_map factors STRENGTH_SL_TIGHTEN_FACTOR and STRENGTH_TP_WIDEN_FACTOR.
    """

    def __init__(self, cfg_map: Dict[str, Any]):
        # read params with safe defaults
        def parse_list(val, default):
            if val is None:
                return default
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, str):
                parts = [p.strip() for p in val.split(",") if p.strip()]
                try:
                    return [float(x) for x in parts]
                except Exception:
                    return default
            return default

        self.cfg_map = cfg_map or {}
        self.atr_period = int(cfg_map.get("ATR_PERIOD", cfg_map.get("ATR_PERIOD", 14)))
        self.atr_mult_tp = parse_list(cfg_map.get("ATR_MULT_TP", cfg_map.get("ATR_MULT_TP", "2.0,3.0")), [2.0, 3.0])
        self.atr_mult_sl = float(cfg_map.get("ATR_MULT_SL", cfg_map.get("ATR_MULT_SL", 1.9)))
        self.tp_partial_sizes = parse_list(cfg_map.get("TP_PARTIAL_SIZES", cfg_map.get("TP_PARTIAL_SIZES", "0.5,0.5")), [0.5, 0.5])

        # normalize partial sizes to match TP count
        if len(self.tp_partial_sizes) != len(self.atr_mult_tp):
            n = len(self.atr_mult_tp) if len(self.atr_mult_tp) > 0 else 1
            base = 1.0 / n
            self.tp_partial_sizes = [base] * n

        self.trailing_start_atr = float(cfg_map.get("TRAILING_START_ATR", cfg_map.get("TRAILING_START_ATR", 1.5)))
        self.trailing_step_atr = float(cfg_map.get("TRAILING_STEP_ATR", cfg_map.get("TRAILING_STEP_ATR", 0.25)))
        self.breakeven_atr = float(cfg_map.get("BREAKEVEN_ATR", cfg_map.get("BREAKEVEN_ATR", 1.0)))
        self.breakeven_buffer_pts = float(cfg_map.get("BREAKEVEN_BUFFER_PTS", cfg_map.get("BREAKEVEN_BUFFER_PTS", 0.03)))
        # slippage/fee
        self.slippage_pct = float(cfg_map.get("SLIPPAGE_PCT", cfg_map.get("SLIPPAGE_PCT", 0.0)))
        self.fee_pct = float(cfg_map.get("TRADE_FEE_PCT", cfg_map.get("TRADE_FEE_PCT", 0.0004)))

        # strength-related config
        # fraction by which SL distance is reduced for strength=1 (e.g. 0.2 => 20% tighter)
        self.strength_sl_tighten = float(cfg_map.get("STRENGTH_SL_TIGHTEN_FACTOR", 0.2))
        # fraction by which TP multipliers are increased for strength=1 (e.g. 0.3 => 30% wider TP)
        self.strength_tp_widen = float(cfg_map.get("STRENGTH_TP_WIDEN_FACTOR", 0.3))

    @staticmethod
    def compute_atr_series(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def plan_exit(self, side: str, entry_price: float, atr_value: Optional[float], strength: float = 0.0) -> Dict[str, Any]:
        """
        Return initial plan: stop_loss, tp_levels (list), partial_qty_fractions (list)
        (partial fractions sum to 1)

        Strength modifies SL/TP:
        - SL distance = base_sl_distance * (1 - strength * strength_sl_tighten)
        - TP multipliers = base_tp_mult * (1 + strength * strength_tp_widen)
        """
        strength = max(0.0, min(1.0, float(strength or 0.0)))

        if atr_value is None or atr_value <= 0 or not math.isfinite(atr_value):
            # fallback to percent-based defaults
            sl = entry_price * (0.99 if side == "LONG" else 1.01)
            tp_levels = []
        else:
            base_sl_mult = float(self.atr_mult_sl)
            # tighten SL distance for strong signals
            adj_sl_mult = base_sl_mult * max(0.0, (1.0 - strength * self.strength_sl_tighten))
            sl = (entry_price - adj_sl_mult * atr_value) if side == "LONG" else (entry_price + adj_sl_mult * atr_value)

            # adjust TP multipliers
            adj_tp_mults = [m * (1.0 + strength * self.strength_tp_widen) for m in self.atr_mult_tp]
            tp_levels = [
                (entry_price + mult * atr_value) if side == "LONG" else (entry_price - mult * atr_value)
                for mult in adj_tp_mults
            ]

        return {"stop_loss": float(sl), "tp_levels": [float(x) for x in tp_levels], "tp_fracs": self.tp_partial_sizes.copy()}

    def apply_slippage(self, price: float, side: str, is_entry: bool) -> float:
        """Simulate taker slippage on entry/exit; fee is accounted separately."""
        if self.slippage_pct and self.slippage_pct != 0:
            if side == "LONG":
                return price * (1.0 + self.slippage_pct) if is_entry else price * (1.0 - self.slippage_pct)
            else:
                return price * (1.0 - self.slippage_pct) if is_entry else price * (1.0 + self.slippage_pct)
        return price


class ExecutionSimulator:
    """
    Candle-based execution/fill model.

    Config options (via cfg_map):
      - SPREAD_PCT: default explicit spread percentage (0.001 -> 0.1%)
      - SPREAD_ATR_MULT: if SPREAD_PCT not set, spread = ATR * SPREAD_ATR_MULT
      - DEPTH_PCT: fraction of candle volume considered available at the best side (0..1)
      - IMPACT_LINEAR: linear impact coefficient (0..1)
      - MIN_LIQUIDITY: min available volume per candle (absolute)
    """

    def __init__(self, cfg_map: Dict[str, Any]):
        def _getf(k, default):
            try:
                return float(cfg_map.get(k, cfg_map.get(k.lower(), default)))
            except Exception:
                return default

        self.spread_pct = _getf("SPREAD_PCT", None)  # prefer explicit
        self.spread_atr_mult = _getf("SPREAD_ATR_MULT", 0.2)
        self.depth_pct = _getf("DEPTH_PCT", 0.2)
        self.impact_linear = _getf("IMPACT_LINEAR", 0.5)
        self.min_liquidity = _getf("MIN_LIQUIDITY", 0.0)
        # fallback slippage floor
        self.min_slippage = _getf("MIN_SLIPPAGE_PCT", 0.0)

    def estimate_spread(self, atr: Optional[float], price: float) -> float:
        """Return absolute spread amount (price units)."""
        if self.spread_pct is not None:
            return price * self.spread_pct
        if atr and atr > 0:
            return atr * self.spread_atr_mult
        # fallback to tiny spread
        return max(price * 0.0001, 0.0000001)

    def available_at_side(self, candle_volume: float) -> float:
        """Estimate available volume at best bid/ask on that candle."""
        if candle_volume is None or math.isnan(candle_volume):
            return max(self.min_liquidity, 0.0)
        return max(self.min_liquidity, float(candle_volume) * self.depth_pct)

    def simulate_market_fill(
        self,
        side: str,
        desired_qty: float,
        price: float,
        candle: Dict[str, Any],
        atr: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Simulate a market fill on a candle.
        Returns: (filled_qty, avg_fill_price, realized_slippage_pct)
        """
        # compute spread absolute
        spread = self.estimate_spread(atr, price)
        half_spread = spread / 2.0

        # available liquidity at best price
        vol = float(candle.get("volume", 0.0) or 0.0)
        avail = self.available_at_side(vol)

        # fillable qty is min(desired, avail). If desired > avail, partial fill occurs.
        filled_qty = min(desired_qty, avail)

        if filled_qty <= 0:
            # no liquidity -> return zero fill
            return 0.0, price, 0.0

        # market impact: price moves linearly with fraction of avail consumed
        frac = min(1.0, filled_qty / max(avail, 1e-12))
        impact = frac * self.impact_linear  # scaled 0..impact_linear
        # slippage direction: fills for buyer see price = price + half_spread + impact*spread
        if side.upper() == "LONG":
            fill_price = price + half_spread + impact * spread
        else:
            fill_price = price - half_spread - impact * spread

        # ensure minimum slippage
        realized_slippage_pct = (abs(fill_price - price) / max(price, 1e-12))
        if realized_slippage_pct < self.min_slippage:
            # adjust fill price to reflect min_slippage
            if side.upper() == "LONG":
                fill_price = price * (1.0 + self.min_slippage)
            else:
                fill_price = price * (1.0 - self.min_slippage)
            realized_slippage_pct = self.min_slippage

        return float(filled_qty), float(fill_price), float(realized_slippage_pct)


# -------------------------
# Unified Backtester
# -------------------------
class UnifiedBacktester:

    @staticmethod
    def _safe_float(v):
        """
        Convert ATR values safely:
        - Accepts scalar, numpy types, 0-d arrays
        - If Series: extract the first element
        - If None or NaN: return None
        """
        if v is None:
            return None
        # if Series -> use first valid element
        if isinstance(v, pd.Series):
            if len(v) == 0:
                return None
            v = v.iloc[0]
        try:
            f = float(v)
            if math.isnan(f) or not math.isfinite(f):
                return None
            return f
        except Exception:
            return None


    def __init__(self, cfg_map: Optional[Dict[str, Any]] = None, signal_engine: Optional[SignalEngine] = None, exit_client: Optional[Any] = None):
        """
        cfg_map: overrides and parameters (used by ExitSimulator and sizing)
        signal_engine: optional SignalEngine instance (if None, created)
        exit_client: optional live SmartExitManager client (if provided, it will be used for exit order creation simulation)
        """
        self.cfg_map = cfg_map or {}
        try:
            loader = load_config()
            # apply loader defaults if not present in provided map
            merged = dict(loader)
            merged.update(self.cfg_map)
            self.cfg_map = merged
        except Exception:
            pass

        self.signal_engine = signal_engine or (SignalEngine(cfg_map=self.cfg_map) if SignalEngine is not None else None)
        self.exit_client = exit_client  # if provided, expected to be SmartExitManager-like
        # create exit simulator using cfg_map
        self.exit_sim = ExitSimulator(self.cfg_map)
        # create execution simulator
        self.exec_sim = ExecutionSimulator(self.cfg_map)

        self.initial_balance = float(self.cfg_map.get("ACCOUNT_BALANCE", self.cfg_map.get("INITIAL_BALANCE", 11.0)))
        self.max_trade_pct = float(self.cfg_map.get("MAX_TRADE_PCT", 0.03 if self.cfg_map.get("MAX_TRADE_PCT") is None else float(self.cfg_map.get("MAX_TRADE_PCT"))))
        # allow either 0.03 or "3%" style. If >1 assume percent value
        try:
            if self.max_trade_pct > 1.0:
                self.max_trade_pct = float(self.max_trade_pct) / 100.0
        except Exception:
            pass
        self.dry_run = bool(str(self.cfg_map.get("DRY_RUN", False)).lower() in ("1", "true", "yes"))

        # strength-related runtime defaults
        self.min_signal_strength = float(self.cfg_map.get("MIN_SIGNAL_STRENGTH", 0.35))
        # size scaling min (when strength=0 -> multiplier = size_min; when strength=1 -> 1.0)
        self.strength_size_min = float(self.cfg_map.get("STRENGTH_SIZE_MIN", 0.5))

    # -------------------------
    # Position sizing helper
    # -------------------------
    def _size_from_risk(self, entry_price: float, stop_price: float, equity: float, strength: float = 0.0) -> float:
        """Risk-based sizing: risk = equity * max_trade_pct; returns qty in asset units.

        strength modifies size: qty *= size_multiplier where
        size_multiplier = strength_size_min + (1 - strength_size_min) * strength
        """
        if stop_price is None or entry_price is None:
            return 0.0
        per_unit_loss = abs(entry_price - stop_price)
        if per_unit_loss <= 0 or not math.isfinite(per_unit_loss):
            return 0.0
        notional_risk = equity * self.max_trade_pct
        qty = notional_risk / per_unit_loss
        # safety: also cap based on notional / price
        qty_cap = (equity * self.max_trade_pct) / max(entry_price, 1e-9)
        qty = min(qty, qty_cap)

        # strength multiplier
        strength = max(0.0, min(1.0, float(strength or 0.0)))
        size_min = max(0.0, min(1.0, float(self.strength_size_min)))
        size_multiplier = size_min + (1.0 - size_min) * strength
        qty = qty * size_multiplier
        return max(0.0, float(qty))

    # -------------------------
    # Helpers for plotting
    # -------------------------
    def _plot_result_if_requested(self, result: BacktestResult, df: Optional[pd.DataFrame]) -> None:
        """Check cfg_map and call plotting helpers / result.save_plots accordingly."""
        save_plots = str(self.cfg_map.get("SAVE_PLOTS", self.cfg_map.get("AUTO_SAVE_PLOTS", False))).lower() in ("1", "true", "yes")
        auto_plot = str(self.cfg_map.get("AUTO_PLOT", False)).lower() in ("1", "true", "yes")
        plots_dir = str(self.cfg_map.get("PLOTS_DIR", "./plots"))
        if save_plots or auto_plot:
            try:
                result.save_plots(df=df, plots_dir=plots_dir, show=auto_plot)
            except Exception as e:
                logger.debug("Plotting failed: %s", e)

    # -------------------------
    # Run batch backtest from OHLCV DataFrame
    # -------------------------
    def run_batch(self, df: pd.DataFrame, *, debug: bool = False) -> BacktestResult:
        """
        Perform batch backtest:
          - generate signals with self.signal_engine.batch_run()
          - sequentially simulate entries & exits (TP partials, SL, trailing, breakeven)
        """
        if df is None or df.empty:
            raise ValueError("df must be provided")

        df = df.copy()
        # ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]) 
                df = df.set_index("timestamp")
            else:
                raise ValueError("df must have datetime index or timestamp column")

        # compute ATR series once
        df["_atr"] = ExitSimulator.compute_atr_series(df, self.exit_sim.atr_period)

        # generate signals
        signals = self.signal_engine.batch_run(df, debug=debug) if self.signal_engine is not None else pd.DataFrame()
        if signals is None or signals.empty:
            logger.info("No signals to backtest.")
            empty_result = BacktestResult(pd.DataFrame(), pd.Series([], dtype=float), pd.Series([], dtype=float), self.initial_balance, self.initial_balance, {})
            # attempt plotting (no-op)
            self._plot_result_if_requested(empty_result, df)
            return empty_result

        # runtime state
        equity = float(self.initial_balance)
        equity_snap_ts: List[pd.Timestamp] = []
        equity_snap_vals: List[float] = []
        trades: List[TradeRecord] = []

        # iterate signals in chronological order and simulate a trade lifecycle per signal
        for _, srow in signals.sort_values("timestamp").iterrows():
            ts = pd.to_datetime(srow["timestamp"]) if "timestamp" in srow else pd.to_datetime(srow.name)
            # find candle index (nearest)
            if ts not in df.index:
                idx_pos = df.index.get_indexer([ts], method="nearest")[0]
                ts = df.index[idx_pos]
            
            row = df.loc[ts]

            # ATR at entry candle
            atr_raw = row["_atr"] if "_atr" in row and not pd.isna(row["_atr"]) else None
            atr_val = float(atr_raw) if atr_raw is not None else None


            
            # strength filtering
            strength = float(srow.get("strength_score", srow.get("strength", 0.0) or 0.0))
            if strength < float(self.cfg_map.get("MIN_SIGNAL_STRENGTH", self.min_signal_strength)):
                if debug:
                    logger.debug("Skipping signal at %s due to strength %.3f < min %.3f", ts, strength, self.min_signal_strength)
                continue

            if not srow.get("long_entry", False) and not srow.get("short_entry", False):
                continue

            side = "LONG" if srow.get("long_entry", False) else "SHORT"
            entry_price = float(row["close"])

            # Determine exit plan (stop & TP levels) — pass strength into plan_exit
            try:
                if self.exit_client and SmartExitManager is not None:
                    plan = self.exit_client.create_exit_orders(symbol=self.cfg_map.get("SYMBOL", ""), side=side, entry_price=entry_price, qty=0, atr_value=atr_val, strength=strength)
                    sl = plan.get("sl") or plan.get("stop_loss") or plan.get("stop")
                    tp_levels = plan.get("tp_levels") or plan.get("tp_levels_calc") or []
                    tp_fracs = list(self.exit_sim.tp_partial_sizes)
                    stop_price = float(sl) if sl else None
                else:
                    plan = self.exit_sim.plan_exit(side, entry_price, atr_val, strength=strength)
                    stop_price = plan["stop_loss"]
                    tp_levels = plan["tp_levels"]
                    tp_fracs = plan["tp_fracs"]
            except Exception:
                plan = self.exit_sim.plan_exit(side, entry_price, atr_val, strength=strength)
                stop_price = plan["stop_loss"]
                tp_levels = plan["tp_levels"]
                tp_fracs = plan["tp_fracs"]

            # compute qty based on stop risk and strength
            qty = self._size_from_risk(entry_price, stop_price, equity, strength=strength)
            if qty <= 0 or not math.isfinite(qty):
                if debug:
                    logger.debug("Skipping trade at %s due to qty=0", ts)
                continue

            # choose reference price for entry: use row["close"]
            ref_price = float(row.get("close", entry_price))
            filled_qty, entry_exec_price, _ = self.exec_sim.simulate_market_fill(side, qty, ref_price, row, atr=atr_val)
            if filled_qty <= 0:
                if debug:
                    logger.debug("Entry not filled due to liquidity at %s", ts)
                continue
            # if partially filled, adjust qty to filled_qty
            qty = filled_qty
            entry_fee_total = entry_exec_price * qty * self.exit_sim.fee_pct

            # build active trade state
            active = {
                "side": side,
                "entry_ts": ts,
                "entry_price": entry_exec_price,
                "original_qty": qty,
                "qty_remaining": qty,
                "stop": float(stop_price) if stop_price is not None else None,
                "tp_levels": [float(x) for x in tp_levels],
                "tp_fracs": [float(x) for x in tp_fracs],
                "realized": 0.0,
                "mfe": 0.0,
                "mae": 0.0,
                "entry_fee_total": entry_fee_total,
            }

            # simulate forward until exit or until a max holding (use PENDING_EXPIRY_CANDLES)
            max_holding = int(self.cfg_map.get("PENDING_EXPIRY_CANDLES", 10))
            df_idx = df.index.get_loc(ts)
            exit_found = False

            for j in range(df_idx + 1, min(len(df) - 1, df_idx + max_holding) + 1):

                # Always exists because j is guaranteed inside bounds
                later = df.iloc[j]

                high = float(later["high"])
                low = float(later["low"])
                close = float(later["close"])

                # Always safe — no `.get()` and no undefined variable
                atr_raw = later["_atr"] if "_atr" in later and not pd.isna(later["_atr"]) else None
                atr_j = float(atr_raw) if atr_raw is not None else atr_val


                # update MFE/MAE
                if active["side"] == "LONG":
                    active["mfe"] = max(active["mfe"], high - active["entry_price"])
                    active["mae"] = min(active["mae"], low - active["entry_price"])
                else:
                    active["mfe"] = max(active["mfe"], active["entry_price"] - low)
                    active["mae"] = min(active["mae"], active["entry_price"] - high)

                # STOP LOSS hit?
                if active["stop"] is not None:
                    if active["side"] == "LONG" and low <= active["stop"]:
                        # simulate fill at the stop price using candle context
                        filled_qty_exit, exit_price, _ = self.exec_sim.simulate_market_fill(active["side"], active["qty_remaining"], active["stop"], later, atr=atr_j)
                        if filled_qty_exit <= 0:
                            # cannot exit at that candle due to liquidity: skip
                            continue
                        qty_closed = filled_qty_exit
                        pnl = (exit_price - active["entry_price"]) * qty_closed
                        fee_exit = exit_price * qty_closed * self.exit_sim.fee_pct
                        # allocate entry fee proportionally to closed qty
                        entry_fee_alloc = active["entry_fee_total"] * (qty_closed / active["original_qty"]) if active["original_qty"] > 0 else 0.0
                        net = pnl - fee_exit - entry_fee_alloc
                        trades.append(TradeRecord(entry_ts=active["entry_ts"], exit_ts=df.index[j], side=active["side"], entry_price=active["entry_price"], exit_price=exit_price, qty=qty_closed, pnl=pnl, net_pnl=net, reason="SL", details={"mfe": active["mfe"], "mae": active["mae"]}))
                        equity += net
                        exit_found = True
                        break
                    if active["side"] == "SHORT" and high >= active["stop"]:
                        filled_qty_exit, exit_price, _ = self.exec_sim.simulate_market_fill(active["side"], active["qty_remaining"], active["stop"], later, atr=atr_j)
                        if filled_qty_exit <= 0:
                            continue
                        qty_closed = filled_qty_exit
                        pnl = (active["entry_price"] - exit_price) * qty_closed
                        fee_exit = exit_price * qty_closed * self.exit_sim.fee_pct
                        entry_fee_alloc = active["entry_fee_total"] * (qty_closed / active["original_qty"]) if active["original_qty"] > 0 else 0.0
                        net = pnl - fee_exit - entry_fee_alloc
                        trades.append(TradeRecord(entry_ts=active["entry_ts"], exit_ts=df.index[j], side=active["side"], entry_price=active["entry_price"], exit_price=exit_price, qty=qty_closed, pnl=pnl, net_pnl=net, reason="SL", details={"mfe": active["mfe"], "mae": active["mae"]}))
                        equity += net
                        exit_found = True
                        break

                # PARTIAL TP hits - SAFE VERSION
                to_pop = []

                for k, tp in enumerate(active["tp_levels"]):
                    hit = ((active["side"] == "LONG" and high >= tp) or
                        (active["side"] == "SHORT" and low <= tp))
                    if not hit:
                        continue

                    frac = active["tp_fracs"][k] if k < len(active["tp_fracs"]) else 1.0 / len(active["tp_levels"])
                    qty_close = active["qty_remaining"] * frac

                    filled_qty_exit, exit_price, _ = self.exec_sim.simulate_market_fill(
                        active["side"], qty_close, tp, later, atr=atr_j
                    )
                    if filled_qty_exit <= 0:
                        continue

                    qty_closed = filled_qty_exit
                    pnl = (exit_price - active["entry_price"]) * qty_closed if active["side"] == "LONG" \
                        else (active["entry_price"] - exit_price) * qty_closed

                    fee_exit = exit_price * qty_closed * self.exit_sim.fee_pct
                    entry_fee_alloc = active["entry_fee_total"] * (qty_closed / active["original_qty"]) \
                                    if active["original_qty"] > 0 else 0.0

                    net = pnl - fee_exit - entry_fee_alloc
                    active["realized"] += pnl
                    active["qty_remaining"] -= qty_closed

                    trades.append(TradeRecord(
                        entry_ts=active["entry_ts"],
                        exit_ts=df.index[j],
                        side=active["side"],
                        entry_price=active["entry_price"],
                        exit_price=exit_price,
                        qty=qty_closed,
                        pnl=pnl,
                        net_pnl=net,
                        reason=f"TP_partial_{k+1}",
                        details={"tp_level": tp, "mfe": active["mfe"], "mae": active["mae"]}
                    ))

                    equity += net

                    to_pop.append(k)

                # Remove TPs in reverse index order
                for idx in reversed(to_pop):
                    active["tp_levels"].pop(idx)
                    active["tp_fracs"].pop(idx)

                # If no quantity remains, exit entire trade
                if active["qty_remaining"] <= 1e-9:
                    exit_found = True
                    break

                # trailing / breakeven updates
                if atr_j and atr_j > 0:
                    if active["side"] == "LONG":
                        cur_profit = max(0.0, high - active["entry_price"]) 
                    else:
                        cur_profit = max(0.0, active["entry_price"] - low)

                    # trailing
                    if cur_profit >= self.exit_sim.trailing_start_atr * atr_j:
                        if active["side"] == "LONG":
                            proposed = high - self.exit_sim.trailing_step_atr * atr_j
                            if proposed > active["stop"]:
                                active["stop"] = proposed
                        else:
                            proposed = low + self.exit_sim.trailing_step_atr * atr_j
                            if proposed < active["stop"]:
                                active["stop"] = proposed

                    # breakeven
                    if cur_profit >= self.exit_sim.breakeven_atr * atr_j:
                        if active["side"] == "LONG":
                            proposed = active["entry_price"] + self.exit_sim.breakeven_buffer_pts * atr_j
                            if proposed > active["stop"]:
                                active["stop"] = proposed
                        else:
                            proposed = active["entry_price"] - self.exit_sim.breakeven_buffer_pts * atr_j
                            if proposed < active["stop"]:
                                active["stop"] = proposed

            # If no exit found within holding window -> close at last close (timeout)
            if not exit_found:
                last_idx = min(len(df) - 1, df_idx + max_holding)
                last_row = df.iloc[last_idx]
                close_price = float(last_row["close"])
                # simulate a market fill at the last close
                filled_qty_exit, exit_price, _ = self.exec_sim.simulate_market_fill(active["side"], active["qty_remaining"], close_price, last_row, atr=atr_val)
                if filled_qty_exit <= 0:
                    # if cannot fill at all, treat as zero pnl
                    pnl = 0.0
                    net = 0.0
                    trades.append(TradeRecord(entry_ts=active["entry_ts"], exit_ts=df.index[last_idx], side=active["side"], entry_price=active["entry_price"], exit_price=close_price, qty=0.0, pnl=pnl, net_pnl=net, reason="TIMEOUT_NO_FILL", details={"mfe": active["mfe"], "mae": active["mae"]}))
                else:
                    qty_closed = filled_qty_exit
                    pnl = (exit_price - active["entry_price"]) * qty_closed if active["side"] == "LONG" else (active["entry_price"] - exit_price) * qty_closed
                    fee_exit = exit_price * qty_closed * self.exit_sim.fee_pct
                    entry_fee_alloc = active["entry_fee_total"] * (qty_closed / active["original_qty"]) if active["original_qty"] > 0 else 0.0
                    net = pnl - fee_exit - entry_fee_alloc
                    trades.append(TradeRecord(entry_ts=active["entry_ts"], exit_ts=df.index[last_idx], side=active["side"], entry_price=active["entry_price"], exit_price=exit_price, qty=qty_closed, pnl=pnl, net_pnl=net, reason="TIMEOUT", details={"mfe": active["mfe"], "mae": active["mae"]}))
                    equity += net

            # snapshot equity for reporting
            equity_snap_ts.append(ts)
            equity_snap_vals.append(equity)

        # Build outputs
        if trades:
            trades_df = pd.DataFrame([asdict(t) for t in trades])
            trades_df = trades_df.sort_values("entry_ts").reset_index(drop=True)
        else:
            # empty frame with consistent columns
            trades_df = pd.DataFrame(columns=[f.name for f in TradeRecord.__dataclass_fields__.values()])

        # equity series and returns
        if equity_snap_ts:
            equity_series = pd.Series(equity_snap_vals, index=pd.to_datetime(equity_snap_ts)).sort_index()
            returns = equity_series.pct_change().fillna(0)
        else:
            equity_series = pd.Series([self.initial_balance], index=[pd.to_datetime(datetime.utcnow())])
            returns = pd.Series([0.0], index=equity_series.index)

        metrics = self._compute_metrics(equity_series, returns, trades_df)
        result = BacktestResult(trades=trades_df, equity_curve=equity_series, returns=returns, initial_balance=self.initial_balance, final_balance=float(equity), metrics=metrics)
        logger.info("Batch backtest complete: initial=%.2f final=%.2f trades=%d", self.initial_balance, float(equity), len(trades_df))

        # plotting if requested
        try:
            self._plot_result_if_requested(result, df)
        except Exception as e:
            logger.debug("Plotting after batch failed: %s", e)

        return result

    # -------------------------
    # Online backtest (streaming)
    # -------------------------
    def run_online(self, df: pd.DataFrame, *, realtime: bool = False, debug: bool = False) -> BacktestResult:
        """
        Streaming backtest: process df candle-by-candle, evaluate SignalEngine.evaluate()
        on each new closed candle, open positions and manage exits in near-real-time.

        If realtime=True, this can be adapted to wait for new candles (not implemented here).
        """
        if df is None or df.empty:
            raise ValueError("df must be provided")

        df = df.copy()
        # ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                raise ValueError("df must have datetime index or timestamp column")

        # precompute ATR
        df["_atr"] = ExitSimulator.compute_atr_series(df, self.exit_sim.atr_period)

        equity = float(self.initial_balance)
        equity_ts: List[pd.Timestamp] = []
        equity_vals: List[float] = []
        trades: List[TradeRecord] = []
        open_positions: List[Dict[str, Any]] = []

        lookback = getattr(self.signal_engine.breakout_cfg, "lookback", 21) if self.signal_engine is not None else 21

        # process candles one by one (simulate arrival of closed candles)
        for i in range(lookback - 1, len(df)):
            # last N candles including current
            window_df = df.iloc[i - lookback + 1 : i + 1].copy()
            candle = df.iloc[i]
            ts = df.index[i]

            # evaluate signal (optionally pass mtf_frames if available)
            sig: Signal = self.signal_engine.evaluate(window_df, mtf_frames=None, debug=debug) if self.signal_engine is not None else (type("S", (), {"long_entry": False, "short_entry": False}))

            # extract strength and filter
            strength = float(getattr(sig, "ctx", {}).get("strength_score", getattr(sig, "ctx", {}).get("strength", 0.0) or 0.0)) if sig is not None else 0.0
            if strength < float(self.cfg_map.get("MIN_SIGNAL_STRENGTH", self.min_signal_strength)):
                if debug and (getattr(sig, "long_entry", False) or getattr(sig, "short_entry", False)):
                    logger.debug("Realtime: skipping weak signal at %s strength=%.3f", ts, strength)
                # still manage existing positions
            
            # Manage open positions first on this candle (SL/TP/trailing/breakeven)
            for pos in list(open_positions):
                atr_val = float(candle["_atr"]) if not pd.isna(candle["_atr"]) else None

                # SL
                if pos["stop"] is not None:
                    if pos["side"] == "LONG" and float(candle["low"]) <= pos["stop"]:
                        filled_qty_exit, exit_price, _ = self.exec_sim.simulate_market_fill(pos["side"], pos["qty_remaining"], pos["stop"], candle, atr=atr_val)
                        if filled_qty_exit <= 0:
                            continue
                        qty_closed = filled_qty_exit
                        pnl = (exit_price - pos["entry_price"]) * qty_closed
                        fee_exit = exit_price * qty_closed * self.exit_sim.fee_pct
                        entry_fee_alloc = pos["entry_fee_total"] * (qty_closed / pos["original_qty"]) if pos["original_qty"] > 0 else 0.0
                        net = pnl - fee_exit - entry_fee_alloc
                        trades.append(TradeRecord(entry_ts=pos["entry_ts"], exit_ts=ts, side=pos["side"], entry_price=pos["entry_price"], exit_price=exit_price, qty=qty_closed, pnl=pnl, net_pnl=net, reason="SL", details={}))
                        equity += net
                        open_positions.remove(pos)
                        continue
                    if pos["side"] == "SHORT" and float(candle["high"]) >= pos["stop"]:
                        filled_qty_exit, exit_price, _ = self.exec_sim.simulate_market_fill(pos["side"], pos["qty_remaining"], pos["stop"], candle, atr=atr_val)
                        if filled_qty_exit <= 0:
                            continue
                        qty_closed = filled_qty_exit
                        pnl = (pos["entry_price"] - exit_price) * qty_closed
                        fee_exit = exit_price * qty_closed * self.exit_sim.fee_pct
                        entry_fee_alloc = pos["entry_fee_total"] * (qty_closed / pos["original_qty"]) if pos["original_qty"] > 0 else 0.0
                        net = pnl - fee_exit - entry_fee_alloc
                        trades.append(TradeRecord(entry_ts=pos["entry_ts"], exit_ts=ts, side=pos["side"], entry_price=pos["entry_price"], exit_price=exit_price, qty=qty_closed, pnl=pnl, net_pnl=net, reason="SL", details={}))
                        equity += net
                        open_positions.remove(pos)
                        continue

                # partial TP hits
                to_pop = []

                for k, tp in enumerate(pos["tp_levels"]):
                    hit = ((pos["side"] == "LONG" and candle["high"] >= tp) or
                        (pos["side"] == "SHORT" and candle["low"] <= tp))
                    if not hit:
                        continue

                    frac = pos["tp_fracs"][k] if k < len(pos["tp_fracs"]) else 1.0 / len(pos["tp_levels"]) if len(pos["tp_levels"])>0 else 1.0
                    qty_close = pos["qty_remaining"] * frac

                    filled_qty_exit, exit_price, _ = self.exec_sim.simulate_market_fill(
                        pos["side"], qty_close, tp, candle, atr=atr_val
                    )
                    if filled_qty_exit <= 0:
                        continue

                    qty_closed = filled_qty_exit
                    pnl = (exit_price - pos["entry_price"]) * qty_closed if pos["side"] == "LONG" \
                        else (pos["entry_price"] - exit_price) * qty_closed

                    fee_exit = exit_price * qty_closed * self.exit_sim.fee_pct
                    entry_fee_alloc = pos["entry_fee_total"] * (qty_closed / pos["original_qty"]) if pos["original_qty"] > 0 else 0.0
                    net = pnl - fee_exit - entry_fee_alloc

                    trades.append(TradeRecord(
                        entry_ts=pos["entry_ts"],
                        exit_ts=ts,
                        side=pos["side"],
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        qty=qty_closed,
                        pnl=pnl,
                        net_pnl=net,
                        reason=f"TP_partial_{k+1}",
                        details={}
                    ))

                    equity += net
                    pos["qty_remaining"] -= qty_closed
                    to_pop.append(k)

                for idx in reversed(to_pop):
                    pos["tp_levels"].pop(idx)
                    pos["tp_fracs"].pop(idx)

                if pos["qty_remaining"] <= 1e-9:
                    open_positions.remove(pos)
                    break

                # trailing/breakeven updates (same as batch)
                if pos in open_positions:
                    atr_val = float(candle["_atr"]) if not pd.isna(candle["_atr"]) else None
                    if atr_val and atr_val > 0:
                        if pos["side"] == "LONG":
                            cur_profit = float(candle["high"]) - pos["entry_price"]
                        else:
                            cur_profit = pos["entry_price"] - float(candle["low"])

                        if cur_profit >= self.exit_sim.trailing_start_atr * atr_val:
                            if pos["side"] == "LONG":
                                proposed = float(candle["high"]) - self.exit_sim.trailing_step_atr * atr_val
                                if proposed > pos["stop"]:
                                    pos["stop"] = proposed
                            else:
                                proposed = float(candle["low"]) + self.exit_sim.trailing_step_atr * atr_val
                                if proposed < pos["stop"]:
                                    pos["stop"] = proposed

                        if cur_profit >= self.exit_sim.breakeven_atr * atr_val:
                            if pos["side"] == "LONG":
                                proposed = pos["entry_price"] + self.exit_sim.breakeven_buffer_pts * atr_val
                                if proposed > pos["stop"]:
                                    pos["stop"] = proposed
                            else:
                                proposed = pos["entry_price"] - self.exit_sim.breakeven_buffer_pts * atr_val
                                if proposed < pos["stop"]:
                                    pos["stop"] = proposed

            # Now process new signals for this candle (open new positions)
            if getattr(sig, "long_entry", False) or getattr(sig, "short_entry", False):
                # if the signal is below strength threshold skip opening
                sig_strength = float(sig.ctx.get("strength_score", sig.ctx.get("strength", 0.0) or 0.0)) if sig is not None else 0.0
                if sig_strength < float(self.cfg_map.get("MIN_SIGNAL_STRENGTH", self.min_signal_strength)):
                    if debug:
                        logger.debug("Skipping open at %s due to weak realtime signal strength %.3f", ts, sig_strength)
                else:
                    side = "LONG" if getattr(sig, "long_entry", False) else "SHORT"
                    entry_price = float(candle["close"])
                    atr_val = float(candle["_atr"]) if not pd.isna(candle["_atr"]) else None

                    plan = self.exit_sim.plan_exit(side, entry_price, atr_val, strength=sig_strength)
                    stop_price = plan["stop_loss"]
                    tp_levels = plan["tp_levels"]
                    tp_fracs = plan["tp_fracs"]

                    qty = self._size_from_risk(entry_price, stop_price, equity, strength=sig_strength)
                    if qty <= 0 or not math.isfinite(qty):
                        continue

                    # simulate entry fill
                    filled_qty, entry_exec_price, _ = self.exec_sim.simulate_market_fill(side, qty, entry_price, candle, atr=atr_val)
                    if filled_qty <= 0:
                        continue
                    qty = filled_qty
                    entry_fee_total = entry_exec_price * qty * self.exit_sim.fee_pct

                    open_positions.append({
                        "side": side,
                        "entry_ts": ts,
                        "entry_price": entry_exec_price,
                        "original_qty": qty,
                        "qty_remaining": qty,
                        "stop": float(stop_price) if stop_price is not None else None,
                        "tp_levels": [float(x) for x in tp_levels],
                        "tp_fracs": [float(x) for x in tp_fracs],
                        "realized": 0.0,
                        "entry_fee_total": entry_fee_total
                    })

            # snapshot equity
            equity_ts.append(ts)
            equity_vals.append(equity)

        # finish: build outputs
        trades_df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()
        if equity_ts:
            equity_series = pd.Series(equity_vals, index=pd.to_datetime(equity_ts)).sort_index()
            returns = equity_series.pct_change().fillna(0)
        else:
            equity_series = pd.Series([self.initial_balance], index=[pd.to_datetime(datetime.utcnow())])
            returns = pd.Series([0.0], index=equity_series.index)

        metrics = self._compute_metrics(equity_series, returns, trades_df)
        result = BacktestResult(trades=trades_df, equity_curve=equity_series, returns=returns, initial_balance=self.initial_balance, final_balance=float(equity), metrics=metrics)
        logger.info("Online backtest complete: initial=%.2f final=%.2f trades=%d", self.initial_balance, float(equity), len(trades_df))

        # plotting if requested
        try:
            self._plot_result_if_requested(result, df)
        except Exception as e:
            logger.debug("Plotting after online failed: %s", e)

        return result

    # -------------------------
    # Metrics
    # -------------------------
    def _compute_metrics(self, equity_series: pd.Series, returns: pd.Series, trades_df: pd.DataFrame) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if equity_series is None or equity_series.empty:
            return metrics
        initial = float(equity_series.iloc[0])
        final = float(equity_series.iloc[-1])
        metrics["initial_balance"] = initial
        metrics["final_balance"] = final
        metrics["total_return"] = (final / initial - 1.0)
        # rough annualization based on timeframe in cfg_map
        tf = str(self.cfg_map.get("TIMEFRAME", "5m"))
        minutes = 1
        try:
            if tf.endswith("m"):
                minutes = int(tf[:-1])
            elif tf.endswith("h"):
                minutes = 60 * int(tf[:-1])
            elif tf.endswith("d"):
                minutes = 60 * 24 * int(tf[:-1])
        except Exception:
            minutes = 5
        periods_per_year = (24 * 60 / max(1, minutes)) * 365.0
        ann_vol = float(returns.std() * math.sqrt(periods_per_year)) if returns.std() > 0 else 0.0
        sharpe = (returns.mean() * math.sqrt(periods_per_year) / returns.std()) if returns.std() > 0 else 0.0
        # max drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        metrics["max_drawdown"] = float(drawdown.min()) if not drawdown.empty else 0.0
        metrics["annual_volatility"] = ann_vol
        metrics["sharpe"] = sharpe
        # trades stats
        if trades_df is not None and not trades_df.empty:
            # ensure net_pnl exists
            if "net_pnl" not in trades_df.columns and "net_pnl" in trades_df:
                trades_df["net_pnl"] = trades_df["net_pnl"]
            wins = trades_df[trades_df["net_pnl"] > 0] if "net_pnl" in trades_df.columns else pd.DataFrame()
            metrics["n_trades"] = len(trades_df)
            metrics["win_rate"] = float(len(wins) / len(trades_df)) if len(trades_df) > 0 else 0.0
            metrics["avg_win"] = float(wins["net_pnl"].mean()) if not wins.empty else 0.0
            losses = trades_df[trades_df["net_pnl"] <= 0] if "net_pnl" in trades_df.columns else pd.DataFrame()
            metrics["avg_loss"] = float(losses["net_pnl"].mean()) if not losses.empty else 0.0
            profit = wins["net_pnl"].sum() if not wins.empty else 0.0
            loss = -losses["net_pnl"].sum() if not losses.empty else 0.0
            metrics["profit_factor"] = (profit / loss) if loss > 0 else (float("inf") if profit > 0 else 0.0)
        return metrics


    def run_with_params(self, df: Any, params: Dict[str, Any], mode: str = 'batch', **kwargs) -> BacktestResult:
        """Run the backtester merging cfg_map with params. mode='batch'|'online'."""
        merged = dict(self.cfg_map or {})
        merged.update(params or {})
        # create a new instance to avoid state leakage
        bt = self.__class__(cfg_map=merged, signal_engine=self.signal_engine)
        if mode == 'batch':
            return bt.run_batch(df, **kwargs)
        return bt.run_online(df, **kwargs)


    # Add this helper to score results (used by the optimizer if present)

    def score_result(self, result: BacktestResult, objective: str = 'sharpe') -> float:
        """Return a scalar score for a BacktestResult (higher is better)."""
        if result is None:
            return -float('inf')
        metrics = getattr(result, 'metrics', {}) or {}
        if objective == 'sharpe':
            return float(metrics.get('sharpe', -1e9))
        if objective == 'total_return':
            return float(metrics.get('total_return', -1e9))
        if objective == 'profit_factor':
            return float(metrics.get('profit_factor', -1e9))
        # default
        return float(metrics.get('sharpe', -1e9))


    # Auto-plot integration: call plotting helpers if available

    def _maybe_auto_plot(self, result: BacktestResult, prefix: Optional[str] = None) -> None:
        """Try to call src.backtest_plots plotting helpers. Safe no-op if not found."""
        try:
            # local import to avoid hard dependency
            from src import backtest_plots as bp  # type: ignore
        except Exception:
            try:
                import backtest_plots as bp  # type: ignore
            except Exception:
                return
        try:
            outdir = self.cfg_map.get('PLOT_OUTPUT_DIR') if isinstance(self.cfg_map, dict) else None
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            name = prefix or 'backtest'
            # attempt several common helpers if present
            if hasattr(bp, 'plot_equity'):
                path = None
                if outdir:
                    path = os.path.join(outdir, f"{name}_equity.png")
                bp.plot_equity(result.equity_curve, trades=result.trades if hasattr(result, 'trades') else None, out_path=path)
            if hasattr(bp, 'plot_drawdown'):
                path = None
                if outdir:
                    path = os.path.join(outdir, f"{name}_drawdown.png")
                bp.plot_drawdown(result.equity_curve, out_path=path)
        except Exception:
            # plotting is best-effort
            return
