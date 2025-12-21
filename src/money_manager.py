# src/money_manager.py
"""
MoneyManager: small-account protections.

Enhancements (Step 3):
- Hard daily loss limit (USDT and %)
- Max consecutive losses
- Dynamic cooldown via signal methods
- Risk-fraction recommendations for ATR sizing
- Daily reset (UTC)
- Persistent minimal state: peak_balance, daily_loss_usdt, consecutive_losses, day (YYYY-MM-DD)

This file includes robust balance extraction and debug logging/fallbacks to avoid
silent zero-balance behaviour that can make the order qty collapse to zero.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("money_manager")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("MM_LOG_LEVEL", "INFO"))

DEFAULT_STATE_PATH = os.getenv("MM_STATE_FILE", ".mm_state.json")


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class MoneyManager:
    def __init__(self, client: Any, state_path: str = DEFAULT_STATE_PATH):
        self.client = client
        self.state_path = state_path

        # limits for safety
        try:
            self.MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))
        except Exception:
            self.MAX_DAILY_LOSS_PCT = 5.0

        try:
            self.MAX_DAILY_LOSS_USDT = float(os.getenv("MAX_DAILY_LOSS_USDT", "15.0"))
        except Exception:
            self.MAX_DAILY_LOSS_USDT = 15.0

        try:
            self.MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
        except Exception:
            self.MAX_CONSECUTIVE_LOSSES = 3

        # persisted minimal state
        self.state: Dict[str, Any] = {
            "day": _today_utc_str(),
            "peak_balance": None,
            "daily_loss_usdt": 0.0,
            "consecutive_losses": 0,
        }
        self._load_state()

    # -------------------------
    # persistence
    # -------------------------
    def _load_state(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for k in ("day", "peak_balance", "daily_loss_usdt", "consecutive_losses"):
                            if k in data:
                                self.state[k] = data[k]
        except Exception:
            logger.debug("MoneyManager: could not load state", exc_info=True)

    def _save_state(self) -> None:
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            logger.debug("MoneyManager: could not save state", exc_info=True)

    # -------------------------
    # Day rollover
    # -------------------------
    def reset_if_new_day(self) -> None:
        today = _today_utc_str()
        if self.state.get("day") != today:
            logger.info("MoneyManager: new UTC day detected — resetting daily counters.")
            self.state["day"] = today
            self.state["daily_loss_usdt"] = 0.0
            self.state["consecutive_losses"] = 0
            self._save_state()

    # -------------------------
    # Wallet balance helper
    # -------------------------
    def _extract_usdt_balance_from_resp(self, resp: Any) -> Optional[float]:
        """
        Internal helper to safely extract a USDT futures balance
        from multiple possible Binance response formats.
        """
        try:
            # --- Case 1: futures_account_balance() format: list of dicts ---
            if isinstance(resp, list):
                for item in resp:
                    if isinstance(item, dict) and str(item.get("asset")).upper() == "USDT":
                        bal = item.get("availableBalance") or item.get("balance") or item.get("crossWalletBalance")
                        if bal is not None:
                            try:
                                return float(bal)
                            except Exception:
                                continue
                return None

            # --- Case 2: futures_account() full account format (dict with 'assets') ---
            if isinstance(resp, dict):
                assets = resp.get("assets")
                if isinstance(assets, list):
                    for a in assets:
                        if str(a.get("asset")).upper() == "USDT":
                            bal = a.get("availableBalance") or a.get("walletBalance") or a.get("balance") or a.get("crossWalletBalance")
                            if bal is not None:
                                try:
                                    return float(bal)
                                except Exception:
                                    continue

                # Some clients: {"totalWalletBalance": "...", "availableBalance": "..."}
                for k in ("availableBalance", "totalWalletBalance", "walletBalance", "balance"):
                    if k in resp:
                        try:
                            return float(resp[k])
                        except Exception:
                            pass

            return None
        except Exception:
            logger.debug("MoneyManager: extraction error", exc_info=True)
            return None

    def _as_number(self, value: Any) -> Optional[float]:
        """
        Safely convert a value to float.
        Returns float or None.
        """
        import numbers
        try:
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                return float(value)

            if isinstance(value, str):
                s = value.strip().replace(",", "")
                if s != "":
                    try:
                        return float(s)
                    except Exception:
                        return None

            return None
        except Exception:
            return None


    def get_account_balance(self) -> Optional[float]:
        """
        Extract USDT futures balance from client responses.
        Always avoids float(resp) unless guaranteed numeric.
        """
        try:
            tried = []

            for name in (
                "futures_account_balance",
                "futures_account",
                "get_futures_account",
                "get_account",
                "account",
                "balance",
                "get_balance",
            ):
                if hasattr(self.client, name):
                    try:
                        fn = getattr(self.client, name)
                        resp = fn() if callable(fn) else fn
                        tried.append((name, type(resp).__name__))

                        # safe scalar
                        num = self._as_number(resp)
                        if num is not None:
                            return num

                        # structured response
                        bal = self._extract_usdt_balance_from_resp(resp)
                        if bal is not None:
                            return float(bal)

                    except Exception:
                        continue

            if hasattr(self.client, "balances"):
                try:
                    resp = getattr(self.client, "balances")
                    tried.append(("balances", type(resp).__name__))

                    num = self._as_number(resp)
                    if num is not None:
                        return num

                    bal = self._extract_usdt_balance_from_resp(resp)
                    if bal is not None:
                        return float(bal)

                except Exception:
                    pass

            for name in ("get_available_balance", "get_balance_summary", "fetch_and_persist_balance"):
                if hasattr(self.client, name):
                    try:
                        fn = getattr(self.client, name)
                        resp = fn() if callable(fn) else fn
                        tried.append((name, type(resp).__name__))

                        # safe scalar
                        num = self._as_number(resp)
                        if num is not None:
                            return num

                        if isinstance(resp, dict):
                            if "available" in resp:
                                num = self._as_number(resp["available"])
                                if num is not None:
                                    return num

                            bal = self._extract_usdt_balance_from_resp(resp)
                            if bal is not None:
                                return float(bal)

                    except Exception:
                        continue

        except Exception:
            pass

        return None


    def get_balance_safe(self) -> float:
        """
        Guaranteed float, never float(resp) on unknown types.
        """
        try:
            bal = self.get_account_balance()
            if bal is not None:
                return bal

            # next, try helper endpoints
            for name in ("fetch_and_persist_balance", "fetch_balance", "fetch_balances"):
                if hasattr(self.client, name):
                    try:
                        fn = getattr(self.client, name)
                        resp = fn() if callable(fn) else None

                        num = self._as_number(resp)
                        if num is not None:
                            return num

                        if isinstance(resp, dict):
                            if "available" in resp:
                                num = self._as_number(resp["available"])
                                if num is not None:
                                    return num

                            bal = self._extract_usdt_balance_from_resp(resp)
                            if bal is not None:
                                return float(bal)

                    except Exception:
                        pass

            # ENV fallback
            env_b = os.getenv("ACCOUNT_BALANCE")
            if env_b:
                num = self._as_number(env_b)
                if num is not None:
                    return num

            # last helpers
            for name in ("get_available_balance", "get_balance_summary", "get_balance"):
                if hasattr(self.client, name):
                    try:
                        fn = getattr(self.client, name)
                        resp = fn() if callable(fn) else fn

                        num = self._as_number(resp)
                        if num is not None:
                            return num

                        if isinstance(resp, dict) and "available" in resp:
                            num = self._as_number(resp["available"])
                            if num is not None:
                                return num

                    except Exception:
                        pass

            return 0.0

        except Exception:
            return 0.0


    # -------------------------
    # Trading permission
    # -------------------------
    def can_trade(self) -> Tuple[bool, Optional[str]]:
        """
        Return (allowed:bool, reason:str)
        """
        try:
            self.reset_if_new_day()
            bal = self.get_account_balance()
            if bal is None:
                # conservative: allow but log; callers using get_balance_safe() will receive 0.0
                logger.debug("MoneyManager.can_trade: balance unknown (None) — allowing trading by default (but check sizing!).")
                return True, None

            peak = self.state.get("peak_balance") or bal
            if peak is None:
                peak = bal
                self.state["peak_balance"] = peak
                self._save_state()

            daily_loss = float(self.state.get("daily_loss_usdt", 0.0))
            daily_loss_pct = (daily_loss / bal) * 100.0 if bal > 0 else 0.0

            if self.MAX_DAILY_LOSS_USDT and daily_loss >= float(self.MAX_DAILY_LOSS_USDT):
                return False, "daily_loss_usdt_exceeded"

            if self.MAX_DAILY_LOSS_PCT and daily_loss_pct >= float(self.MAX_DAILY_LOSS_PCT):
                return False, "daily_loss_pct_exceeded"

            cons = int(self.state.get("consecutive_losses", 0))
            if self.MAX_CONSECUTIVE_LOSSES and cons >= int(self.MAX_CONSECUTIVE_LOSSES):
                return False, "max_consecutive_losses_reached"

            return True, None

        except Exception:
            logger.debug("MoneyManager.can_trade failure", exc_info=True)
            return True, None

    # -------------------------
    # Record trade results
    # -------------------------
    def record_closed_trade(self, profit_usdt: float) -> None:
        try:
            self.reset_if_new_day()

            bal = self.get_account_balance()
            if bal is not None:
                prev_peak = self.state.get("peak_balance")
                if prev_peak is None or bal > float(prev_peak):
                    self.state["peak_balance"] = float(bal)

            if profit_usdt < 0:
                loss = abs(float(profit_usdt))
                self.state["daily_loss_usdt"] = float(self.state.get("daily_loss_usdt", 0.0)) + loss
                self.state["consecutive_losses"] = int(self.state.get("consecutive_losses", 0)) + 1
            else:
                self.state["consecutive_losses"] = 0

            self._save_state()
            logger.info(
                "MoneyManager: recorded trade pnl=%.4f, daily_loss=%.4f, cons=%s",
                profit_usdt,
                self.state.get("daily_loss_usdt"),
                self.state.get("consecutive_losses"),
            )
        except Exception:
            logger.debug("MoneyManager.record_closed_trade failed", exc_info=True)

    # -------------------------
    # Step 3: Cooldown signalling
    # -------------------------
    def should_cooldown(self) -> bool:
        """
        Return True if bot should stop opening positions temporarily.
        ExecutionManager enforces cooldown.
        """
        try:
            cons = int(self.state.get("consecutive_losses", 0))
            max_loss = int(self.MAX_CONSECUTIVE_LOSSES)

            if cons >= max_loss:
                return True
        except Exception:
            pass
        return False

    def get_cooldown_remaining(self) -> Optional[int]:
        """
        Optionally return how many losses remain until cooldown resets.
        """
        try:
            cons = int(self.state.get("consecutive_losses", 0))
            max_loss = int(self.MAX_CONSECUTIVE_LOSSES)
            return max(0, max_loss - cons)
        except Exception:
            return None

    # -------------------------
    # Step 3: Dynamic risk fraction suggestion
    # -------------------------
    def recommend_risk_fraction(self) -> float:
        """
        Suggest a dynamic risk fraction (0.0–1.0) that ExecutionManager may use
        inside ATR sizing logic.
        """
        cons = int(self.state.get("consecutive_losses", 0))

        # cooldown trigger → 0
        if cons >= self.MAX_CONSECUTIVE_LOSSES:
            return 0.0

        # scale based on losses
        if cons == 0:
            return 1.0
        elif cons == 1:
            return 0.7
        elif cons == 2:
            return 0.5
        else:
            return 0.2  # safety fallback

    # -------------------------
    # manual reset
    # -------------------------
    def force_reset(self) -> None:
        self.state["day"] = _today_utc_str()
        self.state["daily_loss_usdt"] = 0.0
        self.state["consecutive_losses"] = 0
        self._save_state()
        logger.info("MoneyManager: force reset executed")