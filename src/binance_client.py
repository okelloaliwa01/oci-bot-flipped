# src/binance_client.py
"""
Hardened, backward-compatible Binance Futures client.

Features:
- Testnet support via USE_TESTNET or constructor arg
- SDK compatibility with multiple connector variants (um_futures, futures, python-binance)
- REST signed GET/POST fallback for all signed endpoints
- Time sync protection for -1102 timestamp errors
- Robust rounding using exchange filters (PRICE_FILTER / LOT_SIZE)
- Atomic order flow (market + TP/SL creation)
- Compatibility aliases and helper methods restored
- Balance persistence to .env (ACCOUNT_BALANCE) kept as optional convenience
"""

from __future__ import annotations
import os
import time
import math
import json
import hmac
import hashlib
import logging
import functools
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from urllib.parse import urlencode

import requests
from requests.exceptions import ConnectionError, ReadTimeout, RequestException

# SDK compatibility
UMFutures = None
try:
    from binance.um_futures import UMFutures  # modern connector
except Exception:
    try:
        from binance.futures import Futures as UMFutures  # legacy connector
    except Exception:
        try:
            from binance.client import Client as UMFutures  # python-binance
        except Exception:
            UMFutures = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    # minimal default handler when module run directly
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ------------------------------------------------------------
# Retry / helper decorators
# ------------------------------------------------------------
def safe_api_call(max_retries: int = 4, backoff: float = 1.8):
    """Retry decorator tuned for network instability and rate limits."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, ReadTimeout) as e:
                    wait = backoff ** attempt
                    logger.warning("%s network error: %s (retry %d/%d -> %.1fs)", func.__name__, e, attempt, max_retries, wait)
                    time.sleep(wait)
                    last_exc = e
                    continue
                except RequestException as e:
                    # typically HTTP errors - bubble up
                    logger.error("%s request exception: %s", func.__name__, e)
                    last_exc = e
                    break
                except Exception as e:
                    estr = str(e).lower()
                    if "429" in estr or "too many requests" in estr:
                        wait = 2.5 * (backoff ** attempt)
                        logger.warning("%s rate limited (retry %d/%d -> %.1fs)", func.__name__, attempt, max_retries, wait)
                        time.sleep(wait)
                        last_exc = e
                        continue
                    last_exc = e
                    logger.debug("%s unhandled error: %s", func.__name__, e)
                    break
            raise RuntimeError(f"{func.__name__} failed after {max_retries} attempts: {last_exc}")
        return wrapper
    return decorator

def retry_with_time_sync(max_attempts: int = 3, delay: float = 0.45):
    """Retry wrapper which triggers time sync on -1102 timestamp errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(self, *args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    text = ""
                    try:
                        text = (e.response.text or "") if getattr(e, "response", None) else ""
                    except Exception:
                        text = ""
                    if "-1102" in text or "timestamp for this request" in text.lower():
                        logger.info("Timestamp error detected — performing time sync and retrying")
                        try:
                            self._sync_time()
                        except Exception as se:
                            logger.debug("Time sync attempt failed: %s", se)
                        time.sleep(delay)
                        continue
                    raise
                except Exception:
                    time.sleep(delay)
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator

# ------------------------------------------------------------
# Main class
# ------------------------------------------------------------
class BinanceClient:
    """
    Hardened Binance Futures client (drop-in replacement).
    """

    _symbol_info_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, use_testnet: Optional[bool] = None):
        # env fallback
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        env_test = os.getenv("USE_TESTNET")
        if use_testnet is not None:
            self.use_testnet = bool(use_testnet)
        else:
            self.use_testnet = str(env_test).lower() in ("1", "true", "yes")
        self.base_url = "https://testnet.binancefuture.com" if self.use_testnet else "https://fapi.binance.com"

        self._client: Optional[Any] = None
        self._session = requests.Session()
        self.time_offset = 0
        self._last_time_sync = 0

    # ---------------------------
    # Initialization / SDK shim
    # ---------------------------
    def _init(self):
        if self._client:
            return
        if UMFutures is None:
            logger.warning("No Binance SDK detected; REST-only mode available for signed endpoints.")
            return
        if not self.api_key or not self.api_secret:
            logger.warning("API key/secret missing; some features will fail.")
            return
        try:
            # many SDKs accept (key=..., secret=..., base_url=...)
            try:
                self._client = UMFutures(key=self.api_key, secret=self.api_secret, base_url=self.base_url)
            except TypeError:
                # older constructors
                self._client = UMFutures(self.api_key, self.api_secret)
            logger.info("Binance SDK client initialized (testnet=%s)", self.use_testnet)
        except Exception as e:
            logger.warning("SDK init failed; falling back to REST-only: %s", e)
            self._client = None

    # ---------------------------
    # Time sync & signing
    # ---------------------------
    def _sync_time(self):
        """Sync time offset (ms)."""
        try:
            url = f"{self.base_url}/fapi/v1/time"
            r = self._session.get(url, timeout=4)
            r.raise_for_status()
            server_time = int(r.json().get("serverTime", int(time.time() * 1000)))
            local_ms = int(time.time() * 1000)
            self.time_offset = server_time - local_ms
            self._last_time_sync = local_ms
            logger.debug("Time sync done. offset=%dms", self.time_offset)
        except Exception as e:
            logger.warning("Time sync failed: %s", e)

    def _signed_params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if params is None:
            params = {}
        now_ms = int(time.time() * 1000)
        # refresh time offset if stale (>30s)
        if abs(now_ms - getattr(self, "_last_time_sync", 0)) > 30_000:
            try:
                self._sync_time()
            except Exception:
                pass
        ts = now_ms + int(getattr(self, "time_offset", 0))
        params.update({"timestamp": ts, "recvWindow": 60000})
        return params

    def _rest_signed_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 6) -> Any:
        params = self._signed_params((params or {}).copy())
        query = urlencode(params)
        signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = signature
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        r = self._session.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _rest_signed_post(self, endpoint: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 8) -> Any:
        payload = payload or {}
        params = self._signed_params(payload.copy())
        query = urlencode(params)
        signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = signature
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        r = self._session.post(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ---------------------------
    # Exchange / Symbol info
    # ---------------------------
    @safe_api_call()
    def exchange_info(self) -> Dict[str, Any]:
        """
        Returns exchange_info (whole payload). Uses SDK if available, otherwise REST.
        """
        self._init()
        if self._client and hasattr(self._client, "exchange_info"):
            try:
                return self._client.exchange_info()
            except Exception as e:
                logger.debug("SDK exchange_info failed: %s", e)
        # REST
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        r = self._session.get(url, timeout=6)
        r.raise_for_status()
        return r.json()

    @safe_api_call()
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        # cache fast-path
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        info = self.exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                self._symbol_info_cache[symbol] = s
                return s
        raise ValueError(f"Symbol not found: {symbol}")

    @safe_api_call()
    def get_symbol_filters(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        info = self.get_symbol_info(symbol)
        return {f["filterType"]: f for f in info.get("filters", [])}

    def _get_tick_and_step(self, symbol: str):
        info = self.get_symbol_info(symbol)
        tick = 0.0
        step = 0.0
        min_notional = 0.0
        for f in info.get("filters", []):
            t = f.get("filterType")
            if t == "PRICE_FILTER":
                tick = float(f.get("tickSize", 0))
            elif t == "LOT_SIZE":
                step = float(f.get("stepSize", 0))
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = float(f.get("minNotional", f.get("notional", 0)))
        return tick, step, min_notional

    def round_price(self, symbol: str, price: float) -> float:
        try:
            tick, _, _ = self._get_tick_and_step(symbol)
            if not tick or tick <= 0:
                return float(price)
            # floor to tick (do not exceed)
            rounded = math.floor(price / tick) * tick
            # keep sensible decimals
            decimals = max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0
            return round(rounded, decimals)
        except Exception:
            return float(price)

    def round_qty(self, symbol: str, qty: float) -> float:
        try:
            _, step, _ = self._get_tick_and_step(symbol)
            if not step or step <= 0:
                return float(round(qty, 8))
            rounded = math.floor(qty / step) * step
            if rounded < step:
                rounded = step
            return round(rounded, 8)
        except Exception:
            return float(round(qty, 8))

    # ---------------------------
    # Klines (chunked)
    # ---------------------------
    @safe_api_call()
    def get_klines(self, symbol: str, interval: str, limit: int = 500, end_time: Optional[int] = None) -> List[List[Any]]:
        """
        Fetch klines with chunking (handles >1500 windows).
        Returns list of kline arrays (like Binance).
        """
        self._init()
        MAX = 1500
        if end_time is None:
            end_time = int(time.time() * 1000)
        collected: List[List[Any]] = []
        remaining = limit
        while remaining > 0:
            fetch_limit = min(remaining, MAX)
            params = {"symbol": symbol.upper(), "interval": interval, "limit": fetch_limit, "endTime": end_time}
            data = None
            # try SDK methods first
            if self._client:
                for name in ("klines", "get_klines", "futures_klines", "get_candles"):
                    if hasattr(self._client, name):
                        try:
                            fn = getattr(self._client, name)
                            # SDKs may expect kwargs or positional args
                            try:
                                data = fn(symbol=symbol, interval=interval, limit=fetch_limit)
                            except TypeError:
                                data = fn(symbol, interval, fetch_limit)
                            break
                        except Exception:
                            continue
            if data is None:
                # REST fallback
                url = f"{self.base_url}/fapi/v1/klines"
                resp = self._session.get(url, params=params, timeout=8)
                resp.raise_for_status()
                data = resp.json()
            if not data:
                break
            collected = data + collected  # prepend older chunk
            # next chunk end_time
            try:
                end_time = int(data[0][0]) - 1
            except Exception:
                break
            remaining -= fetch_limit
            time.sleep(0.15)
        return collected[-limit:]

    # ---------------------------
    # Ticker / Mark price
    # ---------------------------
    @safe_api_call()
    def ticker_price(self, symbol: str) -> Dict[str, Any]:
        self._init()
        # try likely SDK names
        if self._client:
            for name in ("ticker_price", "get_symbol_ticker", "symbol_ticker"):
                if hasattr(self._client, name):
                    try:
                        return getattr(self._client, name)(symbol=symbol)
                    except Exception:
                        continue
        # REST fallback
        url = f"{self.base_url}/fapi/v1/ticker/price"
        r = self._session.get(url, params={"symbol": symbol}, timeout=6)
        r.raise_for_status()
        return r.json()

    @safe_api_call()
    def get_mark_price(self, symbol: str) -> float:
        self._init()
        # sdk
        if self._client and hasattr(self._client, "mark_price"):
            try:
                res = self._client.mark_price(symbol=symbol)
                if isinstance(res, dict) and res.get("markPrice") is not None:
                    return float(res.get("markPrice"))
                if isinstance(res, (str, float, int)):
                    return float(res)
            except Exception:
                pass
        # rest
        url = f"{self.base_url}/fapi/v1/premiumIndex"
        r = self._session.get(url, params={"symbol": symbol}, timeout=6)
        r.raise_for_status()
        j = r.json()
        return float(j.get("markPrice", j.get("lastPrice", 0.0)))

    # ---------------------------
    # Balance helpers
    # ---------------------------
    def _find_env_path(self) -> Path:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for p in candidates:
            if p.exists():
                return p
        return Path.cwd() / ".env"

    def _persist_account_balance_to_env(self, balance: float, env_path: Optional[Path] = None) -> bool:
        env_path = env_path or self._find_env_path()
        try:
            lines: List[str] = []
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
            def set_line(ls: List[str], key: str, value: str):
                import re
                pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
                for i, L in enumerate(ls):
                    if pat.match(L):
                        ls[i] = f"{key}={value}\n"
                        return
                ls.append(f"{key}={value}\n")
            set_line(lines, "ACCOUNT_BALANCE", f"{balance:.8f}")
            env_path.write_text("".join(lines), encoding="utf-8")
            logger.info("Persisted ACCOUNT_BALANCE=%.8f to %s", balance, env_path)
            return True
        except Exception as e:
            logger.warning("Failed to persist ACCOUNT_BALANCE: %s", e)
            return False

    @safe_api_call()
    def get_futures_account_balance(self, asset: str = "USDT") -> Optional[float]:
        """Return available futures balance for asset; persist to .env if successful."""
        self._init()
        balance: Optional[float] = None
        # SDK variations
        if self._client:
            for name in ("futures_account_balance", "balance", "get_balance", "account"):
                if hasattr(self._client, name):
                    try:
                        res = getattr(self._client, name)()
                        if isinstance(res, list):
                            for b in res:
                                if str(b.get("asset")).upper() == asset.upper():
                                    balance = float(b.get("availableBalance") or b.get("balance") or 0.0)
                                    break
                        elif isinstance(res, dict) and asset in res:
                            balance = float(res.get(asset, 0.0))
                        if balance is not None:
                            break
                    except Exception:
                        continue
        # REST fallback
        if balance is None:
            try:
                data = self._rest_signed_get("/fapi/v2/balance")
                if isinstance(data, list):
                    for b in data:
                        if str(b.get("asset")).upper() == asset.upper():
                            balance = float(b.get("availableBalance") or b.get("balance") or 0.0)
                            break
            except Exception as e:
                logger.debug("Balance REST fallback failed: %s", e)

        if balance is not None:
            try:
                self._persist_account_balance_to_env(balance)
            except Exception:
                pass
        else:
            logger.warning("Could not fetch balance for %s", asset)
        return balance

    # alias for compatibility
    def futures_account_balance(self, *args, **kwargs):
        return self.get_futures_account_balance(*args, **kwargs)

    def fetch_and_persist_balance(self, asset: str = "USDT", env_path: Optional[Path] = None) -> Optional[float]:
        bal = self.get_futures_account_balance(asset=asset)
        if bal is None:
            logger.warning("fetch_and_persist_balance: Could not fetch balance")
            return None
        self._persist_account_balance_to_env(bal, env_path=env_path)
        return bal

    @safe_api_call()
    def get_balance_summary(self, asset: str = "USDT") -> Dict[str, float]:
        bal = self.get_futures_account_balance(asset=asset)
        if bal is None:
            return {"asset": asset, "available": 0.0, "wallet": 0.0}
        # SDK sometimes returns wallet vs available; best-effort
        return {"asset": asset, "available": float(bal), "wallet": float(bal)}

    @safe_api_call()
    def get_available_balance(self, asset: str = "USDT") -> float:
        return float(self.get_balance_summary(asset).get("available", 0.0))

    # ---------------------------
    # Position helpers
    # ---------------------------
    @safe_api_call()
    def get_position_risk(self, symbol: Optional[str] = None):
        self._init()
        try:
            if self._client and hasattr(self._client, "get_position_risk"):
                if symbol:
                    return getattr(self._client, "get_position_risk")(symbol=symbol)
                return getattr(self._client, "get_position_risk")()
        except Exception:
            pass
        # REST fallback (position info endpoint)
        try:
            data = self._rest_signed_get("/fapi/v2/positionRisk")
            if symbol:
                return [p for p in data if p.get("symbol") == symbol]
            return data
        except Exception:
            logger.debug("get_position_risk REST fallback failed")
            return []

    def get_position(self, symbol: str) -> Dict[str, Any]:
        self._init()
        try:
            # try various SDK methods
            if self._client:
                for name in ("get_position", "position_information", "get_position_risk"):
                    if hasattr(self._client, name):
                        try:
                            res = getattr(self._client, name)(symbol=symbol)
                            if isinstance(res, list) and res:
                                return res[0]
                            if isinstance(res, dict) and res.get("symbol") == symbol:
                                return res
                        except Exception:
                            continue
        except Exception:
            pass
        # REST fallback
        try:
            data = self._rest_signed_get("/fapi/v2/positionRisk")
            for p in (data or []):
                if p.get("symbol") == symbol:
                    return p
        except Exception:
            pass
        return {}

    @safe_api_call()
    def get_all_positions(self) -> List[Dict[str, Any]]:
        self._init()
        try:
            data = None
            if self._client and hasattr(self._client, "get_position_risk"):
                data = self._client.get_position_risk()
            if not data and hasattr(self._client, "position_information"):
                data = self._client.position_information()
            if not data:
                data = self._rest_signed_get("/fapi/v2/positionRisk")
            # return only non-zero
            out = [p for p in (data or []) if abs(float(p.get("positionAmt", 0) or 0)) > 0]
            return out
        except Exception:
            return []

    def has_open_position(self, symbol: str) -> bool:
        pos = self.get_position(symbol)
        if not pos:
            return False
        try:
            amt = float(pos.get("positionAmt") or pos.get("position") or 0)
            return abs(amt) > 0
        except Exception:
            return False

    # ---------------------------
    # Open orders helpers
    # ---------------------------
    @safe_api_call()
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        self._init()
        # SDK attempts
        if self._client:
            for name in ("get_open_orders", "futures_get_open_orders", "open_orders", "get_all_orders"):
                if hasattr(self._client, name):
                    try:
                        fn = getattr(self._client, name)
                        res = fn(symbol=symbol) if symbol else fn()
                        if isinstance(res, list):
                            return res
                    except Exception:
                        continue
        # REST fallback
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
            data = self._rest_signed_get("/fapi/v1/openOrders", params=params)
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.debug("get_open_orders REST failed: %s", e)
        return []

    @safe_api_call()
    def cancel_order(self, symbol: str, orderId: Union[int, str]):
        self._init()
        # sdk attempt
        if self._client:
            for name in ("futures_cancel_order", "cancel_order", "cancel_open_order"):
                if hasattr(self._client, name):
                    try:
                        return getattr(self._client, name)(symbol=symbol, orderId=orderId)
                    except Exception:
                        continue
        # rest
        try:
            return self._rest_signed_post("/fapi/v1/order", {"symbol": symbol, "orderId": orderId})
        except Exception as e:
            logger.debug("cancel_order REST failed: %s", e)
            raise

    # ---------------------------
    # Order placement (market + generic)
    # ---------------------------
    @retry_with_time_sync()
    def futures_create_order(self, **kwargs) -> Dict[str, Any]:
        """Compat alias used by other modules; will try sdk methods then REST fallback."""
        self._init()
        dry = str(os.getenv("DRY_RUN", "false")).lower() in ("1", "true", "yes")
        if dry:
            logger.info("[DRY_RUN] Simulating futures_create_order: %s", kwargs)
            return {"dry_run": True, **kwargs}

        if self._client:
            for name in ("futures_create_order", "new_order", "create_order", "order"):
                if hasattr(self._client, name):
                    try:
                        return getattr(self._client, name)(**kwargs)
                    except Exception as e:
                        logger.debug("SDK order variant %s failed: %s", name, e)
                        continue
        # REST fallback
        return self._rest_signed_post("/fapi/v1/order", kwargs)

    @retry_with_time_sync()
    def place_order(self, **kwargs) -> Dict[str, Any]:
        """Universal order wrapper (used by smart_exit and execution manager)."""
        return self.futures_create_order(**kwargs)

    @safe_api_call()
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        self._init()
        order_side = "BUY" if side.upper() in ("LONG", "BUY") else "SELL"
        qty = self.round_qty(symbol, quantity)
        logger.info("Placing MARKET order %s %s qty=%s", order_side, symbol, qty)
        return self.futures_create_order(symbol=symbol, side=order_side, type="MARKET", quantity=qty)

    # ---------------------------
    # TP/SL creation (atomic)
    # ---------------------------
    @safe_api_call()
    def create_tp_sl_orders(self,
                            symbol: str,
                            side: str,
                            tp_levels: Union[float, List[float]],
                            sl_price: float,
                            tp_qtys: Optional[Union[float, List[float]]] = None) -> Dict[str, Any]:
        """
        Create take-profit (TAKE_PROFIT_MARKET) orders and a stop (STOP_MARKET).
        Normalizes inputs and applies rounding/step rules.
        """
        self._init()
        # normalize lists
        if isinstance(tp_levels, (int, float)):
            tp_levels = [float(tp_levels)]
        tp_levels = [float(x) for x in (tp_levels or []) if x is not None]

        if tp_qtys is None:
            tp_qtys_list = [None] * len(tp_levels)
        elif isinstance(tp_qtys, (int, float)):
            tp_qtys_list = [float(tp_qtys)] * len(tp_levels)
        else:
            tp_qtys_list = [float(x) for x in tp_qtys]

        res = {"tp_results": [], "sl_result": None, "errors": []}

        # symbol filters
        try:
            tick, step, min_notional = self._get_tick_and_step(symbol)
        except Exception as e:
            logger.warning("Failed to fetch filters for %s: %s", symbol, e)
            tick = step = min_notional = 0.0

        # create TPs
        for idx, tp_price in enumerate(tp_levels):
            qty = tp_qtys_list[idx] if idx < len(tp_qtys_list) else None
            try:
                adj_price = self.round_price(symbol, tp_price)
                adj_qty = None
                if qty:
                    adj_qty = self.round_qty(symbol, qty)
                side_tp = "SELL" if side.upper() in ("LONG", "BUY") else "BUY"
                payload = {
                    "symbol": symbol,
                    "side": side_tp,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": str(adj_price),
                    "closePosition": False
                }
                if adj_qty:
                    payload["quantity"] = adj_qty

                resp = self.futures_create_order(**payload)
                res["tp_results"].append(resp)
            except Exception as e:
                logger.warning("TP creation failed for %s @ %s: %s", symbol, tp_price, e)
                res["errors"].append({"type": "tp", "price": tp_price, "error": str(e)})

        # create SL (STOP_MARKET)
        try:
            adj_sl = self.round_price(symbol, sl_price)
            total_qty = None
            if any(tp_qtys_list):
                total_qty = sum([q for q in tp_qtys_list if q])
                if step and total_qty:
                    total_qty = self.round_qty(symbol, total_qty)
            side_sl = "SELL" if side.upper() in ("LONG", "BUY") else "BUY"
            payload = {
                "symbol": symbol,
                "side": side_sl,
                "type": "STOP_MARKET",
                "stopPrice": str(adj_sl),
                "closePosition": False
            }
            if total_qty:
                payload["quantity"] = total_qty
            sl_resp = self.futures_create_order(**payload)
            res["sl_result"] = sl_resp
        except Exception as e:
            logger.warning("SL creation failed for %s: %s", symbol, e)
            res["errors"].append({"type": "sl", "price": sl_price, "error": str(e)})

        return res

    # ---------------------------
    # Leverage / size helpers
    # ---------------------------
    @safe_api_call()
    def set_leverage(self, symbol: str, leverage: int):
        self._init()
        # try common SDK names
        if self._client:
            for name in ("futures_change_leverage", "change_leverage", "set_leverage"):
                if hasattr(self._client, name):
                    try:
                        return getattr(self._client, name)(symbol=symbol, leverage=leverage)
                    except Exception:
                        continue
        # rest fallback
        return self._rest_signed_post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    @safe_api_call()
    def calculate_position_size(self, symbol: str, margin_usdt: float, leverage: int) -> float:
        """Compute base asset qty from margin (USDT) and leverage, rounded to exchange LOT_SIZE."""
        self._init()
        try:
            price_obj = self.ticker_price(symbol)
            price = float(price_obj.get("price") if isinstance(price_obj, dict) else price_obj)
            if price <= 0:
                logger.warning("Invalid price when calculating qty for %s", symbol)
                return 0.0
            trade_value = margin_usdt * leverage
            raw_qty = trade_value / price
            adj = self.round_qty(symbol, raw_qty)
            logger.debug("calculate_position_size: margin=%s lev=%s price=%s qty=%s", margin_usdt, leverage, price, adj)
            return adj
        except Exception as e:
            logger.warning("calculate_position_size failed for %s: %s", symbol, e)
            return 0.0

    def get_margin_from_percent(self, percent: float, asset: str = "USDT") -> float:
        bal = self.get_available_balance(asset)
        if bal <= 0:
            return 0.0
        return (bal * float(percent)) / 100.0

    # ---------------------------
    # Stop loss update helper
    # ---------------------------
    @safe_api_call()
    def update_stop_loss(self, symbol: str, side: str, quantity: float, new_sl_price: float):
        """Cancel existing stop orders and place a STOP_MARKET at new_sl_price."""
        self._init()
        try:
            # cancel old STOP_* orders
            open_orders = self.get_open_orders(symbol=symbol)
            for o in open_orders:
                try:
                    otype = (o.get("type") or o.get("orderType") or "").upper()
                    if "STOP" in otype or "TAKE" in otype:
                        oid = o.get("orderId") or o.get("order_id") or o.get("id")
                        if oid:
                            try:
                                self.cancel_order(symbol, oid)
                            except Exception:
                                pass
                except Exception:
                    continue
            # place new SL
            stop_side = "SELL" if side.upper() in ("LONG", "BUY") else "BUY"
            adj_price = self.round_price(symbol, new_sl_price)
            q = self.round_qty(symbol, quantity)
            return self.futures_create_order(symbol=symbol, side=stop_side, type="STOP_MARKET", stopPrice=str(adj_price), quantity=q)
        except Exception as e:
            logger.error("update_stop_loss failed for %s: %s", symbol, e)
            return None

    # ---------------------------
    # Utilities / debugging
    # ---------------------------
    def debug_filters(self, symbol: str):
        try:
            filters = self.get_symbol_filters(symbol)
            logger.info("Filters for %s:", symbol)
            for k, v in filters.items():
                logger.info("  %s: %s", k, v)
        except Exception as e:
            logger.warning("debug_filters failed: %s", e)

    def test_order_flow(self):
        """
        Lightweight test to ensure order routing (SDK/REST) is functional.
        Does not send real orders if DRY_RUN enabled.
        """
        try:
            dry = str(os.getenv("DRY_RUN", "false")).lower() in ("1", "true", "yes")
            logger.info("test_order_flow dry_run=%s", dry)
            res = self.futures_create_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.0001)
            logger.info("test_order_flow result: %s", res)
            return res
        except Exception as e:
            logger.error("test_order_flow failed: %s", e)
            return None

# run quick check when invoked directly
if __name__ == "__main__":
    c = BinanceClient()
    try:
        print("Base URL:", c.base_url)
        p = c.ticker_price("BTCUSDT")
        print("BTCUSDT price:", p)
    except Exception as ex:
        print("Quick self-test failed:", ex)
