# src/binance_client.py
"""
Patched BinanceClient wrapper
- preserves your retry decorator and behavior
- exposes .client property for compatibility
- robust balance fetching + REST fallback
- writes ACCOUNT_BALANCE to .env so future runs see it automatically
"""

import os
import time
import functools
import logging
from typing import Any, Dict, List, Optional, Union
from requests.exceptions import ConnectionError, ReadTimeout, RequestException
import requests
import hmac
import hashlib
from urllib.parse import urlencode
from pathlib import Path
import json

from config import USE_TESTNET

# ==========================================================
# ✅ Universal Binance Futures SDK Import Compatibility
# ==========================================================
UMFutures = None

"""if UMFutures is None:
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "binance-futures-connector"], check=True)
        from binance.um_futures import UMFutures
        logging.info("✅ Auto-installed and loaded binance-futures-connector")
    except Exception as e:
        logging.warning(f"⚠️ Auto-install failed, REST-only mode active: {e}")"""


try:
    # Preferred modern connector (binance-futures-connector)
    from binance.um_futures import UMFutures
    logging.info("✅ Using UMFutures from binance-futures-connector")
except ImportError:
    try:
        # Legacy "binance-futures"
        from binance.futures import Futures as UMFutures
        logging.info("✅ Using Futures from binance-futures")
    except ImportError:
        try:
            # Older "python-binance" (used in some forks)
            from binance.client import Client as UMFutures
            logging.info("✅ Using Client from python-binance (limited support)")
        except ImportError:
            UMFutures = None
            logging.warning("⚠️ No Binance SDK found. Will use REST fallback only.")


#logger = logging.getLogger("binance_client")
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
try:
    from binance.error import ClientError as BinanceAPIException
except Exception:
    BinanceAPIException = Exception  # fallback for SDK variants




# ======================================================================
# Retry Decorator
# ======================================================================
def _safe_api_call(max_retries: int = 5, backoff: float = 2.0):
    """Generic retry decorator for Binance API calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, ReadTimeout) as e:
                    wait = backoff ** attempt
                    logger.warning("[%s] Network issue: %s (retry %d/%d) -> waiting %.1fs",
                                   func.__name__, e, attempt, max_retries, wait)
                    time.sleep(wait)
                    last_exc = e
                    continue
                except RequestException as e:
                    logger.error("[%s] RequestException (no retry): %s", func.__name__, e)
                    last_exc = e
                    break
                except Exception as e:
                    estr = str(e).lower()
                    if "429" in estr or "too many requests" in estr:
                        wait = 5 * (backoff ** attempt)
                        logger.warning("[%s] Rate limit hit (retry %d/%d) -> waiting %.1fs",
                                       func.__name__, attempt, max_retries, wait)
                        time.sleep(wait)
                        last_exc = e
                        continue
                    logger.warning("[%s] Unhandled exception: %s", func.__name__, e)
                    last_exc = e
                    break
            raise Exception(f"[FATAL] {func.__name__} failed after {max_retries} retries. Last error: {last_exc}")
        return wrapper
    return decorator


# ======================================================================
# BinanceClient
# ======================================================================
class BinanceClient:
    """
    Unified Binance Futures client with:
    - Multi-TP/SL creation
    - Exchange info caching
    - Symbol filter helpers
    - Balance / leverage / position utilities
    - Signed REST fallback with time sync
    """

    _symbol_info_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 use_testnet: bool = USE_TESTNET):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.use_testnet = use_testnet
        self._client: Optional[UMFutures] = None
        self._base_url = "https://testnet.binancefuture.com" if self.use_testnet else "https://fapi.binance.com"
        self._session: Optional[requests.Session] = None
        self.time_offset: int = 0
        #self.place_order = self.place_order

    # ---------- compatibility property ----------
    @property
    def client(self) -> Optional[UMFutures]:
        """Expose 'client' to be compatible with other modules expecting BinanceClient().client"""
        if self._client is None:
            try:
                self._init()
            except Exception as e:
                logger.warning("Client initialization failed: %s", e)
                return None
        return self._client

    # ---------------- Initialization ----------------
    def _init(self):
        if UMFutures is None:
            raise ImportError("binance.um_futures not installed. Run: pip install binance-futures")
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing BINANCE_API_KEY / BINANCE_API_SECRET in environment or init args.")
        if not self._client:
            self._client = UMFutures(key=self.api_key, secret=self.api_secret, base_url=self._base_url)
            logger.info("Binance UMFutures client initialized (testnet=%s).", self.use_testnet)
        if not self._session:
            self._session = requests.Session()

    # ---------------- Time sync and signed params ----------------
    def _sync_time_offset(self):
        """Synchronize local time offset (ms) with Binance server time."""
        try:
            # attempt SDK call first
            try:
                self._init()
                server_time_resp = self._client.time()
                if isinstance(server_time_resp, dict) and "serverTime" in server_time_resp:
                    server_time = int(server_time_resp["serverTime"])
                elif isinstance(server_time_resp, (int, float)):
                    server_time = int(server_time_resp)
                else:
                    raise Exception("UMFutures.time returned unexpected format")
            except Exception:
                # REST fallback
                url = f"{self._base_url}/fapi/v1/time"
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                server_time = int(resp.json().get("serverTime", int(time.time() * 1000)))

            local_ms = int(time.time() * 1000)
            self.time_offset = server_time - local_ms
            logger.info("[Time Sync] serverTime=%d local_ms=%d offset=%dms", server_time, local_ms, self.time_offset)
        except Exception as e:
            logger.warning("[Time Sync] Failed to sync time: %s", e)

    def _signed_params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Attach signed params with an automatically refreshed server timestamp."""
        import time, logging
        logger = logging.getLogger(__name__)

        if params is None:
            params = {}

        now_ms = int(time.time() * 1000)

        try:
            # Refresh offset if missing or stale (older than 30 seconds)
            if not hasattr(self, "time_offset") or not hasattr(self, "_last_time_sync") \
            or abs(now_ms - getattr(self, "_last_time_sync", 0)) > 30_000:
                self._sync_time_offset()
                self._last_time_sync = now_ms
                logger.debug("[TimeSync] Refreshed offset for signing")
        except Exception as e:
            logger.debug("[TimeSync] Failed to refresh offset: %s", e)

        # Use adjusted timestamp
        ts = now_ms + int(getattr(self, "time_offset", 0))
        params.update({"timestamp": ts, "recvWindow": 60000})
        return params


    def _rest_signed_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 5
    ) -> Dict[str, Any]:
        """
        Robust signed GET request to Binance:
        - uses _signed_params() to handle timestamp + recvWindow
        - signs with HMAC-SHA256
        - raises for HTTP errors
        """
        import requests, hmac, hashlib
        from urllib.parse import urlencode

        if params is None:
            params = {}

        signed = self._signed_params(params.copy())
        query_string = urlencode(signed)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        signed["signature"] = signature

        url = f"{self._base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key, "Content-Type": "application/x-www-form-urlencoded"}
        sess = self._session or requests

        resp = sess.get(url, headers=headers, params=signed, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # ==================================================================
    # Market & Symbol Info
    # ==================================================================
    @_safe_api_call()
    def get_klines(self, symbol: str, interval: str, limit: int = 100, end_time: Optional[int] = None) -> List[List[Any]]:
        """
        Fetch historical klines with automatic chunking for large requests (>1500 limit).
        """
        # initialize client (if available)
        try:
            self._init()
        except Exception:
            # if SDK missing, attempt REST /fapi/v1/klines fallback
            pass

        MAX_LIMIT = 1500
        all_klines: List[List[Any]] = []
        import math

        if end_time is None:
            end_time = int(time.time() * 1000)

        remaining = limit
        chunk_count = max(1, math.ceil(limit / MAX_LIMIT))
        logger.info(f"[get_klines] Fetching {limit} candles for {symbol} ({interval}) in {chunk_count} chunks...")

        for chunk in range(chunk_count):
            fetch_limit = min(remaining, MAX_LIMIT)
            params = {"symbol": symbol.upper(), "interval": interval, "limit": fetch_limit, "endTime": end_time}
            data = None
            try:
                if self._client:
                    # UMFutures klines signature differs per wrapper; attempt a few methods
                    try:
                        data = self._client.klines(**params)
                    except Exception:
                        data = self._client.futures_klines(symbol=params["symbol"], interval=params["interval"], limit=fetch_limit, endTime=end_time)
                else:
                    # REST fallback
                    url = f"{self._base_url}/fapi/v1/klines"
                    resp = (self._session or requests).get(url, params=params, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                logger.error(f"[get_klines] Error fetching chunk {chunk+1}/{chunk_count}: {e}")
                break

            if not data:
                logger.warning(f"[get_klines] No data returned for chunk {chunk+1}/{chunk_count}")
                break

            # prepend older chunk to beginning (so final slice returns last 'limit' items)
            all_klines = data + all_klines
            end_time = int(data[0][0]) - 1
            remaining -= fetch_limit

            logger.info(f"[get_klines] Chunk {chunk+1}/{chunk_count}: got {len(data)} candles, remaining={remaining}")

            if len(data) < fetch_limit:
                break

            time.sleep(0.25)

        logger.info(f"[get_klines] ✅ Completed. Total candles fetched: {len(all_klines)}")
        return all_klines[-limit:]

    @_safe_api_call()
    def exchange_info(self):
        self._init()
        return self._client.exchange_info()

    @_safe_api_call()
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Return exchange_info for a single symbol (cached)."""
        self._init()
        symbol = symbol.upper()
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        info = self._client.exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                self._symbol_info_cache[symbol] = s
                return s
        raise ValueError(f"Symbol {symbol} not found in exchange_info")

    @_safe_api_call()
    def get_symbol_filters(self, symbol: str) -> dict:
        info = self.get_symbol_info(symbol)
        return {f["filterType"]: f for f in info.get("filters", [])}

    # ==================================================================
    # Price / Balance / Positions
    # ==================================================================
    @_safe_api_call()
    def ticker_price(self, symbol: str) -> Dict[str, Any]:
        self._init()
        # try SDK variant first
        for m in ("ticker_price", "get_symbol_ticker", "symbol_ticker"):
            if hasattr(self._client, m):
                try:
                    return getattr(self._client, m)(symbol=symbol)
                except Exception:
                    continue
        # fallback REST
        url = f"{self._base_url}/fapi/v1/ticker/price"
        resp = (self._session or requests).get(url, params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_latest_price(self, symbol: str) -> float:
        """
        Returns the latest market price for a given symbol.
        Wrapper for ticker_price(), compatible with external modules.
        """
        try:
            data = self.ticker_price(symbol)
            if isinstance(data, dict):
                price_str = data.get("price")
                if price_str is not None:
                    return float(price_str)
                else:
                    logger.warning(f"⚠️ get_latest_price() returned None for {symbol}")
                    return 0.0
            elif isinstance(data, (float, int, str)):
                return float(data)
            else:
                logger.warning(f"⚠️ Unexpected price type for {symbol}: {type(data)}")
                return 0.0
        except Exception as e:
            logger.warning(f"⚠️ get_latest_price() failed for {symbol}: {e}")
            return 0.0



    @_safe_api_call()
    def get_mark_price(self, symbol: str) -> float:
        self._init()
        try:
            data = self._client.mark_price(symbol=symbol)
            if isinstance(data, dict):
                return float(data.get("markPrice", 0.0))
            # if SDK returns list or raw, try to parse
            return float(data)
        except Exception:
            # REST fallback
            url = f"{self._base_url}/fapi/v1/premiumIndex"
            resp = (self._session or requests).get(url, params={"symbol": symbol}, timeout=5)
            resp.raise_for_status()
            j = resp.json()
            return float(j.get("markPrice", j.get("lastPrice", 0.0)))

    @_safe_api_call()
    def balance(self):
        self._init()
        # try a few SDK methods
        for m in ("balance", "futures_account_balance", "get_account"):
            if hasattr(self._client, m):
                try:
                    return getattr(self._client, m)()
                except Exception:
                    continue
        # REST fallback
        resp = self._rest_signed_get("/fapi/v2/balance")
        if resp.status_code == 200:
            return resp.json()
        resp.raise_for_status()
        return []

    @_safe_api_call()
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Universal open-orders retriever with REST fallback."""
        self._init()
        possible_methods = ["get_open_orders", "futures_get_open_orders", "open_orders", "get_all_open_orders", "query_open_orders"]
        for mname in possible_methods:
            if hasattr(self._client, mname):
                try:
                    method = getattr(self._client, mname)
                    res = method(symbol=symbol) if symbol else method()
                    if isinstance(res, list):
                        return res
                except Exception as e:
                    if "orderId" in str(e):
                        continue
                    logger.debug("[%s] variant failed: %s", mname, e)

        try:
            endpoint = "/fapi/v1/openOrders"
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol.upper()
            resp = self._rest_signed_get(endpoint, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            else:
                logger.warning("get_open_orders REST fallback HTTP %s: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.warning("get_open_orders REST fallback failed: %s", e)
        return []

    def cancel_order(self, symbol: str, orderId: int):
        """Cancel a specific order by ID."""
        self._init()
        try:
            return self._client.cancel_order(symbol=symbol, orderId=orderId)
        except Exception as e:
            logger.error(f"[cancel_order] Failed to cancel {symbol} {orderId}: {e}")
            raise

    @_safe_api_call()
    def get_position_risk(self, symbol: Optional[str] = None):
        self._init()
        if symbol:
            try:
                return self._client.get_position_risk(symbol=symbol)
            except Exception:
                # fallback naming
                return self._client.get_position_risk()
        return self._client.get_position_risk()

    def get_position(self, symbol: str) -> Dict[str, Any]:
        self._init()
        try:
            positions = self._client.get_position_risk()
            for p in positions:
                if p.get("symbol") == symbol:
                    return p
            return {}
        except Exception:
            try:
                info = self._client.position_information(symbol=symbol)
                if isinstance(info, list) and info:
                    return info[0]
                if isinstance(info, dict):
                    return info
            except Exception:
                pass
            return {}

    @_safe_api_call()
    def get_all_positions(self) -> list:
        self._init()
        try:
            positions = self._client.get_position_risk()
        except Exception:
            positions = self._client.position_information() if hasattr(self._client, "position_information") else []
        return [p for p in positions or [] if abs(float(p.get("positionAmt", 0))) > 0]

    @_safe_api_call()
    def has_open_position(self, symbol: str) -> bool:
        positions = self.get_position_risk(symbol)
        return any(abs(float(p.get("positionAmt", 0))) > 0 for p in positions)

    # ==================================================================
    # Balance helpers (new + persist)
    # ==================================================================
    def _find_env_path(self) -> Optional[Path]:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for p in candidates:
            if p.exists():
                return p
        return Path.cwd() / ".env"


        ...
    def has_open_position(self, symbol):
        ...
        return False

    # ------------------------------------------------------
    # Attribute forwarding to underlying client
    # ------------------------------------------------------
    def __getattr__(self, name):
        """
        Forward unknown attribute access to the underlying client instance.
        This allows existing code to call methods like `futures_create_order`,
        `ticker_price`, `get_symbol_info`, etc., without changing call sites.
        """
        self._init()
        return getattr(self._client, name)



    def _persist_account_balance_to_env(self, balance: float, env_path: Optional[Path] = None):
        try:
            env_path = env_path or self._find_env_path()
            lines = []
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            def set_line(ls, key, value):
                import re
                pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
                for i, L in enumerate(ls):
                    if pat.match(L):
                        ls[i] = f"{key}={value}\n"
                        return
                ls.append(f"{key}={value}\n")

            set_line(lines, "ACCOUNT_BALANCE", f"{balance:.8f}")
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            logger.info("✅ Persisted ACCOUNT_BALANCE=%.8f to %s", balance, env_path.resolve())
            return True
        except Exception as e:
            logger.warning("Failed to persist ACCOUNT_BALANCE: %s", e)
            return False

    # ==================================================================
    # Balance Fetch with Auto-Persist
    # ==================================================================
    @_safe_api_call()
    def get_futures_account_balance(self, asset: str = "USDT") -> Optional[float]:
        """
        Returns the available futures balance in USDT (auto-persisted to .env).
        """
        balance = None
        try:
            self._init()
            for m in ("futures_account_balance", "balance", "get_balance"):
                if hasattr(self._client, m):
                    try:
                        res = getattr(self._client, m)()
                        if isinstance(res, list):
                            for b in res:
                                if b.get("asset") == asset:
                                    balance = float(
                                        b.get("availableBalance")
                                        or b.get("balance")
                                        or b.get("crossWalletBalance")
                                        or 0
                                    )
                                    break
                        elif isinstance(res, dict) and asset in res:
                            balance = float(res.get(asset))
                        if balance is not None:
                            break
                    except Exception:
                        continue

            # REST fallback
            if balance is None:
                resp = self._rest_signed_get("/fapi/v2/balance")
                if resp.status_code == 200:
                    for b in resp.json():
                        if b.get("asset") == asset:
                            balance = float(b.get("availableBalance", b.get("balance", 0)))
                            break

            # Fallback to /account
            if balance is None:
                resp = self._rest_signed_get("/fapi/v1/account")
                if resp.status_code == 200:
                    acc = resp.json()
                    for b in acc.get("assets", []):
                        if b.get("asset") == asset:
                            balance = float(b.get("availableBalance", b.get("walletBalance", 0)))
                            break
        except Exception as e:
            logger.warning("⚠️ Error fetching futures balance: %s", e)
            balance = None

        # Auto persist balance
        if balance is not None:
            self._persist_account_balance_to_env(balance)
        else:
            logger.warning("Could not auto-persist balance — value is None")

        return balance

    def fetch_and_persist_balance(self, asset: str = "USDT", env_path: Optional[Path] = None) -> Optional[float]:
        """
        Convenience: fetch the available balance and persist to .env as ACCOUNT_BALANCE.
        Returns the balance or None.
        """
        try:
            bal = self.get_futures_account_balance(asset=asset)
            if bal is None:
                logger.warning("Could not fetch balance to persist.")
                return None
            persisted = self._persist_account_balance_to_env(bal, env_path=env_path)
            if not persisted:
                logger.warning("Failed to persist ACCOUNT_BALANCE to .env.")
            return bal
        except Exception as e:
            logger.warning("fetch_and_persist_balance failed: %s", e)
            return None

    # ==================================================================
    # Rounding Helpers
    # ==================================================================
    def _get_tick_and_step(self, symbol: str):
        info = self.get_symbol_info(symbol)
        tick, step, notional = 0.0, 0.0, 0.0
        for f in info.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick = float(f.get("tickSize", 0))
            elif f.get("filterType") == "LOT_SIZE":
                step = float(f.get("stepSize", 0))
            elif f.get("filterType") == "MIN_NOTIONAL":
                notional = float(f.get("minNotional", f.get("notional", 0)))
        return tick, step, notional

    def _adjust_price(self, symbol: str, price: float) -> float:
        try:
            tick, _, _ = self._get_tick_and_step(symbol)
            if tick and price:
                q = int(price // tick)
                return round(q * tick, max(0, len(str(tick).split(".")[1].rstrip("0"))))
            return price
        except Exception:
            return price

    def _adjust_quantity(self, symbol: str, qty: float) -> float:
        try:
            _, step, _ = self._get_tick_and_step(symbol)
            if step and qty:
                q_mult = int(qty // step)
                return round(max(q_mult * step, step), 8)
            return round(qty, 8)
        except Exception:
            return round(qty, 8)

    # ==================================================================
    # Market Order & TP/SL
    # ==================================================================
    @_safe_api_call()
    def place_market_order(self, symbol: str, side: str, quantity: float):
        self._init()
        order_side = "BUY" if side.upper() == "LONG" else "SELL"
        qty = self._adjust_quantity(symbol, quantity)
        logger.info("Rounded quantity for %s: %s", symbol, qty)
        # try different SDK order method names for compatibility
        for m in ("new_order", "futures_create_order", "create_order"):
            if hasattr(self._client, m):
                try:
                    return getattr(self._client, m)(symbol=symbol, side=order_side, type="MARKET", quantity=qty)
                except Exception:
                    continue
        # last-resort REST (not signed here; production should use signed)
        url = f"{self._base_url}/fapi/v1/order"
        payload = {"symbol": symbol, "side": order_side, "type": "MARKET", "quantity": qty}
        resp = (self._session or requests).post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    
        # === Universal order placement (for TP/SL compatibility) ===
    def place_order(self, **kwargs) -> dict:
        """
        Generic order wrapper used by ExecutionManager.create_tp_sl_orders().
        Accepts kwargs like:
            symbol='BTCUSDT', side='BUY', type='STOP_MARKET',
            stopPrice='112300', quantity='0.001', reduceOnly=True

        Automatically tries multiple compatible SDK variants, then REST fallback.
        """
        self._init()
        possible_methods = ["new_order", "create_order", "futures_create_order"]

        for m in possible_methods:
            if hasattr(self._client, m):
                try:
                    logger.debug("[place_order] trying variant: %s", m)
                    return getattr(self._client, m)(**kwargs)
                except Exception as e:
                    logger.debug("[place_order] variant %s failed: %s", m, e)
                    continue

        # --- REST fallback ---
        try:
            logger.debug("[place_order] using REST fallback")
            if hasattr(self.client, "_request_futures_api"):
                return self._request_futures_api('POST', 'order', True, data=kwargs)
            else:
                return self._rest_signed_post("/fapi/v1/order", kwargs)

        except Exception as e:
            logger.error("[place_order] REST fallback failed: %s", e)
            raise


    @_safe_api_call()
    def create_tp_sl_orders(self, symbol: str, side: str,
                            tp_levels: Union[float, List[float]],
                            sl_price: float,
                            tp_qtys: Optional[Union[float, List[float]]] = None) -> Dict[str, Any]:
        """Create TAKE_PROFIT_MARKET & STOP_MARKET orders."""
        self._init()
        if isinstance(tp_levels, (int, float)):
            tp_levels = [float(tp_levels)]
        else:
            tp_levels = [float(x) for x in tp_levels]

        if tp_qtys is None:
            tp_qtys = [None] * len(tp_levels)
        elif isinstance(tp_qtys, (int, float)):
            tp_qtys = [float(tp_qtys)] * len(tp_levels)
        else:
            tp_qtys = [float(x) for x in tp_qtys]

        results = {"tp_results": [], "sl_result": None, "errors": []}
        try:
            tick, step, min_notional = self._get_tick_and_step(symbol)
        except Exception as e:
            logger.warning("Symbol filters fetch failed for %s: %s", symbol, e)
            tick = step = min_notional = 0.0

        # --- TP creation loop ---
        for tp_price, qty in zip(tp_levels, tp_qtys):
            try:
                adj_price = tp_price
                if tick:
                    adj_price = float(int(tp_price // tick) * tick)
                    if adj_price == tp_price:
                        adj_price = round(adj_price + tick, 8)
                adj_qty = qty if qty is not None else 0.0
                if adj_qty and step:
                    adj_qty = float(int(adj_qty // step) * step) if adj_qty >= step else step
                if adj_qty == 0.0:
                    adj_qty = None  # let the caller decide or set from position

                side_tp = "SELL" if side.upper() == "LONG" else "BUY"
                # try SDK create
                if hasattr(self._client, "new_order"):
                    resp = self._client.new_order(
                        symbol=symbol,
                        side=side_tp,
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=str(adj_price),
                        closePosition=False,
                        quantity=adj_qty,
                    )
                else:
                    # fallback REST signed: build payload and post
                    payload = {"symbol": symbol, "side": side_tp, "type": "TAKE_PROFIT_MARKET", "stopPrice": str(adj_price)}
                    if adj_qty:
                        payload["quantity"] = adj_qty
                    resp = self._rest_signed_post("/fapi/v1/order", payload)
                results["tp_results"].append(resp)
            except Exception as e:
                logger.warning("TP creation error for %s at %s: %s", symbol, tp_price, e)
                results["errors"].append({"type": "tp", "price": tp_price, "qty": qty, "error": str(e)})

        # --- SL creation ---
        try:
            adj_sl = float(int(sl_price // tick) * tick) if tick else sl_price
            total_qty = sum([q for q in (tp_qtys or []) if q]) if tp_qtys else None
            adj_total_qty = float(int(total_qty // step) * step) if total_qty and step else total_qty
            side_sl = "SELL" if side.upper() == "LONG" else "BUY"
            if hasattr(self._client, "new_order"):
                sl_resp = self._client.new_order(
                    symbol=symbol,
                    side=side_sl,
                    type="STOP_MARKET",
                    stopPrice=str(adj_sl),
                    closePosition=False,
                    quantity=adj_total_qty,
                )
            else:
                payload = {"symbol": symbol, "side": side_sl, "type": "STOP_MARKET", "stopPrice": str(adj_sl)}
                if adj_total_qty:
                    payload["quantity"] = adj_total_qty
                sl_resp = self._rest_signed_post("/fapi/v1/order", payload)
            results["sl_result"] = sl_resp
        except Exception as e:
            logger.warning("SL creation error for %s: %s", symbol, e)
            results["errors"].append({"type": "sl", "price": sl_price, "error": str(e)})

        return results
    

        # ============================================================
    # 🔹 Compatibility: futures_create_order Alias
    # ============================================================
    def futures_create_order(self, **kwargs) -> dict:
        """
        Compatibility alias for futures_create_order() used by ExecutionManager & SmartExitManager.
        Falls back to create_order() or new_order() automatically.
        """
        self._init()

        # --- Dry-run simulation ---
        if os.getenv("DRY_RUN", "False").lower() == "true":
            logger.info("[DRY_RUN] Simulated futures_create_order: %s", kwargs)
            return {
                "symbol": kwargs.get("symbol"),
                "side": kwargs.get("side"),
                "type": kwargs.get("type"),
                "quantity": kwargs.get("quantity"),
                "price": kwargs.get("price"),
                "dry_run": True,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }

        # --- Try native SDK futures order ---
        if hasattr(self._client, "futures_create_order"):
            try:
                logger.debug("[futures_create_order] using native SDK futures_create_order()")
                return self._client.futures_create_order(**kwargs)
            except BinanceAPIException as e:
                logger.error("[futures_create_order] BinanceAPIException: %s", e)
            except Exception as e:
                logger.debug("[futures_create_order] SDK futures_create_order failed: %s", e)

        # --- Fallback: create_order() / new_order() ---
        for m in ("create_order", "new_order"):
            if hasattr(self._client, m):
                try:
                    logger.debug("[futures_create_order] fallback variant: %s", m)
                    return getattr(self._client, m)(**kwargs)
                except Exception as e:
                    logger.debug("[futures_create_order] fallback %s failed: %s", m, e)

        # --- Final REST fallback ---
        try:
            logger.debug("[futures_create_order] using REST fallback")
            if hasattr(self.client, "_request_futures_api"):
                return self._request_futures_api('POST', 'order', True, data=kwargs)
            else:
                return self._rest_signed_post("/fapi/v1/order", kwargs)

        except Exception as e:
            logger.error("[futures_create_order] REST fallback failed: %s", e)
            raise
    

    # ============================================================
    # 🧪 SELF-TEST SNIPPET
    # ============================================================
    def test_order_flow(self):
        """
        Simple local test to confirm SDK vs REST behavior.
        """
        logger.info("[TEST] Starting BinanceClient order path test...")
        try:
            result = self.futures_create_order(
                symbol="BTCUSDT",
                side="BUY",
                type="MARKET",
                quantity=0.001
            )
            logger.info("[TEST] Order path succeeded: %s", result)
        except Exception as e:
            logger.error("[TEST] Order path failed: %s", e)

    # REST POST helper (signed)
    def _rest_signed_post(self, endpoint: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Dict[str, Any]:
        """Perform a signed POST request to Binance Futures REST endpoint, with full debug and error handling."""
        import requests, hmac, hashlib, time, logging
        from urllib.parse import urlencode

        logger = logging.getLogger(__name__)
        if payload is None:
            payload = {}

        try:
            signed = self._signed_params(payload.copy())
            query_string = urlencode(signed)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            signed["signature"] = signature

            url = f"{self._base_url}{endpoint}"
            headers = {
                "X-MBX-APIKEY": self.api_key,
                "Content-Type": "application/json",
            }

            sess = self._session or requests
            logger.debug(f"[REST] POST {url} payload={signed}")

            resp = sess.post(url, headers=headers, json=signed, timeout=timeout)
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                # Log the error response body if available
                msg = getattr(e.response, "text", "")
                logger.error(f"[REST ERROR] {e} | body={msg}")
                raise  # rethrow to propagate upward

            return resp.json()

        except Exception as e:
            logger.exception(f"[REST POST FAILED] {endpoint} | {e}")
            raise


    # ==================================================================
    # Leverage / Balance / Position Size
    # ==================================================================
    @_safe_api_call()
    def set_leverage(self, symbol: str, leverage: int):
        self._init()
        try:
            return self._client.change_leverage(symbol=symbol, leverage=leverage)
        except Exception:
            return self._client.set_leverage(symbol=symbol, leverage=leverage)

    @_safe_api_call()
    def get_balance_summary(self, asset: str = "USDT") -> Dict[str, float]:
        self._init()
        try:
            balances = None
            try:
                balances = self._client.balance()
            except Exception as e:
                logger.debug("SDK balance call failed: %s", e)

            if isinstance(balances, list):
                for b in balances:
                    if b.get("asset") == asset:
                        return {
                            "asset": asset,
                            "available": float(b.get("availableBalance", b.get("balance", 0))),
                            "wallet": float(b.get("crossWalletBalance", b.get("walletBalance", 0)))
                        }

            # REST fallback
            resp = self._rest_signed_get("/fapi/v2/balance")
            if resp.status_code == 200:
                for b in resp.json():
                    if b.get("asset") == asset:
                        return {
                            "asset": asset,
                            "available": float(b.get("availableBalance", b.get("balance", 0))),
                            "wallet": float(b.get("crossWalletBalance", b.get("walletBalance", 0)))
                        }

            return {"asset": asset, "available": 0.0, "wallet": 0.0}

        except Exception as e:
            logger.warning("get_balance_summary() failed: %s", e)
            return {"asset": asset, "available": 0.0, "wallet": 0.0}


    @_safe_api_call()
    def get_available_balance(self, asset: str = "USDT") -> float:
        """Return only the available balance in float (safe wrapper)."""
        summary = self.get_balance_summary(asset)
        return float(summary.get("available", 0.0))

    @_safe_api_call()
    def get_margin_from_percent(self, percent: float, asset: str = "USDT") -> float:
        """
        Compute dynamic margin (USDT) from percent of available balance.
        """
        try:
            self._init()
            balance = self.get_available_balance(asset)
            if balance <= 0:
                logger.warning("Balance 0 or unavailable for %s", asset)
                return 0.0
            margin = (balance * float(percent)) / 100.0
            logger.info("Dynamic margin from %.2f%% of %.2f USDT = %.2f", percent, balance, margin)
            return margin
        except Exception as e:
            logger.warning("Error calculating margin from percent: %s", e)
            return 0.0

    @_safe_api_call()
    def calculate_position_size(self, symbol: str, margin_usdt: float, leverage: int) -> float:
        """
        Compute position size (qty) based on margin and leverage.
        Ensures rounding to exchange precision.
        """
        self._init()
        try:
            price_data = self.ticker_price(symbol)
            price = float(price_data.get("price") if isinstance(price_data, dict) else price_data)
            if price <= 0:
                logger.warning("Invalid price for %s", symbol)
                return 0.0

            trade_value = margin_usdt * leverage
            qty = trade_value / price
            rounded = self._adjust_quantity(symbol, qty)
            logger.info(
                "Position size for %s: qty=%.6f (Margin %.2f USDT, Lev %dx, Price %.2f)",
                symbol, rounded, margin_usdt, leverage, price,
            )
            return rounded
        except Exception as e:
            logger.warning("calculate_position_size failed for %s: %s", symbol, e)
            return 0.0

    # ==================================================================
    # Stop-Loss Update
    # ==================================================================
    @_safe_api_call()
    def update_stop_loss(self, symbol: str, side: str, quantity: float, new_sl_price: float):
        """Cancel previous SL and place new STOP_MARKET."""
        self._init()
        opposite = "SELL" if side.upper() == "LONG" else "BUY"
        new_sl_price = self._adjust_price(symbol, new_sl_price)
        quantity = self._adjust_quantity(symbol, quantity)
        try:
            open_orders = self.get_open_orders(symbol=symbol)
            for order in open_orders:
                otype = order.get("type") or order.get("orderType", "")
                if "STOP" in otype.upper():
                    oid = order.get("orderId") or order.get("id")
                    try:
                        if hasattr(self._client, "futures_cancel_order"):
                            self._client.futures_cancel_order(symbol=symbol, orderId=oid)
                        else:
                            self._client.cancel_order(symbol=symbol, orderId=oid)
                    except Exception:
                        pass
            return self._client.new_order(
                symbol=symbol,
                side=opposite,
                type="STOP_MARKET",
                stopPrice=str(new_sl_price),
                closePosition=False,
                quantity=quantity,
            )
        except Exception as e:
            logger.error("Failed to update SL for %s: %s", symbol, e)
            return None
        
    def get_futures_balance_usdt(self):
        """
        Returns total USDT balance from futures account dynamically.
        """
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["balance"])
            return 0.0
        except Exception as e:
            print(f"⚠️ Error fetching futures balance: {e}")
            return 0.0


    # ==================================================================
    # Debug
    # ==================================================================
    def debug_filters(self, symbol: str):
        filters = self.get_symbol_filters(symbol)
        logger.info("Filters for %s:", symbol)
        for k, v in filters.items():
            logger.info("  %s: %s", k, v)


# If run directly, quick test (won't crash if credentials missing)
if __name__ == "__main__":
    c = BinanceClient()
    try:
        bal = c.get_futures_account_balance()
        print("Available USDT futures balance:", bal)
    except Exception as e:
        print("Client test failed:", e)
