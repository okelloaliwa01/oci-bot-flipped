# filters.py
"""
Utility functions for retrieving Binance Futures symbol filters (lot size, tick size, etc.)
Keeps your folder structure the same but cleans architecture and removes unsafe code.
"""

from binance.client import Client
from typing import Dict, Any, Optional


def get_binance_client(api_key: str, api_secret: str) -> Client:
    """Create and return a Binance Futures client instance."""
    return Client(api_key, api_secret)


def get_symbol_filters(client: Client, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch symbol-specific futures filters from Binance.

    Args:
        client (Client): Binance client instance.
        symbol (str): Symbol to look for, e.g., "BTCUSDT".

    Returns:
        dict or None: A dictionary of filters, or None if symbol not found.
    """
    try:
        exchange_info = client.futures_exchange_info()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch futures exchange info: {e}")

    for s in exchange_info.get("symbols", []):
        if s.get("symbol") == symbol.upper():
            return {f["filterType"]: f for f in s.get("filters", [])}

    return None


def get_lot_size(filters: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract LOT_SIZE filter if available."""
    return filters.get("LOT_SIZE")


def get_price_filter(filters: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract PRICE_FILTER filter if available."""
    return filters.get("PRICE_FILTER")


def get_market_lot_size(filters: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract MARKET_LOT_SIZE filter if available."""
    return filters.get("MARKET_LOT_SIZE")
