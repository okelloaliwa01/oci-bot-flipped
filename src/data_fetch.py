#src/data_fetch.py
import pandas as pd
from datetime import datetime
from binance_client import BinanceClient

def fetch_closed_candles(symbol, interval, limit, client=None):
    """Returns a pandas DataFrame with columns: open_time, open, high, low, close, volume, close_time"""
    if client is None:
        client = BinanceClient()
    raw = client.get_klines(symbol, interval, limit)
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    records = []
    for k in raw:
        records.append({
            'open_time': int(k[0]),
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5]),
            'close_time': int(k[6])
        })
    df = pd.DataFrame(records)
    # sort oldest -> newest
    df = df.sort_values('open_time').reset_index(drop=True)
    return df
