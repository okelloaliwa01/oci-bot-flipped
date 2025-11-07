# src/utils/atr_helper.py
import pandas as pd

def calculate_atr(client, symbol, interval='1m', period=14):
    klines = client.get_klines(symbol, interval, period + 1)
    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume','c1','c2','c3','c4','c5','c6'
    ])
    df[['high','low','close']] = df[['high','low','close']].astype(float)
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = (df['high'] - df['close'].shift()).abs()
    df['L-PC'] = (df['low'] - df['close'].shift()).abs()
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    atr = df['TR'].rolling(period).mean().iloc[-1]
    return float(atr)
