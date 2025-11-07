import os
import sys

# --- Ensure src/ is in the Python path ---
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# check_binance_client_path.py
from src.binance_client import BinanceClient
print("BinanceClient loaded from:", BinanceClient.__module__)
print("File path:", BinanceClient.__doc__)
print(dir(BinanceClient))
