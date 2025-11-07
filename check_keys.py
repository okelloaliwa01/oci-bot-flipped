from binance.um_futures import UMFutures
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# ✅ Use mainnet base URL
client = UMFutures(key=api_key, secret=api_secret, base_url="https://fapi.binance.com")

print("🔍 Testing Binance Futures MAINNET API keys...")

try:
    res = client.balance()
    print("✅ Keys are valid! Account balance:")
    print(res)
except Exception as e:
    print("❌ Keys invalid or permission issue:")
    print(e)
