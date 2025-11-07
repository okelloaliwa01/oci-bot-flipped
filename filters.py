from binance.client import Client

# Replace with your API keys
api_key = "W4dYvsci0tU1bLIU0wsofLGBmno5xe2Jg2QAPdDTfOPhxFxT5fgzvegJ55pHs6vQ"
api_secret = "KcpAfk41yYbvNZCHxmCZPNqieBZxZ3lmBNORFsFx84hD3IHcWGNhTdea2viNcrxd"
client = Client(api_key, api_secret)

# Get all futures symbol info
exchange_info = client.futures_exchange_info()

# Find the filters for BTCUSDT
for symbol in exchange_info['symbols']:
    if symbol['symbol'] == 'BTCUSDT':
        print(symbol['filters'])
        break
