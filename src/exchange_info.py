import logging

class ExchangeInfoManager:
    def __init__(self, client):
        self.client = client
        self.cache = {}

    def get_filters(self, symbol: str):
        """
        Fetch and parse symbol filters dynamically from Binance.
        Cached so we never spam the API.
        """
        if symbol in self.cache:
            return self.cache[symbol]

        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    filters = {}

                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            filters["minQty"] = float(f["minQty"])
                            filters["maxQty"] = float(f["maxQty"])
                            filters["stepSize"] = float(f["stepSize"])

                        if f["filterType"] == "PRICE_FILTER":
                            filters["tickSize"] = float(f["tickSize"])

                        if f["filterType"] == "MIN_NOTIONAL":
                            filters["minNotional"] = float(f.get("notional", 0.0))

                    self.cache[symbol] = filters
                    return filters
        except Exception as e:
            logging.error("ExchangeInfoManager failed to fetch filters: %s", e)

        return {
            "minQty": 0.0,
            "maxQty": 999999,
            "stepSize": 0.000001,
            "tickSize": 0.01,
            "minNotional": 0.0,
        }