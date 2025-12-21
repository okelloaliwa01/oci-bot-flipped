import math

def round_step(value, step):
    if step <= 0:
        return value
    return math.floor(value / step) * step

def apply_symbol_filters(qty, price, filters):
    """
    Returns a safe quantity satisfying:
    - stepSize
    - minQty
    - minNotional
    """
    step = filters.get("stepSize", 0.000001)
    min_qty = filters.get("minQty", 0.0)
    min_notional = filters.get("minNotional", 0.0)

    # Apply step size
    qty = round_step(qty, step)

    # Enforce minimum quantity
    if qty < min_qty:
        qty = min_qty

    # Enforce minimum notional
    if qty * price < min_notional:
        qty = min_notional / price
        qty = round_step(qty, step)

    return qty