"""
Корректный расчёт PnL и размера позиции (Kelly Criterion).
Заменяет упрощённые расчёты в base.py.
"""
import logging
from typing import Dict
import math

logger = logging.getLogger(__name__)


def calculate_pnl(
    entry: float,
    exit: float,
    qty: float,
    side: str,
    fees_pct: float = 0.06,
    funding_pct: float = 0.0,
) -> Dict:
    """
    Точный PnL в USDT.

    Returns: {pnl_usd, pnl_pct, gross, fees, funding}
    """
    if side.upper() in ("BUY", "LONG"):
        gross = qty * (exit - entry)
    else:  # SELL / SHORT
        gross = qty * (entry - exit)

    notional_in = qty * entry
    notional_out = qty * exit
    fees = (notional_in + notional_out) * (fees_pct / 100)
    funding = notional_in * (funding_pct / 100)

    pnl_usd = gross - fees - funding
    pnl_pct = pnl_usd / notional_in * 100 if notional_in > 0 else 0

    return {
        "pnl_usd": round(pnl_usd, 4),
        "pnl_pct": round(pnl_pct, 4),
        "gross": round(gross, 4),
        "fees": round(fees, 4),
        "funding": round(funding, 4),
    }


def calculate_r_multiple(entry: float, exit: float, stop_loss: float, side: str) -> float:
    """
    R-multiple: насколько R (risk units) заработали/потеряли.
    Универсальная метрика для ML labelling.
    """
    if side.upper() in ("BUY", "LONG"):
        gain = exit - entry
        risk = entry - stop_loss
    else:
        gain = entry - exit
        risk = stop_loss - entry
    if risk <= 0:
        return 0
    return round(gain / risk, 3)


def fixed_risk_position_size(
    balance: float,
    entry: float,
    stop_loss: float,
    risk_pct: float = 1.0,
    leverage: int = 1,
    min_qty: float = 0.001,
) -> float:
    """
    Размер позиции на основе фиксированного риска.
    При срабатывании SL потеряем ровно risk_pct% от depo.
    """
    risk_usd = balance * (risk_pct / 100)
    sl_distance = abs(entry - stop_loss)
    if sl_distance <= 0 or entry <= 0:
        return 0
    qty = risk_usd / sl_distance
    return max(min_qty, round(qty, 4))


def kelly_position_size(
    balance: float,
    entry: float,
    stop_loss: float,
    take_profit: float,
    win_probability: float,
    side: str,
    kelly_fraction: float = 0.25,  # 1/4 Kelly — безопасный дефолт
    max_risk_pct: float = 2.0,
    min_qty: float = 0.001,
) -> Dict:
    """
    Kelly Criterion: оптимальный размер ставки.

    f* = (b·p - q) / b
    где p = P(win), q = 1-p, b = RR ratio

    fraction_kelly: множитель (0.25 = quarter Kelly, рекомендуется)
    """
    sl_dist = abs(entry - stop_loss)
    tp_dist = abs(take_profit - entry)
    if sl_dist <= 0 or tp_dist <= 0:
        return {"qty": 0, "risk_pct": 0, "kelly_pct": 0}

    b = tp_dist / sl_dist  # RR ratio
    p = max(0.01, min(0.99, win_probability))
    q = 1 - p

    # Kelly formula
    full_kelly = (b * p - q) / b
    used_kelly = max(0, full_kelly * kelly_fraction)

    # Кэп на риск
    risk_pct = min(used_kelly * 100, max_risk_pct)
    risk_usd = balance * (risk_pct / 100)

    qty = risk_usd / sl_dist
    qty = max(min_qty, round(qty, 4))

    return {
        "qty": qty,
        "risk_pct": round(risk_pct, 3),
        "kelly_pct": round(full_kelly * 100, 3),
        "fractional_kelly_pct": round(used_kelly * 100, 3),
        "rr": round(b, 2),
        "win_prob": round(p, 3),
    }


def adaptive_sl_tp(
    entry: float,
    atr: float,
    side: str,
    atr_multiplier_sl: float = 1.5,
    rr_target: float = 2.5,
) -> Dict:
    """
    Адаптивные SL/TP на основе ATR.
    SL = entry ± k*ATR
    TP = entry ± rr_target * k*ATR
    """
    sl_distance = atr * atr_multiplier_sl
    tp_distance = sl_distance * rr_target

    if side.upper() in ("BUY", "LONG"):
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        sl = entry + sl_distance
        tp = entry - tp_distance

    return {
        "stop_loss": round(sl, 6),
        "take_profit": round(tp, 6),
        "sl_pct": round(sl_distance / entry * 100, 4),
        "tp_pct": round(tp_distance / entry * 100, 4),
        "rr_ratio": rr_target,
    }


def estimate_slippage(orderbook_data: Dict, qty: float, side: str) -> float:
    """
    Оценка проскальзывания через ордербук.
    orderbook_data: {bids: [[price, size], ...], asks: [...]}
    """
    if not orderbook_data or "bids" not in orderbook_data or "asks" not in orderbook_data:
        return 0.001  # 0.1% дефолт
    levels = orderbook_data["asks"] if side.upper() == "BUY" else orderbook_data["bids"]
    if not levels:
        return 0.001

    filled = 0
    weighted_price = 0
    for lvl in levels:
        try:
            price = float(lvl[0])
            size = float(lvl[1])
        except Exception:
            continue
        take = min(size, qty - filled)
        weighted_price += price * take
        filled += take
        if filled >= qty:
            break

    if filled <= 0:
        return 0.01
    avg_price = weighted_price / filled
    best_price = float(levels[0][0])
    slippage = abs(avg_price - best_price) / best_price
    return slippage
