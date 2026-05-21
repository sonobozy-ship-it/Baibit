"""
normalize_take_profit() — общий модуль нормализации TP для всех стратегий.
Применяется ПОСЛЕ генерации сигнала, ДО открытия позиции.

Правила (RR enforcement):
  1. sl_dist = abs(entry - stop_loss)
  2. min_rr: scalp-like = 1.2, normal = 1.5
  3. min_tp_dist = sl_dist * min_rr
  4. Find nearest resistance (BUY) or support (SELL) in last 100 candles
  5. max_allowed_dist = dist_to_level * (0.70 for scalp, 0.80 for normal)
  6. new_tp_dist = min(original_tp_dist, max_allowed_dist)
  7. If new_tp_dist < min_tp_dist: log reason and return None (block trade)
  8. Otherwise adjust TP only if it makes it more conservative
"""
import logging
from typing import Optional

import pandas as pd

from strategies.base import TradingSignal

try:
    import pandas_ta as ta  # type: ignore
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

logger = logging.getLogger(__name__)


def _calc_atr(df: pd.DataFrame, length: int = 14) -> float:
    try:
        if _HAS_TA:
            atr_s = ta.atr(df["high"], df["low"], df["close"], length=length)
            if atr_s is not None and not atr_s.isna().all():
                val = float(atr_s.iloc[-1])
                if val > 0:
                    return val
    except Exception:
        pass
    hl = (df["high"] - df["low"]).iloc[-length:]
    return float(hl.mean()) if len(hl) > 0 else 0.0


def _find_levels(df: pd.DataFrame, n_bars: int = 100) -> tuple:
    """Nearest resistance (max high) and support (min low) over last n_bars candles."""
    recent = df.iloc[-n_bars:-1]  # exclude current candle
    if recent.empty:
        recent = df.iloc[-n_bars:]
    resistance = float(recent["high"].max())
    support = float(recent["low"].min())
    return resistance, support


def normalize_take_profit(
    signal: TradingSignal,
    df: pd.DataFrame,
    scalp_mode: bool = False,
) -> Optional[TradingSignal]:
    """
    Normalises TP using RR enforcement.

    Returns None if the trade should be blocked (RR cannot be met).
    Returns the signal with an adjusted take_profit otherwise.
    """
    if df is None or len(df) < 20:
        return signal

    entry = signal.entry_price
    side = signal.action  # "BUY" or "SELL"
    sl_dist = abs(entry - signal.stop_loss)

    if sl_dist <= 0:
        return signal

    # RR minimums
    min_rr = 1.2 if scalp_mode else 1.5
    min_tp_dist = sl_dist * min_rr

    # Level proximity factor
    level_mult = 0.70 if scalp_mode else 0.80

    resistance, support = _find_levels(df, n_bars=100)
    original_tp = signal.take_profit
    original_tp_dist = abs(original_tp - entry)

    if side == "BUY":
        dist_to_level = resistance - entry
        if dist_to_level <= 0:
            logger.info(
                f"[TP_NORM] {signal.symbol} LONG blocked — "
                f"price already above resistance {resistance:.6f}"
            )
            return None

        max_allowed_dist = dist_to_level * level_mult
        new_tp_dist = min(original_tp_dist, max_allowed_dist)

        if new_tp_dist < min_tp_dist:
            logger.info(
                f"[TP_NORM] {signal.symbol} LONG blocked — "
                f"new_tp_dist={new_tp_dist:.6f} < min_tp_dist={min_tp_dist:.6f} "
                f"(RR={new_tp_dist/sl_dist:.2f} < {min_rr}) "
                f"dist_to_res={dist_to_level:.6f} scalp={scalp_mode}"
            )
            return None

        new_tp = entry + new_tp_dist
        # Only adjust if new TP is more conservative (closer)
        if new_tp < original_tp:
            signal.take_profit = round(new_tp, 8)

    else:  # SELL
        dist_to_level = entry - support
        if dist_to_level <= 0:
            logger.info(
                f"[TP_NORM] {signal.symbol} SHORT blocked — "
                f"price already below support {support:.6f}"
            )
            return None

        max_allowed_dist = dist_to_level * level_mult
        new_tp_dist = min(original_tp_dist, max_allowed_dist)

        if new_tp_dist < min_tp_dist:
            logger.info(
                f"[TP_NORM] {signal.symbol} SHORT blocked — "
                f"new_tp_dist={new_tp_dist:.6f} < min_tp_dist={min_tp_dist:.6f} "
                f"(RR={new_tp_dist/sl_dist:.2f} < {min_rr}) "
                f"dist_to_sup={dist_to_level:.6f} scalp={scalp_mode}"
            )
            return None

        new_tp = entry - new_tp_dist
        # Only adjust if new TP is more conservative (closer to entry for short)
        if new_tp > original_tp:
            signal.take_profit = round(new_tp, 8)

    actual_rr = abs(signal.take_profit - entry) / sl_dist if sl_dist > 0 else 0
    logger.debug(
        f"[TP_NORM] {signal.symbol} {side} entry={entry:.6f} "
        f"tp_orig={original_tp:.6f} → tp_adj={signal.take_profit:.6f} "
        f"RR={actual_rr:.2f} sl_dist={sl_dist:.6f} "
        f"res={resistance:.6f} sup={support:.6f} scalp={scalp_mode}"
    )

    return signal
