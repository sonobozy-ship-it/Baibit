"""
normalize_take_profit() — общий модуль нормализации TP для всех стратегий.
Применяется ПОСЛЕ генерации сигнала, ДО открытия позиции.

Правила:
  1. TP = min(ATR*0.5, distance_to_level*0.6, avg_body*1.5)   [scalp: ATR*0.35, level*0.5]
  2. Кеп: TP не дальше ATR*0.7   [scalp: ATR*0.35]
  3. Блокирует вход если до resistance/support < ATR*0.5
  4. Минимум: TP не ближе чем 0.5R (risk)
  5. Логирует детали по каждому решению
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


def _find_levels(df: pd.DataFrame, n_bars: int = 50) -> tuple:
    """Ближайший resistance (max high) и support (min low) за последние n_bars свечей."""
    recent = df.iloc[-n_bars:-1]  # исключаем текущую свечу
    if recent.empty:
        recent = df.iloc[-n_bars:]
    resistance = float(recent["high"].max())
    support = float(recent["low"].min())
    return resistance, support


def _avg_candle_body(df: pd.DataFrame, n: int = 20) -> float:
    bodies = abs(df["close"] - df["open"]).iloc[-n:]
    return float(bodies.mean()) if len(bodies) > 0 else 0.0


def normalize_take_profit(
    signal: TradingSignal,
    df: pd.DataFrame,
    scalp_mode: bool = False,
) -> Optional[TradingSignal]:
    """
    Нормализует TP сигнала по общим правилам.

    Возвращает None если вход заблокирован (слишком близко к уровню).
    Возвращает сигнал с скорректированным take_profit.
    """
    if df is None or len(df) < 20:
        return signal

    entry = signal.entry_price
    side = signal.action  # "BUY" or "SELL"

    atr = _calc_atr(df)
    if atr <= 0:
        return signal

    resistance, support = _find_levels(df)
    avg_body = _avg_candle_body(df)
    sl_dist = abs(entry - signal.stop_loss)
    original_tp = signal.take_profit

    # Мультипликаторы ATR для скальп-режима vs обычного
    atr_tp_mult = 0.35 if scalp_mode else 0.5
    atr_cap_mult = 0.35 if scalp_mode else 0.7
    level_dist_mult = 0.5 if scalp_mode else 0.6

    if side == "BUY":
        dist_to_level = resistance - entry

        # Фильтр качества входа: слишком близко к resistance
        if dist_to_level <= 0 or dist_to_level < atr * 0.5:
            logger.info(
                f"[TP_NORM] {signal.symbol} LONG заблокирован — "
                f"до resistance {dist_to_level:.6f} < ATR*0.5={atr*0.5:.6f}"
            )
            return None

        # Кандидаты TP (дистанция от entry)
        candidates = [atr * atr_tp_mult, dist_to_level * level_dist_mult]
        if not scalp_mode and avg_body > 0:
            candidates.append(avg_body * 1.5)

        new_tp_dist = min(c for c in candidates if c > 0)
        new_tp_dist = min(new_tp_dist, atr * atr_cap_mult)  # кеп
        new_tp_dist = max(new_tp_dist, sl_dist * 0.5)       # минимум 0.5R

        new_tp = entry + new_tp_dist

        # Корректируем только если новый TP ближе (консервативнее) чем оригинал
        if new_tp < original_tp:
            signal.take_profit = round(new_tp, 8)

    else:  # SELL
        dist_to_level = entry - support

        if dist_to_level <= 0 or dist_to_level < atr * 0.5:
            logger.info(
                f"[TP_NORM] {signal.symbol} SHORT заблокирован — "
                f"до support {dist_to_level:.6f} < ATR*0.5={atr*0.5:.6f}"
            )
            return None

        candidates = [atr * atr_tp_mult, dist_to_level * level_dist_mult]
        if not scalp_mode and avg_body > 0:
            candidates.append(avg_body * 1.5)

        new_tp_dist = min(c for c in candidates if c > 0)
        new_tp_dist = min(new_tp_dist, atr * atr_cap_mult)
        new_tp_dist = max(new_tp_dist, sl_dist * 0.5)

        new_tp = entry - new_tp_dist

        if new_tp > original_tp:
            signal.take_profit = round(new_tp, 8)

    rr = abs(signal.take_profit - entry) / sl_dist if sl_dist > 0 else 0
    logger.debug(
        f"[TP_NORM] {signal.symbol} {side} entry={entry:.6f} "
        f"tp_orig={original_tp:.6f} → tp_adj={signal.take_profit:.6f} "
        f"RR={rr:.2f} ATR={atr:.6f} "
        f"res={resistance:.6f} sup={support:.6f} scalp={scalp_mode}"
    )

    return signal
