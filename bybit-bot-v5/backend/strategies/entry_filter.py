"""
entry_filter.py — фильтр подтверждения входа (anti-knife).

Цель: не ловить ножи. Входить только после подтверждения разворота,
а не при первом касании поддержки/сопротивления.

Правила LONG:
  1. Запрет если последние 3 свечи красные (falling knife)
  2. Запрет если последние 2 красные + текущая красная + цена ниже EMA9
  3. После 2-х красных — обязательна зелёная текущая свеча
  4. Запрет если текущая красная + цена ниже EMA9
  5. Проверка разворотной свечи (c-1) + подтверждения (c0)

Правила SHORT (зеркально):
  1. Запрет если последние 3 свечи зелёные
  2. Запрет если последние 2 зелёные + текущая зелёная + цена выше EMA9
  3. После 2-х зелёных — обязательна красная текущая свеча
  4. Запрет если текущая зелёная + цена выше EMA9

В scalp_mode: только запрет ножей (правило 1+2), остальное — предупреждения.
"""
import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Возможные причины блокировки (для логирования)
FALLING_KNIFE = "falling_knife_block"
BULL_RUN      = "bull_run_block"
BAD_MOMENTUM  = "bad_micro_momentum"
NO_REVERSAL   = "no_reversal_confirmation"
WEAK_VOLUME   = "weak_volume"
VALID_ENTRY   = "valid_entry"
VALID_REV     = "valid_reversal_entry"
NO_DATA       = "insufficient_data"


def _dir(o: float, c: float) -> str:
    return "G" if c > o else "R"


def _ema9(close: pd.Series) -> float:
    return float(close.astype(float).ewm(span=9, adjust=False).mean().iloc[-1])


def validate_entry_confirmation(
    df: pd.DataFrame,
    side: str,
    scalp_mode: bool = False,
) -> Dict:
    """
    Проверяет подтверждение разворота перед открытием позиции.

    side: "BUY" или "SELL"
    scalp_mode: в режиме скальпа только грубые блоки (нож), но не требуем
                полного подтверждения разворота.

    Returns:
        entry_allowed: bool
        entry_block_reason: str  (пустая если разрешён)
        last_3_candles_direction: "GRG" / "RRR" и т.п.
        ema9_position: "above" | "below"
        short_term_momentum: float
        reversal_candle_detected: bool
        confirmation_candle_detected: bool
    """
    min_bars = 6
    if df is None or len(df) < min_bars:
        return _ok(NO_DATA)

    # Индексы: c0 = текущая (закрытая) свеча, c_1 = предыдущая, ...
    c0  = df.iloc[-1]
    c_1 = df.iloc[-2]
    c_2 = df.iloc[-3]
    c_3 = df.iloc[-4]

    d0  = _dir(float(c0["open"]),  float(c0["close"]))
    d_1 = _dir(float(c_1["open"]), float(c_1["close"]))
    d_2 = _dir(float(c_2["open"]), float(c_2["close"]))
    last3 = d_2 + d_1 + d0

    price   = float(c0["close"])
    ema9_v  = _ema9(df["close"])
    ema9_pos = "above" if price > ema9_v else "below"

    # Краткосрочный импульс: текущая цена vs цена 3 свечи назад
    stm = price - float(c_3["close"])

    # Среднее тело свечи (последние 20)
    bodies = abs(df["close"] - df["open"]).astype(float).iloc[-20:]
    avg_body = float(bodies.mean()) if len(bodies) > 0 else 0.0

    # Размеры тел текущей и предыдущей
    body_c0  = abs(float(c0["close"])  - float(c0["open"]))
    body_c_1 = abs(float(c_1["close"]) - float(c_1["open"]))

    # Объём
    vol_ok = True
    if "volume" in df.columns:
        vol_ma  = float(df["volume"].rolling(20).mean().iloc[-1])
        vol_cur = float(df["volume"].iloc[-1])
        vol_ok  = vol_cur >= vol_ma * 0.7 if vol_ma > 0 else True

    # Разворотная свеча: c_1 (предыдущая) + подтверждение c0 (текущая)
    if side == "BUY":
        rev_candle  = (d_1 == "G") and (body_c_1 >= avg_body * 0.7)
        conf_candle = rev_candle and (float(c0["close"]) > float(c_1["high"]))
    else:
        rev_candle  = (d_1 == "R") and (body_c_1 >= avg_body * 0.7)
        conf_candle = rev_candle and (float(c0["close"]) < float(c_1["low"]))

    meta = {
        "last_3_candles_direction": last3,
        "ema9_position":            ema9_pos,
        "short_term_momentum":      round(stm, 8),
        "reversal_candle_detected": rev_candle,
        "confirmation_candle_detected": conf_candle,
    }

    # ── LONG проверки ──────────────────────────────────────────────────────────
    if side == "BUY":

        # Правило 1: все 3 красные → классический нож
        if last3 == "RRR":
            return _block(FALLING_KNIFE, **meta)

        # Правило 2: 2 красные + текущая красная + ниже EMA9 + отрицательный импульс
        if d_1 == "R" and d0 == "R" and price < ema9_v and stm < 0:
            return _block(BAD_MOMENTUM, **meta)

        # Правило 3: после 2-х красных — текущая должна быть зелёной
        if d_2 == "R" and d_1 == "R":
            if d0 != "G":
                return _block(NO_REVERSAL, **meta)
            # Текущая зелёная после 2-х красных — разворот есть, но нужен объём
            if not vol_ok and not scalp_mode:
                return _block(WEAK_VOLUME, **meta)
            return _ok(VALID_REV, **meta)

        # Правило 4: текущая красная + ниже EMA9 = плохой momentum
        # (в scalp_mode пропускаем: даём возможность торговать)
        if not scalp_mode and d0 == "R" and price < ema9_v:
            return _block(BAD_MOMENTUM, **meta)

        # Вход разрешён
        reason = VALID_REV if (rev_candle or conf_candle) else VALID_ENTRY
        return _ok(reason, **meta)

    # ── SHORT проверки ─────────────────────────────────────────────────────────
    else:

        # Правило 1: все 3 зелёные → бегущий поезд
        if last3 == "GGG":
            return _block(BULL_RUN, **meta)

        # Правило 2: 2 зелёные + текущая зелёная + выше EMA9 + положительный импульс
        if d_1 == "G" and d0 == "G" and price > ema9_v and stm > 0:
            return _block(BAD_MOMENTUM, **meta)

        # Правило 3: после 2-х зелёных — текущая должна быть красной
        if d_2 == "G" and d_1 == "G":
            if d0 != "R":
                return _block(NO_REVERSAL, **meta)
            if not vol_ok and not scalp_mode:
                return _block(WEAK_VOLUME, **meta)
            return _ok(VALID_REV, **meta)

        # Правило 4: текущая зелёная + выше EMA9 = плохой momentum для шорта
        if not scalp_mode and d0 == "G" and price > ema9_v:
            return _block(BAD_MOMENTUM, **meta)

        reason = VALID_REV if (rev_candle or conf_candle) else VALID_ENTRY
        return _ok(reason, **meta)


# ── Вспомогательные конструкторы ───────────────────────────────────────────────

def _block(reason: str, **meta) -> Dict:
    return {
        "entry_allowed":      False,
        "entry_block_reason": reason,
        **_defaults(meta),
    }


def _ok(reason: str, **meta) -> Dict:
    return {
        "entry_allowed":      True,
        "entry_block_reason": "",
        **_defaults(meta),
    }


def _defaults(meta: Dict) -> Dict:
    return {
        "last_3_candles_direction":  meta.get("last_3_candles_direction", "???"),
        "ema9_position":             meta.get("ema9_position", "?"),
        "short_term_momentum":       meta.get("short_term_momentum", 0.0),
        "reversal_candle_detected":  meta.get("reversal_candle_detected", False),
        "confirmation_candle_detected": meta.get("confirmation_candle_detected", False),
    }
