"""
Quant Filters — фильтры качества сигнала перед открытием позиции.

1. FlatMarketFilter  : ADX(14) > 20, ATR(14) > ATR_MA20 × 0.8
2. MTFConfirmation   : 4H EMA50 vs EMA200 — только по тренду
3. VolumeFilter      : Volume > SMA(Volume,20) × 1.3
4. DynamicSLTP       : SL = ATR×1.5, TP = ATR×3, min RR ≥ 2.0
5. QuantFilter       : единый фасад, применяется в trading loop

Использование:
    qf = QuantFilter()
    result = qf.check_all(df_1h, action="BUY", entry=price, df_4h=df_4h)
    if not result["pass"]:
        continue   # NO TRADE
    signal.stop_loss  = result["sl"]
    signal.take_profit = result["tp"]
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── внутренние индикаторы (без внешних зависимостей) ─────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx_value(df: pd.DataFrame, period: int = 14) -> float:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up   = high.diff()
    down = (-low.diff())
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    a = 1 / period
    atr_e = tr.ewm(alpha=a, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(alpha=a, adjust=False).mean() / (atr_e + 1e-9)
    di_m  = 100 * dm_m.ewm(alpha=a, adjust=False).mean() / (atr_e + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    return float(dx.ewm(alpha=a, adjust=False).mean().iloc[-1])


# ── 1. Фильтр флэта ───────────────────────────────────────────────────────────

class FlatMarketFilter:
    """
    NO TRADE если ADX(14) ≤ adx_min ИЛИ ATR(14) ≤ ATR_MA20 × atr_ratio_min.
    Цель: не торговать в боковике.
    """

    def __init__(self, adx_min: float = 20.0, atr_ratio_min: float = 0.8):
        self.adx_min      = adx_min
        self.atr_ratio_min = atr_ratio_min

    def check(self, df: pd.DataFrame) -> Dict:
        if len(df) < 25:
            return {"pass": False, "reason": "Мало свечей для ADX/ATR"}

        adx_val  = _adx_value(df, 14)
        atr_s    = _atr_series(df, 14)
        atr_val  = float(atr_s.iloc[-1])
        atr_ma20 = float(atr_s.rolling(20).mean().iloc[-1])

        adx_ok = adx_val > self.adx_min
        atr_ok = atr_ma20 > 0 and atr_val > atr_ma20 * self.atr_ratio_min
        passed = adx_ok and atr_ok

        ratio = round(atr_val / atr_ma20, 3) if atr_ma20 > 0 else 0.0
        reason = (
            "OK" if passed
            else f"FLAT: ADX={adx_val:.1f}({'✅' if adx_ok else '❌'}) "
                 f"ATR/MA20={ratio:.2f}({'✅' if atr_ok else '❌'})"
        )
        return {
            "pass":    passed,
            "adx":     round(adx_val, 2),
            "adx_ok":  adx_ok,
            "atr":     round(atr_val, 8),
            "atr_ma20": round(atr_ma20, 8),
            "atr_ratio": ratio,
            "atr_ok":  atr_ok,
            "reason":  reason,
        }


# ── 2. MTF подтверждение (4H) ─────────────────────────────────────────────────

class MTFConfirmation:
    """
    LONG только если EMA50_4H > EMA200_4H.
    SHORT только если EMA50_4H < EMA200_4H.
    Если 4H данных нет — пропускаем (не блокируем).
    """

    def check(self, df_4h: Optional[pd.DataFrame], action: str) -> Dict:
        if df_4h is None or len(df_4h) < 205:
            return {
                "pass": True,
                "reason": "4H: нет данных — фильтр пропущен",
                "direction": "NEUTRAL",
            }

        close  = df_4h["close"].astype(float)
        ema50  = float(_ema(close, 50).iloc[-1])
        ema200 = float(_ema(close, 200).iloc[-1])

        if ema50 > ema200:
            direction    = "BULL"
            allowed      = "BUY"
        elif ema50 < ema200:
            direction    = "BEAR"
            allowed      = "SELL"
        else:
            direction    = "NEUTRAL"
            allowed      = None

        passed = direction == "NEUTRAL" or action == allowed
        sign   = ">" if ema50 > ema200 else "<"
        reason = (
            f"4H {direction}: EMA50={ema50:.4f} {sign} EMA200={ema200:.4f}"
            + ("" if passed else f" → БЛОКИРУЕТ {action}")
        )
        return {
            "pass":       passed,
            "direction":  direction,
            "ema50_4h":   round(ema50, 6),
            "ema200_4h":  round(ema200, 6),
            "action":     action,
            "reason":     reason,
        }


# ── 3. Объёмный фильтр ────────────────────────────────────────────────────────

class VolumeFilter:
    """
    NO TRADE если Volume < SMA(Volume,20) × multiplier.
    Цель: входить только при реальном движении объёма.
    """

    def __init__(self, multiplier: float = 1.3):
        self.multiplier = multiplier

    def check(self, df: pd.DataFrame) -> Dict:
        if len(df) < 21:
            return {"pass": False, "reason": "Мало свечей для объёмного фильтра"}

        vol     = df["volume"].astype(float)
        cur_vol = float(vol.iloc[-1])
        sma20   = float(vol.rolling(20).mean().iloc[-1])
        ratio   = cur_vol / sma20 if sma20 > 0 else 0.0
        passed  = ratio >= self.multiplier

        return {
            "pass":    passed,
            "volume":  round(cur_vol, 2),
            "sma20":   round(sma20, 2),
            "ratio":   round(ratio, 3),
            "reason":  (
                f"Vol={cur_vol:.0f} {'≥' if passed else '<'} "
                f"SMA20×{self.multiplier}={sma20 * self.multiplier:.0f}"
            ),
        }


# ── 4. Динамический SL/TP ────────────────────────────────────────────────────

class DynamicSLTP:
    """
    SL = entry ± ATR × sl_mult
    TP = entry ± ATR × tp_mult
    Пропускаем сигнал если RR < min_rr.

    Также даёт уровни частичного закрытия:
      TP1 (30% объёма) — на 1/3 пути к TP
      TP2 (30% объёма) — на 2/3 пути к TP
      Остаток (40%)   — трейлинг ATR × trail_mult
    """

    def __init__(
        self,
        sl_mult:    float = 1.5,
        tp_mult:    float = 3.0,
        min_rr:     float = 2.0,
        trail_mult: float = 1.2,
    ):
        self.sl_mult    = sl_mult
        self.tp_mult    = tp_mult
        self.min_rr     = min_rr
        self.trail_mult = trail_mult

    def calculate(
        self,
        df: pd.DataFrame,
        action: str,
        entry_price: float,
    ) -> Dict:
        if len(df) < 15 or entry_price <= 0:
            return {"valid": False, "reason": "Нет данных для ATR SL/TP"}

        atr_s   = _atr_series(df, 14)
        atr_val = float(atr_s.iloc[-1])
        if atr_val <= 0:
            return {"valid": False, "reason": "ATR = 0"}

        sl_dist = atr_val * self.sl_mult
        tp_dist = atr_val * self.tp_mult
        rr      = tp_dist / sl_dist  # = tp_mult / sl_mult

        if action == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        valid = rr >= self.min_rr
        return {
            "valid":    valid,
            "sl":       round(sl, 8),
            "tp":       round(tp, 8),
            "atr":      round(atr_val, 8),
            "sl_dist":  round(sl_dist, 8),
            "tp_dist":  round(tp_dist, 8),
            "rr":       round(rr, 3),
            "reason":   f"ATR={atr_val:.6f} SL={sl:.6f} TP={tp:.6f} RR={rr:.2f}",
        }

    def partial_tp_levels(
        self, entry: float, tp: float, side: str
    ) -> Dict:
        """
        Рассчитывает TP1 и TP2 для частичного закрытия.
        TP1 = 1/3 пути → закрыть 30%
        TP2 = 2/3 пути → закрыть 30%
        Остаток (40%) → trailing
        """
        if side == "Buy":
            tp1 = entry + (tp - entry) * 0.333
            tp2 = entry + (tp - entry) * 0.667
        else:
            tp1 = entry - (entry - tp) * 0.333
            tp2 = entry - (entry - tp) * 0.667
        return {
            "tp1":         round(tp1, 8),
            "tp2":         round(tp2, 8),
            "tp_full":     round(tp, 8),
            "tp1_hit":     False,
            "tp2_hit":     False,
            "remaining_qty_pct": 1.0,
        }


# ── 5. Единый фасад ───────────────────────────────────────────────────────────

class QuantFilter:
    """
    Применяет все фильтры последовательно.
    Если хоть один не прошёл — NO TRADE.
    Если все прошли — возвращает ATR-based SL/TP.

    Пример:
        result = state.quant_filter.check_all(df_1h, "BUY", price, df_4h)
        if not result["pass"]:
            continue
        signal.stop_loss   = result["sl"]
        signal.take_profit = result["tp"]
    """

    def __init__(
        self,
        adx_min:           float = 20.0,
        atr_ratio_min:     float = 0.8,
        volume_mult:       float = 1.3,
        sl_mult:           float = 1.5,
        tp_mult:           float = 3.0,
        min_rr:            float = 2.0,
        trail_mult:        float = 1.2,
        flat_enabled:      bool  = True,
        mtf_enabled:       bool  = True,
        volume_enabled:    bool  = True,
    ):
        self.flat_filter   = FlatMarketFilter(adx_min, atr_ratio_min)
        self.mtf           = MTFConfirmation()
        self.vol_filter    = VolumeFilter(volume_mult)
        self.dynamic_sltp  = DynamicSLTP(sl_mult, tp_mult, min_rr, trail_mult)

        self.flat_enabled   = flat_enabled
        self.mtf_enabled    = mtf_enabled
        self.volume_enabled = volume_enabled

    def check_all(
        self,
        df: pd.DataFrame,
        action: str,
        entry_price: float,
        df_4h: Optional[pd.DataFrame] = None,
        strategy_id: str = "",
    ) -> Dict:
        """
        Полная проверка всех фильтров.
        Возвращает:
          {"pass": True/False, "sl": float, "tp": float, "atr": float,
           "rr": float, "reason": str, "details": {...}}
        """
        details: Dict = {}

        # 1. Фильтр флэта
        if self.flat_enabled:
            flat_r = self.flat_filter.check(df)
            details["flat"] = flat_r
            if not flat_r["pass"]:
                logger.debug(f"[QuantFilter] {strategy_id}: {flat_r['reason']}")
                return {"pass": False, "reason": flat_r["reason"], "details": details}

        # 2. MTF тренд
        if self.mtf_enabled:
            mtf_r = self.mtf.check(df_4h, action)
            details["mtf"] = mtf_r
            if not mtf_r["pass"]:
                logger.debug(f"[QuantFilter] {strategy_id}: {mtf_r['reason']}")
                return {"pass": False, "reason": mtf_r["reason"], "details": details}

        # 3. Объём
        if self.volume_enabled:
            vol_r = self.vol_filter.check(df)
            details["volume"] = vol_r
            if not vol_r["pass"]:
                logger.debug(f"[QuantFilter] {strategy_id}: {vol_r['reason']}")
                return {"pass": False, "reason": vol_r["reason"], "details": details}

        # 4. Динамический SL/TP (последний — переопределяет SL/TP сигнала)
        sltp = self.dynamic_sltp.calculate(df, action, entry_price)
        details["sltp"] = sltp
        if not sltp["valid"]:
            return {"pass": False, "reason": sltp["reason"], "details": details}

        return {
            "pass":    True,
            "reason":  "OK",
            "sl":      sltp["sl"],
            "tp":      sltp["tp"],
            "atr":     sltp["atr"],
            "rr":      sltp["rr"],
            "details": details,
        }

    def partial_tp_levels(
        self, entry: float, tp: float, side: str
    ) -> Dict:
        """Прокси к DynamicSLTP.partial_tp_levels."""
        return self.dynamic_sltp.partial_tp_levels(entry, tp, side)
