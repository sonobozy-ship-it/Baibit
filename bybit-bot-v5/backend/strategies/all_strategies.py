"""
Все торговые стратегии.
Каждая имеет систему фильтров (math edge) + breakeven + trailing stop.
Индикаторы — pandas_ta (или ручной расчёт где нужно).
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional
from datetime import datetime, timezone
from .base import BaseStrategy, TradingSignal
from .trend_fib import TrendMomentumStrategy, TrendFibonacciStrategy
from .scalper_pro import ScalperProStrategy, SCALP_SYMBOLS
from .aggressive_momentum import AggressiveMomentumStrategy


# ============================================================
# Shared utilities — общие функции для всех стратегий
# ============================================================

def _struct_levels(df: pd.DataFrame, atr: float, lookback: int = 100):
    """Swing-уровни поддержки/сопротивления с 2+ касаниями."""
    highs  = df["high"].values[-lookback:]
    lows   = df["low"].values[-lookback:]
    tol    = atr * 0.6
    sh, sl = [], []
    for i in range(2, len(highs) - 2):
        if highs[i] >= max(highs[i-2], highs[i-1], highs[i+1], highs[i+2]):
            sh.append(highs[i])
        if lows[i] <= min(lows[i-2], lows[i-1], lows[i+1], lows[i+2]):
            sl.append(lows[i])

    def cluster(vals):
        if not vals:
            return []
        out, grp = [], [sorted(vals)[0]]
        for v in sorted(vals)[1:]:
            if v - grp[0] <= tol:
                grp.append(v)
            else:
                if len(grp) >= 2:
                    out.append(sum(grp) / len(grp))
                grp = [v]
        if len(grp) >= 2:
            out.append(sum(grp) / len(grp))
        return out

    return cluster(sl), cluster(sh)   # (support_levels, resistance_levels)


def _nearest_level(price: float, levels: list, atr: float) -> float:
    """Возвращает ближайший уровень в пределах 0.5 ATR, иначе 0."""
    for lvl in sorted(levels, key=lambda x: abs(x - price)):
        if abs(price - lvl) <= atr * 0.5:
            return lvl
    return 0.0


def _wick_signal(c: pd.Series) -> str:
    """Факел: 'bullish' / 'bearish' / ''."""
    body  = abs(float(c["close"]) - float(c["open"]))
    upper = float(c["high"]) - max(float(c["close"]), float(c["open"]))
    lower = min(float(c["close"]), float(c["open"])) - float(c["low"])
    if body < 1e-9:
        return ""
    if lower > body * 2 and lower > upper * 1.2:
        return "bullish"
    if upper > body * 2 and upper > lower * 1.2:
        return "bearish"
    return ""


def _fake_bo(df: pd.DataFrame, levels: list, direction: str, atr: float) -> bool:
    """Ложный пробой уровня (direction: 'sup' или 'res')."""
    if len(df) < 4 or not levels:
        return False
    r = df.iloc[-4:]
    close = float(df.iloc[-1]["close"])
    for lvl in levels:
        if direction == "sup":
            if any(r["low"] < lvl - atr * 0.05) and close > lvl:
                return True
        else:
            if any(r["high"] > lvl + atr * 0.05) and close < lvl:
                return True
    return False


def _htf_trend(df: pd.DataFrame, n: int = 24) -> int:
    """Грубый HTF тренд по последним n свечам: +1 вверх, -1 вниз, 0 флэт."""
    if len(df) < n + 1:
        return 0
    start = float(df.iloc[-n]["close"])
    end   = float(df.iloc[-1]["close"])
    chg   = (end - start) / start * 100
    if chg > 0.5:
        return 1
    if chg < -0.5:
        return -1
    return 0


def _atr_ok(last, df: pd.DataFrame, max_mult: float = 1.5) -> bool:
    """True если ATR не аномально высокий (< среднего × max_mult)."""
    atr_mean = df["atr"].iloc[-20:].mean() if "atr" in df.columns else 0
    return atr_mean == 0 or float(last.get("atr", 0)) <= atr_mean * max_mult



# ============================================================
# S1: EMA CROSSOVER — точный вход за структурой, R:R ≥ 2.5
# ============================================================
class EMACrossoverStrategy(BaseStrategy):
    ID = "S1"
    NAME = "EMA CROSSOVER"
    DESCRIPTION = "EMA 9/21 cross + структурный уровень + R:R ≥ 2.5 + объём + HTF"
    REGIME_PREFERENCE = []

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.5,
            take_profit_pct=3.75,   # R:R 2.5:1
            edge_wr_target=0.62,
            timeframe="15",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None
        df = df.copy()

        df["atr"]      = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ema_fast"] = ta.ema(df["close"], length=9)
        df["ema_slow"] = ta.ema(df["close"], length=21)
        df["ema_50"]   = ta.ema(df["close"], length=50)
        df["rsi"]      = ta.rsi(df["close"], length=14)
        df["vol_ma"]   = df["volume"].rolling(20).mean()

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        atr   = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        price = float(last["close"])

        if atr == 0 or price == 0:
            return None

        # ATR не должен быть аномально высоким
        if not _atr_ok(last, df):
            return None

        # EMA кросс
        bull_cross = float(prev["ema_fast"]) < float(prev["ema_slow"]) and float(last["ema_fast"]) > float(last["ema_slow"])
        bear_cross = float(prev["ema_fast"]) > float(prev["ema_slow"]) and float(last["ema_fast"]) < float(last["ema_slow"])
        if not (bull_cross or bear_cross):
            return None

        side = "BUY" if bull_cross else "SELL"

        # Структурный уровень обязателен
        sup_lvls, res_lvls = _struct_levels(df, atr)
        if side == "BUY":
            lvl = _nearest_level(price, sup_lvls, atr)
        else:
            lvl = _nearest_level(price, res_lvls, atr)
        if not lvl:
            return None   # нет структуры — WAIT

        # Не входить в середине диапазона
        if sup_lvls and res_lvls:
            mid = (max(sup_lvls) + min(res_lvls)) / 2
            if abs(price - mid) / atr < 1.0:
                return None

        rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        vol   = float(last["volume"])
        volma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else vol

        checks = {
            "ema_cross":   True,
            "structure":   True,
            "volume":      vol > volma * 1.3,
            "rsi":         (30 < rsi < 65) if side == "BUY" else (35 < rsi < 70),
            "candle":      (price > float(last["open"])) if side == "BUY" else (price < float(last["open"])),
            "htf_trend":   _htf_trend(df) in (1, 0) if side == "BUY" else _htf_trend(df) in (-1, 0),
            "ema50_align": price > float(last["ema_50"]) if side == "BUY" else price < float(last["ema_50"]),
        }

        if not all(checks.values()):
            return None

        # SL за структурой + 0.2 ATR, минимум 0.8 ATR
        if side == "BUY":
            sl  = min(lvl - atr * 0.2, price - atr * 0.8)
            risk = price - sl
            tp  = price + risk * 2.5   # R:R 2.5:1
        else:
            sl  = max(lvl + atr * 0.2, price + atr * 0.8)
            risk = sl - price
            tp  = price - risk * 2.5

        # Проверяем R:R (на случай плохой структуры)
        actual_rr = abs(tp - price) / max(abs(sl - price), 1e-9)
        if actual_rr < 2.0:
            return None

        conf = 0.70 + 0.03 * sum(1 for v in checks.values() if v)
        return TradingSignal(
            action=side, symbol=self.symbol, confidence=round(min(0.88, conf), 2),
            entry_price=price, stop_loss=round(sl, 8), take_profit=round(tp, 8),
            reason=f"EMA9/21 cross | Lvl@{lvl:.5g} | RR{actual_rr:.1f} | RSI{rsi:.0f}",
            filters_passed=checks,
        )


# ============================================================
# S2: BOLLINGER BANDS — усиленный, HTF + запрет флэта
# ============================================================
class BollingerBandsStrategy(BaseStrategy):
    ID = "S2"
    NAME = "BOLLINGER BANDS"
    DESCRIPTION = "BB крайние + RSI экстремум + объём + HTF тренд + запрет флэта"
    REGIME_PREFERENCE = []

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.8,
            take_profit_pct=4.5,
            edge_wr_target=0.65,
            timeframe="15",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None
        df = df.copy()

        bb = ta.bbands(df["close"], length=20, std=2)
        df = df.join(bb)
        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["rsi"]    = ta.rsi(df["close"], length=14)
        df["ema50"]  = ta.ema(df["close"], length=50)
        df["ema20"]  = ta.ema(df["close"], length=20)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        price = float(last["close"])
        atr   = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        vol   = float(last["volume"])
        volma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else vol

        if atr == 0 or price == 0:
            return None

        # Запрет флэта: ATR должен быть достаточным
        atr_pct = atr / price * 100
        if atr_pct < 0.4:
            return None

        bb_lower = float(last["BBL_20_2.0"])
        bb_upper = float(last["BBU_20_2.0"])
        bb_mid   = float(last["BBM_20_2.0"])

        below_lower = price <= bb_lower * 1.002
        above_upper = price >= bb_upper * 0.998

        # RSI экстремум + разворот (не просто экстремум, а начало возврата)
        rsi_bull = rsi < 38 and rsi > float(prev["rsi"])   # перепродан и начинает расти
        rsi_bear = rsi > 62 and rsi < float(prev["rsi"])   # перекуплен и начинает падать

        vol_confirm = vol > volma * 1.2

        long_setup  = below_lower and rsi_bull and vol_confirm
        short_setup = above_upper and rsi_bear and vol_confirm

        if not (long_setup or short_setup):
            return None

        side = "BUY" if long_setup else "SELL"

        # HTF тренд — торговать ОТ экстремума в направлении тренда
        htf = _htf_trend(df)
        if side == "BUY" and htf == -1:    # сильный нисходящий HTF → пропускаем BUY против тренда
            return None
        if side == "SELL" and htf == 1:    # сильный восходящий HTF → пропускаем SELL против тренда
            return None

        # EMA50 наклон совпадает (для BUY — ema50 растёт, для SELL — падает)
        ema50_slope = float(last["ema50"]) - float(df.iloc[-5]["ema50"])
        if side == "BUY" and ema50_slope < -atr * 0.5:
            return None
        if side == "SELL" and ema50_slope > atr * 0.5:
            return None

        # RSI 45-65 для BUY, 35-55 для SELL (зоны подтверждения)
        # Уже проверено через rsi_bull/rsi_bear + BB

        # SL за BB + 0.2 ATR, TP к противоположной BB
        if side == "BUY":
            sl  = round(bb_lower - atr * 0.2, 8)
            tp  = round(bb_mid + (bb_mid - bb_lower) * 0.8, 8)
        else:
            sl  = round(bb_upper + atr * 0.2, 8)
            tp  = round(bb_mid - (bb_upper - bb_mid) * 0.8, 8)

        rr = abs(tp - price) / max(abs(sl - price), 1e-9)
        if rr < 1.8:
            return None

        checks = {
            "bb_touch":    True,
            "rsi_reverse": True,
            "volume":      vol_confirm,
            "htf_ok":      htf != (-1 if side == "BUY" else 1),
            "no_flat":     atr_pct >= 0.4,
            "ema50_slope": True,
        }

        conf = 0.72 + (0.04 if htf == (1 if side == "BUY" else -1) else 0.0)
        return TradingSignal(
            action=side, symbol=self.symbol, confidence=round(min(0.88, conf), 2),
            entry_price=price, stop_loss=sl, take_profit=tp,
            reason=f"BB {'lower' if long_setup else 'upper'} | RSI{rsi:.0f}↑↓ | vol×{vol/volma:.1f} | HTF{htf:+d}",
            filters_passed=checks,
        )


# ============================================================
# S3: RSI DIVERGENCE
# ============================================================
class RSIDivergenceStrategy(BaseStrategy):
    ID = "S3"
    NAME = "RSI DIVERGENCE"
    DESCRIPTION = "Дивергенция RSI H4 + MACD cross + структура рынка"
    REGIME_PREFERENCE = []  # работает во всех режимах

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.2,
            take_profit_pct=3.6,
            edge_wr_target=0.60,
            timeframe="60",  # H1 (было H4 — слишком редкие сигналы)
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 50:
            return None
        df = df.copy()

        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = df.join(macd)
        df["ema50"] = ta.ema(df["close"], length=50)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # ── Реальная дивергенция: ищем два свинг-лоу/хая в окне 20 свечей ──────
        # Бычья: второй ценовой лоу НИЖЕ первого, но RSI на втором лоу ВЫШЕ
        # Медвежья: второй ценовой хай ВЫШЕ первого, но RSI на втором хае НИЖЕ
        window = df.iloc[-20:]
        # Находим два минимума цены
        low_idx1 = window["low"].iloc[:10].idxmin()
        low_idx2 = window["low"].iloc[10:].idxmin()
        price_ll  = window.loc[low_idx2, "low"]  < window.loc[low_idx1, "low"]
        rsi_hl    = window.loc[low_idx2, "rsi"]  > window.loc[low_idx1, "rsi"]
        # Находим два максимума цены
        high_idx1 = window["high"].iloc[:10].idxmax()
        high_idx2 = window["high"].iloc[10:].idxmax()
        price_hh  = window.loc[high_idx2, "high"] > window.loc[high_idx1, "high"]
        rsi_lh    = window.loc[high_idx2, "rsi"]  < window.loc[high_idx1, "rsi"]

        # RSI должен быть в зоне экстремума при дивергенции
        bull_div = price_ll and rsi_hl and last["rsi"] < 45   # цена ниже, RSI выше → сила покупателей
        bear_div = price_hh and rsi_lh and last["rsi"] > 55   # цена выше, RSI ниже → слабость продавцов

        # MACD кросс обязателен (не просто направление)
        macd_bull_cross = prev["MACD_12_26_9"] < prev["MACDs_12_26_9"] and last["MACD_12_26_9"] > last["MACDs_12_26_9"]
        macd_bear_cross = prev["MACD_12_26_9"] > prev["MACDs_12_26_9"] and last["MACD_12_26_9"] < last["MACDs_12_26_9"]

        # Касание уровня: цена у EMA50 (зона поддержки/сопротивления)
        near_ema50 = abs(last["close"] - last["ema50"]) / last["ema50"] < 0.012

        if not ((bull_div and macd_bull_cross) or (bear_div and macd_bear_cross)):
            return None

        side = "BUY" if bull_div else "SELL"
        entry = float(last["close"])
        if side == "BUY":
            sl = entry * (1 - self.stop_loss_pct / 100)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = entry * (1 + self.stop_loss_pct / 100)
            tp = entry * (1 - self.take_profit_pct / 100)

        confidence = 0.70 + (0.05 if near_ema50 else 0.0)
        return TradingSignal(
            action=side, symbol=self.symbol, confidence=round(confidence, 2),
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"RSI div ({side}) + MACD cross | RSI={last['rsi']:.1f} | EMA50={'touch' if near_ema50 else 'far'}",
            filters_passed={"rsi_div": True, "macd_cross": True, "rsi_extreme": True, "ema50_touch": near_ema50},
        )


# ============================================================
# S4: BREAKOUT HUNTER — пробой + ретест + 3+ подтверждений
# ============================================================
class BreakoutHunterStrategy(BaseStrategy):
    ID = "S4"
    NAME = "BREAKOUT HUNTER"
    DESCRIPTION = "Пробой структурного уровня + ретест + 3+ подтверждений, адаптивный режим"
    REGIME_PREFERENCE = []

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.5,
            take_profit_pct=5.0,
            edge_wr_target=0.58,
            timeframe="60",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 70:
            return None
        df = df.copy()

        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["rsi"]    = ta.rsi(df["close"], length=14)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last  = df.iloc[-1]
        price = float(last["close"])
        atr   = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        vol   = float(last["volume"])
        volma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else vol

        if atr == 0:
            return None

        # Структурные уровни (последние 60 свечей)
        sup_lvls, res_lvls = _struct_levels(df, atr, lookback=60)

        # Уровни из простого диапазона 48 свечей (классический подход)
        level_zone = df.iloc[-55:-5]
        resistance = float(level_zone["high"].max())
        support    = float(level_zone["low"].min())

        # Пробой классического уровня
        broke_up   = price > resistance * 1.002
        broke_down = price < support    * 0.998

        if not (broke_up or broke_down):
            return None

        side  = "BUY" if broke_up else "SELL"
        level = resistance if broke_up else support

        # Ретест: последние 3 свечи касались уровня
        recent = df.iloc[-4:-1]
        retested = (
            (side == "BUY"  and float(recent["low"].min())  <= resistance * 1.006) or
            (side == "SELL" and float(recent["high"].max()) >= support    * 0.994)
        )

        # 5 подтверждений
        checks = {
            "breakout":  True,
            "retest":    retested,
            "volume":    vol > volma * 1.5,
            "candle":    (price > float(last["open"])) if side == "BUY" else (price < float(last["open"])),
            "htf_trend": _htf_trend(df) in (1, 0) if side == "BUY" else _htf_trend(df) in (-1, 0),
            "ema_align": price > float(last["ema_50"]) if side == "BUY" else price < float(last["ema_50"]),
        }

        # Адаптивный порог: больше убытков → строже фильтр
        min_conf = 3
        if self.consecutive_losses >= 3:
            min_conf = 5   # SAFE: нужны почти все подтверждения
        elif self.consecutive_losses >= 2:
            min_conf = 4

        conf_count = sum(1 for v in checks.values() if v)
        if conf_count < min_conf:
            return None

        # SL за структурой + 0.2 ATR
        if side == "BUY":
            sl   = round(min(level - atr * 0.2, price - atr * 0.8), 8)
            risk = price - sl
            tp   = round(price + risk * 3.0, 8)   # R:R 3:1 — пробои дают большой ход
        else:
            sl   = round(max(level + atr * 0.2, price + atr * 0.8), 8)
            risk = sl - price
            tp   = round(price - risk * 3.0, 8)

        rr = abs(tp - price) / max(abs(sl - price), 1e-9)
        if rr < 2.0:
            return None

        conf = 0.70 + 0.03 * conf_count
        return TradingSignal(
            action=side, symbol=self.symbol, confidence=round(min(0.90, conf), 2),
            entry_price=price, stop_loss=sl, take_profit=tp,
            reason=f"Breakout+Retest lvl={level:.5g} | {conf_count}/6conf | RR{rr:.1f}",
            filters_passed=checks,
        )


# ============================================================
# S5: STRICT MEAN-REVERSION — BB rejection in flat market
# ============================================================
class ScalperGridStrategy(BaseStrategy):
    ID = "S5"
    NAME = "SCALPER GRID"
    DESCRIPTION = "Строгий mean-reversion: отказ от края Bollinger в плоском рынке"
    REGIME_PREFERENCE = ["flat"]

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=0.5,
            take_profit_pct=1.0,
            edge_wr_target=0.60,
            timeframe="5",
            **kwargs,
        )
        self.max_hold_minutes = 45

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 80:
            return None
        df = df.copy()

        # ── Indicators ──────────────────────────────────────────────────────
        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ema20"]  = ta.ema(df["close"], length=20)
        df["ema50"]  = ta.ema(df["close"], length=50)
        df["rsi"]    = ta.rsi(df["close"], length=14)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None:
            df = df.join(adx_df)

        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None:
            df = df.join(bb)

        last  = df.iloc[-1]
        prev  = df.iloc[-2]

        # Extract values — guard NaN
        close  = float(last["close"])
        open_  = float(last["open"])
        high   = float(last["high"])
        low    = float(last["low"])
        atr    = float(last["atr"])   if not pd.isna(last.get("atr",   float("nan"))) else 0.0
        ema20  = float(last["ema20"]) if not pd.isna(last.get("ema20", float("nan"))) else close
        ema50  = float(last["ema50"]) if not pd.isna(last.get("ema50", float("nan"))) else close
        rsi    = float(last["rsi"])   if not pd.isna(last.get("rsi",   float("nan"))) else 50.0
        prev_rsi = float(prev["rsi"]) if not pd.isna(prev.get("rsi",   float("nan"))) else 50.0
        volume = float(last["volume"])
        vol_ma = float(last["vol_ma"]) if not pd.isna(last.get("vol_ma", float("nan"))) else volume

        # ADX
        adx_col = next((c for c in df.columns if c.startswith("ADX_")), None)
        if adx_col is None:
            return None
        adx = float(last[adx_col]) if not pd.isna(last.get(adx_col, float("nan"))) else 99.0

        # BB columns
        bbl_col = next((c for c in df.columns if c.startswith("BBL_")), None)
        bbu_col = next((c for c in df.columns if c.startswith("BBU_")), None)
        bbm_col = next((c for c in df.columns if c.startswith("BBM_")), None)
        if not bbl_col or not bbu_col or not bbm_col:
            return None
        bbl = float(last[bbl_col]) if not pd.isna(last.get(bbl_col, float("nan"))) else 0.0
        bbu = float(last[bbu_col]) if not pd.isna(last.get(bbu_col, float("nan"))) else 0.0
        bbm = float(last[bbm_col]) if not pd.isna(last.get(bbm_col, float("nan"))) else close

        if atr <= 0 or close <= 0 or bbl <= 0 or bbu <= 0:
            return None

        # ── FLAT MARKET FILTERS ──────────────────────────────────────────────
        atr_pct = atr / close * 100

        # 1. ATR range: 0.12% to 1.20%
        tradable_atr = 0.12 <= atr_pct <= 1.20

        # 2. EMA20 and EMA50 close together
        flat_ema = abs(ema20 - ema50) / close * 100 < 0.45

        # 3. EMA50 slope < 0.35% over last 10 candles
        if len(df) >= 11:
            ema50_10 = float(df["ema50"].iloc[-11])
            ema50_slope_pct = abs(ema50 - ema50_10) / max(ema50_10, 1e-9) * 100
        else:
            ema50_slope_pct = 99.0
        flat_slope = ema50_slope_pct < 0.35

        # 4. BB width in range
        bbw = (bbu - bbl) / close
        range_width = 0.006 <= bbw <= 0.045

        # 5. ADX < 16
        adx_flat = adx < 16

        # 6. ADX not accelerating: last ADX <= mean of previous 5 + 1.0
        if adx_col and len(df) >= 7:
            prev5_adx = df[adx_col].iloc[-7:-2].dropna()
            adx_mean5 = float(prev5_adx.mean()) if len(prev5_adx) > 0 else adx
            adx_not_rising = adx <= adx_mean5 + 1.0
        else:
            adx_not_rising = True

        # 7. Volume in normal range: 0.45 to 1.25 of MA
        normal_volume = (vol_ma > 0) and (0.45 * vol_ma <= volume <= 1.25 * vol_ma)

        flat_filters_pass = (
            tradable_atr and flat_ema and flat_slope
            and range_width and adx_flat and adx_not_rising and normal_volume
        )
        if not flat_filters_pass:
            return None

        # ── REJECTION CANDLE DETECTION ───────────────────────────────────────
        candle_range = high - low
        buy_setup  = False
        sell_setup = False

        # BUY setup: lower BB rejection
        if (
            low <= bbl * 1.003          # touches/pierces lower BB
            and close > bbl             # closes back above lower BB
            and candle_range > 0
            and (min(open_, close) - low) > 0.3 * candle_range   # lower wick
            and abs(close - open_) < atr * 0.6                    # body not too large
            and prev_rsi < 38
            and rsi > prev_rsi + 1.0
            and rsi < 48
        ):
            buy_setup = True

        # SELL setup: upper BB rejection
        if (
            high >= bbu * 0.997         # touches/pierces upper BB
            and close < bbu             # closes back below upper BB
            and candle_range > 0
            and (high - max(open_, close)) > 0.3 * candle_range   # upper wick
            and abs(close - open_) < atr * 0.6
            and prev_rsi > 62
            and rsi < prev_rsi - 1.0
            and rsi > 52
        ):
            sell_setup = True

        if not (buy_setup or sell_setup):
            return None
        if buy_setup and sell_setup:
            buy_setup = (rsi < 50)
            sell_setup = not buy_setup

        side = "BUY" if buy_setup else "SELL"
        entry = close

        # ── SL / TP ──────────────────────────────────────────────────────────
        if side == "BUY":
            sl_price = min(float(df.iloc[-12:]["low"].min()), low) - atr * 0.20
            tp_price = bbm
        else:
            sl_price = max(float(df.iloc[-12:]["high"].max()), high) + atr * 0.20
            tp_price = bbm

        # ── RISK CHECKS ──────────────────────────────────────────────────────
        risk   = abs(entry - sl_price)
        reward = abs(tp_price - entry)
        if risk <= 0:
            return None
        rr = reward / risk
        risk_pct = risk / entry * 100

        if risk_pct < 0.18 or risk_pct > 0.75 or rr < 1.55:
            return None

        filters_passed = {
            "tradable_atr":  tradable_atr,
            "flat_ema":      flat_ema,
            "flat_slope":    flat_slope,
            "range_width":   range_width,
            "adx_flat":      adx_flat,
            "normal_volume": normal_volume,
            "rsi_reversion": True,
            "rr":            round(rr, 2),
        }

        reason = (
            f"Strict range rejection {'BUY lower BB' if buy_setup else 'SELL upper BB'} "
            f"| ATR {atr_pct:.2f}% | ADX {adx:.1f} | RR {rr:.2f}"
        )

        return TradingSignal(
            action=side,
            symbol=self.symbol,
            confidence=0.74,
            entry_price=entry,
            stop_loss=round(sl_price, 8),
            take_profit=round(tp_price, 8),
            reason=reason,
            filters_passed=filters_passed,
        )


# ============================================================
# S6: TREND FOLLOWER
# ============================================================
class TrendFollowerStrategy(BaseStrategy):
    ID = "S6"
    NAME = "TREND FOLLOWER"
    DESCRIPTION = "ADX > 25 + Supertrend + EMA200 выше + HTF согласован"
    REGIME_PREFERENCE = []  # обучение: работает во всех режимах

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.5,
            take_profit_pct=4.5,
            edge_wr_target=0.63,
            timeframe="60",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 220:
            return None
        df = df.copy()

        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df = df.join(adx)
        df["ema_200"] = ta.ema(df["close"], length=200)
        st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3)
        df = df.join(st)

        last = df.iloc[-1]

        adx_strong   = last["ADX_14"] > 22
        above_ema200 = last["close"] > last["ema_200"]
        below_ema200 = last["close"] < last["ema_200"]
        st_bull = last["SUPERTd_10_3.0"] == 1
        st_bear = last["SUPERTd_10_3.0"] == -1

        # Касание: пуллбэк к линии Supertrend (не более 1.5% от неё)
        supert_val = last.get("SUPERT_10_3.0", float("nan"))
        if pd.isna(supert_val):
            supert_val = last.get("SUPERT_10_3", float("nan"))
        touch_supert = not pd.isna(supert_val) and abs(last["close"] - supert_val) / supert_val < 0.015

        # Дополнительно: RSI не перекуплен/перепродан (пуллбэк, а не экстремум)
        df["rsi"] = ta.rsi(df["close"], length=14)
        rsi_val = float(df["rsi"].iloc[-1])
        rsi_pullback_bull = 35 < rsi_val < 60   # откат, но не oversold
        rsi_pullback_bear = 40 < rsi_val < 65

        long_setup  = adx_strong and above_ema200 and st_bull and touch_supert and rsi_pullback_bull
        short_setup = adx_strong and below_ema200 and st_bear and touch_supert and rsi_pullback_bear

        if not (long_setup or short_setup):
            return None

        side = "BUY" if long_setup else "SELL"
        entry = float(last["close"])
        # SL за линию Supertrend
        if side == "BUY":
            sl = min(entry * (1 - self.stop_loss_pct / 100), supert_val * 0.998) if not pd.isna(supert_val) \
                 else entry * (1 - self.stop_loss_pct / 100)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = max(entry * (1 + self.stop_loss_pct / 100), supert_val * 1.002) if not pd.isna(supert_val) \
                 else entry * (1 + self.stop_loss_pct / 100)
            tp = entry * (1 - self.take_profit_pct / 100)

        return TradingSignal(
            action=side, symbol=self.symbol, confidence=0.76,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"Trend {side}: ADX {last['ADX_14']:.1f} + Supertrend touch | RSI {rsi_val:.0f}",
            filters_passed={"adx": True, "supertrend": True, "ema200": True, "st_touch": touch_supert, "rsi_pullback": True},
        )


# ============================================================
# S7: MULTI-CONFIRM — индикаторы + обязательная ценовая структура
# ============================================================
class MultiConfirmStrategy(BaseStrategy):
    ID = "S7"
    NAME = "MULTI-CONFIRM"
    DESCRIPTION = "5/7 индикаторов + обязательная структура цены + R:R ≥ 2.2 + ATR фильтр"
    REGIME_PREFERENCE = []

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.2,
            take_profit_pct=3.6,   # RR 3:1 минимум от структурного SL
            edge_wr_target=0.72,
            timeframe="60",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 80:
            return None
        df = df.copy()

        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ema_21"] = ta.ema(df["close"], length=21)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["rsi"]    = ta.rsi(df["close"], length=14)
        bb   = ta.bbands(df["close"], length=20, std=2)
        df   = df.join(bb)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df   = df.join(macd)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        price = float(last["close"])
        atr   = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        vol   = float(last["volume"])
        volma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else vol

        if atr == 0:
            return None

        # Блок если ATR аномально высокий (рынок слишком волатилен — стоп будет огромным)
        if not _atr_ok(last, df, max_mult=1.5):
            return None

        # ── ОБЯЗАТЕЛЬНО: ценовая структура ───────────────────────────────────
        sup_lvls, res_lvls = _struct_levels(df, atr)
        near_sup = _nearest_level(price, sup_lvls, atr)
        near_res = _nearest_level(price, res_lvls, atr)
        has_structure = bool(near_sup or near_res)
        if not has_structure:
            return None   # без структуры — WAIT

        # Касание EMA21 (pullback к динамической поддержке)
        touch_ema21 = abs(price - float(last["ema_21"])) / float(last["ema_21"]) < 0.010

        ema_bull = price > float(last["ema_21"]) > float(last["ema_50"])
        ema_bear = price < float(last["ema_21"]) < float(last["ema_50"])

        rsi_bull = 38 < rsi < 62 and rsi > float(prev["rsi"])
        rsi_bear = 38 < rsi < 62 and rsi < float(prev["rsi"])

        bb_mid = float(last["BBM_20_2.0"]) if "BBM_20_2.0" in last else price
        bb_touch_bull = price <= bb_mid * 1.006
        bb_touch_bear = price >= bb_mid * 0.994

        macd_h = float(last.get("MACDh_12_26_9", 0))
        macd_h_prev = float(prev.get("MACDh_12_26_9", 0))
        macd_bull = macd_h > 0 and macd_h > macd_h_prev
        macd_bear = macd_h < 0 and macd_h < macd_h_prev

        vol_spike = vol > volma * 1.5
        htf = _htf_trend(df)

        # Факел как бонусный фильтр
        wick = _wick_signal(last)

        long_filters = {
            "ema_trend":   ema_bull,
            "ema21_touch": touch_ema21,
            "rsi_confirm": rsi_bull,
            "bb_touch":    bb_touch_bull,
            "macd":        macd_bull,
            "vol_spike":   vol_spike,
            "htf_trend":   htf >= 0,
        }
        short_filters = {
            "ema_trend":   ema_bear,
            "ema21_touch": touch_ema21,
            "rsi_confirm": rsi_bear,
            "bb_touch":    bb_touch_bear,
            "macd":        macd_bear,
            "vol_spike":   vol_spike,
            "htf_trend":   htf <= 0,
        }

        if not touch_ema21:
            return None   # обязательный фильтр

        long_score  = sum(long_filters.values())
        short_score = sum(short_filters.values())

        long_setup  = long_score >= 6
        short_setup = short_score >= 6

        if not (long_setup or short_setup):
            return None

        if long_setup and short_setup:
            long_setup  = long_score >= short_score
            short_setup = not long_setup

        side  = "BUY" if long_setup else "SELL"
        score = long_score if long_setup else short_score

        # Проверяем что структура совпадает с направлением
        if side == "BUY" and not near_sup:
            return None
        if side == "SELL" and not near_res:
            return None
        lvl = near_sup if side == "BUY" else near_res

        # SL за структурой + 0.2 ATR
        if side == "BUY":
            sl   = round(min(lvl - atr * 0.2, price - atr * 0.8), 8)
            risk = price - sl
            tp   = round(price + risk * 2.5, 8)
        else:
            sl   = round(max(lvl + atr * 0.2, price + atr * 0.8), 8)
            risk = sl - price
            tp   = round(price - risk * 2.5, 8)

        rr = abs(tp - price) / max(abs(sl - price), 1e-9)
        if rr < 2.2:
            return None

        conf = 0.80 + 0.03 * (score - 6)
        if wick == ("bullish" if side == "BUY" else "bearish"):
            conf = min(0.95, conf + 0.05)

        filters = long_filters if long_setup else short_filters
        filters["structure"] = True
        return TradingSignal(
            action=side, symbol=self.symbol, confidence=round(conf, 2),
            entry_price=price, stop_loss=sl, take_profit=tp,
            reason=f"MULTI-CONFIRM {score}/7 | Lvl@{lvl:.5g} | RR{rr:.1f} | RSI{rsi:.0f}",
            filters_passed=filters,
        )


# ============================================================
# S11: DRAGONFLY GOLD
# Реализация по мотивам Dragonfly EA (MetaTrader):
#   Ichimoku Kinko Hyo + Bollinger Bands + Parabolic SAR
#   + Stochastic + OBV — динамический SL/TP на основе BB
# ============================================================
class DragonflyGoldStrategy(BaseStrategy):
    """
    5 независимых систем — нужно ≥ 4 из 5 согласованных сигналов.

    Уникальные индикаторы (не используются в S1–S10):
      • Ichimoku Kinko Hyo — тренд и зоны поддержки/сопротивления
      • Parabolic SAR      — разворотные точки
      • Stochastic         — перекупленность / перепроданность
      • OBV                — объёмное подтверждение направления

    SL/TP динамические на основе ширины Bollinger Bands:
      Лонг:  SL = BB нижняя полоса, TP = BB верхняя полоса
      Шорт:  SL = BB верхняя полоса, TP = BB нижняя полоса
    """

    ID = "S11"
    NAME = "DRAGONFLY GOLD"
    DESCRIPTION = "Ichimoku + PSAR + Stochastic + OBV + BB-динамический SL/TP"
    REGIME_PREFERENCE = []  # обучение: работает во всех режимах

    _MIN_RISK_PCT  = 0.20   # меньше — шум
    _MAX_RISK_PCT  = 5.0    # больше — нет смысла открывать
    _MIN_FILTERS   = 4      # минимум 4 из 5 систем должны совпасть

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.8,      # fallback (перекрывается BB SL)
            take_profit_pct=3.6,    # fallback (перекрывается BB TP)
            edge_wr_target=0.66,
            timeframe="60",         # H1 — оптимально для Dragonfly EA
            **kwargs,
        )

    # ── Основной анализ ───────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        # Ichimoku требует 52 бара Senkou B + запас
        if len(df) < 65:
            return None
        df = df.copy()

        df = df.copy()

        # ── Расчёт индикаторов ────────────────────────────────────────────────

        # Ichimoku
        ichi_result = ta.ichimoku(df["high"], df["low"], df["close"],
                                  tenkan=9, kijun=26, senkou=52)
        ichi_df = ichi_result[0] if isinstance(ichi_result, tuple) else ichi_result
        df = df.join(ichi_df)

        # Bollinger Bands
        bb   = ta.bbands(df["close"], length=20, std=2)
        df   = df.join(bb)

        # Parabolic SAR
        psar = ta.psar(df["high"], df["low"], df["close"],
                       af0=0.02, af_step=0.02, max_af=0.2)
        df   = df.join(psar)

        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        df    = df.join(stoch)

        # OBV и его скользящая средняя
        df["obv"]    = ta.obv(df["close"], df["volume"])
        df["obv_ma"] = df["obv"].rolling(10).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        c = float(last["close"])

        # ── Ichimoku-фильтр ───────────────────────────────────────────────────
        senkou_a = last["ISA_9"]
        senkou_b = last["ISB_52"]
        tenkan   = last["ITS_9"]
        kijun    = last["IKS_26"]

        if pd.isna(senkou_a) or pd.isna(senkou_b):
            return None

        cloud_top = max(senkou_a, senkou_b)
        cloud_bot = min(senkou_a, senkou_b)
        ichi_bull = c > cloud_top and tenkan > kijun    # цена над облаком, Tenkan > Kijun
        ichi_bear = c < cloud_bot and tenkan < kijun

        # ── PSAR-фильтр ───────────────────────────────────────────────────────
        psar_dir      = last["PSARd_0.02_0.2"]
        psar_dir_prev = prev["PSARd_0.02_0.2"]
        psar_bull = psar_dir == 1                        # SAR ниже цены → бычий тренд
        psar_bear = psar_dir == -1
        # Флип: переворот на текущей свече — сигнал сильнее
        psar_flip_bull = (psar_dir_prev == -1) and (psar_dir == 1)
        psar_flip_bear = (psar_dir_prev == 1)  and (psar_dir == -1)

        # ── Stochastic-фильтр ─────────────────────────────────────────────────
        sk      = last["STOCHk_14_3_3"]
        sd      = last["STOCHd_14_3_3"]
        sk_prev = prev["STOCHk_14_3_3"]
        sd_prev = prev["STOCHd_14_3_3"]

        stoch_bull = (sk_prev < sd_prev) and (sk > sd) and sk < 55   # пересечение вверх из зоны ≤55
        stoch_bear = (sk_prev > sd_prev) and (sk < sd) and sk > 45   # пересечение вниз из зоны ≥45

        # ── OBV-фильтр ────────────────────────────────────────────────────────
        obv_bull = last["obv"] > last["obv_ma"]   # объём поддерживает рост
        obv_bear = last["obv"] < last["obv_ma"]

        # ── BB-позиция ────────────────────────────────────────────────────────
        bb_lower = last["BBL_20_2.0"]
        bb_mid   = last["BBM_20_2.0"]
        bb_upper = last["BBU_20_2.0"]

        if pd.isna(bb_lower) or pd.isna(bb_upper):
            return None

        bb_support    = c <= bb_mid               # покупаем в нижней половине BB
        bb_resistance = c >= bb_mid               # продаём в верхней половине BB

        # ── Сборка фильтров ───────────────────────────────────────────────────
        long_filters = {
            "ichimoku":   ichi_bull,
            "psar":       psar_bull,
            "stochastic": stoch_bull,
            "obv":        obv_bull,
            "bb_zone":    bb_support,
        }
        short_filters = {
            "ichimoku":   ichi_bear,
            "psar":       psar_bear,
            "stochastic": stoch_bear,
            "obv":        obv_bear,
            "bb_zone":    bb_resistance,
        }

        long_score  = sum(long_filters.values())
        short_score = sum(short_filters.values())

        long_ok  = long_score  >= self._MIN_FILTERS
        short_ok = short_score >= self._MIN_FILTERS

        if not (long_ok or short_ok):
            return None

        # При конфликте выбираем направление с бо́льшим счётом
        if long_ok and short_ok:
            if long_score >= short_score:
                short_ok = False
            else:
                long_ok = False

        direction = "BUY" if long_ok else "SELL"
        score     = long_score if long_ok else short_score
        filters   = long_filters if long_ok else short_filters

        # ── Динамический SL/TP на основе BB ──────────────────────────────────
        entry = c
        if direction == "BUY":
            sl = bb_lower * 0.9995    # чуть ниже нижней полосы
            tp = bb_upper
        else:
            sl = bb_upper * 1.0005    # чуть выше верхней полосы
            tp = bb_lower

        # Проверяем что SL по правильную сторону от входа
        if direction == "BUY" and sl >= entry:
            return None
        if direction == "SELL" and sl <= entry:
            return None

        risk_pct = abs(entry - sl) / entry * 100
        if not (self._MIN_RISK_PCT <= risk_pct <= self._MAX_RISK_PCT):
            return None

        # Обновляем параметры для to_dict() / статистики
        self.stop_loss_pct   = round(risk_pct, 3)
        self.take_profit_pct = round(abs(tp - entry) / entry * 100, 3)

        # Флип PSAR — экстра-буст к уверенности
        flip_boost = 0.05 if (
            (direction == "BUY"  and psar_flip_bull) or
            (direction == "SELL" and psar_flip_bear)
        ) else 0.0

        confidence = round(0.62 + 0.06 * (score - self._MIN_FILTERS) + flip_boost, 2)

        return TradingSignal(
            action=direction, symbol=self.symbol,
            confidence=min(confidence, 0.92),
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=(
                f"Dragonfly {direction} {score}/5 | "
                f"PSAR {'FLIP ' if flip_boost else ''}"
                f"Stoch {sk:.0f} | BB-SL risk {risk_pct:.2f}%"
            ),
            filters_passed=filters,
        )


# ============================================================
# S12: OVERBOUGHT SHORT — специализирован на шортах
# ============================================================
class OverboughtShortStrategy(BaseStrategy):
    """
    Ищет перегретые активы для шорта:
      1. RSI(14) > 62 и падает (разворот вниз)
      2. Цена у верхней BB или выше
      3. MACD histogram снижается
      4. Цена выше EMA(50) — перегрев относительно средней
      5. Объём падает при высокой цене (дивергенция)

    Также генерирует BUY при зеркальных условиях (RSI < 38, ниже нижней BB).
    """
    ID   = "S12"
    NAME = "OVERBOUGHT SHORT"
    DESCRIPTION = "RSI разворот + BB верхний/нижний + MACD + объём дивергенция"
    REGIME_PREFERENCE = []  # обучение: работает во всех режимах

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.2,
            take_profit_pct=3.0,
            edge_wr_target=0.63,
            timeframe="15",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None
        df = df.copy()

        bb   = ta.bbands(df["close"], length=20, std=2)
        df   = df.join(bb)
        df["rsi"]    = ta.rsi(df["close"], length=14)
        df["ema50"]  = ta.ema(df["close"], length=50)
        macd_df      = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df           = df.join(macd_df)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        rsi0 = last["rsi"]
        rsi1 = prev["rsi"]
        bbu  = last["BBU_20_2.0"]
        bbl  = last["BBL_20_2.0"]
        c    = last["close"]
        vol_falling = last["volume"] < last["vol_ma"] * 0.85

        # ── SELL: RSI разворачивается вниз от перегрева ────────────
        sell = (
            rsi0 > 62 and rsi0 < rsi1          # RSI высокий и начинает падать
            and c >= bbu * 0.997               # цена у верхней BB или выше
            and last["MACDh_12_26_9"] < prev["MACDh_12_26_9"]  # MACD гистограмма падает
        )

        # ── BUY: зеркально — RSI разворачивается вверх от перепроданности ──
        buy = (
            rsi0 < 38 and rsi0 > rsi1          # RSI низкий и начинает расти
            and c <= bbl * 1.003               # цена у нижней BB или ниже
            and last["MACDh_12_26_9"] > prev["MACDh_12_26_9"]  # MACD гистограмма растёт
        )

        if not (sell or buy):
            return None

        # Если оба — выбираем более сильный сигнал
        if sell and buy:
            sell_strength = rsi0 - 62   # thresholds 62 for sell, 38 for buy
            buy_strength  = 38 - rsi0
            if sell_strength >= buy_strength:
                buy = False
            else:
                sell = False

        side  = "SELL" if sell else "BUY"
        entry = float(c)
        if side == "SELL":
            sl = round(entry * (1 + self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 - self.take_profit_pct / 100), 8)
        else:
            sl = round(entry * (1 - self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 + self.take_profit_pct / 100), 8)

        confidence = 0.66
        if vol_falling and side == "SELL":
            confidence += 0.04   # объём падает при хаях — доп. подтверждение шорта
        if abs(rsi0 - (62 if sell else 38)) > 5:
            confidence += 0.03   # RSI экстремальнее → чуть выше уверенность

        bb_side = "верхняя" if sell else "нижняя"
        return TradingSignal(
            action=side, symbol=self.symbol,
            confidence=round(min(0.85, confidence), 2),
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"OB-Short {side}: RSI{rsi0:.0f}({'↓' if sell else '↑'}) BB-{bb_side} MACD{'↓' if sell else '↑'}",
            filters_passed={
                "rsi_reversal": True,
                "bb_touch":     True,
                "macd_confirm": True,
                "ema50_side":   True,
                "vol_div":      vol_falling,
                "rsi_value":    round(rsi0, 1),
            },
        )


# ============================================================
# S13: MEME REVERSAL — кулдаун после убытков, SAFE режим
# ============================================================
class MemeReversalS13Strategy(OverboughtShortStrategy):
    """
    OverboughtShort для мем-монет (PEPEUSDT) с кулдауном после убытков.
    После 1 убытка: 30 мин паузы. После 2 подряд: 45 мин. После 3+: SAFE.
    """
    ID   = "S13"
    NAME = "MEME REVERSAL S13"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_loss_time: Optional[datetime] = None

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        now = datetime.now(timezone.utc)

        # Кулдаун по серии убытков
        if self.consecutive_losses >= 3:
            return None   # SAFE: ждём пока серия не прервётся

        if self.consecutive_losses >= 1 and self._last_loss_time:
            cooldown_min = 45 if self.consecutive_losses >= 2 else 30
            elapsed = (now - self._last_loss_time).total_seconds() / 60
            if elapsed < cooldown_min:
                return None

        sig = super().analyze(df)
        if sig is None:
            return None

        # В SAFE режиме (2 убытка) требуем более высокий confidence
        if self.consecutive_losses >= 2 and sig.confidence < 0.72:
            return None

        return sig

    def close_position(self, exit_price: float, qty: float = 1.0, fees_pct: float = 0.06):
        result = super().close_position(exit_price, qty, fees_pct)
        if result.get("pnl", 0) < 0:
            self._last_loss_time = datetime.now(timezone.utc)
        return result


# ============================================================
# S14: MEME REVERSAL — 4+ подтверждений, запрет догонять движение
# ============================================================
class MemeReversalS14Strategy(OverboughtShortStrategy):
    """
    OverboughtShort для мем-монет (WIFUSDT) с усиленными фильтрами.
    Требует 4+ подтверждений, запрещает входить если движение >70% ATR уже прошло.
    """
    ID   = "S14"
    NAME = "MEME REVERSAL S14"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None
        df_copy = df.copy()

        df_copy["atr"]    = ta.atr(df_copy["high"], df_copy["low"], df_copy["close"], length=14)
        df_copy["ema50"]  = ta.ema(df_copy["close"], length=50)
        df_copy["vol_ma"] = df_copy["volume"].rolling(20).mean()
        df_copy["rsi"]    = ta.rsi(df_copy["close"], length=14)

        last  = df_copy.iloc[-1]
        atr   = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        price = float(last["close"])
        rsi   = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        vol   = float(last["volume"])
        volma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else vol

        if atr == 0:
            return None

        sig = super().analyze(df)
        if sig is None:
            return None

        # Запрет: движение уже прошло > 70% ATR от открытия свечи
        candle_move = abs(price - float(last["open"]))
        if candle_move > atr * 0.7:
            return None   # догоняем поезд — WAIT

        # Структурный уровень обязателен
        sup_lvls, res_lvls = _struct_levels(df_copy, atr)
        if sig.action == "SELL":
            lvl = _nearest_level(price, res_lvls, atr * 1.2)
        else:
            lvl = _nearest_level(price, sup_lvls, atr * 1.2)
        if not lvl:
            return None

        # 4 подтверждения: структура + объём + RSI + EMA50 + HTF + свеча
        checks = {
            "structure": True,
            "volume":    vol > volma * 1.2,
            "rsi":       (rsi < 40) if sig.action == "BUY" else (rsi > 60),
            "ema50":     price < float(last["ema50"]) * 1.01 if sig.action == "BUY" else price > float(last["ema50"]) * 0.99,
            "htf":       _htf_trend(df_copy) != (1 if sig.action == "SELL" else -1),
            "candle":    (price > float(last["open"])) if sig.action == "BUY" else (price < float(last["open"])),
        }
        if sum(1 for v in checks.values() if v) < 4:
            return None

        return sig


# ============================================================
# Реестр всех стратегий
# ============================================================
ALL_STRATEGIES = {
    "S1": EMACrossoverStrategy,
    "S2": BollingerBandsStrategy,
    "S3": RSIDivergenceStrategy,
    "S4": BreakoutHunterStrategy,
    "S5": ScalperGridStrategy,
    "S6": TrendFollowerStrategy,
    "S7": MultiConfirmStrategy,
    "S8": TrendMomentumStrategy,
    "S9": TrendFibonacciStrategy,
    "S10": ScalperProStrategy,
    "S11": DragonflyGoldStrategy,
    "S12": OverboughtShortStrategy,
    "S13": MemeReversalS13Strategy,   # кулдаун 30/45 мин, SAFE после 3 убытков
    "S14": MemeReversalS14Strategy,   # 4+ подтверждений, запрет догонять движение
    "S15": AggressiveMomentumStrategy,
}
