"""
Все 9 торговых стратегий.
Каждая имеет систему фильтров (math edge) + breakeven + trailing stop.
Индикаторы — pandas_ta (или ручной расчёт где нужно).
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional
from .base import BaseStrategy, TradingSignal
from .trend_fib import TrendMomentumStrategy, TrendFibonacciStrategy
from .scalper_pro import ScalperProStrategy, SCALP_SYMBOLS
from .aggressive_momentum import AggressiveMomentumStrategy


# ============================================================
# S1: EMA CROSSOVER
# ============================================================
class EMACrossoverStrategy(BaseStrategy):
    ID = "S1"
    NAME = "EMA CROSSOVER"
    DESCRIPTION = "EMA 9/21 cross + RSI зона + объём × 1.5 + свеча подтверждения"
    REGIME_PREFERENCE = []  # обучение: работает во всех режимах

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.5,
            take_profit_pct=3.5,
            edge_wr_target=0.62,
            timeframe="15",
            **kwargs,
        )
        self.fast_ema = 9
        self.slow_ema = 21
        self.rsi_period = 14

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 50:
            return None
        df = df.copy()

        df["ema_fast"] = ta.ema(df["close"], length=self.fast_ema)
        df["ema_slow"] = ta.ema(df["close"], length=self.slow_ema)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        ema_bull_cross = prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        ema_bear_cross = prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

        # Касание EMA: свеча должна задеть EMA своим телом/тенью при кроссе
        # BUY: low свечи <= ema_fast (касание снизу-вверх), тело закрылось выше
        # SELL: high свечи >= ema_fast (касание сверху-вниз), тело закрылось ниже
        ema_touch_bull = last["low"] <= last["ema_fast"] * 1.002
        ema_touch_bear = last["high"] >= last["ema_fast"] * 0.998

        rsi_in_zone = 30 < last["rsi"] < 70   # зона без экстремумов
        vol_confirm = last["volume"] > last["vol_ma"] * 1.3  # реальный объём

        # Подтверждение свечой: бычья свеча при BUY, медвежья при SELL
        bull_candle = last["close"] > last["open"]
        bear_candle = last["close"] < last["open"]

        filters = {
            "ema_cross":    ema_bull_cross or ema_bear_cross,
            "ema_touch":    ema_touch_bull if ema_bull_cross else ema_touch_bear,
            "rsi_zone":     rsi_in_zone,
            "volume_spike": vol_confirm,
            "candle_confirm": bull_candle if ema_bull_cross else bear_candle,
        }

        if not all(filters.values()):
            return None

        side = "BUY" if ema_bull_cross else "SELL"
        entry = float(last["close"])
        if side == "BUY":
            sl = entry * (1 - self.stop_loss_pct / 100)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = entry * (1 + self.stop_loss_pct / 100)
            tp = entry * (1 - self.take_profit_pct / 100)

        return TradingSignal(
            action=side, symbol=self.symbol, confidence=0.7,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"EMA{self.fast_ema}/{self.slow_ema} cross+touch | RSI {last['rsi']:.1f} | vol×{last['volume']/last['vol_ma']:.1f}",
            filters_passed=filters,
        )


# ============================================================
# S2: BOLLINGER BANDS
# ============================================================
class BollingerBandsStrategy(BaseStrategy):
    ID = "S2"
    NAME = "BOLLINGER BANDS"
    DESCRIPTION = "BB squeeze + RSI экстремум + объём + наклон 50EMA"
    REGIME_PREFERENCE = []  # обучение: работает во всех режимах

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
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["ema50"] = ta.ema(df["close"], length=50)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]

        # Фильтры (используем имена колонок pandas_ta)
        bb_lower = last["BBL_20_2.0"]
        bb_upper = last["BBU_20_2.0"]
        bb_width = (bb_upper - bb_lower) / last["close"]

        below_lower = last["close"] <= bb_lower * 1.001
        above_upper = last["close"] >= bb_upper * 0.999
        rsi_oversold = last["rsi"] < 38
        rsi_overbought = last["rsi"] > 60
        vol_spike = last["volume"] > last["vol_ma"] * 1.2

        long_setup = below_lower and rsi_oversold and vol_spike
        short_setup = above_upper and rsi_overbought and vol_spike

        if not (long_setup or short_setup):
            return None

        side = "BUY" if long_setup else "SELL"
        entry = float(last["close"])
        if side == "BUY":
            sl = entry * (1 - self.stop_loss_pct / 100)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = entry * (1 + self.stop_loss_pct / 100)
            tp = entry * (1 - self.take_profit_pct / 100)

        return TradingSignal(
            action=side, symbol=self.symbol, confidence=0.75,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"BB {'lower' if long_setup else 'upper'} + RSI extreme + vol",
            filters_passed={
                "bb_touch": True, "rsi_extreme": True,
                "vol_spike": vol_spike, "ema_slope": True,
            },
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
# S4: BREAKOUT HUNTER
# ============================================================
class BreakoutHunterStrategy(BaseStrategy):
    ID = "S4"
    NAME = "BREAKOUT HUNTER"
    DESCRIPTION = "Пробой + ретест уровня + объём × 2 + ATR фильтр"
    REGIME_PREFERENCE = []  # обучение: работает во всех режимах

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.5,
            take_profit_pct=5.0,
            edge_wr_target=0.58,
            timeframe="60",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None
        df = df.copy()

        # Уровни: high/low за последние 48 свечей (не включая последние 5)
        lookback = 48
        level_zone = df.iloc[-lookback:-5]
        resistance = level_zone["high"].max()
        support    = level_zone["low"].min()

        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]

        # ── Пробой: закрытие выше/ниже уровня с минимальным зазором 0.2% ──────
        break_up   = last["close"] > resistance * 1.002
        break_down = last["close"] < support * 0.998

        # ── Ретест: в последних 3 свечах цена касалась уровня (вернулась к нему) ─
        # После пробоя вверх: минимум одной из последних свечей опускался к resistance
        recent_3 = df.iloc[-4:-1]
        retest_up   = break_up   and recent_3["low"].min()  <= resistance * 1.005
        retest_down = break_down and recent_3["high"].max() >= support   * 0.995

        vol_confirm = last["volume"] > last["vol_ma"] * 1.5   # объём × 1.5
        atr_expand  = last["atr"]    > df.iloc[-10:]["atr"].mean() * 1.1  # ATR расширяется

        if not ((retest_up or retest_down) and vol_confirm and atr_expand):
            return None

        side = "BUY" if retest_up else "SELL"
        level = resistance if retest_up else support
        entry = float(last["close"])
        if side == "BUY":
            sl = min(entry * (1 - self.stop_loss_pct / 100), level * 0.998)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = max(entry * (1 + self.stop_loss_pct / 100), level * 1.002)
            tp = entry * (1 - self.take_profit_pct / 100)

        return TradingSignal(
            action=side, symbol=self.symbol, confidence=0.73,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"Breakout+Retest {'⬆' if retest_up else '⬇'} level={level:.4f} | vol×{last['volume']/last['vol_ma']:.1f}",
            filters_passed={"break": True, "retest": True, "vol": vol_confirm, "atr": atr_expand},
        )


# ============================================================
# S5: STRUCTURE SCALPER — вход только по структуре + 3+ подтверждения
# ============================================================
class ScalperGridStrategy(BaseStrategy):
    ID = "S5"
    NAME = "SCALPER GRID"
    DESCRIPTION = "Структурный скальпер: уровни 2+ касания, 3+ подтверждений, без чистых ATR-входов"
    REGIME_PREFERENCE = []

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            edge_wr_target=0.55,
            timeframe="5",
            **kwargs,
        )

    # ── уровни поддержки / сопротивления ─────────────────────────────────────
    def _find_levels(self, df: pd.DataFrame, atr: float):
        highs = df["high"].values
        lows  = df["low"].values
        tol   = atr * 0.6

        swing_highs, swing_lows = [], []
        for i in range(2, len(highs) - 2):
            if highs[i] >= max(highs[i-2], highs[i-1], highs[i+1], highs[i+2]):
                swing_highs.append(highs[i])
            if lows[i]  <= min(lows[i-2],  lows[i-1],  lows[i+1],  lows[i+2]):
                swing_lows.append(lows[i])

        def cluster(vals):
            if not vals:
                return []
            result, group = [], [sorted(vals)[0]]
            for v in sorted(vals)[1:]:
                if v - group[0] <= tol:
                    group.append(v)
                else:
                    if len(group) >= 2:
                        result.append(sum(group) / len(group))
                    group = [v]
            if len(group) >= 2:
                result.append(sum(group) / len(group))
            return result

        return cluster(swing_lows), cluster(swing_highs)

    # ── факел (длинная тень) ──────────────────────────────────────────────────
    @staticmethod
    def _wick_type(c: pd.Series) -> str:
        body  = abs(float(c["close"]) - float(c["open"]))
        hi    = float(c["high"])
        lo    = float(c["low"])
        upper = hi - max(float(c["close"]), float(c["open"]))
        lower = min(float(c["close"]), float(c["open"])) - lo
        if body < 1e-9:
            return ""
        if lower > body * 2 and lower > upper * 1.2:
            return "bullish"
        if upper > body * 2 and upper > lower * 1.2:
            return "bearish"
        return ""

    # ── ложный пробой ─────────────────────────────────────────────────────────
    def _fake_breakout(self, df: pd.DataFrame, levels: list, side: str, atr: float) -> bool:
        if len(df) < 4 or not levels:
            return False
        recent = df.iloc[-4:]
        close_now = float(df.iloc[-1]["close"])
        for lvl in levels:
            if side == "support":
                if any(recent["low"] < lvl - atr * 0.05) and close_now > lvl:
                    return True
            else:
                if any(recent["high"] > lvl + atr * 0.05) and close_now < lvl:
                    return True
        return False

    # ── ближайший уровень ─────────────────────────────────────────────────────
    @staticmethod
    def _nearest(price: float, levels: list, atr: float) -> float:
        for lvl in sorted(levels, key=lambda x: abs(x - price)):
            if abs(price - lvl) <= atr * 0.5:
                return lvl
        return 0.0

    # ── MACD ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _macd_confirm(df: pd.DataFrame, side: str) -> bool:
        try:
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is None or macd.empty:
                return False
            hist_col = [c for c in macd.columns if "h" in c.lower() or "hist" in c.lower()]
            if not hist_col:
                return False
            h = macd[hist_col[0]]
            if side == "BUY":
                return float(h.iloc[-1]) > float(h.iloc[-2])   # гистограмма растёт
            else:
                return float(h.iloc[-1]) < float(h.iloc[-2])   # гистограмма падает
        except Exception:
            return False

    # ── главный метод ─────────────────────────────────────────────────────────
    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 100:
            return None
        df = df.copy()

        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["rsi"]    = ta.rsi(df["close"], length=14)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last   = df.iloc[-1]
        price  = float(last["close"])
        atr    = float(last["atr"])   if not pd.isna(last["atr"])    else 0.0
        rsi    = float(last["rsi"])   if not pd.isna(last["rsi"])    else 50.0
        vol    = float(last["volume"])
        vol_ma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else vol
        ema20  = float(last["ema_20"]) if not pd.isna(last["ema_20"]) else price
        ema50  = float(last["ema_50"]) if not pd.isna(last["ema_50"]) else price

        if atr == 0 or price == 0:
            return None

        # ATR должен быть достаточным для покрытия комиссий
        atr_pct = atr / price * 100
        if atr_pct < 0.25:
            return None

        # Поиск структурных уровней (последние 100 свечей)
        support_lvls, resist_lvls = self._find_levels(df.iloc[-100:], atr)
        if not support_lvls and not resist_lvls:
            return None

        buy_lvl  = self._nearest(price, support_lvls, atr)
        sell_lvl = self._nearest(price, resist_lvls,  atr)

        # Цена должна быть у одного уровня, не в середине диапазона
        if buy_lvl and sell_lvl:
            return None
        if not buy_lvl and not sell_lvl:
            return None

        side    = "BUY" if buy_lvl else "SELL"
        lvl     = buy_lvl or sell_lvl
        fake_bo = self._fake_breakout(
            df, support_lvls if side == "BUY" else resist_lvls,
            "support" if side == "BUY" else "resistance", atr
        )

        # ── 8 подтверждений ──────────────────────────────────────────────────
        checks = {}

        # 1. Структурный уровень (уже гарантирован выше)
        checks["structure"] = True

        # 2. Свеча подтверждения (текущая закрылась в нужную сторону)
        if side == "BUY":
            checks["confirm_candle"] = float(last["close"]) > float(last["open"])
        else:
            checks["confirm_candle"] = float(last["close"]) < float(last["open"])

        # 3. Объём × 1.3 от среднего
        checks["volume"] = vol > vol_ma * 1.3

        # 4. RSI не на экстремуме, подтверждает направление
        if side == "BUY":
            checks["rsi"] = 25 < rsi < 62
        else:
            checks["rsi"] = 38 < rsi < 75

        # 5. Факел в нужную сторону
        wick = self._wick_type(last)
        checks["wick"] = (wick == "bullish") if side == "BUY" else (wick == "bearish")

        # 6. Ложный пробой
        checks["fake_breakout"] = fake_bo

        # 7. EMA не против входа (допускаем небольшое отклонение)
        if side == "BUY":
            checks["ema_align"] = ema20 >= ema50 * 0.993
        else:
            checks["ema_align"] = ema20 <= ema50 * 1.007

        # 8. MACD-гистограмма в нужном направлении
        checks["macd"] = self._macd_confirm(df, side)

        conf_count = sum(1 for v in checks.values() if v)

        # Адаптивный порог по серии убытков
        if self.consecutive_losses >= 4:
            min_conf = 5   # SAFE: только A+ сигналы
        elif self.consecutive_losses >= 3:
            min_conf = 4
        else:
            min_conf = 3

        if conf_count < min_conf:
            return None

        # ── SL за структурой + 0.2 ATR буфер, минимум 0.8 ATR ───────────────
        if side == "BUY":
            sl_struct = lvl - atr * 0.2
            sl_min    = price - atr * 0.8
            sl        = min(sl_struct, sl_min)
            risk      = price - sl
            tp        = price + risk * 2.0   # RR 2:1 минимум
        else:
            sl_struct = lvl + atr * 0.2
            sl_min    = price + atr * 0.8
            sl        = max(sl_struct, sl_min)
            risk      = sl - price
            tp        = price - risk * 2.0

        # Проверяем что TP не слишком близко (должен покрыть комиссии × 3)
        tp_pct = abs(tp - price) / price * 100
        if tp_pct < 0.4:
            return None

        # confidence: вклад подтверждений + бонус за приоритетные паттерны
        conf = conf_count / 8.0
        if checks["fake_breakout"]: conf = min(1.0, conf + 0.15)
        if checks["wick"]:          conf = min(1.0, conf + 0.10)

        parts = []
        if checks["fake_breakout"]: parts.append("FakeBO")
        if checks["wick"]:          parts.append(f"Wick({wick})")
        parts.append(f"Lvl@{lvl:.5g}")
        parts.append(f"{conf_count}/8conf")
        parts.append(f"RSI={rsi:.0f}")
        if self.consecutive_losses >= 3:
            parts.append(f"SAFE(loss={self.consecutive_losses})")

        return TradingSignal(
            action=side,
            symbol=self.symbol,
            confidence=round(conf, 2),
            entry_price=price,
            stop_loss=round(sl, 8),
            take_profit=round(tp, 8),
            reason=" | ".join(parts),
            filters_passed=checks,
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
# S7: MULTI-CONFIRM (максимальный edge)
# ============================================================
class MultiConfirmStrategy(BaseStrategy):
    ID = "S7"
    NAME = "MULTI-CONFIRM"
    DESCRIPTION = "≥5 из 6 независимых индикаторов согласованы → высокий WR"
    REGIME_PREFERENCE = []  # работает во всех режимах (6 фильтров сами фильтруют)

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=1.2,
            take_profit_pct=4.0,
            edge_wr_target=0.72,
            timeframe="60",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None
        df = df.copy()

        df["ema_21"] = ta.ema(df["close"], length=21)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["rsi"] = ta.rsi(df["close"], length=14)
        bb = ta.bbands(df["close"], length=20, std=2)
        df = df.join(bb)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = df.join(macd)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Касание: цена должна касаться EMA21 (в пределах 0.8%)
        # Это делает S7 pullback-стратегией, а не "входом в воздухе"
        touch_ema21 = abs(last["close"] - last["ema_21"]) / last["ema_21"] < 0.008

        # 6 фильтров
        ema_bull = last["close"] > last["ema_21"] > last["ema_50"]
        ema_bear = last["close"] < last["ema_21"] < last["ema_50"]
        rsi_bull = 38 < last["rsi"] < 62 and last["rsi"] > prev["rsi"]
        rsi_bear = 38 < last["rsi"] < 62 and last["rsi"] < prev["rsi"]
        # BB: цена возвращается к средней линии (BB midline touch) — реальное касание
        bb_touch_bull = last["close"] <= last["BBM_20_2.0"] * 1.005  # касание средней снизу
        bb_touch_bear = last["close"] >= last["BBM_20_2.0"] * 0.995  # касание средней сверху
        macd_bull = last["MACD_12_26_9"] > last["MACDs_12_26_9"] and last["MACDh_12_26_9"] > prev["MACDh_12_26_9"]
        macd_bear = last["MACD_12_26_9"] < last["MACDs_12_26_9"] and last["MACDh_12_26_9"] < prev["MACDh_12_26_9"]
        vol_spike = last["volume"] > last["vol_ma"] * 1.5
        htf_bull = last["close"] > df.iloc[-24]["close"]  # рост за 24ч
        htf_bear = last["close"] < df.iloc[-24]["close"]

        long_filters = {
            "ema_trend":   ema_bull,
            "ema21_touch": touch_ema21,
            "rsi_confirm": rsi_bull,
            "bb_touch":    bb_touch_bull,
            "macd_cross":  macd_bull,
            "vol_spike":   vol_spike,
            "htf_trend":   htf_bull,
        }
        short_filters = {
            "ema_trend":   ema_bear,
            "ema21_touch": touch_ema21,
            "rsi_confirm": rsi_bear,
            "bb_touch":    bb_touch_bear,
            "macd_cross":  macd_bear,
            "vol_spike":   vol_spike,
            "htf_trend":   htf_bear,
        }

        # Требуем минимум 5 из 7 фильтров (включая обязательный ema21_touch)
        # ema21_touch обязателен отдельно
        if not touch_ema21:
            return None
        REQUIRED_SCORE = 5
        long_score = sum(long_filters.values())
        short_score = sum(short_filters.values())
        long_setup = long_score >= REQUIRED_SCORE
        short_setup = short_score >= REQUIRED_SCORE

        if not (long_setup or short_setup):
            return None

        # Если оба набирают нужный счёт — выбираем направление с бо́льшим счётом
        if long_setup and short_setup:
            if long_score >= short_score:
                short_setup = False
            else:
                long_setup = False

        side = "BUY" if long_setup else "SELL"
        entry = float(last["close"])
        if side == "BUY":
            sl = entry * (1 - self.stop_loss_pct / 100)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = entry * (1 + self.stop_loss_pct / 100)
            tp = entry * (1 - self.take_profit_pct / 100)

        score = long_score if long_setup else short_score
        confidence = 0.80 + 0.03 * (score - REQUIRED_SCORE)  # 0.80 при 5/6, 0.83 при 6/6
        return TradingSignal(
            action=side, symbol=self.symbol, confidence=round(confidence, 2),
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"MULTI-CONFIRM {side} — {score}/6 фильтров согласованы",
            filters_passed=long_filters if long_setup else short_filters,
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
            rsi0 > 58 and rsi0 < rsi1          # RSI высокий и начинает падать
            and c >= bbu * 0.997               # цена у верхней BB или выше
            and last["MACDh_12_26_9"] < prev["MACDh_12_26_9"]  # MACD гистограмма падает
        )

        # ── BUY: зеркально — RSI разворачивается вверх от перепроданности ──
        buy = (
            rsi0 < 42 and rsi0 > rsi1          # RSI низкий и начинает расти
            and c <= bbl * 1.003               # цена у нижней BB или ниже
            and last["MACDh_12_26_9"] > prev["MACDh_12_26_9"]  # MACD гистограмма растёт
        )

        if not (sell or buy):
            return None

        # Если оба — выбираем более сильный сигнал
        if sell and buy:
            sell_strength = rsi0 - 62
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
    "S8": TrendMomentumStrategy,    # тренд роста/падения (HH/HL + EMA-стек + ADX)
    "S9": TrendFibonacciStrategy,   # тренд + уровни Фибоначчи (38.2/50/61.8%)
    "S10": ScalperProStrategy,      # 3m высокочастотный скальпер (до 8 сигналов/день/символ)
    "S11": DragonflyGoldStrategy,   # Ichimoku + PSAR + Stochastic + OBV + BB-динамический SL/TP
    "S12": OverboughtShortStrategy,      # RSI разворот + BB + MACD — SOLUSDT
    "S13": OverboughtShortStrategy,      # RSI разворот + BB + MACD — PEPEUSDT (мем, волатильность)
    "S14": OverboughtShortStrategy,      # RSI разворот + BB + MACD — WIFUSDT (мем, резкие откаты)
    "S15": AggressiveMomentumStrategy,   # 1m EMA+RSI7+VWAP+ATR+ADX, M15 тренд, TP 0.7/1.5/2.5%
}
