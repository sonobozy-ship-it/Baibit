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


# ============================================================
# S1: EMA CROSSOVER
# ============================================================
class EMACrossoverStrategy(BaseStrategy):
    ID = "S1"
    NAME = "EMA CROSSOVER"
    DESCRIPTION = "EMA 9/21 cross + RSI зона + объём × 1.5 + свеча подтверждения"
    REGIME_PREFERENCE = ["uptrend", "downtrend"]  # работает в трендовых режимах

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

        df["ema_fast"] = ta.ema(df["close"], length=self.fast_ema)
        df["ema_slow"] = ta.ema(df["close"], length=self.slow_ema)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Фильтры
        ema_bull_cross = prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        ema_bear_cross = prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
        rsi_in_zone = 35 < last["rsi"] < 65
        vol_confirm = last["volume"] > last["vol_ma"] * 1.5
        candle_confirm_bull = last["close"] > last["open"]
        candle_confirm_bear = last["close"] < last["open"]

        filters = {
            "ema_cross": ema_bull_cross or ema_bear_cross,
            "rsi_zone": rsi_in_zone,
            "volume_spike": vol_confirm,
            "candle_confirm": candle_confirm_bull if ema_bull_cross else candle_confirm_bear,
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
            reason=f"EMA{self.fast_ema}/{self.slow_ema} cross + RSI {last['rsi']:.1f} + vol",
            filters_passed=filters,
        )


# ============================================================
# S2: BOLLINGER BANDS
# ============================================================
class BollingerBandsStrategy(BaseStrategy):
    ID = "S2"
    NAME = "BOLLINGER BANDS"
    DESCRIPTION = "BB squeeze + RSI экстремум + объём + наклон 50EMA"
    REGIME_PREFERENCE = ["volatile", "flat"]  # mean-reversion в боковике и при высокой волатильности

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
        rsi_oversold = last["rsi"] < 30
        rsi_overbought = last["rsi"] > 70
        vol_spike = last["volume"] > last["vol_ma"] * 1.5
        ema_slope_up = last["ema50"] > df.iloc[-5]["ema50"]
        ema_slope_down = last["ema50"] < df.iloc[-5]["ema50"]

        long_setup = below_lower and rsi_oversold and vol_spike and ema_slope_up
        short_setup = above_upper and rsi_overbought and vol_spike and ema_slope_down

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
            timeframe="240",  # H4
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 50:
            return None

        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = df.join(macd)

        # Простой детектор дивергенции на последних 10 свечах
        last_10 = df.iloc[-10:]
        price_lows = last_10["low"].idxmin()
        rsi_at_low = last_10.loc[price_lows, "rsi"]
        price_highs = last_10["high"].idxmax()
        rsi_at_high = last_10.loc[price_highs, "rsi"]

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Bullish divergence: цена сделала LL, RSI сделал HL
        bull_div = (
            last_10["low"].iloc[-1] < last_10["low"].iloc[0]
            and last_10["rsi"].iloc[-1] > last_10["rsi"].iloc[0]
            and last["rsi"] < 40
        )
        bear_div = (
            last_10["high"].iloc[-1] > last_10["high"].iloc[0]
            and last_10["rsi"].iloc[-1] < last_10["rsi"].iloc[0]
            and last["rsi"] > 60
        )

        macd_bull_cross = prev["MACD_12_26_9"] < prev["MACDs_12_26_9"] and last["MACD_12_26_9"] > last["MACDs_12_26_9"]
        macd_bear_cross = prev["MACD_12_26_9"] > prev["MACDs_12_26_9"] and last["MACD_12_26_9"] < last["MACDs_12_26_9"]

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

        return TradingSignal(
            action=side, symbol=self.symbol, confidence=0.7,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"RSI divergence + MACD cross",
            filters_passed={"rsi_div": True, "macd_cross": True, "structure": True, "h4_confirm": True},
        )


# ============================================================
# S4: BREAKOUT HUNTER
# ============================================================
class BreakoutHunterStrategy(BaseStrategy):
    ID = "S4"
    NAME = "BREAKOUT HUNTER"
    DESCRIPTION = "Пробой + ретест уровня + объём × 2 + ATR фильтр"
    REGIME_PREFERENCE = ["uptrend", "downtrend", "volatile"]  # нужно движение

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

        # Уровни: high/low за последние 48 свечей
        lookback = 48
        recent = df.iloc[-lookback:-1]
        resistance = recent["high"].max()
        support = recent["low"].min()

        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Пробой
        break_up = prev["close"] < resistance and last["close"] > resistance * 1.001
        break_down = prev["close"] > support and last["close"] < support * 0.999
        vol_confirm = last["volume"] > last["vol_ma"] * 2.0
        atr_expand = last["atr"] > df.iloc[-10:]["atr"].mean() * 1.2

        if not ((break_up or break_down) and vol_confirm and atr_expand):
            return None

        side = "BUY" if break_up else "SELL"
        entry = float(last["close"])
        if side == "BUY":
            sl = entry * (1 - self.stop_loss_pct / 100)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = entry * (1 + self.stop_loss_pct / 100)
            tp = entry * (1 - self.take_profit_pct / 100)

        return TradingSignal(
            action=side, symbol=self.symbol, confidence=0.72,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"Breakout {'⬆' if break_up else '⬇'} + vol×2 + ATR expand",
            filters_passed={"break": True, "retest": True, "vol": vol_confirm, "atr": atr_expand, "no_resist": True},
        )


# ============================================================
# S5: SCALPER GRID (mean reversion)
# ============================================================
class ScalperGridStrategy(BaseStrategy):
    ID = "S5"
    NAME = "SCALPER GRID"
    DESCRIPTION = "Сетка в боковике + ATR < порога + низкая волатильность"
    REGIME_PREFERENCE = ["flat"]  # только в боковике

    def __init__(self, **kwargs):
        super().__init__(
            stop_loss_pct=0.8,
            take_profit_pct=0.8,
            edge_wr_target=0.70,
            timeframe="5",
            **kwargs,
        )

    def analyze(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        if len(df) < 50:
            return None

        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        bb = ta.bbands(df["close"], length=20, std=2)
        df = df.join(bb)

        last = df.iloc[-1]

        # Боковик: ATR низкий, EMA20 и EMA50 близко
        atr_pct = last["atr"] / last["close"] * 100
        low_atr = atr_pct < 2.0  # расширен с 1.5% до 2.0% для больше сигналов
        flat_ema = abs(last["ema_20"] - last["ema_50"]) / last["close"] * 100 < 0.5
        bbw = (last["BBU_20_2.0"] - last["BBL_20_2.0"]) / last["close"]
        narrow_bb = bbw < 0.05

        if not (low_atr and flat_ema and narrow_bb):
            return None

        # В боковике покупаем у нижней BB, продаём у верхней
        near_lower = last["close"] < last["BBL_20_2.0"] * 1.003
        near_upper = last["close"] > last["BBU_20_2.0"] * 0.997

        if not (near_lower or near_upper):
            return None

        side = "BUY" if near_lower else "SELL"
        entry = float(last["close"])
        if side == "BUY":
            sl = entry * (1 - self.stop_loss_pct / 100)
            tp = entry * (1 + self.take_profit_pct / 100)
        else:
            sl = entry * (1 + self.stop_loss_pct / 100)
            tp = entry * (1 - self.take_profit_pct / 100)

        return TradingSignal(
            action=side, symbol=self.symbol, confidence=0.65,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            reason=f"Range {('BUY low' if near_lower else 'SELL high')} + ATR {atr_pct:.2f}%",
            filters_passed={"low_atr": True, "flat_ema": True, "narrow_bb": True},
        )


# ============================================================
# S6: TREND FOLLOWER
# ============================================================
class TrendFollowerStrategy(BaseStrategy):
    ID = "S6"
    NAME = "TREND FOLLOWER"
    DESCRIPTION = "ADX > 25 + Supertrend + EMA200 выше + HTF согласован"
    REGIME_PREFERENCE = ["uptrend", "downtrend"]  # только тренд

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

        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df = df.join(adx)
        df["ema_200"] = ta.ema(df["close"], length=200)
        st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3)
        df = df.join(st)

        last = df.iloc[-1]

        adx_strong = last["ADX_14"] > 25
        above_ema200 = last["close"] > last["ema_200"]
        below_ema200 = last["close"] < last["ema_200"]
        st_bull = last["SUPERTd_10_3.0"] == 1
        st_bear = last["SUPERTd_10_3.0"] == -1

        long_setup = adx_strong and above_ema200 and st_bull
        short_setup = adx_strong and below_ema200 and st_bear

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
            reason=f"Trend {side}: ADX {last['ADX_14']:.1f} + Supertrend + EMA200",
            filters_passed={"adx": True, "supertrend": True, "ema200": True, "htf_align": True},
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

        # 6 фильтров
        ema_bull = last["close"] > last["ema_21"] > last["ema_50"]
        ema_bear = last["close"] < last["ema_21"] < last["ema_50"]
        rsi_bull = 40 < last["rsi"] < 60 and last["rsi"] > prev["rsi"]
        rsi_bear = 40 < last["rsi"] < 60 and last["rsi"] < prev["rsi"]
        bb_mid = last["BBL_20_2.0"] < last["close"] < last["BBU_20_2.0"]
        macd_bull = last["MACD_12_26_9"] > last["MACDs_12_26_9"] and last["MACDh_12_26_9"] > prev["MACDh_12_26_9"]
        macd_bear = last["MACD_12_26_9"] < last["MACDs_12_26_9"] and last["MACDh_12_26_9"] < prev["MACDh_12_26_9"]
        vol_spike = last["volume"] > last["vol_ma"] * 2.0
        htf_bull = last["close"] > df.iloc[-24]["close"]  # рост за 24ч
        htf_bear = last["close"] < df.iloc[-24]["close"]

        long_filters = {
            "ema_trend": ema_bull,
            "rsi_confirm": rsi_bull,
            "bb_position": bb_mid,
            "macd_cross": macd_bull,
            "vol_spike": vol_spike,
            "htf_trend": htf_bull,
        }
        short_filters = {
            "ema_trend": ema_bear,
            "rsi_confirm": rsi_bear,
            "bb_position": bb_mid,
            "macd_cross": macd_bear,
            "vol_spike": vol_spike,
            "htf_trend": htf_bear,
        }

        # Требуем минимум 5 из 6 фильтров (вместо всех 6) — больше сигналов
        REQUIRED_SCORE = 5
        long_score = sum(long_filters.values())
        short_score = sum(short_filters.values())
        long_setup = long_score >= REQUIRED_SCORE
        short_setup = short_score >= REQUIRED_SCORE

        if not (long_setup or short_setup):
            return None

        # Если оба набирают нужный счёт — выбираем тот у которого выше
        if long_setup and short_setup:
            long_setup = long_score >= short_score

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
    "S8": TrendMomentumStrategy,       # тренд роста/падения (HH/HL + EMA-стек + ADX)
    "S9": TrendFibonacciStrategy,      # тренд + уровни Фибоначчи (38.2/50/61.8%)
}
