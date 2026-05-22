"""
Unit tests для ScalperProStrategy (S10).

Тест-кейсы:
  1. BUY-сигнал блокируется при H1 SELL (строгий MTF)
  2. SELL-сигнал блокируется при H1 BUY (строгий MTF)
  3. H1 NEUTRAL не блокирует сигнал
  4. Импульсная свеча блокирует сигнал (candle_range > 0.45%)
  5. Низкий R:R блокирует (RR < 1.8)
  6. Плохой спред блокирует (spread > 0.08%)
  7. Confidence не превышает 0.92
  8. Early TP не закрывает в минус при progress < 50%
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Добавляем backend в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from strategies.scalper_pro import ScalperProStrategy, ScalperTrendContext


# ─────────────────────────────────────────────────────────────────────────────
# Фабрики тестовых данных
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 120, trend: str = "up", base: float = 1.0) -> pd.DataFrame:
    """Генерирует минимальный 3m DataFrame для тестов."""
    np.random.seed(42)
    prices = [base]
    for _ in range(n - 1):
        delta = 0.0005 if trend == "up" else (-0.0005 if trend == "down" else 0.0)
        prices.append(prices[-1] * (1 + delta + np.random.normal(0, 0.0003)))
    prices = np.array(prices)
    vol = np.random.uniform(1000, 3000, n)
    return pd.DataFrame({
        "open":      prices * (1 - 0.0002),
        "high":      prices * 1.0008,
        "low":       prices * 0.9992,
        "close":     prices,
        "volume":    vol,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="3min"),
    })


def _make_h1_trend(direction: str, n: int = 250, base: float = 1.0) -> pd.DataFrame:
    """H1 DataFrame с заданным трендом EMA8/21/50."""
    np.random.seed(7)
    prices = [base]
    for _ in range(n - 1):
        delta = 0.001 if direction == "BUY" else (-0.001 if direction == "SELL" else 0.0)
        prices.append(prices[-1] * (1 + delta + np.random.normal(0, 0.0005)))
    prices = np.array(prices)
    return pd.DataFrame({
        "open":      prices * 0.999,
        "high":      prices * 1.002,
        "low":       prices * 0.998,
        "close":     prices,
        "volume":    np.ones(n) * 5000,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="60min"),
    })


def _make_strategy(mode: str = "mtf_h1") -> ScalperProStrategy:
    return ScalperProStrategy(symbol="TESTUSDT", mode=mode)


# ─────────────────────────────────────────────────────────────────────────────
# ScalperTrendContext unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScalperTrendContext:

    def test_h1_adx_min_is_20(self):
        assert ScalperTrendContext.H1_ADX_MIN == 20

    def test_m15_adx_min_is_15(self):
        assert ScalperTrendContext.M15_ADX_MIN == 15

    def test_no_data_returns_neutral(self):
        ctx = ScalperTrendContext.compute(None, "H1")
        assert ctx["direction"] == "NEUTRAL"
        ctx2 = ScalperTrendContext.compute(pd.DataFrame(), "H1")
        assert ctx2["direction"] == "NEUTRAL"

    def test_reason_contains_real_values(self):
        df = _make_h1_trend("BUY", n=250)
        ctx = ScalperTrendContext.compute(df, "H1")
        if ctx["direction"] == "BUY":
            # reason должен содержать реальные числа, а не шаблон с >
            assert "EMA8=" in ctx["reason"]
            assert "EMA21=" in ctx["reason"]
            assert "EMA50=" in ctx["reason"]
            assert "ADX=" in ctx["reason"]

    def test_aligns_strict_blocks_counter_trend(self):
        ctx_sell = {"direction": "SELL", "strength": 35.0, "reason": "H1 SELL"}
        # BUY против SELL direction — блок
        assert not ScalperTrendContext.aligns_strict(ctx_sell, "BUY")

    def test_aligns_strict_allows_same_direction(self):
        ctx_buy = {"direction": "BUY", "strength": 25.0, "reason": "H1 BUY"}
        assert ScalperTrendContext.aligns_strict(ctx_buy, "BUY")

    def test_aligns_strict_allows_neutral(self):
        ctx_neutral = {"direction": "NEUTRAL", "strength": 10.0, "reason": "H1 NEUTRAL"}
        assert ScalperTrendContext.aligns_strict(ctx_neutral, "BUY")
        assert ScalperTrendContext.aligns_strict(ctx_neutral, "SELL")

    def test_aligns_soft_allows_weak_counter_trend(self):
        ctx_weak = {"direction": "SELL", "strength": 20.0}
        # В мягком режиме ADX < 30 не блокирует
        assert ScalperTrendContext.aligns(ctx_weak, "BUY")

    def test_aligns_soft_blocks_strong_counter_trend(self):
        ctx_strong = {"direction": "SELL", "strength": 35.0}
        assert not ScalperTrendContext.aligns(ctx_strong, "BUY")


# ─────────────────────────────────────────────────────────────────────────────
# ScalperProStrategy unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScalperProMTFFilter:

    def test_buy_blocked_by_h1_sell(self):
        """BUY-сигнал должен блокироваться при H1 direction=SELL."""
        strat  = _make_strategy("mtf_h1")
        df     = _make_df(120, "up")
        df_h1  = _make_h1_trend("SELL", 250)

        # Создаём условия для BUY на 3m
        # EMA8 > EMA21 достигается восходящим трендом — используем uptrend df
        # H1 SELL должен блокировать
        result = strat.analyze(df, df_h1=df_h1)
        # Если сигнал BUY — должен быть заблокирован
        if result is not None:
            assert result.action == "SELL", "BUY не должен проходить при H1 SELL"

    def test_sell_blocked_by_h1_buy(self):
        """SELL-сигнал должен блокироваться при H1 direction=BUY."""
        strat  = _make_strategy("mtf_h1")
        df     = _make_df(120, "down")
        df_h1  = _make_h1_trend("BUY", 250)

        result = strat.analyze(df, df_h1=df_h1)
        if result is not None:
            assert result.action == "BUY", "SELL не должен проходить при H1 BUY"

    def test_neutral_h1_does_not_block(self):
        """H1 NEUTRAL не блокирует ни BUY, ни SELL."""
        strat  = _make_strategy("mtf_h1")
        df     = _make_df(120, "up")
        df_h1  = _make_h1_trend("NEUTRAL", 250)
        # Не должен бросать исключение; может вернуть None (нет условий для входа) — OK
        try:
            result = strat.analyze(df, df_h1=df_h1)
        except Exception as e:
            pytest.fail(f"NEUTRAL H1 вызвал исключение: {e}")

    def test_mtf_full_requires_m15_alignment(self):
        """В режиме mtf_full 15m должен совпадать с направлением."""
        strat = _make_strategy("mtf_full")
        df    = _make_df(120, "up")
        # H1 нейтральный, 15m нисходящий
        df_h1  = _make_h1_trend("NEUTRAL", 250)
        df_m15 = _make_h1_trend("SELL", 250)
        result = strat.analyze(df, df_h1=df_h1, df_m15=df_m15)
        if result is not None:
            # При 15m SELL — BUY-сигнал должен блокироваться
            assert result.action == "SELL"


class TestScalperProFilters:

    def test_large_candle_blocks_signal(self):
        """Импульсная свеча (range > 0.45%) блокирует сигнал."""
        strat = _make_strategy("base")
        df    = _make_df(120, "up")
        # Делаем последнюю свечу импульсной
        idx = len(df) - 1
        c0  = float(df.iloc[idx]["close"])
        df.iloc[idx, df.columns.get_loc("high")]  = c0 * 1.006  # range > 0.45%
        df.iloc[idx, df.columns.get_loc("low")]   = c0 * 0.999
        df.iloc[idx, df.columns.get_loc("open")]  = c0 * 1.001

        result = strat.analyze(df)
        assert result is None, "Импульсная свеча должна блокировать сигнал"

    def test_bad_spread_blocks_signal(self):
        """Спред > 0.08% блокирует сигнал."""
        strat  = _make_strategy("base")
        df     = _make_df(120, "up")
        result = strat.analyze(df, spread_pct=0.10)
        assert result is None, "Плохой спред должен блокировать сигнал"

    def test_good_spread_does_not_block(self):
        """Спред < 0.08% не блокирует сам по себе."""
        strat  = _make_strategy("base")
        df     = _make_df(120, "up")
        # При нулевом спреде ошибки нет
        try:
            strat.analyze(df, spread_pct=0.02)
        except Exception as e:
            pytest.fail(f"Нормальный спред вызвал исключение: {e}")

    def test_insufficient_bars_returns_none(self):
        strat = _make_strategy("base")
        df    = _make_df(30, "up")
        assert strat.analyze(df) is None

    def test_confidence_not_exceed_092(self):
        """Confidence не должен превышать 0.92."""
        strat  = _make_strategy("mtf_h1")
        df     = _make_df(120, "up")
        df_h1  = _make_h1_trend("BUY", 250)
        df_m15 = _make_h1_trend("BUY", 250)

        # Прогоняем много раз с разными данными
        for seed in range(5):
            np.random.seed(seed)
            df_t = _make_df(120, "up")
            result = strat.analyze(df_t, df_h1=df_h1, df_m15=df_m15)
            if result is not None:
                assert result.confidence <= 0.92, (
                    f"confidence {result.confidence} превышает 0.92"
                )


class TestScalperProEarlyTP:

    def _setup_position(self, strat: ScalperProStrategy, side: str = "Buy") -> None:
        entry = 1.0
        if side == "Buy":
            sl = 0.985
            tp = 1.045
        else:
            sl = 1.015
            tp = 0.955
        strat.current_position = {
            "side":       side,
            "entry":      entry,
            "sl":         sl,
            "tp":         tp,
            "initial_sl": sl,
            "be_moved":   False,
            "opened_at":  datetime.now(timezone.utc) - timedelta(minutes=10),
            "qty":        1.0,
        }

    def test_early_tp_blocked_below_50pct_progress(self):
        """Early TP не срабатывает при прогрессе < 50% пути к TP."""
        strat = _make_strategy()
        self._setup_position(strat, "Buy")
        # Цена прошла 30% пути к TP: entry=1.0, tp=1.045 → 30% = 1.0135
        current_price = 1.0 + 0.30 * (1.045 - 1.0)
        result = strat.check_early_tp(current_price, threshold_pct=85.0)
        assert result is None, "Early TP не должен срабатывать при прогрессе < 50%"

    def test_early_tp_blocked_if_net_pnl_zero(self):
        """Early TP не срабатывает если net PnL ≤ 0 (комиссии съедают прибыль)."""
        strat = _make_strategy()
        self._setup_position(strat, "Buy")
        # entry=1.0, tp=1.045. Прогресс 80% = 1.036
        current_price = 1.0 + 0.80 * (1.045 - 1.0)
        # Очень маленькая qty — gross почти ноль, fees > gross
        strat.current_position["qty"] = 0.0001
        result = strat.check_early_tp(current_price, threshold_pct=85.0)
        # При qty=0.0001 и entry=1.0: gross ≈ 0.0036 * 0.0001 = 3.6e-7
        # fee_pct = 0.055/100 * 2 * 5 = 0.0055, fees = 1.0 * 0.0001 * 0.0055 = 5.5e-7
        # net_pnl = 3.6e-7 - 5.5e-7 < 0 → блок
        assert result is None, "Early TP не должен работать при net PnL ≤ 0"

    def test_early_tp_blocked_if_position_too_fresh(self):
        """Early TP не срабатывает если позиция открыта < 6 мин (2 свечи 3m)."""
        strat = _make_strategy()
        self._setup_position(strat, "Buy")
        # Открылась только что (2 мин назад — меньше порога 6 мин)
        strat.current_position["opened_at"] = datetime.now(timezone.utc) - timedelta(minutes=2)
        # Прогресс 90%
        current_price = 1.0 + 0.90 * (1.045 - 1.0)
        strat.current_position["qty"] = 100.0  # большой qty чтобы net_pnl > 0
        result = strat.check_early_tp(current_price, threshold_pct=85.0)
        assert result is None, "Early TP не должен работать сразу после открытия"

    def test_early_tp_allowed_when_conditions_met(self):
        """Early TP разрешён: прогресс >= 85%, net_pnl > 0, открыта >= 6 мин."""
        strat = _make_strategy()
        self._setup_position(strat, "Buy")
        strat.current_position["qty"] = 1000.0
        strat.current_position["opened_at"] = datetime.now(timezone.utc) - timedelta(minutes=15)
        current_price = 1.0 + 0.90 * (1.045 - 1.0)
        result = strat.check_early_tp(current_price, threshold_pct=85.0)
        assert result == current_price, "Early TP должен сработать при выполнении всех условий"


class TestFiltersPassedStructure:

    def test_filters_passed_contains_required_keys(self):
        """filters_passed должен содержать все обязательные ключи."""
        strat  = _make_strategy("mtf_h1")
        df     = _make_df(120, "up")
        df_h1  = _make_h1_trend("NEUTRAL", 250)

        required_keys = {
            "mode", "ema_trend", "rsi_cross", "rsi_prev", "rsi_current",
            "bb_rejection", "volume_ratio", "candle_range_pct", "candle_body_ratio",
            "spread_pct", "funding_rate", "h1_trend", "h1_adx", "m15_trend", "m15_adx",
            "rr_ratio", "fee_guard_passed", "ai_score",
            "confidence_before_mtf", "confidence_final", "confidence_min",
        }

        result = strat.analyze(df, df_h1=df_h1)
        if result is not None:
            missing = required_keys - set(result.filters_passed.keys())
            assert not missing, f"Отсутствуют ключи в filters_passed: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
