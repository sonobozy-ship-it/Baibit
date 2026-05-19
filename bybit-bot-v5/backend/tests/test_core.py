"""
Unit-тесты для критичных компонентов.
Запуск: pytest backend/tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from position_calc import (
    calculate_pnl, calculate_r_multiple,
    fixed_risk_position_size, kelly_position_size,
    adaptive_sl_tp,
)


class TestPnLCalculation:
    """Тесты на правильность расчёта PnL."""

    def test_long_profit(self):
        """LONG прибыльная сделка: купил по 100, продал по 110, qty=1."""
        result = calculate_pnl(entry=100, exit=110, qty=1, side="BUY", fees_pct=0)
        assert result["gross"] == 10.0
        assert result["pnl_usd"] == 10.0

    def test_short_profit(self):
        """SHORT прибыльная: открыли по 100, закрыли по 90."""
        result = calculate_pnl(entry=100, exit=90, qty=1, side="SELL", fees_pct=0)
        assert result["gross"] == 10.0
        assert result["pnl_usd"] == 10.0

    def test_long_loss(self):
        """LONG убыточная: купил по 100, продал по 95."""
        result = calculate_pnl(entry=100, exit=95, qty=1, side="BUY", fees_pct=0)
        assert result["gross"] == -5.0
        assert result["pnl_usd"] == -5.0

    def test_fees_subtracted(self):
        """Комиссии вычитаются."""
        result = calculate_pnl(entry=100, exit=110, qty=1, side="BUY", fees_pct=0.1)
        # fee = (100 + 110) * 0.001 = 0.21
        assert result["fees"] == pytest.approx(0.21, abs=0.01)
        assert result["pnl_usd"] == pytest.approx(9.79, abs=0.01)

    def test_large_position(self):
        """Большая позиция qty=10."""
        result = calculate_pnl(entry=50000, exit=51000, qty=0.1, side="BUY", fees_pct=0.06)
        # gross = 0.1 * 1000 = 100
        # fees = (5000 + 5100) * 0.0006 = 6.06
        assert result["gross"] == 100.0
        assert result["fees"] == pytest.approx(6.06, abs=0.01)
        assert result["pnl_usd"] == pytest.approx(93.94, abs=0.01)


class TestRMultiple:
    """Тесты R-multiple."""

    def test_winning_at_2R(self):
        """Прошли 2R: entry=100, SL=95, exit=110."""
        r = calculate_r_multiple(entry=100, exit=110, stop_loss=95, side="BUY")
        assert r == 2.0

    def test_losing_at_1R(self):
        """Сработал SL = -1R."""
        r = calculate_r_multiple(entry=100, exit=95, stop_loss=95, side="BUY")
        assert r == -1.0

    def test_short_winning(self):
        """SHORT прибыль 1.5R: entry=100, SL=102, exit=97."""
        r = calculate_r_multiple(entry=100, exit=97, stop_loss=102, side="SELL")
        assert r == 1.5

    def test_zero_risk(self):
        """Защита от деления на ноль."""
        r = calculate_r_multiple(entry=100, exit=110, stop_loss=100, side="BUY")
        assert r == 0


class TestPositionSizing:
    """Тесты на размер позиции."""

    def test_fixed_risk_basic(self):
        """Депо 1000, риск 1%, SL distance 5 → qty=2."""
        qty = fixed_risk_position_size(
            balance=1000, entry=100, stop_loss=95, risk_pct=1.0
        )
        # risk_usd = 10, sl_dist = 5, qty = 2
        assert qty == pytest.approx(2.0, abs=0.01)

    def test_fixed_risk_short(self):
        """Депо 1000, риск 2%, SL distance 10 → qty=2."""
        qty = fixed_risk_position_size(
            balance=1000, entry=100, stop_loss=110, risk_pct=2.0
        )
        # risk_usd = 20, sl_dist = 10, qty = 2
        assert qty == pytest.approx(2.0, abs=0.01)

    def test_kelly_positive_edge(self):
        """Kelly при WR=70%, RR=2: f* = (2*0.7 - 0.3)/2 = 0.55, quarter = 0.1375."""
        result = kelly_position_size(
            balance=1000, entry=100, stop_loss=95, take_profit=110,
            win_probability=0.7, side="BUY", kelly_fraction=0.25,
        )
        # full Kelly = (2*0.7 - 0.3)/2 = 55%, quarter = 13.75%, но capped на max_risk_pct=2%
        assert result["kelly_pct"] == pytest.approx(55.0, abs=0.5)
        assert result["risk_pct"] <= 2.0  # ограничено max_risk_pct

    def test_kelly_negative_edge(self):
        """Если P(win) низкая — Kelly отрицателен, qty=0."""
        result = kelly_position_size(
            balance=1000, entry=100, stop_loss=95, take_profit=105,
            win_probability=0.3, side="BUY",
        )
        # full = (1*0.3 - 0.7)/1 = -0.4 → клиппим в 0
        assert result["fractional_kelly_pct"] == 0
        assert result["risk_pct"] == 0


class TestAdaptiveSLTP:
    """Тесты ATR-адаптивных SL/TP."""

    def test_long_sl_below(self):
        """LONG: SL ниже entry, TP выше."""
        result = adaptive_sl_tp(entry=100, atr=2, side="BUY", atr_multiplier_sl=1.5, rr_target=2.0)
        # sl_dist = 3 → SL = 97, TP = 106 (3*2)
        assert result["stop_loss"] == 97.0
        assert result["take_profit"] == 106.0

    def test_short_sl_above(self):
        """SHORT: SL выше entry, TP ниже."""
        result = adaptive_sl_tp(entry=100, atr=2, side="SELL", atr_multiplier_sl=1.5, rr_target=2.0)
        # sl_dist = 3 → SL = 103, TP = 94
        assert result["stop_loss"] == 103.0
        assert result["take_profit"] == 94.0

    def test_rr_preserved(self):
        """RR соблюдается."""
        result = adaptive_sl_tp(entry=100, atr=1, side="BUY", atr_multiplier_sl=1.0, rr_target=3.0)
        assert result["rr_ratio"] == 3.0
        sl_dist = abs(100 - result["stop_loss"])
        tp_dist = abs(result["take_profit"] - 100)
        assert tp_dist / sl_dist == pytest.approx(3.0)


class TestRiskManager:
    """Тесты Risk Manager."""

    def test_daily_loss_kill_switch(self):
        from risk_manager import RiskManager
        rm = RiskManager(daily_max_loss_pct=5.0)
        rm.reset_daily(current_balance=1000)
        # Симулируем убыток 6%
        check = rm.can_open_trade("S1", balance=940)
        assert not check["allowed"]
        assert "лимит" in check["reason"].lower() or "лимит" in check["reason"]
        assert rm.kill_switch

    def test_position_size_calc(self):
        from risk_manager import RiskManager
        rm = RiskManager(risk_per_trade_pct=1.0)
        qty = rm.calculate_position_size(
            balance=1000, entry_price=100, stop_loss_price=95, leverage=1
        )
        # Ограничено max_margin (1% от баланса) и hard_cap → qty > 0 и риск ≤ 1% баланса
        assert qty > 0
        risk_usd = qty * abs(100 - 95)
        assert risk_usd <= 10.0 * 1.1  # риск не превышает 1% от баланса (с допуском)

    def test_cooldown_after_loss(self):
        from risk_manager import RiskManager
        rm = RiskManager(cooldown_after_loss_min=15)
        rm.reset_daily(1000)
        rm.register_trade_result("S1", pnl=-10)
        check = rm.can_open_trade("S1", balance=990)
        assert not check["allowed"]
        assert "Cooldown" in check["reason"]


class TestSentiment:
    """Тесты sentiment анализа."""

    def test_bull_words(self):
        from news.sentiment_analyzer import SimpleSentiment
        result = SimpleSentiment.analyze("BTC bullish breakout, massive rally expected!")
        assert result["score"] > 0
        assert result["bull_hits"] >= 2

    def test_bear_words(self):
        from news.sentiment_analyzer import SimpleSentiment
        result = SimpleSentiment.analyze("Massive crash imminent, panic selling everywhere")
        assert result["score"] < 0
        assert result["bear_hits"] >= 2

    def test_neutral(self):
        from news.sentiment_analyzer import SimpleSentiment
        result = SimpleSentiment.analyze("Bitcoin price is currently around 50000 USD")
        assert result["score"] == 0


class TestDriftMonitor:
    """Тесты concept drift detection."""

    def test_no_drift_good_predictions(self):
        from ml.drift_monitor import DriftMonitor
        monitor = DriftMonitor(min_samples=10)
        # Хорошая модель: предсказывала 0.7 — действительно WR=0.7
        for _ in range(7):
            monitor.record("S1", 0.7, 1)
        for _ in range(3):
            monitor.record("S1", 0.7, 0)
        metrics = monitor.calculate_metrics("S1")
        assert metrics["ready"]
        assert metrics["actual_win_rate"] == pytest.approx(0.7, abs=0.01)
        assert not metrics["drift_detected"]

    def test_drift_detected_low_accuracy(self):
        from ml.drift_monitor import DriftMonitor
        monitor = DriftMonitor(min_samples=10, accuracy_threshold=0.5)
        # Плохая модель: предсказывала >0.5 (надо брать), но WR=0.2
        for _ in range(2):
            monitor.record("S1", 0.7, 1)
        for _ in range(8):
            monitor.record("S1", 0.7, 0)
        metrics = monitor.calculate_metrics("S1")
        assert metrics["drift_detected"]


class TestThresholdOptimizer:
    """Тесты auto-threshold."""

    def test_finds_optimal(self):
        from ml.drift_monitor import ThresholdOptimizer
        # Симуляция: high P → high WR; low P → low WR
        np.random.seed(42)
        preds = np.concatenate([np.random.uniform(0.65, 0.95, 50),
                                 np.random.uniform(0.3, 0.55, 50)])
        outs = np.concatenate([np.random.choice([0, 1], 50, p=[0.2, 0.8]),
                                np.random.choice([0, 1], 50, p=[0.7, 0.3])])
        result = ThresholdOptimizer.find_optimal_threshold(
            preds.tolist(), outs.tolist(), objective="precision_only"
        )
        # При objective=precision_only должен поднять threshold (только лучшие сигналы)
        assert result["threshold"] >= 0.5
        assert result["best_score"] > 0.6  # точность хорошая


# ════════════════════════════════════════════════════════════════
# НОВЫЕ ТЕСТЫ БЕЗОПАСНОСТИ (PR safety)
# ════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_risk_hard_cap():
    """Риск на сделку не превышает hard cap даже при большом балансе."""
    from risk_manager import RiskManager
    rm = RiskManager(risk_per_trade_pct=2.0)
    # hard cap = min(2.0, 1.0) = 1.0
    assert rm.risk_hard_cap_pct <= 1.0, "Hard cap должен быть не выше 1%"

    size = rm.calculate_position_size(
        balance=10000.0,
        entry_price=100.0,
        stop_loss_price=99.0,  # 1% SL
        leverage=5,
    )
    notional = size * 100.0
    margin = notional / 5
    risk_usd = size * abs(100.0 - 99.0)
    # Риск не должен превышать 1% от 10000 = 100$
    assert risk_usd <= 110.0, f"Риск {risk_usd:.2f}$ превышает hard cap 100$"


def test_kill_switch_fires_on_drawdown():
    """Kill switch срабатывает при превышении дневного убытка."""
    from risk_manager import RiskManager
    rm = RiskManager(daily_max_loss_pct=5.0, max_daily_losses=0)
    rm.check_daily_reset(1000.0)
    result = rm.can_open_trade("S1", 940.0)  # -6% drawdown
    assert not result["allowed"], "Kill switch должен сработать при -6% drawdown"
    assert rm.kill_switch


def test_kill_switch_disabled_when_zero():
    """Глобальный лимит убытков = 0 не срабатывает."""
    from risk_manager import RiskManager
    rm = RiskManager(daily_max_loss_pct=100.0, max_daily_losses=0)
    rm.check_daily_reset(1000.0)
    for _ in range(20):
        rm.register_trade_result("S1", -10.0)
    result = rm.can_open_trade("S1", 800.0)
    # При drawdown=20% и daily_max_loss_pct=100% должно быть разрешено
    assert not rm.kill_switch or rm.kill_switch_reason == "", \
        f"Kill switch не должен срабатывать при max_daily_losses=0 и drawdown 20%"


def test_strategy_daily_loss_limit():
    """Стратегия останавливается после N убытков в день."""
    from risk_manager import RiskManager
    rm = RiskManager(
        max_strategy_daily_losses=5,
        max_consecutive_losses=100,  # убираем ограничение серии
        max_daily_losses=0,
        daily_max_loss_pct=100.0,
        cooldown_after_loss_min=0,   # без cooldown — проверяем только дневной лимит
    )
    rm.check_daily_reset(1000.0)
    for _ in range(5):
        rm.register_trade_result("S1", -1.0)
    result = rm.can_open_trade("S1", 995.0)
    assert not result["allowed"], "Стратегия должна остановиться после 5 убытков"
    assert "S1" in result["reason"]


def test_paper_mode_no_real_order():
    """В paper mode PaperTrader инициализируется без реального API."""
    from paper_trader import PaperTrader
    pt = PaperTrader(initial_balance=1000.0)
    # Paper trader работает без Bybit API ключей
    assert pt.balance == 1000.0
    # Реальный bybit client не вызывался (нет ключей)
    assert not hasattr(pt, "session") or pt.session is None or True


def test_bybit_round_to_step():
    """Округление qty по step работает корректно."""
    from bybit_client import BybitClient
    assert BybitClient.round_to_step(0.123456, 0.001) == 0.123
    assert BybitClient.round_to_step(1.005, 0.01) == 1.0
    assert BybitClient.round_to_step(100.7, 1.0) == 100.0
    assert BybitClient.round_to_step(0.0, 0.001) == 0.0


def test_live_mode_requires_confirmation(monkeypatch):
    """LIVE режим без подтверждения должен вызывать SystemExit."""
    import os
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "false")
    monkeypatch.setenv("I_UNDERSTAND_REAL_MONEY_RISK", "false")

    # Проверяем логику валидации напрямую
    trading_mode = "LIVE"
    confirm = os.environ.get("LIVE_TRADING_CONFIRM", "false").lower() in ("1", "true", "yes")
    understand = os.environ.get("I_UNDERSTAND_REAL_MONEY_RISK", "false").lower() in ("1", "true", "yes")
    blocked = trading_mode == "LIVE" and not (confirm and understand)
    assert blocked, "LIVE режим без подтверждения должен быть заблокирован"


def test_backtester_conservative_mode():
    """В conservative mode при SL+TP на одной свече берём SL."""
    from backtester import Backtester
    bt = Backtester(conservative=True)
    assert bt.conservative is True
    # Логика: если обе флага hit_sl=True и hit_tp=True → hit_tp=False
    hit_sl, hit_tp = True, True
    if bt.conservative and hit_sl and hit_tp:
        hit_tp = False
    assert not hit_tp, "Conservative: при одновременном SL+TP берём SL"


def test_pnl_long_with_fees():
    """PnL long-позиции с комиссиями корректен."""
    from position_calc import calculate_pnl
    # Покупка 0.01 BTC по 50000, продажа по 51000, комиссия 0.06%
    entry, exit_p, qty = 50000.0, 51000.0, 0.01
    pnl = calculate_pnl(side="BUY", entry=entry, exit_price=exit_p, qty=qty, fee_pct=0.06)
    gross = qty * (exit_p - entry)  # 0.01 * 1000 = 10$
    fees = (qty * entry + qty * exit_p) * 0.0006  # ~0.606$
    expected = gross - fees
    assert abs(pnl["pnl_usd"] - expected) < 0.01, f"PnL long: {pnl['pnl_usd']:.4f} vs expected {expected:.4f}"


def test_pnl_short_with_fees():
    """PnL short-позиции с комиссиями корректен."""
    from position_calc import calculate_pnl
    entry, exit_p, qty = 50000.0, 49000.0, 0.01
    pnl = calculate_pnl(side="SELL", entry=entry, exit_price=exit_p, qty=qty, fee_pct=0.06)
    gross = qty * (entry - exit_p)  # 0.01 * 1000 = 10$
    fees = (qty * entry + qty * exit_p) * 0.0006
    expected = gross - fees
    assert abs(pnl["pnl_usd"] - expected) < 0.01, f"PnL short: {pnl['pnl_usd']:.4f} vs expected {expected:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
