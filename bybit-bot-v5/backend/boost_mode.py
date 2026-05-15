"""
Boost Mode — режим разгона депозита.

Математика:
  $10 → $100 за 7 дней = 10x = ~38.3% в день (compound)
  $10 → $150 за 7 дней = 15x = ~47.1% в день

Реализация через три фазы с автоматической адаптацией:

  Фаза 1 «Разгон»    ($10  → 3x старта):  плечо 15x, риск 80% баланса
  Фаза 2 «Рост»      (3x  → 8x старта):   плечо 12x, риск 60% баланса
  Фаза 3 «Закрепление» (8x+ старта):       плечо  8x, риск 40% баланса

Символы для малого депозита (маленький мин. размер лота):
  SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, BNBUSDT

Активные стратегии в boost-режиме:
  S4 Breakout  — сильные движения H1, TP 3-4%
  S5 Scalper   — быстрые сделки 5m, TP 2%
  S8 Trend     — трендовые входы H1 на откате
  S9 Fib+Trend — качественные входы H1 (фаза 3)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Конфигурация фаз
# ────────────────────────────────────────────────────────────

@dataclass
class BoostPhaseConfig:
    name:           str
    leverage:       int
    risk_pct:       float   # % баланса как маржа за сделку
    tp_pct:         float   # % движения цены (TP)
    sl_pct:         float   # % движения цены (SL)
    max_positions:  int
    daily_loss_limit_pct: float
    allowed_strategies: List[str]
    cooldown_after_loss_min: int


PHASES: List[BoostPhaseConfig] = [
    BoostPhaseConfig(
        name="Разгон",
        leverage=15,
        risk_pct=80.0,
        tp_pct=2.5,
        sl_pct=1.2,
        max_positions=1,
        daily_loss_limit_pct=35.0,
        allowed_strategies=["S4", "S5", "S8"],
        cooldown_after_loss_min=20,
    ),
    BoostPhaseConfig(
        name="Рост",
        leverage=12,
        risk_pct=60.0,
        tp_pct=3.0,
        sl_pct=1.5,
        max_positions=2,
        daily_loss_limit_pct=25.0,
        allowed_strategies=["S4", "S8", "S9"],
        cooldown_after_loss_min=30,
    ),
    BoostPhaseConfig(
        name="Закрепление",
        leverage=8,
        risk_pct=40.0,
        tp_pct=3.5,
        sl_pct=1.8,
        max_positions=2,
        daily_loss_limit_pct=20.0,
        allowed_strategies=["S7", "S8", "S9"],
        cooldown_after_loss_min=45,
    ),
]

# Символы с маленьким минимальным лотом (подходят для малого депозита)
BOOST_SYMBOLS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT", "MATICUSDT"]


# ────────────────────────────────────────────────────────────
# Математический анализ
# ────────────────────────────────────────────────────────────

class BoostCalculator:
    """
    Статический калькулятор: показывает что нужно для достижения цели
    и оценивает вероятность через симуляцию Монте-Карло.
    """

    @staticmethod
    def required_daily_return(initial: float, target: float, days: int) -> float:
        """Минимальная дневная доходность (compound) для достижения цели."""
        if days <= 0 or initial <= 0:
            return float("inf")
        return (target / initial) ** (1.0 / days) - 1.0

    @staticmethod
    def monte_carlo(
        initial: float,
        target: float,
        days: int,
        trades_per_day: int,
        win_rate: float,
        profit_per_win: float,   # доля от баланса (напр. 0.30 = 30%)
        loss_per_loss: float,    # доля от баланса при стопе
        n_sims: int = 5_000,
        rng_seed: int = 42,
    ) -> Dict:
        """
        Монте-Карло симуляция.
        Возвращает вероятность успеха, медианный и перцентильные балансы.
        """
        rng = np.random.default_rng(rng_seed)
        total_trades = days * trades_per_day
        outcomes = rng.random((n_sims, total_trades))  # shape (n_sims, trades)

        balances = np.full(n_sims, float(initial))
        for t in range(total_trades):
            win_mask = outcomes[:, t] < win_rate
            balances = np.where(win_mask,
                                balances * (1.0 + profit_per_win),
                                balances * (1.0 - loss_per_loss))
            balances = np.maximum(balances, 0.0)  # не уходим в минус

        success_rate = float((balances >= target).mean())
        return {
            "success_rate":   round(success_rate * 100, 1),
            "median_balance": round(float(np.median(balances)), 2),
            "p10_balance":    round(float(np.percentile(balances, 10)), 2),   # пессимизм
            "p90_balance":    round(float(np.percentile(balances, 90)), 2),   # оптимизм
            "p25_balance":    round(float(np.percentile(balances, 25)), 2),
            "p75_balance":    round(float(np.percentile(balances, 75)), 2),
            "ruin_rate":      round(float((balances < initial * 0.1).mean()) * 100, 1),
        }

    @classmethod
    def full_analysis(
        cls,
        initial: float,
        target: float,
        days: int,
    ) -> Dict:
        """
        Полный анализ плана разгона.
        Возвращает несколько сценариев с вероятностями.
        """
        req_daily = cls.required_daily_return(initial, target, days)
        multiplier = target / initial

        scenarios = []
        for trades_per_day, win_rate, leverage, tp_pct, sl_pct, label in [
            # trades/day, WR,   lev, tp%,  sl%,  label
            (3,           0.55, 10,  3.0,  1.5,  "Консервативный"),
            (3,           0.60, 12,  3.0,  1.5,  "Умеренный"),
            (4,           0.60, 12,  2.5,  1.2,  "Агрессивный"),
            (5,           0.62, 15,  2.5,  1.2,  "Очень агрессивный"),
            (5,           0.65, 15,  3.0,  1.2,  "Оптимистичный (лучший случай)"),
        ]:
            profit_per_win = (tp_pct / 100) * leverage  # % прибыли от маржи
            loss_per_loss  = (sl_pct / 100) * leverage  # % убытка от маржи

            mc = cls.monte_carlo(
                initial, target, days,
                trades_per_day, win_rate,
                profit_per_win, loss_per_loss,
            )
            expected_daily = (
                trades_per_day * (win_rate * profit_per_win - (1 - win_rate) * loss_per_loss)
            )
            scenarios.append({
                "label":           label,
                "trades_per_day":  trades_per_day,
                "win_rate_pct":    round(win_rate * 100, 0),
                "leverage":        leverage,
                "tp_pct":          tp_pct,
                "sl_pct":          sl_pct,
                "profit_per_win":  round(profit_per_win * 100, 1),
                "loss_per_loss":   round(loss_per_loss * 100, 1),
                "expected_daily_return_pct": round(expected_daily * 100, 1),
                **mc,
            })

        # Итоговый вывод
        best_realistic = next(
            (s for s in scenarios if s["success_rate"] >= 20), scenarios[-1]
        )

        return {
            "initial":          initial,
            "target":           target,
            "days":             days,
            "multiplier":       round(multiplier, 1),
            "required_daily_pct": round(req_daily * 100, 2),
            "scenarios":        scenarios,
            "best_realistic":   best_realistic,
            "symbols":          BOOST_SYMBOLS,
            "disclaimer": (
                "Разгон депозита — высокий риск. Вероятность потери >90% начального "
                "капитала существенна. Используйте только средства, которые вы готовы потерять."
            ),
        }


# ────────────────────────────────────────────────────────────
# Сессия разгона
# ────────────────────────────────────────────────────────────

@dataclass
class DayRecord:
    date:           str
    start_balance:  float
    end_balance:    float
    trades:         int
    wins:           int
    losses:         int
    pnl_usd:        float
    pnl_pct:        float


@dataclass
class BoostSession:
    initial_balance:     float
    target_balance:      float
    deadline_days:       int
    started_at:          str = field(default_factory=lambda: datetime.utcnow().isoformat())
    current_balance:     float = 0.0
    peak_balance:        float = 0.0
    session_trades:      int = 0
    session_wins:        int = 0
    session_losses:      int = 0
    consecutive_losses:  int = 0
    day_records:         List[DayRecord] = field(default_factory=list)
    active:              bool = True
    stopped_reason:      str = ""

    # Состояние текущего дня
    day_start_balance:   float = 0.0
    day_trades:          int = 0
    day_wins:            int = 0
    day_losses:          int = 0
    day_start_str:       str = ""

    def __post_init__(self):
        if self.current_balance == 0.0:
            self.current_balance = self.initial_balance
        if self.peak_balance == 0.0:
            self.peak_balance = self.initial_balance
        if self.day_start_balance == 0.0:
            self.day_start_balance = self.initial_balance
        if not self.day_start_str:
            self.day_start_str = datetime.utcnow().date().isoformat()

    @property
    def elapsed_days(self) -> float:
        delta = datetime.utcnow() - datetime.fromisoformat(self.started_at)
        return delta.total_seconds() / 86400

    @property
    def days_remaining(self) -> float:
        return max(0.0, self.deadline_days - self.elapsed_days)

    @property
    def progress_pct(self) -> float:
        if self.target_balance <= self.initial_balance:
            return 100.0
        return min(100.0, (self.current_balance - self.initial_balance) /
                   (self.target_balance - self.initial_balance) * 100)

    @property
    def phase_index(self) -> int:
        ratio = self.current_balance / self.initial_balance
        if ratio < 3.0:
            return 0   # Разгон
        elif ratio < 8.0:
            return 1   # Рост
        else:
            return 2   # Закрепление

    @property
    def phase(self) -> BoostPhaseConfig:
        return PHASES[self.phase_index]

    @property
    def required_daily_return_pct(self) -> float:
        if self.days_remaining <= 0:
            return float("inf")
        r = BoostCalculator.required_daily_return(
            self.current_balance, self.target_balance, max(1, int(self.days_remaining + 0.5))
        )
        return round(r * 100, 2)

    @property
    def session_win_rate(self) -> float:
        if not self.session_trades:
            return 0.0
        return round(self.session_wins / self.session_trades * 100, 1)

    @property
    def drawdown_from_peak_pct(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return round((self.peak_balance - self.current_balance) / self.peak_balance * 100, 2)

    @property
    def day_drawdown_pct(self) -> float:
        if self.day_start_balance <= 0:
            return 0.0
        return round((self.day_start_balance - self.current_balance) / self.day_start_balance * 100, 2)


# ────────────────────────────────────────────────────────────
# Менеджер boost-режима
# ────────────────────────────────────────────────────────────

class BoostManager:
    """
    Управляет сессией разгона.
    Интегрируется с торговым циклом через is_active / get_risk_params.
    """

    PAUSE_ON_CONSEC_LOSSES = 3   # пауза после N убытков подряд
    PAUSE_DURATION_MIN     = 60  # на сколько минут
    EMERGENCY_DD_PCT       = 40  # % просадки от пика → аварийная остановка

    def __init__(self, persist_path: str = "data/boost_session.json"):
        self.session:    Optional[BoostSession] = None
        self._pause_until: Optional[datetime]  = None
        self._persist    = Path(persist_path)
        self._load()

    # ── управление сессией ────────────────────────────────

    def start(
        self,
        initial_balance: float,
        target_balance: float,
        deadline_days: int = 7,
    ) -> Dict:
        if self.session and self.session.active:
            return {"success": False, "error": "Сессия уже активна. Сначала остановите её."}

        self.session = BoostSession(
            initial_balance=initial_balance,
            target_balance=target_balance,
            deadline_days=deadline_days,
        )
        self._pause_until = None
        self._save()
        analysis = BoostCalculator.full_analysis(initial_balance, target_balance, deadline_days)
        logger.info(
            f"[Boost] Старт: ${initial_balance:.2f} → ${target_balance:.2f} "
            f"за {deadline_days} дн. Требуется: {analysis['required_daily_pct']}%/день"
        )
        return {"success": True, "session": self._session_dict(), "analysis": analysis}

    def stop(self, reason: str = "Ручная остановка") -> Dict:
        if not self.session:
            return {"success": False, "error": "Нет активной сессии"}
        self.session.active = False
        self.session.stopped_reason = reason
        self._save()
        logger.info(f"[Boost] Остановлен: {reason}")
        return {"success": True, "final": self._session_dict()}

    # ── проверки в торговом цикле ─────────────────────────

    @property
    def is_active(self) -> bool:
        return bool(self.session and self.session.active)

    def strategy_allowed(self, strategy_id: str) -> bool:
        if not self.is_active:
            return True   # вне boost — всё разрешено
        return strategy_id in self.session.phase.allowed_strategies

    def can_open_trade(self, balance: float) -> Dict:
        """Дополнительные boost-проверки (вызываются ПОСЛЕ стандартного RiskManager)."""
        if not self.is_active:
            return {"allowed": True}

        sess = self.session
        sess.current_balance = balance
        sess.peak_balance    = max(sess.peak_balance, balance)

        # Дедлайн истёк
        if sess.days_remaining <= 0:
            if balance >= sess.target_balance:
                self.stop("🎉 Цель достигнута! Дедлайн истёк.")
            else:
                self.stop("⏰ Дедлайн истёк без достижения цели.")
            return {"allowed": False, "reason": sess.stopped_reason}

        # Цель достигнута
        if balance >= sess.target_balance:
            self.stop(f"🎉 Цель ${sess.target_balance:.2f} достигнута!")
            return {"allowed": False, "reason": sess.stopped_reason}

        # Аварийная просадка от пика
        if sess.drawdown_from_peak_pct >= self.EMERGENCY_DD_PCT:
            self.stop(
                f"🚨 Аварийная просадка {sess.drawdown_from_peak_pct:.1f}% от пика"
            )
            return {"allowed": False, "reason": sess.stopped_reason}

        # Дневной лимит убытков
        if sess.day_drawdown_pct >= sess.phase.daily_loss_limit_pct:
            return {
                "allowed": False,
                "reason": (
                    f"Дневной лимит убытков {sess.phase.daily_loss_limit_pct}% "
                    f"(потеряно {sess.day_drawdown_pct:.1f}% за день)"
                ),
            }

        # Пауза после серии убытков
        if self._pause_until and datetime.utcnow() < self._pause_until:
            rem = (self._pause_until - datetime.utcnow()).total_seconds() / 60
            return {"allowed": False, "reason": f"Пауза после серии убытков. Осталось {rem:.0f} мин"}

        return {"allowed": True}

    def get_risk_params(self) -> Optional[Dict]:
        """
        Возвращает параметры риска для текущей фазы.
        Торговый цикл применяет их вместо стандартных.
        """
        if not self.is_active:
            return None
        ph = self.session.phase
        return {
            "leverage":         ph.leverage,
            "risk_pct":         ph.risk_pct,
            "max_positions":    ph.max_positions,
            "tp_override_pct":  ph.tp_pct,   # переопределяет TP стратегии
            "sl_override_pct":  ph.sl_pct,   # переопределяет SL стратегии
            "daily_loss_limit": ph.daily_loss_limit_pct,
            "phase_name":       ph.name,
            "phase_index":      self.session.phase_index,
        }

    def apply_sl_tp(
        self,
        entry: float,
        side: str,   # "BUY" / "SELL"
        original_sl: float,
        original_tp: float,
    ) -> Tuple[float, float]:
        """
        Переопределяет SL/TP под параметры фазы.
        Берём более консервативный из двух SL (ближе к entry).
        """
        if not self.is_active:
            return original_sl, original_tp

        ph = self.session.phase
        if side == "BUY":
            boost_sl = entry * (1 - ph.sl_pct / 100)
            boost_tp = entry * (1 + ph.tp_pct / 100)
            sl = max(boost_sl, original_sl)   # ближайший стоп
            tp = max(boost_tp, original_tp)   # лучший TP
        else:
            boost_sl = entry * (1 + ph.sl_pct / 100)
            boost_tp = entry * (1 - ph.tp_pct / 100)
            sl = min(boost_sl, original_sl)
            tp = min(boost_tp, original_tp)

        return round(sl, 8), round(tp, 8)

    def calculate_qty(
        self,
        balance: float,
        entry_price: float,
        leverage: int,
    ) -> float:
        """
        Compound-sizing: используем % баланса как маржу.
        qty = (balance × risk_pct / 100) × leverage / entry_price
        """
        if not self.is_active:
            return 0.0
        ph = self.session.phase
        margin   = balance * (ph.risk_pct / 100)
        notional = margin * leverage
        qty      = notional / entry_price
        return round(qty, 6)

    # ── регистрация результатов ───────────────────────────

    def register_trade(self, pnl_usd: float, balance: float):
        if not self.is_active:
            return
        sess = self.session
        sess.current_balance = balance
        sess.peak_balance    = max(sess.peak_balance, balance)
        sess.session_trades += 1
        sess.day_trades     += 1

        if pnl_usd > 0:
            sess.session_wins      += 1
            sess.day_wins          += 1
            sess.consecutive_losses = 0
        else:
            sess.session_losses    += 1
            sess.day_losses        += 1
            sess.consecutive_losses += 1
            if sess.consecutive_losses >= self.PAUSE_ON_CONSEC_LOSSES:
                self._pause_until = datetime.utcnow() + timedelta(minutes=self.PAUSE_DURATION_MIN)
                logger.warning(
                    f"[Boost] {sess.consecutive_losses} убытков подряд → "
                    f"пауза {self.PAUSE_DURATION_MIN} мин"
                )

        phase_old = sess.phase_index
        # Проверяем переход фазы
        phase_new = sess.phase_index
        if phase_new != phase_old:
            logger.info(
                f"[Boost] Переход в фазу '{PHASES[phase_new].name}' "
                f"(баланс=${balance:.2f}, x{balance/sess.initial_balance:.1f})"
            )

        # Проверяем переход дня
        today = datetime.utcnow().date().isoformat()
        if today != sess.day_start_str:
            sess.day_records.append(DayRecord(
                date=sess.day_start_str,
                start_balance=sess.day_start_balance,
                end_balance=balance,
                trades=sess.day_trades,
                wins=sess.day_wins,
                losses=sess.day_losses,
                pnl_usd=round(balance - sess.day_start_balance, 4),
                pnl_pct=round((balance - sess.day_start_balance) / sess.day_start_balance * 100, 2),
            ))
            sess.day_start_balance = balance
            sess.day_start_str     = today
            sess.day_trades = sess.day_wins = sess.day_losses = 0

        self._save()

    # ── статус / персистентность ──────────────────────────

    def get_status(self) -> Dict:
        if not self.session:
            return {"active": False, "message": "Нет активной сессии"}
        sess = self.session
        return {
            "active":              sess.active,
            "initial_balance":     sess.initial_balance,
            "current_balance":     sess.current_balance,
            "target_balance":      sess.target_balance,
            "progress_pct":        round(sess.progress_pct, 1),
            "multiplier":          round(sess.current_balance / sess.initial_balance, 2),
            "phase":               sess.phase.name,
            "phase_index":         sess.phase_index,
            "days_remaining":      round(sess.days_remaining, 1),
            "elapsed_days":        round(sess.elapsed_days, 1),
            "required_daily_pct":  sess.required_daily_return_pct,
            "peak_balance":        sess.peak_balance,
            "drawdown_from_peak":  sess.drawdown_from_peak_pct,
            "day_drawdown":        sess.day_drawdown_pct,
            "session_trades":      sess.session_trades,
            "session_win_rate":    sess.session_win_rate,
            "consecutive_losses":  sess.consecutive_losses,
            "paused_until":        self._pause_until.isoformat() if self._pause_until else None,
            "stopped_reason":      sess.stopped_reason,
            "day_records":         [asdict(r) for r in sess.day_records],
            "risk_params":         self.get_risk_params(),
            "allowed_strategies":  sess.phase.allowed_strategies,
        }

    def _session_dict(self) -> Dict:
        if not self.session:
            return {}
        s = self.session
        return {
            "initial_balance":    s.initial_balance,
            "current_balance":    s.current_balance,
            "target_balance":     s.target_balance,
            "started_at":         s.started_at,
            "deadline_days":      s.deadline_days,
            "active":             s.active,
            "stopped_reason":     s.stopped_reason,
            "progress_pct":       round(s.progress_pct, 1),
            "phase":              s.phase.name,
        }

    def _save(self):
        try:
            self._persist.parent.mkdir(parents=True, exist_ok=True)
            if self.session:
                data = {
                    "session": asdict(self.session),
                    "pause_until": self._pause_until.isoformat() if self._pause_until else None,
                }
                self._persist.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"[Boost] Сохранение: {e}")

    def _load(self):
        try:
            if self._persist.exists():
                data = json.loads(self._persist.read_text())
                sess_data = data.get("session", {})
                if sess_data:
                    # Восстанавливаем DayRecord objects
                    sess_data["day_records"] = [
                        DayRecord(**r) for r in sess_data.get("day_records", [])
                    ]
                    self.session = BoostSession(**{
                        k: v for k, v in sess_data.items()
                        if k in BoostSession.__dataclass_fields__
                    })
                if data.get("pause_until"):
                    self._pause_until = datetime.fromisoformat(data["pause_until"])
                logger.info(
                    f"[Boost] Восстановлена сессия: active={self.session.active}, "
                    f"balance=${self.session.current_balance:.2f}"
                )
        except Exception as e:
            logger.warning(f"[Boost] Загрузка сессии: {e}")
            self.session = None
