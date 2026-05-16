"""
Boost Mode — режим разгона депозита.

Три режима риска:

  safe       — плечо 3x, риск 15%/сделку, R:R 3:1, стоп дня 8%
  moderate   — плечо 5x, риск 20%/сделку, R:R 3:1, стоп дня 12%  (дефолт)
  aggressive — плечо 8-15x, риск 40-80%/сделку (исходный)

Математика (moderate, $10 → $100):
  Нужно ~8-11%/день → реально при 4-5 сделках/день с 60% WR
  MC-вероятность за 30 дней: ~50-60%

Рекомендуемые цели:
  safe:       $10 → $30  за 21 день (~50% MC)
  moderate:   $10 → $100 за 30 дней (~50% MC)  |  $10→$50 за 14 дней (~55% MC)
  aggressive: $10 → $100 за  7 дней (~20% MC)  — высокий риск

Символы для малого депозита:
  SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, BNBUSDT
"""
from __future__ import annotations

import json
import logging
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
    name:                 str
    leverage:             int
    risk_pct:             float   # % баланса как маржа за сделку
    tp_pct:               float   # % движения цены (TP)
    sl_pct:               float   # % движения цены (SL)
    max_positions:        int
    daily_loss_limit_pct: float
    allowed_strategies:   List[str]
    cooldown_after_loss_min: int


# ── Агрессивный (исходный, сохранён для совместимости) ───────
PHASES_AGGRESSIVE: List[BoostPhaseConfig] = [
    BoostPhaseConfig(
        name="Разгон", leverage=15, risk_pct=80.0, tp_pct=2.5, sl_pct=1.2,
        max_positions=1, daily_loss_limit_pct=35.0,
        allowed_strategies=["S4", "S5", "S8"], cooldown_after_loss_min=20,
    ),
    BoostPhaseConfig(
        name="Рост", leverage=12, risk_pct=60.0, tp_pct=3.0, sl_pct=1.5,
        max_positions=2, daily_loss_limit_pct=25.0,
        allowed_strategies=["S4", "S8", "S9", "S11"], cooldown_after_loss_min=30,
    ),
    BoostPhaseConfig(
        name="Закрепление", leverage=8, risk_pct=40.0, tp_pct=3.5, sl_pct=1.8,
        max_positions=2, daily_loss_limit_pct=20.0,
        allowed_strategies=["S7", "S8", "S9", "S11"], cooldown_after_loss_min=45,
    ),
]

# ── Умеренный (дефолт) ────────────────────────────────────────
# Плечо 5x, риск 20%, R:R 4:1 (TP=4%, SL=1%), стоп дня 12%
# 70-80% MC-вероятность: $10→$30 за 21 день, $10→$50 за 30 дней
PHASES_MODERATE: List[BoostPhaseConfig] = [
    BoostPhaseConfig(
        name="Старт", leverage=5, risk_pct=20.0, tp_pct=4.0, sl_pct=1.0,
        max_positions=1, daily_loss_limit_pct=12.0,
        allowed_strategies=["S4", "S8", "S9", "S11"], cooldown_after_loss_min=30,
    ),
    BoostPhaseConfig(
        name="Рост", leverage=5, risk_pct=20.0, tp_pct=4.0, sl_pct=1.0,
        max_positions=2, daily_loss_limit_pct=12.0,
        allowed_strategies=["S4", "S8", "S9", "S11"], cooldown_after_loss_min=35,
    ),
    BoostPhaseConfig(
        name="Закрепление", leverage=4, risk_pct=15.0, tp_pct=5.0, sl_pct=1.0,
        max_positions=2, daily_loss_limit_pct=10.0,
        allowed_strategies=["S7", "S8", "S9", "S11"], cooldown_after_loss_min=45,
    ),
]

# ── Безопасный ────────────────────────────────────────────────
# Плечо 3x, риск 15%, R:R 4:1 (TP=4%, SL=1%), стоп дня 8%
# 70-80% MC-вероятность: $10→$20 за 21 день, $10→$30 за 30 дней
PHASES_SAFE: List[BoostPhaseConfig] = [
    BoostPhaseConfig(
        name="Накопление", leverage=3, risk_pct=15.0, tp_pct=4.0, sl_pct=1.0,
        max_positions=1, daily_loss_limit_pct=8.0,
        allowed_strategies=["S8", "S9"], cooldown_after_loss_min=60,
    ),
    BoostPhaseConfig(
        name="Рост", leverage=3, risk_pct=15.0, tp_pct=4.0, sl_pct=1.0,
        max_positions=1, daily_loss_limit_pct=8.0,
        allowed_strategies=["S8", "S9", "S11"], cooldown_after_loss_min=60,
    ),
    BoostPhaseConfig(
        name="Финиш", leverage=3, risk_pct=15.0, tp_pct=5.0, sl_pct=1.0,
        max_positions=2, daily_loss_limit_pct=6.0,
        allowed_strategies=["S7", "S8", "S9", "S11"], cooldown_after_loss_min=60,
    ),
]

# ── Скальпинг ────────────────────────────────────────────────
# 3m таймфрейм, 10 символов одновременно, до 50 прибыльных сделок/день
# Плечо 5x, риск 8%/сделку (маленький риск, много сделок), R:R 2.5:1
PHASES_SCALP: List[BoostPhaseConfig] = [
    BoostPhaseConfig(
        name="Разгон-Скальп", leverage=5, risk_pct=8.0, tp_pct=0.30, sl_pct=0.12,
        max_positions=5, daily_loss_limit_pct=8.0,
        allowed_strategies=["S10"],   # только ScalperPro
        cooldown_after_loss_min=5,
    ),
    BoostPhaseConfig(
        name="Рост-Скальп", leverage=5, risk_pct=8.0, tp_pct=0.30, sl_pct=0.12,
        max_positions=6, daily_loss_limit_pct=8.0,
        allowed_strategies=["S10"],
        cooldown_after_loss_min=5,
    ),
    BoostPhaseConfig(
        name="Закреп-Скальп", leverage=4, risk_pct=6.0, tp_pct=0.35, sl_pct=0.12,
        max_positions=6, daily_loss_limit_pct=6.0,
        allowed_strategies=["S10", "S8", "S9", "S11"],  # в фазе 3 добавляем качественные
        cooldown_after_loss_min=10,
    ),
]

BOOST_MODES: Dict[str, List[BoostPhaseConfig]] = {
    "safe":       PHASES_SAFE,
    "moderate":   PHASES_MODERATE,
    "aggressive": PHASES_AGGRESSIVE,
    "scalp":      PHASES_SCALP,
}

# Параметры безопасности по режиму
_MODE_SAFETY = {
    "safe":       {"emergency_dd": 15, "pause_losses": 2},
    "moderate":   {"emergency_dd": 25, "pause_losses": 2},
    "aggressive": {"emergency_dd": 40, "pause_losses": 3},
    "scalp":      {"emergency_dd": 20, "pause_losses": 5},  # частые убытки нормальны
}

# Символы с маленьким минимальным лотом (подходят для малого депозита)
BOOST_SYMBOLS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT", "MATICUSDT"]


# ────────────────────────────────────────────────────────────
# Математический анализ
# ────────────────────────────────────────────────────────────

class BoostCalculator:
    """
    Статический калькулятор: показывает что нужно для достижения цели
    и оценивает вероятность через симуляцию Монте-Карло.

    profit_per_win / loss_per_loss — доля от БАЛАНСА (с учётом risk_pct).
    """

    @staticmethod
    def required_daily_return(initial: float, target: float, days: int) -> float:
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
        profit_per_win: float,   # доля от БАЛАНСА при победе
        loss_per_loss: float,    # доля от БАЛАНСА при стопе
        n_sims: int = 5_000,
        rng_seed: int = 42,
    ) -> Dict:
        rng = np.random.default_rng(rng_seed)
        total_trades = days * trades_per_day
        outcomes = rng.random((n_sims, total_trades))

        balances = np.full(n_sims, float(initial))
        for t in range(total_trades):
            win_mask = outcomes[:, t] < win_rate
            balances = np.where(
                win_mask,
                balances * (1.0 + profit_per_win),
                balances * (1.0 - loss_per_loss),
            )
            balances = np.maximum(balances, 0.0)

        success_rate = float((balances >= target).mean())
        return {
            "success_rate":   round(success_rate * 100, 1),
            "median_balance": round(float(np.median(balances)), 2),
            "p10_balance":    round(float(np.percentile(balances, 10)), 2),
            "p90_balance":    round(float(np.percentile(balances, 90)), 2),
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
        mode: str = "moderate",
    ) -> Dict:
        """
        Полный анализ плана разгона.
        Учитывает risk_pct: profit = risk_pct * leverage * tp_pct.
        """
        req_daily = cls.required_daily_return(initial, target, days)
        multiplier = target / initial

        # Сценарии (trades_per_day, WR, leverage, tp_pct, sl_pct, risk_pct, label)
        # R:R 4:1 (TP=4%, SL=1%) — цель 70-80% MC-вероятности
        if mode == "scalp":
            # 3m скальпинг: маленький R:R 2.5:1, но МНОГО сделок в день
            # profit/loss как % от баланса: 8% риск × 5x × 0.3% TP = 1.2%
            scenario_defs = [
                # tpd  WR    lev   tp%    sl%   risk%   label
                (30,  0.58,  5,   0.30,  0.12,  8.0, "30 сд/д (5 сим × 6) 58% WR"),
                (40,  0.60,  5,   0.30,  0.12,  8.0, "40 сд/д (10 сим × 4) 60% WR"),
                (50,  0.60,  5,   0.30,  0.12,  8.0, "50 сд/д (10 сим × 5) 60% WR"),
                (50,  0.62,  5,   0.30,  0.12,  8.0, "50 сд/д 62% WR"),
                (60,  0.62,  5,   0.30,  0.12,  8.0, "60 сд/д (10 сим × 6) оптимист."),
            ]
        elif mode == "safe":
            scenario_defs = [
                # tpd   WR     lev  tp%  sl%  risk%  label
                (2,    0.58,   3,   4.0, 1.0, 15.0, "Осторожный 2сд/д R:R 4:1"),
                (3,    0.58,   3,   4.0, 1.0, 15.0, "Базовый 3сд/д R:R 4:1"),
                (3,    0.60,   3,   4.0, 1.0, 15.0, "Базовый+ 60% WR"),
                (4,    0.60,   3,   4.0, 1.0, 15.0, "Активный 4сд/д R:R 4:1"),
                (4,    0.62,   3,   4.0, 1.0, 15.0, "Оптимистичный 62% WR"),
            ]
        elif mode == "aggressive":
            scenario_defs = [
                (3,    0.55,  10,   3.0, 1.5, 80.0, "Консервативный"),
                (3,    0.60,  12,   3.0, 1.5, 80.0, "Умеренный"),
                (4,    0.60,  12,   2.5, 1.2, 80.0, "Агрессивный"),
                (5,    0.62,  15,   2.5, 1.2, 80.0, "Очень агрессивный"),
                (5,    0.65,  15,   3.0, 1.2, 80.0, "Оптимистичный"),
            ]
        else:  # moderate (default) — R:R 4:1, цель 70-80%
            scenario_defs = [
                (3,    0.58,   5,   4.0, 1.0, 20.0, "Базовый 3сд/д R:R 4:1"),
                (3,    0.60,   5,   4.0, 1.0, 20.0, "Базовый+ 60% WR"),
                (4,    0.60,   5,   4.0, 1.0, 20.0, "Активный 4сд/д R:R 4:1"),
                (4,    0.62,   5,   4.0, 1.0, 20.0, "Активный+ 62% WR"),
                (5,    0.62,   5,   4.0, 1.0, 20.0, "Оптимистичный 5сд/д"),
            ]

        scenarios = []
        for tpd, wr, lev, tp_p, sl_p, risk_p, label in scenario_defs:
            # Прибыль/убыток как доля от БАЛАНСА
            pw = (risk_p / 100) * lev * (tp_p / 100)
            pl = (risk_p / 100) * lev * (sl_p / 100)

            mc = cls.monte_carlo(initial, target, days, tpd, wr, pw, pl)
            expected_daily = tpd * (wr * pw - (1 - wr) * pl)
            scenarios.append({
                "label":                    label,
                "trades_per_day":           tpd,
                "win_rate_pct":             round(wr * 100, 0),
                "leverage":                 lev,
                "risk_pct":                 risk_p,
                "tp_pct":                   tp_p,
                "sl_pct":                   sl_p,
                "rr_ratio":                 round(tp_p / sl_p, 1),
                "profit_per_win_balance":   round(pw * 100, 2),
                "loss_per_loss_balance":    round(pl * 100, 2),
                "expected_daily_return_pct": round(expected_daily * 100, 2),
                **mc,
            })

        best_realistic = next(
            (s for s in scenarios if s["success_rate"] >= 30), scenarios[-1]
        )

        # Достижимые цели при 70% и 80% вероятности
        prob_targets = cls.find_targets_for_probabilities(initial, days, mode)

        # Советы
        tips = _realistic_tips(initial, target, days, mode)

        return {
            "initial":             initial,
            "target":              target,
            "days":                days,
            "mode":                mode,
            "multiplier":          round(multiplier, 1),
            "required_daily_pct":  round(req_daily * 100, 2),
            "scenarios":           scenarios,
            "best_realistic":      best_realistic,
            "prob_targets":        prob_targets,
            "symbols":             BOOST_SYMBOLS,
            "tips":                tips,
            "disclaimer": (
                "Разгон депозита — высокий риск потери капитала. "
                "Используйте только средства, которые вы готовы потерять."
            ),
        }

    @classmethod
    def find_targets_for_probabilities(
        cls,
        initial: float,
        days:    int,
        mode:    str = "moderate",
        probs:   List[float] = None,
        n_sims:  int = 3_000,
    ) -> List[Dict]:
        """
        Для каждой целевой вероятности (70%, 75%, 80%) находит максимально
        достижимый баланс бинарным поиском по Monte Carlo.
        """
        if probs is None:
            probs = [0.70, 0.75, 0.80]

        # Параметры «базового» сценария для режима
        _params = {
            "safe":       dict(trades_per_day=3,  wr=0.60, lev=3,  tp=4.0,  sl=1.0,  risk=15.0),
            "moderate":   dict(trades_per_day=4,  wr=0.60, lev=5,  tp=4.0,  sl=1.0,  risk=20.0),
            "aggressive": dict(trades_per_day=4,  wr=0.62, lev=12, tp=3.0,  sl=1.5,  risk=80.0),
            "scalp":      dict(trades_per_day=40, wr=0.60, lev=5,  tp=0.30, sl=0.12, risk=8.0),
        }
        p = _params.get(mode, _params["moderate"])
        pw = (p["risk"] / 100) * p["lev"] * (p["tp"] / 100)
        pl = (p["risk"] / 100) * p["lev"] * (p["sl"] / 100)

        results = []
        for target_prob in probs:
            lo, hi = initial, initial * 500.0
            for _ in range(18):          # 18 итераций → точность < 1%
                mid = (lo + hi) / 2.0
                mc  = cls.monte_carlo(initial, mid, days, p["trades_per_day"],
                                      p["wr"], pw, pl, n_sims=n_sims)
                if mc["success_rate"] / 100 >= target_prob:
                    lo = mid
                else:
                    hi = mid
            results.append({
                "probability_pct":   round(target_prob * 100),
                "target":            round((lo + hi) / 2, 1),
                "multiplier":        round((lo + hi) / 2 / initial, 1),
                "days":              days,
                "mode":              mode,
                "trades_per_day":    p["trades_per_day"],
                "win_rate_pct":      round(p["wr"] * 100),
                "leverage":          p["lev"],
                "rr_ratio":          round(p["tp"] / p["sl"], 0),
            })
        return results


def _realistic_tips(initial: float, target: float, days: int, mode: str) -> List[str]:
    """Советы исходя из реалистичности цели."""
    multiplier = target / initial
    tips = []

    if mode == "moderate":
        tips.append("💡 Moderate: 5× плечо, R:R 4:1, стоп дня 12% → цель 70-80% MC-вероятности.")
        if multiplier > 5 and days <= 14:
            tips.append(
                f"⚠️  {multiplier:.1f}x за {days} дней — ниже 50% MC. "
                f"Для 70%+: смотри prob_targets в ответе."
            )
        elif multiplier <= 3 and days >= 21:
            tips.append("✅ Консервативная цель — вероятность успеха 80%+.")

    elif mode == "safe":
        tips.append("🛡  Safe: 3× плечо, R:R 4:1, стоп дня 8%, пауза после 2 убытков — минимальный риск руина.")
        if multiplier > 3 and days <= 14:
            tips.append(
                f"⚠️  Safe-режим: {multiplier:.1f}x за {days} дней — ниже 50% MC. "
                f"Для 70%+: смотри prob_targets."
            )

    elif mode == "aggressive":
        tips.append(
            "🔥 Aggressive: 8-15× плечо. Вероятность потери >90% депозита существенна. "
            "Рекомендуем moderate после стабильного профита на paper-trading."
        )

    actual_prob = _quick_mc_pct(initial, target, days, mode)
    tips.append(
        f"📊 Ваша цель {multiplier:.1f}x за {days} дн. → MC ≈{actual_prob:.0f}% "
        f"(режим {mode}). Для 70-80% — используй prob_targets."
    )
    return tips


def _quick_mc_pct(initial: float, target: float, days: int, mode: str) -> float:
    """Быстрая оценка MC-вероятности для подсказки."""
    if mode == "safe":
        pw, pl, wr, tpd = 0.15*3*0.04, 0.15*3*0.01, 0.60, 3
    elif mode == "aggressive":
        pw, pl, wr, tpd = 0.80*12*0.03, 0.80*12*0.015, 0.60, 4
    elif mode == "scalp":
        pw, pl, wr, tpd = 0.08*5*0.003, 0.08*5*0.0012, 0.60, 40
    else:  # moderate
        pw, pl, wr, tpd = 0.20*5*0.04, 0.20*5*0.01, 0.60, 4
    mc = BoostCalculator.monte_carlo(initial, target, days, tpd, wr, pw, pl, n_sims=2000)
    return mc["success_rate"]


# ────────────────────────────────────────────────────────────
# Сессия разгона
# ────────────────────────────────────────────────────────────

@dataclass
class DayRecord:
    date:          str
    start_balance: float
    end_balance:   float
    trades:        int
    wins:          int
    losses:        int
    pnl_usd:       float
    pnl_pct:       float


@dataclass
class BoostSession:
    initial_balance:    float
    target_balance:     float
    deadline_days:      int
    mode:               str = "moderate"
    started_at:         str = field(default_factory=lambda: datetime.utcnow().isoformat())
    current_balance:    float = 0.0
    peak_balance:       float = 0.0
    session_trades:     int = 0
    session_wins:       int = 0
    session_losses:     int = 0
    consecutive_losses: int = 0
    day_records:        List[DayRecord] = field(default_factory=list)
    active:             bool = True
    stopped_reason:     str = ""
    day_start_balance:  float = 0.0
    day_trades:         int = 0
    day_wins:           int = 0
    day_losses:         int = 0
    day_start_str:      str = ""

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
        return (datetime.utcnow() - datetime.fromisoformat(self.started_at)).total_seconds() / 86400

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
        if self.initial_balance <= 0:
            return 0
        ratio = self.current_balance / self.initial_balance
        if ratio < 3.0:
            return 0
        elif ratio < 8.0:
            return 1
        return 2

    @property
    def phases(self) -> List[BoostPhaseConfig]:
        return BOOST_MODES.get(self.mode, PHASES_MODERATE)

    @property
    def phase(self) -> BoostPhaseConfig:
        return self.phases[min(self.phase_index, len(self.phases) - 1)]

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
    PAUSE_DURATION_MIN = 60

    def __init__(self, persist_path: str = "data/boost_session.json"):
        self.session:      Optional[BoostSession] = None
        self._pause_until: Optional[datetime]     = None
        self._persist      = Path(persist_path)
        self._load()

    # ── управление сессией ────────────────────────────────

    def start(
        self,
        initial_balance: float,
        target_balance:  float,
        deadline_days:   int = 7,
        mode:            str = "moderate",
    ) -> Dict:
        if mode not in BOOST_MODES:
            return {"success": False, "error": f"Неизвестный режим '{mode}'. Доступны: {list(BOOST_MODES)}"}
        if initial_balance <= 0:
            return {"success": False, "error": "Начальный баланс должен быть > 0"}
        if self.session and self.session.active:
            return {"success": False, "error": "Сессия уже активна. Сначала остановите её."}

        self.session = BoostSession(
            initial_balance=initial_balance,
            target_balance=target_balance,
            deadline_days=deadline_days,
            mode=mode,
        )
        self._pause_until = None
        self._save()

        analysis = BoostCalculator.full_analysis(initial_balance, target_balance, deadline_days, mode)
        safety = _MODE_SAFETY[mode]
        logger.info(
            f"[Boost] Старт ({mode}): ${initial_balance:.2f} → ${target_balance:.2f} "
            f"за {deadline_days} дн. Требуется: {analysis['required_daily_pct']}%/день | "
            f"Плечо: {self.session.phase.leverage}x | "
            f"Стоп просадки: {safety['emergency_dd']}%"
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
            return True
        allowed = self.session.phase.allowed_strategies
        if strategy_id in allowed:
            return True
        # SC_XXX — это экземпляры ScalperPro (S10), проверяем по паттерну
        if strategy_id.startswith("SC_") and "S10" in allowed:
            return True
        return False

    @property
    def _emergency_dd(self) -> int:
        if not self.session:
            return 40
        return _MODE_SAFETY.get(self.session.mode, {}).get("emergency_dd", 40)

    @property
    def _pause_on_losses(self) -> int:
        if not self.session:
            return 3
        return _MODE_SAFETY.get(self.session.mode, {}).get("pause_losses", 3)

    def can_open_trade(self, balance: float) -> Dict:
        if not self.is_active:
            return {"allowed": True}

        sess = self.session
        sess.current_balance = balance
        sess.peak_balance    = max(sess.peak_balance, balance)

        if sess.days_remaining <= 0:
            if balance >= sess.target_balance:
                self.stop("🎉 Цель достигнута! Дедлайн истёк.")
            else:
                self.stop("⏰ Дедлайн истёк без достижения цели.")
            return {"allowed": False, "reason": sess.stopped_reason}

        if balance >= sess.target_balance:
            self.stop(f"🎉 Цель ${sess.target_balance:.2f} достигнута!")
            return {"allowed": False, "reason": sess.stopped_reason}

        if sess.drawdown_from_peak_pct >= self._emergency_dd:
            self.stop(
                f"🚨 Аварийная просадка {sess.drawdown_from_peak_pct:.1f}% от пика "
                f"(лимит {self._emergency_dd}%)"
            )
            return {"allowed": False, "reason": sess.stopped_reason}

        if sess.day_drawdown_pct >= sess.phase.daily_loss_limit_pct:
            return {
                "allowed": False,
                "reason": (
                    f"Дневной лимит убытков {sess.phase.daily_loss_limit_pct}% "
                    f"(потеряно {sess.day_drawdown_pct:.1f}% за день)"
                ),
            }

        if self._pause_until and datetime.utcnow() < self._pause_until:
            rem = (self._pause_until - datetime.utcnow()).total_seconds() / 60
            return {"allowed": False, "reason": f"Пауза после серии убытков. Осталось {rem:.0f} мин"}

        return {"allowed": True}

    def get_risk_params(self) -> Optional[Dict]:
        if not self.is_active:
            return None
        ph = self.session.phase
        return {
            "leverage":         ph.leverage,
            "risk_pct":         ph.risk_pct,
            "max_positions":    ph.max_positions,
            "tp_override_pct":  ph.tp_pct,
            "sl_override_pct":  ph.sl_pct,
            "daily_loss_limit": ph.daily_loss_limit_pct,
            "phase_name":       ph.name,
            "phase_index":      self.session.phase_index,
            "mode":             self.session.mode,
        }

    def apply_sl_tp(
        self,
        entry: float,
        side:  str,
        original_sl: float,
        original_tp: float,
    ) -> Tuple[float, float]:
        if not self.is_active:
            return original_sl, original_tp

        ph = self.session.phase
        if side == "BUY":
            boost_sl = entry * (1 - ph.sl_pct / 100)
            boost_tp = entry * (1 + ph.tp_pct / 100)
            sl = max(boost_sl, original_sl)   # ближайший стоп (безопаснее)
            tp = max(boost_tp, original_tp)   # дальний TP  (выгоднее)
        else:
            boost_sl = entry * (1 + ph.sl_pct / 100)
            boost_tp = entry * (1 - ph.tp_pct / 100)
            sl = min(boost_sl, original_sl)
            tp = min(boost_tp, original_tp)

        return round(sl, 8), round(tp, 8)

    def calculate_qty(
        self,
        balance:     float,
        entry_price: float,
        leverage:    int,
    ) -> float:
        """Compound sizing: risk_pct% баланса как маржа × плечо / цена."""
        if not self.is_active:
            return 0.0
        ph      = self.session.phase
        margin  = balance * (ph.risk_pct / 100)
        notional = margin * leverage
        return round(notional / entry_price, 6)

    # ── регистрация результатов ───────────────────────────

    def register_trade(self, pnl_usd: float, balance: float):
        if not self.is_active:
            return
        sess = self.session
        sess.current_balance = balance
        sess.peak_balance    = max(sess.peak_balance, balance)
        sess.session_trades += 1
        sess.day_trades     += 1

        prev_phase = sess.phase_index
        if pnl_usd > 0:
            sess.session_wins      += 1
            sess.day_wins          += 1
            sess.consecutive_losses = 0
        else:
            sess.session_losses    += 1
            sess.day_losses        += 1
            sess.consecutive_losses += 1
            if sess.consecutive_losses >= self._pause_on_losses:
                self._pause_until = datetime.utcnow() + timedelta(minutes=self.PAUSE_DURATION_MIN)
                logger.warning(
                    f"[Boost] {sess.consecutive_losses} убытков подряд → "
                    f"пауза {self.PAUSE_DURATION_MIN} мин"
                )

        # Лог перехода фазы
        new_phase = sess.phase_index
        if new_phase != prev_phase:
            logger.info(
                f"[Boost] ▶ Фаза '{sess.phase.name}' "
                f"(баланс=${balance:.2f}, ×{balance/sess.initial_balance:.1f}, режим={sess.mode})"
            )

        # Смена дня
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
                pnl_pct=round(
                    (balance - sess.day_start_balance) / sess.day_start_balance * 100
                    if sess.day_start_balance else 0.0, 2
                ),
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
        safety = _MODE_SAFETY.get(sess.mode, {})
        return {
            "active":             sess.active,
            "mode":               sess.mode,
            "initial_balance":    sess.initial_balance,
            "current_balance":    sess.current_balance,
            "target_balance":     sess.target_balance,
            "progress_pct":       round(sess.progress_pct, 1),
            "multiplier":         round(sess.current_balance / sess.initial_balance, 2),
            "phase":              sess.phase.name,
            "phase_index":        sess.phase_index,
            "days_remaining":     round(sess.days_remaining, 1),
            "elapsed_days":       round(sess.elapsed_days, 1),
            "required_daily_pct": sess.required_daily_return_pct,
            "peak_balance":       sess.peak_balance,
            "drawdown_from_peak": sess.drawdown_from_peak_pct,
            "day_drawdown":       sess.day_drawdown_pct,
            "emergency_dd_limit": safety.get("emergency_dd"),
            "session_trades":     sess.session_trades,
            "session_win_rate":   sess.session_win_rate,
            "consecutive_losses": sess.consecutive_losses,
            "paused_until":       self._pause_until.isoformat() if self._pause_until else None,
            "stopped_reason":     sess.stopped_reason,
            "day_records":        [asdict(r) for r in sess.day_records],
            "risk_params":        self.get_risk_params(),
            "allowed_strategies": sess.phase.allowed_strategies,
        }

    def _session_dict(self) -> Dict:
        if not self.session:
            return {}
        s = self.session
        return {
            "initial_balance":  s.initial_balance,
            "current_balance":  s.current_balance,
            "target_balance":   s.target_balance,
            "started_at":       s.started_at,
            "deadline_days":    s.deadline_days,
            "mode":             s.mode,
            "active":           s.active,
            "stopped_reason":   s.stopped_reason,
            "progress_pct":     round(s.progress_pct, 1),
            "phase":            s.phase.name,
        }

    def _save(self):
        try:
            self._persist.parent.mkdir(parents=True, exist_ok=True)
            if self.session:
                data = {
                    "session":     asdict(self.session),
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
                    f"[Boost] Сессия восстановлена: mode={self.session.mode}, "
                    f"active={self.session.active}, balance=${self.session.current_balance:.2f}"
                )
        except Exception as e:
            logger.warning(f"[Boost] Загрузка сессии: {e}")
            self.session = None
