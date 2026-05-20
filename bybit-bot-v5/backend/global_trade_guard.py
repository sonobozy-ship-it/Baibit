"""
GlobalTradeGuard — единый защитный слой перед открытием любой сделки.

Принцип:
    Стратегия предлагает → Guard разрешает или блокирует → Risk Manager считает объём.

Применяется ГЛОБАЛЬНО ко всем стратегиям (S1-S14, SC_*, FUSION).
Никакая стратегия не может открыть сделку в обход Guard.

Проверки:
    1. Тренд 1H (EMA50 > EMA200 = UP, иначе DOWN)
    2. Старший тренд 4H (опционально, если данные есть)
    3. ADX — сила тренда (< 20 = флэт, торгуем только по тренду)
    4. ATR — волатильность (слишком низкая = нет движения)
    5. Volume spike — подтверждение объёмом
    6. ML confidence >= порога
    7. Risk/Reward >= 1.5
    8. Risk per trade <= MAX_RISK_PER_TRADE_USDT
    9. Leverage <= MAX_LEVERAGE
    10. Cooldown после MAX_CONSECUTIVE_LOSSES убытков
    11. Anti-FOMO: не входить в середину импульса (цена ушла > 1.5% от EMA9)
    12. Торговля только по тренду: LONG только если 1H UP, SHORT только если 1H DOWN
        (мягкий фильтр — предупреждает, не всегда блокирует)

Каждое решение логируется в JSON-формате.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardDecision:
    """Результат проверки Guard."""
    approved: bool
    blocked_by: List[str] = field(default_factory=list)
    warnings: List[str]   = field(default_factory=list)
    ml_probability: Optional[float] = None
    risk_reward: Optional[float]    = None
    decision: str = "WAIT"   # "TRADE" или "WAIT"

    def to_log_dict(self, strategy_id: str, raw_signal: str) -> Dict:
        return {
            "strategy":      strategy_id,
            "raw_signal":    raw_signal,
            "final_signal":  raw_signal if self.approved else "WAIT",
            "blocked_by":    self.blocked_by,
            "warnings":      self.warnings,
            "ml_probability": self.ml_probability,
            "risk_reward":   self.risk_reward,
            "decision":      self.decision,
            "timestamp":     datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Упрощённый ADX без внешних зависимостей."""
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


# ─────────────────────────────────────────────────────────────────────────────

class GlobalTradeGuard:
    """
    Единый защитный фильтр.
    Вызывается из trading_loop ПОСЛЕ того как стратегия сгенерировала сигнал.
    """

    # Мягкие фильтры (блокируют в нормальном режиме, но не в strong-trend)
    SOFT_BLOCK_AGAINST_1H_TREND = True
    # Минимальный ADX для подтверждения тренда (при торговле по тренду)
    ADX_TREND_MIN = 20.0
    # Максимальный сдвиг от EMA9 для anti-FOMO (в %)
    ANTI_FOMO_PCT = 1.5
    # Минимальный объём (× средний) для подтверждения
    VOLUME_SPIKE_MIN = 1.2

    def __init__(self):
        # Счётчик убытков подряд per strategy: sid → count
        self._consec_losses: Dict[str, int] = {}
        # Время последнего кулдауна per strategy: sid → datetime
        self._cooldown_until: Dict[str, datetime] = {}

    # ── Публичный API ────────────────────────────────────────────────────────

    def check(
        self,
        *,
        strategy_id: str,
        signal_action: str,       # "BUY" или "SELL"
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        qty: float,
        leverage: int,
        balance: float,
        df: Optional[pd.DataFrame] = None,         # свечи рабочего ТФ
        df_h1: Optional[pd.DataFrame] = None,      # свечи 1H
        df_h4: Optional[pd.DataFrame] = None,      # свечи 4H (опционально)
        ml_probability: Optional[float] = None,
        signal_confidence: float = 0.0,
        scalp_mode: bool = False,                  # агрессивный скальпинг: ослабленные фильтры
    ) -> GuardDecision:
        """
        Возвращает GuardDecision с approved=True/False и списком причин блокировки.
        В scalp_mode: пониженный порог confidence (0.55), R:R>=1.2, нет H1 фильтра.
        После 3+ убытков в scalp_mode: возврат к нормальным порогам.
        """
        blocked: List[str] = []
        warnings: List[str] = []

        # Адаптивные пороги: в scalp_mode ослабляем, но ужесточаем после убытков
        consec = self._consec_losses.get(strategy_id, 0)
        if scalp_mode and consec < 3:
            min_confidence = 0.55   # ослабленный порог
            min_rr = 1.2
        elif scalp_mode and consec < 5:
            min_confidence = cfg.MIN_SIGNAL_CONFIDENCE   # нормальный
            min_rr = 1.3
        else:
            min_confidence = cfg.MIN_SIGNAL_CONFIDENCE
            min_rr = 1.5

        # ── 1. Кулдаун после серии убытков ──────────────────────────────────
        if self._is_in_cooldown(strategy_id):
            blocked.append("cooldown_active")

        # ── 2. Risk/Reward ────────────────────────────────────────────────────
        rr = self._calc_rr(signal_action, entry_price, stop_loss, take_profit)
        if rr is not None and rr < min_rr:
            blocked.append(f"low_rr_{rr:.2f}")

        # ── 3. Risk per trade (USDT) ──────────────────────────────────────────
        risk_usdt = abs(entry_price - stop_loss) * qty * leverage if (entry_price and stop_loss) else 0
        if risk_usdt > cfg.MAX_RISK_PER_TRADE_USDT and cfg.MAX_RISK_PER_TRADE_USDT > 0:
            blocked.append(f"risk_too_high_{risk_usdt:.2f}usdt")

        # ── 4. Leverage cap ──────────────────────────────────────────────────
        if leverage > cfg.MAX_LEVERAGE:
            blocked.append(f"leverage_{leverage}x_exceeds_{cfg.MAX_LEVERAGE}x")

        # ── 5. Signal confidence ─────────────────────────────────────────────
        if signal_confidence < min_confidence and signal_confidence > 0:
            blocked.append(f"low_confidence_{signal_confidence:.2f}")

        # ── 6. ML probability ────────────────────────────────────────────────
        if ml_probability is not None and ml_probability < 0.75:
            if ml_probability < 0.55:
                blocked.append(f"ml_probability_low_{ml_probability:.2f}")
            else:
                warnings.append(f"ml_probability_marginal_{ml_probability:.2f}")

        # ── 7. Технические фильтры на 1H свечах (пропускаем в scalp_mode) ───
        if not scalp_mode and df_h1 is not None and len(df_h1) >= 50:
            self._check_h1_trend(signal_action, df_h1, blocked, warnings)

        # ── 8. Технические фильтры на рабочих свечах ────────────────────────
        if df is not None and len(df) >= 30:
            # В scalp_mode пропускаем anti-FOMO (быстрые движения = суть скальпа)
            self._check_working_tf(signal_action, df, blocked, warnings)

        # ── 9. Четырёхчасовой тренд (опционально, только не в scalp_mode) ───
        if not scalp_mode and df_h4 is not None and len(df_h4) >= 50:
            self._check_h4_trend(signal_action, df_h4, warnings)

        approved = len(blocked) == 0
        decision = GuardDecision(
            approved=approved,
            blocked_by=blocked,
            warnings=warnings,
            ml_probability=ml_probability,
            risk_reward=rr,
            decision="TRADE" if approved else "WAIT",
        )

        if not approved:
            log = decision.to_log_dict(strategy_id, signal_action)
            logger.info(f"[Guard] {strategy_id} BLOCKED: {blocked}")
            logger.debug(f"[Guard] detail: {log}")
        elif warnings:
            logger.debug(f"[Guard] {strategy_id} APPROVED with warnings: {warnings}")

        return decision

    def record_loss(self, strategy_id: str):
        """Записать убыток. При превышении MAX_CONSECUTIVE_LOSSES → кулдаун."""
        self._consec_losses[strategy_id] = self._consec_losses.get(strategy_id, 0) + 1
        if self._consec_losses[strategy_id] >= cfg.MAX_CONSECUTIVE_LOSSES:
            until = datetime.utcnow() + timedelta(minutes=cfg.COOLDOWN_MINUTES)
            self._cooldown_until[strategy_id] = until
            self._consec_losses[strategy_id] = 0
            logger.warning(
                f"[Guard] {strategy_id}: {cfg.MAX_CONSECUTIVE_LOSSES} убытков подряд → "
                f"кулдаун до {until.strftime('%H:%M UTC')}"
            )

    def record_win(self, strategy_id: str):
        """Сброс счётчика при выигрыше."""
        self._consec_losses[strategy_id] = 0

    def reset_all(self):
        """Полный сброс всех счётчиков и кулдаунов (вызывается при переключении режима обучения)."""
        self._consec_losses.clear()
        self._cooldown_until.clear()
        logger.info("[Guard] Все счётчики и кулдауны сброшены")

    def get_status(self) -> Dict:
        now = datetime.utcnow()
        cooldowns = {
            sid: (self._cooldown_until[sid] - now).total_seconds() / 60
            for sid in self._cooldown_until
            if self._cooldown_until[sid] > now
        }
        return {
            "consecutive_losses": dict(self._consec_losses),
            "cooldowns_remaining_min": cooldowns,
        }

    # ── Внутренние методы ────────────────────────────────────────────────────

    def _is_in_cooldown(self, strategy_id: str) -> bool:
        until = self._cooldown_until.get(strategy_id)
        if until and datetime.utcnow() < until:
            return True
        return False

    @staticmethod
    def _calc_rr(action: str, entry: float, sl: float, tp: float) -> Optional[float]:
        if not entry or not sl or not tp:
            return None
        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return None
        return round(reward / risk, 2)

    def _check_h1_trend(
        self,
        action: str,
        df_h1: pd.DataFrame,
        blocked: List[str],
        warnings: List[str],
    ):
        """EMA50/EMA200 тренд на 1H."""
        close = df_h1["close"].astype(float)
        ema50  = float(_ema(close, 50).iloc[-1])
        ema200 = float(_ema(close, 200).iloc[-1]) if len(close) >= 200 else None
        last   = float(close.iloc[-1])

        # Определяем тренд
        if ema200:
            trend_up = ema50 > ema200
        else:
            ema21  = float(_ema(close, 21).iloc[-1])
            trend_up = ema50 > ema21

        if self.SOFT_BLOCK_AGAINST_1H_TREND:
            if action == "BUY" and not trend_up:
                blocked.append("bullish_1h_against_downtrend")
            elif action == "SELL" and trend_up:
                blocked.append("bearish_1h_against_uptrend")

        # Anti-FOMO: цена далеко от EMA9
        ema9 = float(_ema(close, 9).iloc[-1])
        if ema9 > 0:
            deviation_pct = abs(last - ema9) / ema9 * 100
            if deviation_pct > self.ANTI_FOMO_PCT:
                blocked.append(f"anti_fomo_impulse_{deviation_pct:.1f}pct")

        # ADX — торгуем только при наличии тренда
        try:
            adx_val = _adx(df_h1)
            if adx_val < self.ADX_TREND_MIN:
                warnings.append(f"low_adx_1h_{adx_val:.1f}")
        except Exception:
            pass

    def _check_working_tf(
        self,
        action: str,
        df: pd.DataFrame,
        blocked: List[str],
        warnings: List[str],
    ):
        """Объём и ATR на рабочем таймфрейме. (Anti-FOMO — в _check_h1_trend, там пропускается в scalp_mode)"""
        # Подтверждение объёмом
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean().iloc[-1]
            vol_last = float(df["volume"].iloc[-1])
            if vol_ma and vol_ma > 0:
                vol_ratio = vol_last / vol_ma
                if vol_ratio < self.VOLUME_SPIKE_MIN:
                    warnings.append(f"low_volume_{vol_ratio:.2f}x")

        # Минимальная волатильность (ATR)
        try:
            if len(df) >= 14:
                atr_val = _atr(df)
                price   = float(df["close"].iloc[-1])
                atr_pct = atr_val / price * 100 if price > 0 else 0
                if atr_pct < 0.05:
                    warnings.append(f"low_volatility_atr_{atr_pct:.3f}pct")
        except Exception:
            pass

    def _check_h4_trend(
        self,
        action: str,
        df_h4: pd.DataFrame,
        warnings: List[str],
    ):
        """Мягкая проверка 4H тренда (не блокирует, только предупреждает)."""
        close = df_h4["close"].astype(float)
        ema50  = float(_ema(close, 50).iloc[-1])
        ema200 = float(_ema(close, 200).iloc[-1]) if len(close) >= 200 else None
        if ema200:
            trend_up = ema50 > ema200
            if (action == "BUY" and not trend_up) or (action == "SELL" and trend_up):
                warnings.append("against_4h_trend")
