"""
ScalperPro Strategy (S10) + ScalperTrendContext (MTF фильтр).

Три режима работы:
  base     — только 3m фильтры, confidence_min=0.66
  mtf_h1   — 3m + строгий H1-фильтр (ОСНОВНОЙ РЕЖИМ), confidence_min=0.68
  mtf_full — 3m + строгий H1 + 15m должен совпадать, confidence_min=0.72

Ожидаемая доходность (честная формула):
  E = WR × avg_win − (1 − WR) × avg_loss − fees − slippage

  WR 72-78% — исторический ориентир из paper trading.
  Реальный трейдинг даёт ниже из-за проскальзывания, отстаивания в очереди
  и ночных гэпов. Подтверждать только walk-forward backtestом на out-of-sample данных.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import BaseStrategy, TradingSignal
from .market_filters import LevelBuilder, calc_ai_score

logger = logging.getLogger(__name__)


# ── символы для SC_* режима ───────────────────────────────────────────────────
# Не должны пересекаться с S1-S9, S11-S14:
# S1=BTC S2=ETH S4=BNB S5=DOGE S6=XRP S8=DOT S9=NEAR S10=LINK S11=AVAX
SCALP_SYMBOLS = [
    # Пользовательские фавориты (создаются первыми)
    "JASMYUSDT",       # SC_JASM — high volume, volatile meme
    "1000SHIBUSDT",    # SC_SHIB — micro-lot, high vol
    "BILLIEUSDT",      # SC_BILL — active meme token
    # Дополнительные ликвидные
    "TRXUSDT",
    "XLMUSDT",
    "VETUSDT",
    "LTCUSDT",
    "ATOMUSDT",
    "ARBUSDT",
    "SUIUSDT",
    "FTMUSDT",
    "INJUSDT",
]


# ── приватные хелперы ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi_series(series: pd.Series, period: int = 7) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _bbands(series: pd.Series, period: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + k * std, sma - k * std


def _adx_scalar(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up   = high.diff()
    down = (-low.diff())
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    a    = 1.0 / period
    atr_s = tr.ewm(alpha=a, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(alpha=a, adjust=False).mean() / (atr_s + 1e-9)
    di_m  = 100 * dm_m.ewm(alpha=a, adjust=False).mean() / (atr_s + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    return float(dx.ewm(alpha=a, adjust=False).mean().iloc[-1])


# ── ScalperTrendContext ───────────────────────────────────────────────────────

class ScalperTrendContext:
    """
    MTF тренд-контекст для S10 и SC_* стратегий.

    Логика:
      BUY  = EMA8 > EMA21 > EMA50 + цена выше EMA8 + ADX >= adx_min
      SELL = EMA8 < EMA21 < EMA50 + цена ниже EMA8 + ADX >= adx_min
      NEUTRAL = всё остальное

    ADX thresholds:
      H1:  ADX_MIN = 20 — строгий фильтр качественного тренда
      15m: ADX_MIN = 15 — чуть мягче, только для подтверждения
    """

    H1_ADX_MIN  = 20
    M15_ADX_MIN = 15

    @classmethod
    def compute(cls, df: Optional[pd.DataFrame], label: str = "H1") -> Dict:
        adx_min = cls.H1_ADX_MIN if label == "H1" else cls.M15_ADX_MIN

        if df is None or len(df) < 55:
            return {
                "direction": "NEUTRAL",
                "strength":  0.0,
                "reason":    f"{label}:no_data",
            }

        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        e8   = _ema(close, 8)
        e21  = _ema(close, 21)
        e50  = _ema(close, 50)
        adx_val = _adx_scalar(high, low, close, 14)

        c   = float(close.iloc[-1])
        v8  = float(e8.iloc[-1])
        v21 = float(e21.iloc[-1])
        v50 = float(e50.iloc[-1])

        bull = v8 > v21 > v50 and c > v8 and adx_val >= adx_min
        bear = v8 < v21 < v50 and c < v8 and adx_val >= adx_min

        if bull:
            return {
                "direction": "BUY",
                "strength":  round(adx_val, 1),
                "reason":    f"{label} BUY: EMA8={v8:.4f}>EMA21={v21:.4f}>EMA50={v50:.4f}, ADX={adx_val:.0f}",
            }
        if bear:
            return {
                "direction": "SELL",
                "strength":  round(adx_val, 1),
                "reason":    f"{label} SELL: EMA8={v8:.4f}<EMA21={v21:.4f}<EMA50={v50:.4f}, ADX={adx_val:.0f}",
            }
        return {
            "direction": "NEUTRAL",
            "strength":  round(adx_val, 1),
            "reason":    f"{label} NEUTRAL: ADX={adx_val:.0f} or no EMA-stack",
        }

    @staticmethod
    def aligns_strict(trend_ctx: Dict, action: str) -> bool:
        """
        Строгий режим (mtf_h1, mtf_full):
        - NEUTRAL → разрешено (без MTF-бонуса к confidence)
        - direction == action → разрешено
        - direction != action → БЛОКИРОВКА (контртренд полностью запрещён)
        """
        d = trend_ctx["direction"]
        return d == "NEUTRAL" or d == action

    @staticmethod
    def aligns(trend_ctx: Dict, action: str) -> bool:
        """Мягкий режим (только base): слабый контртренд ADX < 30 не блокирует."""
        d = trend_ctx["direction"]
        if d == "NEUTRAL" or d == action:
            return True
        return trend_ctx.get("strength", 0) < 30


# ── ScalperProStrategy ────────────────────────────────────────────────────────

class ScalperProStrategy(BaseStrategy):
    """
    ScalperPro — 3-минутный скальпер.

    Фильтры BUY:
      1. EMA(8) > EMA(21) — краткосрочный тренд вверх
      2. Цена выше EMA(21)
      3. RSI(7) пересёк вверх из oversold (≤35) за последние 2 свечи
      4. Предыдущая или текущая свеча касалась нижней BB, текущая закрылась выше
      5. Объём ≥ 1.3× среднего (фильтр мёртвого рынка)
      6. Свеча не импульсная: range < 0.45%, body < 2.2× среднего тела

    Фильтры SELL — зеркально.

    MTF фильтр (mtf_h1 / mtf_full):
      H1 строгий: контртренд полностью запрещён. NEUTRAL — ок (без бонуса к confidence).
      15m (только mtf_full): направление должно совпадать.

    Expectancy check:
      RR >= 1.8; TP >= 2× estimated_fees.

    Confidence:
      Base = 0.58; max = 0.92; min для сигнала зависит от режима.
    """

    ID   = "S10"
    NAME = "Scalper Pro"
    DESCRIPTION = "3m EMA+RSI+BB скальпер, строгий H1 фильтр, anti-impulse, fee guard"
    REGIME_PREFERENCE = []   # любой рынок — фильтруется внутри

    # ── индикаторы ────────────────────────────────────────────────────────────
    _EMA_FAST   = 8
    _EMA_SLOW   = 21
    _RSI_PERIOD = 7
    _BB_PERIOD  = 20
    _BB_K       = 2.0
    _VOL_MULT   = 1.3
    _RSI_OS     = 35
    _RSI_OB     = 65
    _RSI_WINDOW = 2
    _BB_TOUCH   = 0.008    # 0.8% proximity to BB band
    _MIN_BARS   = 60

    # ── anti-impulse ──────────────────────────────────────────────────────────
    _MAX_CANDLE_RANGE_PCT = 0.45   # % высоты свечи относительно цены
    _MAX_BODY_MULT        = 2.2    # тело свечи vs среднего тела за 20 свечей

    # ── spread / funding guard ────────────────────────────────────────────────
    _MAX_SPREAD_PCT        = 0.08   # % — отклоняем если выше
    _SPREAD_SOFT_LIMIT_PCT = 0.048  # > 60% max → штраф -0.05 к confidence
    _MAX_FUNDING_ABS       = 0.03   # |funding| > 3% → штраф -0.10

    # ── expectancy ────────────────────────────────────────────────────────────
    _MIN_RR           = 1.8
    _MIN_TP_FEE_MULT  = 2.0    # TP distance > 2× estimated_fee_pct × entry

    # ── confidence ────────────────────────────────────────────────────────────
    _BASE_CONFIDENCE  = 0.58
    _MIN_CONFIDENCE: Dict[str, float] = {
        "base":     0.66,
        "mtf_h1":   0.68,
        "mtf_full": 0.72,
    }

    # ── early TP protection ───────────────────────────────────────────────────
    _EARLY_TP_MIN_PROGRESS_PCT = 50.0
    _EARLY_TP_MIN_CANDLES      = 2     # минимум 2 × 3min = 6 мин в позиции

    def __init__(
        self,
        symbol: str = "DOGEUSDT",
        mode:   str = "mtf_h1",
        **kwargs,
    ):
        super().__init__(
            symbol=symbol,
            timeframe="3",
            leverage=kwargs.pop("leverage", 5),
            stop_loss_pct=kwargs.pop("stop_loss_pct", 0.12),
            take_profit_pct=kwargs.pop("take_profit_pct", 0.30),
            breakeven_pct=kwargs.pop("breakeven_pct", 0.15),
            trailing_stop_pct=kwargs.pop("trailing_stop_pct", 0.08),
            max_hold_minutes=kwargs.pop("max_hold_minutes", 30.0),
            **kwargs,
        )
        self.mode = mode if mode in ("base", "mtf_h1", "mtf_full") else "mtf_h1"

    # ── analyze ───────────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        df_h1:               Optional[pd.DataFrame] = None,
        df_m15:              Optional[pd.DataFrame] = None,
        spread_pct:          Optional[float] = None,
        funding_rate:        Optional[float] = None,
        orderbook_imbalance: Optional[float] = None,
    ) -> Optional[TradingSignal]:
        """
        df     — 3m свечи (обязательно, минимум _MIN_BARS)
        df_h1  — H1 свечи (рекомендуется в режиме mtf_h1/mtf_full)
        df_m15 — 15m свечи (только в mtf_full)
        spread_pct          — текущий спред в % (optional)
        funding_rate        — текущая ставка финансирования (optional)
        orderbook_imbalance — дисбаланс стакана bid/ask (optional, пока только для лога)
        """
        if len(df) < self._MIN_BARS:
            return None

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        open_  = df["open"].astype(float)
        volume = df["volume"].astype(float)

        # ── Индикаторы ────────────────────────────────────────────────────────
        ema_fast = _ema(close, self._EMA_FAST)
        ema_slow = _ema(close, self._EMA_SLOW)
        rsi_ser  = _rsi_series(close, self._RSI_PERIOD)
        bb_up, bb_lo = _bbands(close, self._BB_PERIOD, self._BB_K)
        avg_vol  = volume.rolling(20).mean()
        avg_body = (close - open_).abs().rolling(20).mean()

        # Текущая свеча [−1]
        c0    = float(close.iloc[-1])
        o0    = float(open_.iloc[-1])
        h0    = float(high.iloc[-1])
        l0    = float(low.iloc[-1])
        ef0   = float(ema_fast.iloc[-1])
        es0   = float(ema_slow.iloc[-1])
        bbu0  = float(bb_up.iloc[-1])
        bbl0  = float(bb_lo.iloc[-1])
        vol0  = float(volume.iloc[-1])
        avol0 = float(avg_vol.iloc[-1]) + 1e-9
        abody = float(avg_body.iloc[-1]) + 1e-9

        # Предыдущая свеча [−2]
        l1   = float(low.iloc[-2])
        h1   = float(high.iloc[-2])
        bbu1 = float(bb_up.iloc[-2])
        bbl1 = float(bb_lo.iloc[-2])

        # RSI окно
        w        = self._RSI_WINDOW
        rsi_vals = [float(rsi_ser.iloc[-i]) for i in range(1, w + 2)]
        r0       = rsi_vals[0]
        rsi_was_os = any(v < self._RSI_OS for v in rsi_vals[1:])
        rsi_was_ob = any(v > self._RSI_OB for v in rsi_vals[1:])
        vol_ratio  = vol0 / avol0

        # ── Anti-impulse guard ─────────────────────────────────────────────────
        candle_range     = h0 - l0
        candle_range_pct = candle_range / c0 * 100 if c0 > 0 else 0.0
        curr_body        = abs(c0 - o0)
        body_ratio       = curr_body / abody

        if candle_range_pct > self._MAX_CANDLE_RANGE_PCT:
            logger.debug(f"[S10] {self.symbol}: impulse candle range {candle_range_pct:.3f}% > {self._MAX_CANDLE_RANGE_PCT}%")
            return None
        if body_ratio > self._MAX_BODY_MULT:
            logger.debug(f"[S10] {self.symbol}: large body ratio {body_ratio:.2f} > {self._MAX_BODY_MULT}")
            return None

        # ── Spread guard (hard block) ──────────────────────────────────────────
        if spread_pct is not None and spread_pct > self._MAX_SPREAD_PCT:
            logger.debug(f"[S10] {self.symbol}: BAD_SPREAD {spread_pct:.4f}% > {self._MAX_SPREAD_PCT}%")
            return None

        # ── Базовые 3m сигналы ─────────────────────────────────────────────────
        # BUY: EMA тренд + RSI пересёк из oversold + BB отскок подтверждён предыдущей свечой
        prev_touch_bbl = l1 <= bbl1 * 1.003 or l0 <= bbl0 * 1.003
        prev_touch_bbu = h1 >= bbu1 * 0.997 or h0 >= bbu0 * 0.997

        buy = (
            ef0 > es0
            and c0 > es0
            and rsi_was_os
            and r0 >= self._RSI_OS
            and prev_touch_bbl
            and c0 > bbl0
            and vol_ratio >= self._VOL_MULT
        )
        sell = (
            ef0 < es0
            and c0 < es0
            and rsi_was_ob
            and r0 <= self._RSI_OB
            and prev_touch_bbu
            and c0 < bbu0
            and vol_ratio >= self._VOL_MULT
        )

        if not buy and not sell:
            return None

        action = "BUY" if buy else "SELL"

        # ── MTF фильтр ─────────────────────────────────────────────────────────
        h1_ctx  = ScalperTrendContext.compute(df_h1,  "H1")
        m15_ctx = ScalperTrendContext.compute(df_m15, "15m")

        if self.mode in ("mtf_h1", "mtf_full"):
            if not ScalperTrendContext.aligns_strict(h1_ctx, action):
                logger.debug(
                    f"[S10/{self.mode}] {self.symbol}: {action} ← blocked by {h1_ctx['reason']}"
                )
                return None
        else:   # base
            if not ScalperTrendContext.aligns(h1_ctx, action):
                return None

        if self.mode == "mtf_full" and m15_ctx["direction"] not in ("NEUTRAL", action):
            logger.debug(
                f"[S10/mtf_full] {self.symbol}: {action} ← blocked by 15m {m15_ctx['reason']}"
            )
            return None

        # ── SL / TP ────────────────────────────────────────────────────────────
        entry = c0
        if action == "BUY":
            sl = round(entry * (1 - self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 + self.take_profit_pct / 100), 8)
        else:
            sl = round(entry * (1 + self.stop_loss_pct / 100), 8)
            tp = round(entry * (1 - self.take_profit_pct / 100), 8)

        # ── Expectancy check ───────────────────────────────────────────────────
        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        rr     = reward / risk if risk > 0 else 0.0

        if rr < self._MIN_RR:
            logger.debug(f"[S10] {self.symbol}: R:R {rr:.2f} < {self._MIN_RR}")
            return None

        # TP должен покрывать минимум 2× расчётные комиссии
        estimated_fee_pct = 0.055 / 100 * 2 * self.leverage
        if reward < estimated_fee_pct * entry * self._MIN_TP_FEE_MULT:
            logger.debug(f"[S10] {self.symbol}: TP {reward:.6f} < 2× fees, skip")
            return None

        # ── Confidence ─────────────────────────────────────────────────────────
        r1 = rsi_vals[1] if len(rsi_vals) > 1 else r0
        conf = self._BASE_CONFIDENCE

        # RSI cross quality: глубина oversold/overbought до пересечения
        rsi_depth = (self._RSI_OS - r1) if action == "BUY" else (r1 - self._RSI_OB)
        conf += min(0.10, max(0.04, rsi_depth / 100))

        # Volume
        conf += min(0.08, max(0.02, (vol_ratio - self._VOL_MULT) * 0.08))

        # H1 alignment
        if h1_ctx["direction"] == action:
            conf += 0.08

        # 15m alignment
        if m15_ctx["direction"] == action:
            conf += 0.04

        # Clean BB rejection: предыдущая свеча пробила BB, текущая вернулась
        clean_rejection = (
            (l1 <= bbl1 * 1.003 and c0 > bbl0) if action == "BUY"
            else (h1 >= bbu1 * 0.997 and c0 < bbu0)
        )
        if clean_rejection:
            conf += 0.05

        # Spread soft penalty
        if spread_pct is not None and spread_pct > self._SPREAD_SOFT_LIMIT_PCT:
            conf -= 0.05

        # Funding penalty
        if funding_rate is not None and abs(funding_rate) > self._MAX_FUNDING_ABS:
            conf -= 0.10

        conf_before_mtf = round(
            self._BASE_CONFIDENCE
            + min(0.10, max(0.04, rsi_depth / 100))
            + min(0.08, max(0.02, (vol_ratio - self._VOL_MULT) * 0.08)),
            3,
        )
        conf = round(min(0.92, conf), 3)

        # Confidence threshold
        min_conf = self._MIN_CONFIDENCE.get(self.mode, 0.68)
        if conf < min_conf:
            logger.debug(
                f"[S10/{self.mode}] {self.symbol}: confidence {conf} < threshold {min_conf}"
            )
            return None

        # ── AI Score ───────────────────────────────────────────────────────────
        ai_score = calc_ai_score(
            trend=(ef0 > es0) if action == "BUY" else (ef0 < es0),
            volume=vol_ratio >= self._VOL_MULT,
            htf=h1_ctx["direction"] in ("NEUTRAL", action),
            liquidity=True,   # S10 не строит уровни
            rr=rr,
            volatility=True,  # ATR проверяется MarketFilter на уровне main
        )
        if ai_score < 80:
            logger.debug(f"[S10] {self.symbol}: AI score {ai_score} < 80")
            return None

        # ── Reason ─────────────────────────────────────────────────────────────
        parts = [
            f"S10/{self.mode} {action}",
            f"RSI({r1:.0f}→{r0:.0f})",
            f"Vol×{vol_ratio:.1f}",
            f"BB_{'LO' if buy else 'HI'}_reject{'_clean' if clean_rejection else ''}",
            f"RR={rr:.2f}",
            f"AI={ai_score}",
        ]
        if h1_ctx["direction"] != "NEUTRAL":
            parts.append(h1_ctx["reason"])
        if m15_ctx["direction"] != "NEUTRAL":
            parts.append(m15_ctx["reason"])
        if funding_rate is not None and abs(funding_rate) > 0.005:
            parts.append(f"funding={funding_rate:.4f}")

        # ── filters_passed (максимально подробно) ──────────────────────────────
        filters_passed = {
            "mode":                 self.mode,
            "ema_trend":            (ef0 > es0) if action == "BUY" else (ef0 < es0),
            "rsi_cross":            True,
            "rsi_prev":             round(r1, 1),
            "rsi_current":          round(r0, 1),
            "bb_rejection":         True,
            "bb_prev_touch":        (l1 <= bbl1 * 1.003) if action == "BUY" else (h1 >= bbu1 * 0.997),
            "bb_clean_rejection":   clean_rejection,
            "volume_ratio":         round(vol_ratio, 2),
            "candle_range_pct":     round(candle_range_pct, 3),
            "candle_body_ratio":    round(body_ratio, 2),
            "spread_pct":           spread_pct,
            "funding_rate":         funding_rate,
            "orderbook_imbalance":  orderbook_imbalance,
            "h1_trend":             h1_ctx["direction"],
            "h1_adx":               h1_ctx["strength"],
            "h1_reason":            h1_ctx["reason"],
            "m15_trend":            m15_ctx["direction"],
            "m15_adx":              m15_ctx["strength"],
            "rr_ratio":             round(rr, 2),
            "fee_guard_passed":     True,
            "ai_score":             ai_score,
            "confidence_before_mtf": conf_before_mtf,
            "confidence_final":     conf,
            "confidence_min":       min_conf,
        }

        return TradingSignal(
            action=action,
            symbol=self.symbol,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=conf,
            reason=" | ".join(parts),
            filters_passed=filters_passed,
        )

    # ── Early TP override ──────────────────────────────────────────────────────

    def check_early_tp(self, current_price: float, threshold_pct: float = 85.0) -> Optional[float]:
        """
        Строгая защита от раннего TP:
        - прогресс к TP < 50%: НЕ закрывать
        - net PnL ≤ 0 после комиссий: НЕ закрывать
        - позиция открыта < 2 свечей (6 мин на 3m): НЕ закрывать (кроме аварийных сценариев)
        """
        if not self.current_position or threshold_pct <= 0:
            return None

        entry     = self.current_position["entry"]
        tp        = self.current_position["tp"]
        side      = self.current_position["side"]
        qty       = self.current_position.get("qty", 1.0)
        opened_at = self.current_position.get("opened_at")

        # Минимум 2 свечи (6 мин) — строгая защита через datetime для корректного timezone
        if opened_at is not None:
            try:
                now_dt = datetime.now(timezone.utc)
                # Нормализуем opened_at к aware datetime
                if isinstance(opened_at, pd.Timestamp):
                    oa_dt = opened_at.to_pydatetime()
                    if oa_dt.tzinfo is None:
                        oa_dt = oa_dt.replace(tzinfo=timezone.utc)
                elif isinstance(opened_at, datetime):
                    oa_dt = opened_at if opened_at.tzinfo else opened_at.replace(tzinfo=timezone.utc)
                else:
                    oa_dt = None

                if oa_dt is not None:
                    elapsed_min = (now_dt - oa_dt).total_seconds() / 60.0
                    if elapsed_min < self._EARLY_TP_MIN_CANDLES * 3:
                        return None
            except Exception:
                pass

        if side == "Buy":
            tp_dist  = tp - entry
            progress = (current_price - entry) / tp_dist * 100 if tp_dist > 0 else 0
            gross    = (current_price - entry) * qty
        else:
            tp_dist  = entry - tp
            progress = (entry - current_price) / tp_dist * 100 if tp_dist > 0 else 0
            gross    = (entry - current_price) * qty

        effective_threshold = max(self._EARLY_TP_MIN_PROGRESS_PCT, threshold_pct)
        if progress < effective_threshold:
            return None

        fee_pct = 0.055 / 100 * 2 * self.leverage
        fees    = entry * qty * fee_pct
        net_pnl = gross - fees
        if net_pnl <= 0:
            return None

        logger.info(
            f"[S10] {self.symbol}: early TP @ {current_price:.6f} "
            f"({progress:.0f}% progress, net_pnl≈{net_pnl:.4f})"
        )
        return current_price
