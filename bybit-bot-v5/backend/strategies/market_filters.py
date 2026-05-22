"""
Общие рыночные фильтры: MarketFilter, HTFFilter, LevelBuilder, PerSymbolGuard, calc_ai_score.
Единая библиотека качественных фильтров — используется в S5, S10 и при желании в других стратегиях.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Внутренние хелперы (не экспортируются)
# ─────────────────────────────────────────────────────────────────────────────

def _simple_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
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


def _simple_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    return float((100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
# 1. MarketFilter — глобальный ATR + flat-фильтр
# ─────────────────────────────────────────────────────────────────────────────

_ATR_MIN_PCT: Dict[str, float] = {
    "DOGEUSDT": 0.18,   # DOGE — ниже порог из-за меньшей абсолютной волатильности
}
_ATR_MIN_PCT_DEFAULT = 0.25   # для всех остальных символов


class MarketFilter:
    """
    Глобальный рыночный фильтр — проверять ПЕРЕД генерацией любого сигнала.

    Блокирует торговлю если:
    - ATR(14) ниже минимального порога волатильности (мёртвый рынок)
    - рынок флэтовый: EMA50/EMA200 расстояние < 0.12% ИЛИ ADX14 < 18
    """

    @staticmethod
    def check(df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Returns:
            ok       — можно торговать
            reason   — причина блокировки или 'ok'
            atr_pct  — ATR в % от цены
            adx      — значение ADX
            is_flat  — True если рынок флэтовый
        """
        if len(df) < 210:
            return {"ok": True, "reason": "skip:no_data", "atr_pct": 0.0, "adx": 0.0, "is_flat": False}

        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # ATR(14) EMA-сглаженный
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr   = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        c0    = float(close.iloc[-1])
        atr_pct = atr / c0 * 100 if c0 > 0 else 0.0

        atr_min = _ATR_MIN_PCT.get(symbol, _ATR_MIN_PCT_DEFAULT)
        if atr_pct < atr_min:
            return {
                "ok":      False,
                "reason":  f"LOW_ATR:{atr_pct:.3f}%<{atr_min:.2f}%",
                "atr_pct": round(atr_pct, 4),
                "adx":     0.0,
                "is_flat": True,
            }

        ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        ema_dist_pct = abs(ema50 - ema200) / max(ema200, 1e-9) * 100

        adx = _simple_adx(high, low, close, 14)

        is_flat = (ema_dist_pct < 0.12) or (adx < 18)
        if is_flat:
            return {
                "ok":      False,
                "reason":  f"FLAT_MARKET:ema_dist={ema_dist_pct:.3f}%,ADX={adx:.1f}",
                "atr_pct": round(atr_pct, 4),
                "adx":     round(adx, 1),
                "is_flat": True,
            }

        return {
            "ok":      True,
            "reason":  "ok",
            "atr_pct": round(atr_pct, 4),
            "adx":     round(adx, 1),
            "is_flat": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. HTFFilter — H1/15m фильтр тренда для S5
# ─────────────────────────────────────────────────────────────────────────────

class HTFFilter:
    """
    H1 фильтр: EMA50 > EMA200 → bias=BUY; EMA50 < EMA200 → bias=SELL.
    15m подтверждение: RSI(14) > 50 для BUY, < 50 для SELL.

    strict=True (по умолчанию): h1_bias противоположный action → блок.
    strict=False: только предупреждение.
    """

    @staticmethod
    def check(
        df_h1:  Optional[pd.DataFrame],
        df_m15: Optional[pd.DataFrame],
        action: str,
        strict: bool = True,
    ) -> Dict:
        result: Dict = {
            "ok":        True,
            "h1_bias":   "NEUTRAL",
            "h1_ema50":  0.0,
            "h1_ema200": 0.0,
            "m15_rsi":   50.0,
            "reason":    "ok",
        }

        # H1 EMA50 vs EMA200
        if df_h1 is not None and len(df_h1) >= 210:
            c_h1  = df_h1["close"].astype(float)
            e50   = float(c_h1.ewm(span=50,  adjust=False).mean().iloc[-1])
            e200  = float(c_h1.ewm(span=200, adjust=False).mean().iloc[-1])
            result["h1_ema50"]  = round(e50,  6)
            result["h1_ema200"] = round(e200, 6)

            if e50 > e200:
                result["h1_bias"] = "BUY"
            elif e50 < e200:
                result["h1_bias"] = "SELL"

            if strict and result["h1_bias"] != "NEUTRAL" and result["h1_bias"] != action:
                result["ok"]     = False
                result["reason"] = f"NO_HTF:H1_bias={result['h1_bias']}≠{action}"
                return result

        # 15m RSI подтверждение
        if df_m15 is not None and len(df_m15) >= 20:
            rsi_val = _simple_rsi(df_m15["close"].astype(float), 14)
            result["m15_rsi"] = round(rsi_val, 1)
            if action == "BUY" and rsi_val < 50:
                result["ok"]     = False
                result["reason"] = f"NO_15M:RSI={rsi_val:.1f}<50"
                return result
            if action == "SELL" and rsi_val > 50:
                result["ok"]     = False
                result["reason"] = f"NO_15M:RSI={rsi_val:.1f}>50"
                return result

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. LevelBuilder — swing-уровни + distance check
# ─────────────────────────────────────────────────────────────────────────────

class LevelBuilder:
    """
    Строит swing-уровни поддержки/сопротивления из price action.
    Кластеризует близкие уровни по ATR.

    Использование:
      levels = LevelBuilder.build(df)
      ok, dist_pct = LevelBuilder.distance_ok(price, "BUY", levels, threshold_pct=0.25)
    """

    @staticmethod
    def build(df: pd.DataFrame, lookback: int = 100) -> Dict:
        if len(df) < 10:
            return {"supports": [], "resistances": []}

        h = df["high"].values[-lookback:]
        l = df["low"].values[-lookback:]
        c = df["close"].values[-lookback:]

        # ATR для кластеризации
        n = min(14, len(c) - 1)
        tr_vals = [
            max(h[-i] - l[-i], abs(h[-i] - c[-i-1]), abs(l[-i] - c[-i-1]))
            for i in range(1, n + 1)
        ]
        tol = (sum(tr_vals) / len(tr_vals) if tr_vals else 0.0) * 0.8

        sh: List[float] = []
        sl: List[float] = []
        for i in range(2, len(h) - 2):
            if h[i] >= max(h[i-2], h[i-1], h[i+1], h[i+2]):
                sh.append(float(h[i]))
            if l[i] <= min(l[i-2], l[i-1], l[i+1], l[i+2]):
                sl.append(float(l[i]))

        def _cluster(vals: List[float]) -> List[float]:
            if not vals or tol <= 0:
                return vals
            res: List[float] = []
            grp = [sorted(vals)[0]]
            for v in sorted(vals)[1:]:
                if v - grp[0] <= tol:
                    grp.append(v)
                else:
                    res.append(sum(grp) / len(grp))
                    grp = [v]
            res.append(sum(grp) / len(grp))
            return res

        return {"supports": _cluster(sl), "resistances": _cluster(sh)}

    @staticmethod
    def distance_ok(
        price: float,
        action: str,
        levels: Dict,
        threshold_pct: float = 0.25,
    ) -> Tuple[bool, float]:
        """
        BUY: расстояние до ближайшего сопротивления > threshold_pct%.
        SELL: расстояние до ближайшей поддержки > threshold_pct%.
        Returns (ok, distance_pct).
        """
        if not levels:
            return True, 999.0

        if action == "BUY":
            targets = [r for r in levels.get("resistances", []) if r > price]
        else:
            targets = [s for s in levels.get("supports",    []) if s < price]

        if not targets:
            return True, 999.0

        nearest  = min(targets, key=lambda x: abs(x - price))
        dist_pct = abs(nearest - price) / price * 100
        return dist_pct >= threshold_pct, round(dist_pct, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PerSymbolGuard — защита от переторговки
# ─────────────────────────────────────────────────────────────────────────────

class PerSymbolGuard:
    """
    Защита от переторговки на уровне символа.
    - Минимум N минут между сделками (default 15)
    - Кулдаун после SL (по умолчанию 5 свечей)
    - 2h блокировка после 3 подряд SL
    """

    def __init__(self) -> None:
        self._last_trade:    Dict[str, datetime] = {}
        self._sl_until:      Dict[str, datetime] = {}
        self._block_until:   Dict[str, datetime] = {}
        self._consec_losses: Dict[str, int]      = {}

    def can_trade(self, symbol: str, min_interval_min: float = 15.0) -> Dict:
        now = datetime.now(timezone.utc)

        bu = self._block_until.get(symbol)
        if bu and now < bu:
            rem = int((bu - now).total_seconds() / 60)
            return {"ok": False, "reason": f"BLOCKED_2H:{rem}min_left"}

        su = self._sl_until.get(symbol)
        if su and now < su:
            rem = int((su - now).total_seconds() / 60)
            return {"ok": False, "reason": f"SL_COOLDOWN:{rem}min_left"}

        lt = self._last_trade.get(symbol)
        if lt:
            elapsed = (now - lt).total_seconds() / 60
            if elapsed < min_interval_min:
                rem = round(min_interval_min - elapsed, 1)
                return {"ok": False, "reason": f"MIN_INTERVAL:{rem}min_left"}

        return {"ok": True, "reason": "ok"}

    def register_trade(self, symbol: str) -> None:
        self._last_trade[symbol] = datetime.now(timezone.utc)

    def register_sl(
        self,
        symbol: str,
        candle_minutes: float = 5.0,
        candle_cooldown: int = 5,
    ) -> None:
        now = datetime.now(timezone.utc)
        self._sl_until[symbol] = now + timedelta(seconds=candle_minutes * candle_cooldown * 60)

        losses = self._consec_losses.get(symbol, 0) + 1
        self._consec_losses[symbol] = losses
        if losses >= 3:
            self._block_until[symbol]   = now + timedelta(hours=2)
            self._consec_losses[symbol] = 0
            logger.warning(f"[PerSymbolGuard] {symbol}: 3 подряд SL → блок 2h")

    def register_win(self, symbol: str) -> None:
        self._consec_losses[symbol] = 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. calc_ai_score — мультифакторный скоринг сделки
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 5a. calc_quality_score — оценка качества сделки 0-10
# ─────────────────────────────────────────────────────────────────────────────

def calc_quality_score(
    df:            pd.DataFrame,
    signal_action: str,
    spread_pct:    float = 0.0,
    ml_prob:       Optional[float] = None,
    atr_pct:       float = 0.0,
) -> float:
    """
    Trade quality score 0-10. Min 7.5 (scalper: 8.0) to trade.

    Weights: Trend=25%, Momentum=20%, Volume=20%, Spread=15%, Volatility=10%, ML=10%
    """
    if len(df) < 50:
        return 5.0

    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)
    c0     = float(close.iloc[-1])

    score = 0.0

    # Trend (2.5 pts): EMA stack alignment
    e8  = float(close.ewm(span=8,  adjust=False).mean().iloc[-1])
    e21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    e50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    if signal_action == "BUY":
        if e8 > e21 > e50 and c0 > e8:
            score += 2.5
        elif e21 > e50 and c0 > e21:
            score += 1.0
    else:
        if e8 < e21 < e50 and c0 < e8:
            score += 2.5
        elif e21 < e50 and c0 < e21:
            score += 1.0

    # Momentum (2.0 pts): RSI zone
    rsi = _simple_rsi(close, 14)
    if signal_action == "BUY":
        if 50 <= rsi <= 70:
            score += 2.0
        elif 40 <= rsi < 50 or 70 < rsi <= 76:
            score += 1.0
    else:
        if 30 <= rsi <= 50:
            score += 2.0
        elif 24 <= rsi < 30 or 50 < rsi <= 60:
            score += 1.0

    # Volume (2.0 pts): current vs 20-bar avg
    vol_avg   = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / (vol_avg + 1e-9)
    if vol_ratio >= 1.5:
        score += 2.0
    elif vol_ratio >= 1.0:
        score += 1.0

    # Spread (1.5 pts)
    if spread_pct <= 0.03:
        score += 1.5
    elif spread_pct <= 0.06:
        score += 0.8
    elif spread_pct <= 0.09:
        score += 0.3

    # Volatility (1.0 pts): ATR % in healthy range
    if 0.2 <= atr_pct <= 1.5:
        score += 1.0
    elif (0.1 <= atr_pct < 0.2) or (1.5 < atr_pct <= 2.0):
        score += 0.5

    # ML probability (1.0 pts)
    if ml_prob is None:
        score += 0.5
    elif ml_prob >= 0.65:
        score += 1.0
    elif ml_prob >= 0.55:
        score += 0.5

    return round(min(score, 10.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. calc_ai_score — мультифакторный скоринг сделки
# ─────────────────────────────────────────────────────────────────────────────

def calc_ai_score(
    trend:      bool,
    volume:     bool,
    htf:        bool,
    liquidity:  bool,
    rr:         float,
    volatility: bool,
) -> int:
    """
    AI Score 0-100. Торговать если score >= 80.

    Веса:
      trend=25, volume=20, htf=20, liquidity=15, rr=10, volatility=10

    rr: 10 pts если >= 2.5; 5 pts если >= 2.0; 0 если < 2.0.
    """
    score = 0
    if trend:      score += 25
    if volume:     score += 20
    if htf:        score += 20
    if liquidity:  score += 15
    if rr >= 2.5:  score += 10
    elif rr >= 2.0: score += 5
    if volatility: score += 10
    return min(score, 100)
