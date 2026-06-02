"""
pandas_ta compatibility shim — реализация всех используемых индикаторов
через чистый numpy/pandas. Устанавливать pandas-ta не нужно.
"""
import numpy as np
import pandas as pd
from typing import Optional

# Версия для проверки импорта
version = "compat-1.0"


def ema(series: pd.Series, length: int = 9, **kwargs) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int = 20, **kwargs) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14, **kwargs) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, **kwargs) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False, min_periods=length).mean()


def bbands(series: pd.Series, length: int = 20, std: float = 2.0, **kwargs) -> pd.DataFrame:
    mid = series.rolling(window=length, min_periods=length).mean()
    sigma = series.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + std * sigma
    lower = mid - std * sigma
    bw = (upper - lower) / mid.replace(0, np.nan)
    bp = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        f"BBL_{length}_{float(std)}": lower,
        f"BBM_{length}_{float(std)}": mid,
        f"BBU_{length}_{float(std)}": upper,
        f"BBW_{length}_{float(std)}": bw,
        f"BBP_{length}_{float(std)}": bp,
    })


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    **kwargs,
) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDh_{fast}_{slow}_{signal}": hist,
        f"MACDs_{fast}_{slow}_{signal}": signal_line,
    })


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    **kwargs,
) -> pd.DataFrame:
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    raw_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_k = raw_k.rolling(smooth_k, min_periods=1).mean()
    stoch_d = stoch_k.rolling(d, min_periods=1).mean()
    return pd.DataFrame({
        f"STOCHk_{k}_{d}_{smooth_k}": stoch_k,
        f"STOCHd_{k}_{d}_{smooth_k}": stoch_d,
    })


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    **kwargs,
) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=length, min_periods=length).mean()
    mad = tp.rolling(window=length, min_periods=length).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def willr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    **kwargs,
) -> pd.Series:
    highest = high.rolling(length, min_periods=length).max()
    lowest = low.rolling(length, min_periods=length).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    **kwargs,
) -> pd.DataFrame:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    dm_plus_s = pd.Series(dm_plus, index=high.index)
    dm_minus_s = pd.Series(dm_minus, index=high.index)

    tr_s = atr(high, low, close, length=1)  # raw TR per candle
    # Wilders smoothing
    atr_s = tr_s.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    dmp_s = dm_plus_s.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    dmn_s = dm_minus_s.ewm(com=length - 1, adjust=False, min_periods=length).mean()

    dip = 100 * dmp_s / atr_s.replace(0, np.nan)
    dim = 100 * dmn_s / atr_s.replace(0, np.nan)

    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    adx_val = dx.ewm(com=length - 1, adjust=False, min_periods=length).mean()

    return pd.DataFrame({
        f"ADX_{length}": adx_val,
        f"DMP_{length}": dip,
        f"DMN_{length}": dim,
    })


def obv(close: pd.Series, volume: pd.Series, **kwargs) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    **kwargs,
) -> pd.DataFrame:
    """
    Ichimoku Kinko Hyo (упрощённый — без сдвига для сравнения с ценой).
    Senkou A/B не сдвинуты вперёд, что позволяет напрямую сравнивать с close.
    """
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
    senkou_a   = (tenkan_sen + kijun_sen) / 2
    senkou_b   = (high.rolling(senkou).max() + low.rolling(senkou).min()) / 2
    chikou     = close.shift(-kijun)

    return pd.DataFrame({
        f"ITS_{tenkan}": tenkan_sen,
        f"IKS_{kijun}":  kijun_sen,
        f"ISA_{tenkan}": senkou_a,
        f"ISB_{senkou}": senkou_b,
        f"ICS_{kijun}":  chikou,
    }, index=close.index)


def psar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af0: float = 0.02,
    af_step: float = 0.02,
    max_af: float = 0.2,
    **kwargs,
) -> pd.DataFrame:
    """Parabolic SAR. Возвращает PSARl (лонг, ниже цены), PSARs (шорт, выше), PSARd (направление 1/-1)."""
    hi = high.values
    lo = low.values
    n  = len(hi)

    psar_long  = np.full(n, np.nan)
    psar_short = np.full(n, np.nan)
    direction  = np.zeros(n)

    bull = True
    af   = af0
    ep   = hi[0]
    sar  = lo[0]

    for i in range(1, n):
        if bull:
            sar_new = sar + af * (ep - sar)
            # PSAR не может быть выше двух предыдущих минимумов
            sar_new = min(sar_new, lo[i - 1], lo[max(0, i - 2)])
            if lo[i] < sar_new:           # разворот → шорт
                bull    = False
                sar_new = ep
                ep      = lo[i]
                af      = af0
                psar_short[i] = sar_new
                direction[i]  = -1
            else:
                if hi[i] > ep:
                    ep = hi[i]
                    af = min(af + af_step, max_af)
                psar_long[i] = sar_new
                direction[i] = 1
        else:
            sar_new = sar + af * (ep - sar)
            # PSAR не может быть ниже двух предыдущих максимумов
            sar_new = max(sar_new, hi[i - 1], hi[max(0, i - 2)])
            if hi[i] > sar_new:           # разворот → лонг
                bull    = True
                sar_new = ep
                ep      = hi[i]
                af      = af0
                psar_long[i] = sar_new
                direction[i] = 1
            else:
                if lo[i] < ep:
                    ep = lo[i]
                    af = min(af + af_step, max_af)
                psar_short[i] = sar_new
                direction[i]  = -1

        sar = sar_new

    idx = close.index
    return pd.DataFrame({
        f"PSARl_{af0}_{max_af}":  pd.Series(psar_long,  index=idx),
        f"PSARs_{af0}_{max_af}":  pd.Series(psar_short, index=idx),
        f"PSARd_{af0}_{max_af}":  pd.Series(direction,  index=idx),
    })


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
    **kwargs,
) -> pd.DataFrame:
    atr_s = atr(high, low, close, length=length).values
    hl2   = ((high + low) / 2).values
    cl    = close.values
    n     = len(cl)

    upper_basic = hl2 + multiplier * atr_s
    lower_basic = hl2 - multiplier * atr_s

    upper     = upper_basic.copy()
    lower     = lower_basic.copy()
    direction = np.ones(n, dtype=np.int8)
    trend     = cl.copy()

    for i in range(1, n):
        upper[i] = upper_basic[i] if (upper_basic[i] < upper[i-1] or cl[i-1] > upper[i-1]) else upper[i-1]
        lower[i] = lower_basic[i] if (lower_basic[i] > lower[i-1] or cl[i-1] < lower[i-1]) else lower[i-1]

        if direction[i-1] == -1 and cl[i] > upper[i]:
            direction[i] = 1
        elif direction[i-1] == 1 and cl[i] < lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        trend[i] = lower[i] if direction[i] == 1 else upper[i]

    idx = close.index
    col = f"SUPERT_{length}_{float(multiplier)}"
    return pd.DataFrame({
        col:                              pd.Series(trend,     index=idx),
        f"SUPERTd_{length}_{float(multiplier)}": pd.Series(direction.astype(float), index=idx),
        f"SUPERTs_{length}_{float(multiplier)}": pd.Series(trend,     index=idx),
        f"SUPERTl_{length}_{float(multiplier)}": pd.Series(lower,     index=idx),
        f"SUPERTu_{length}_{float(multiplier)}": pd.Series(upper,     index=idx),
    })
