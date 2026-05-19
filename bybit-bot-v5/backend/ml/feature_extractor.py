"""
Feature Extractor — извлечение признаков для ML.
Каждый раз когда стратегия даёт сигнал — мы делаем СНИМОК рыночного состояния:
50+ признаков, которые потом помогут модели предсказать успех сделки.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Извлекает 50+ фич из DataFrame со свечами."""

    FEATURE_NAMES = [
        # --- Цена и тренд (8) ---
        "price", "price_change_1c", "price_change_5c", "price_change_20c",
        "ema_9_slope", "ema_21_slope", "ema_50_slope", "ema_200_slope",

        # --- Положение относительно EMA (4) ---
        "dist_to_ema9_pct", "dist_to_ema21_pct", "dist_to_ema50_pct", "dist_to_ema200_pct",

        # --- Осцилляторы (6) ---
        "rsi_14", "rsi_change", "stoch_k", "stoch_d", "cci_20", "williams_r",

        # --- MACD (3) ---
        "macd_value", "macd_signal", "macd_histogram",

        # --- Волатильность (5) ---
        "atr_pct", "atr_change", "bb_width", "bb_position", "volatility_20",

        # --- Объём (4) ---
        "volume_ratio_to_ma", "volume_change", "volume_trend", "obv_slope",

        # --- Свечной паттерн (4) ---
        "body_size_pct", "upper_wick_pct", "lower_wick_pct", "candle_direction",

        # --- Структура рынка (4) ---
        "higher_highs_count", "lower_lows_count", "range_position", "trend_strength_adx",

        # --- Время (5) ---
        "hour_of_day", "day_of_week", "is_weekend", "is_us_session", "is_asia_session",

        # --- Контекст сигнала (5) ---
        "signal_action_buy", "signal_confidence", "signal_rr_ratio", "signal_sl_pct", "signal_tp_pct",

        # --- Старшие таймфреймы (3) ---
        "htf_trend_direction", "htf_distance_pct", "htf_volatility",

        # --- Производительность стратегии (4) ---
        "strategy_recent_wr", "strategy_recent_pnl", "strategy_consecutive_losses", "strategy_total_trades",

        # --- НОВОЕ: Order Book (5) ---
        "ob_imbalance",          # disbalance bids vs asks
        "ob_spread_pct",         # bid-ask спред
        "ob_depth_ratio",        # глубина 10 уровней bids/asks
        "ob_large_orders_bid",   # сколько крупных ордеров на покупку
        "ob_large_orders_ask",

        # --- НОВОЕ: Funding & OI (3) ---
        "funding_rate",          # текущая ставка финансирования
        "open_interest_change",  # изменение OI
        "long_short_ratio",      # соотношение лонгов/шортов

        # --- НОВОЕ: Sentiment (6) ---
        "sentiment_score",            # общий sentiment [-1, 1]
        "sentiment_magnitude",         # уверенность
        "news_sentiment",
        "twitter_sentiment",
        "news_count_24h",
        "sentiment_freshness_min",

        # --- НОВОЕ: Аномалии (2) ---
        "anomaly_score",         # Isolation Forest score
        "regime_id",             # текущий рыночный режим (KMeans)
    ]

    def __init__(self):
        self.feature_count = len(self.FEATURE_NAMES)

    def extract(
        self,
        df: pd.DataFrame,
        signal_data: Dict,
        strategy_stats: Dict,
        htf_df: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict] = None,
        orderbook_data: Optional[Dict] = None,
        market_meta: Optional[Dict] = None,
        regime_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Извлекает все фичи.

        df: свечи стратегии (последние ~200)
        signal_data: {action, confidence, entry_price, stop_loss, take_profit}
        strategy_stats: {win_rate, pnl, consecutive_losses, trades}
        htf_df: свечи старшего таймфрейма (опционально)
        """
        if len(df) < 50:
            return {}

        features = {}
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last
        price = float(last["close"])

        # --- Считаем все индикаторы один раз ---
        ema_9 = ta.ema(df["close"], length=9)
        ema_21 = ta.ema(df["close"], length=21)
        ema_50 = ta.ema(df["close"], length=50)
        ema_200 = ta.ema(df["close"], length=200) if len(df) >= 200 else None
        rsi = ta.rsi(df["close"], length=14)
        atr = ta.atr(df["high"], df["low"], df["close"], length=14)
        bb = ta.bbands(df["close"], length=20, std=2)
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        cci = ta.cci(df["high"], df["low"], df["close"], length=20)
        wr = ta.willr(df["high"], df["low"], df["close"], length=14)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        obv = ta.obv(df["close"], df["volume"])

        vol_ma = df["volume"].rolling(20).mean()

        # --- 1. Цена и тренд (8) ---
        features["price"] = price
        features["price_change_1c"] = self._pct_change(df["close"].iloc[-1], df["close"].iloc[-2])
        features["price_change_5c"] = self._pct_change(df["close"].iloc[-1], df["close"].iloc[-6]) if len(df) >= 6 else 0
        features["price_change_20c"] = self._pct_change(df["close"].iloc[-1], df["close"].iloc[-21]) if len(df) >= 21 else 0
        features["ema_9_slope"] = self._slope(ema_9, 5)
        features["ema_21_slope"] = self._slope(ema_21, 5)
        features["ema_50_slope"] = self._slope(ema_50, 5)
        features["ema_200_slope"] = self._slope(ema_200, 5) if ema_200 is not None else 0

        # --- 2. Положение относительно EMA (4) ---
        features["dist_to_ema9_pct"] = self._pct_change(price, ema_9.iloc[-1])
        features["dist_to_ema21_pct"] = self._pct_change(price, ema_21.iloc[-1])
        features["dist_to_ema50_pct"] = self._pct_change(price, ema_50.iloc[-1])
        features["dist_to_ema200_pct"] = self._pct_change(price, ema_200.iloc[-1]) if ema_200 is not None else 0

        # --- 3. Осцилляторы (6) ---
        features["rsi_14"] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        features["rsi_change"] = float(rsi.iloc[-1] - rsi.iloc[-5]) if len(rsi) >= 5 else 0
        features["stoch_k"] = float(stoch["STOCHk_14_3_3"].iloc[-1]) if stoch is not None and not pd.isna(stoch["STOCHk_14_3_3"].iloc[-1]) else 50
        features["stoch_d"] = float(stoch["STOCHd_14_3_3"].iloc[-1]) if stoch is not None and not pd.isna(stoch["STOCHd_14_3_3"].iloc[-1]) else 50
        features["cci_20"] = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0
        features["williams_r"] = float(wr.iloc[-1]) if not pd.isna(wr.iloc[-1]) else -50

        # --- 4. MACD (3) ---
        if macd_df is not None and "MACD_12_26_9" in macd_df.columns:
            features["macd_value"] = float(macd_df["MACD_12_26_9"].iloc[-1]) if not pd.isna(macd_df["MACD_12_26_9"].iloc[-1]) else 0
            features["macd_signal"] = float(macd_df["MACDs_12_26_9"].iloc[-1]) if not pd.isna(macd_df["MACDs_12_26_9"].iloc[-1]) else 0
            features["macd_histogram"] = float(macd_df["MACDh_12_26_9"].iloc[-1]) if not pd.isna(macd_df["MACDh_12_26_9"].iloc[-1]) else 0
        else:
            features["macd_value"] = features["macd_signal"] = features["macd_histogram"] = 0

        # --- 5. Волатильность (5) ---
        features["atr_pct"] = float(atr.iloc[-1] / price * 100) if not pd.isna(atr.iloc[-1]) else 1.0
        features["atr_change"] = self._pct_change(atr.iloc[-1], atr.iloc[-10]) if len(atr) >= 10 else 0
        if bb is not None and "BBU_20_2.0" in bb.columns:
            bb_upper = bb["BBU_20_2.0"].iloc[-1]
            bb_lower = bb["BBL_20_2.0"].iloc[-1]
            features["bb_width"] = float((bb_upper - bb_lower) / price * 100) if not pd.isna(bb_upper) else 0
            features["bb_position"] = float((price - bb_lower) / (bb_upper - bb_lower)) if bb_upper != bb_lower else 0.5
        else:
            features["bb_width"] = features["bb_position"] = 0
        features["volatility_20"] = float(df["close"].iloc[-20:].pct_change().std() * 100) if len(df) >= 20 else 0

        # --- 6. Объём (4) ---
        features["volume_ratio_to_ma"] = float(last["volume"] / vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) and vol_ma.iloc[-1] > 0 else 1.0
        features["volume_change"] = self._pct_change(last["volume"], prev["volume"])
        features["volume_trend"] = self._slope(df["volume"], 10)
        features["obv_slope"] = self._slope(obv, 10) if not obv.empty else 0

        # --- 7. Свечной паттерн (4) ---
        body = abs(last["close"] - last["open"])
        candle_range = last["high"] - last["low"]
        features["body_size_pct"] = float(body / candle_range * 100) if candle_range > 0 else 0
        features["upper_wick_pct"] = float((last["high"] - max(last["open"], last["close"])) / candle_range * 100) if candle_range > 0 else 0
        features["lower_wick_pct"] = float((min(last["open"], last["close"]) - last["low"]) / candle_range * 100) if candle_range > 0 else 0
        features["candle_direction"] = 1.0 if last["close"] > last["open"] else -1.0

        # --- 8. Структура рынка (4) ---
        recent = df.iloc[-20:]
        features["higher_highs_count"] = int(sum(1 for i in range(1, len(recent)) if recent["high"].iloc[i] > recent["high"].iloc[i-1]))
        features["lower_lows_count"] = int(sum(1 for i in range(1, len(recent)) if recent["low"].iloc[i] < recent["low"].iloc[i-1]))
        range_high = recent["high"].max()
        range_low = recent["low"].min()
        features["range_position"] = float((price - range_low) / (range_high - range_low)) if range_high != range_low else 0.5
        features["trend_strength_adx"] = float(adx_df["ADX_14"].iloc[-1]) if adx_df is not None and not pd.isna(adx_df["ADX_14"].iloc[-1]) else 20

        # --- 9. Время (5) ---
        now = datetime.utcnow()
        if "timestamp" in df.columns and not df.empty:
            try:
                ts = pd.to_datetime(df["timestamp"].iloc[-1])
                now = ts.to_pydatetime()
            except Exception:
                pass
        features["hour_of_day"] = float(now.hour)
        features["day_of_week"] = float(now.weekday())
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
        features["is_us_session"] = 1.0 if 13 <= now.hour <= 21 else 0.0  # 13:00-21:00 UTC
        features["is_asia_session"] = 1.0 if 0 <= now.hour <= 8 else 0.0

        # --- 10. Контекст сигнала (5) ---
        features["signal_action_buy"] = 1.0 if signal_data.get("action") == "BUY" else 0.0
        features["signal_confidence"] = float(signal_data.get("confidence", 0.5))
        sl = float(signal_data.get("stop_loss", price))
        tp = float(signal_data.get("take_profit", price))
        entry = float(signal_data.get("entry_price", price))
        sl_dist = abs(entry - sl) / entry if entry > 0 else 0
        tp_dist = abs(tp - entry) / entry if entry > 0 else 0
        features["signal_rr_ratio"] = float(tp_dist / sl_dist) if sl_dist > 0 else 1.0
        features["signal_sl_pct"] = float(sl_dist * 100)
        features["signal_tp_pct"] = float(tp_dist * 100)

        # --- 11. HTF (старший таймфрейм) (3) ---
        if htf_df is not None and len(htf_df) >= 50:
            htf_ema = ta.ema(htf_df["close"], length=21)
            htf_last_price = float(htf_df["close"].iloc[-1])
            features["htf_trend_direction"] = 1.0 if htf_last_price > htf_ema.iloc[-1] else -1.0
            features["htf_distance_pct"] = self._pct_change(htf_last_price, htf_ema.iloc[-1])
            features["htf_volatility"] = float(htf_df["close"].iloc[-20:].pct_change().std() * 100) if len(htf_df) >= 20 else 0
        else:
            features["htf_trend_direction"] = 0
            features["htf_distance_pct"] = 0
            features["htf_volatility"] = 0

        # --- 12. Производительность стратегии (rolling, snapshot НА МОМЕНТ сигнала) ---
        # ВАЖНО: эти значения должны быть СНЯТЫ до того как текущая сделка состоится.
        # strategy_stats передаются caller'ом и должны быть БЕЗ учёта текущего сигнала.
        # Используем rolling-окно последних 20 сделок, а не глобальные счётчики.
        features["strategy_recent_wr"] = float(strategy_stats.get("rolling_wr_20", 0.5))
        features["strategy_recent_pnl"] = float(strategy_stats.get("rolling_pnl_20", 0))
        features["strategy_consecutive_losses"] = float(strategy_stats.get("consecutive_losses", 0))
        features["strategy_total_trades"] = float(strategy_stats.get("trades", 0))

        # --- НОВОЕ: 13. Order Book (5) ---
        ob = orderbook_data or {}
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if bids and asks:
            try:
                bids_sum = sum(float(b[1]) for b in bids[:10])
                asks_sum = sum(float(a[1]) for a in asks[:10])
                total_depth = bids_sum + asks_sum
                features["ob_imbalance"] = (bids_sum - asks_sum) / total_depth if total_depth > 0 else 0
                features["ob_depth_ratio"] = bids_sum / asks_sum if asks_sum > 0 else 1
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                features["ob_spread_pct"] = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0
                # large orders = ордера в 3x больше медианы
                median_bid = np.median([float(b[1]) for b in bids[:20]]) if len(bids) >= 20 else 0
                median_ask = np.median([float(a[1]) for a in asks[:20]]) if len(asks) >= 20 else 0
                features["ob_large_orders_bid"] = sum(1 for b in bids[:20] if float(b[1]) > median_bid * 3)
                features["ob_large_orders_ask"] = sum(1 for a in asks[:20] if float(a[1]) > median_ask * 3)
            except Exception:
                features["ob_imbalance"] = features["ob_depth_ratio"] = features["ob_spread_pct"] = 0
                features["ob_large_orders_bid"] = features["ob_large_orders_ask"] = 0
        else:
            features["ob_imbalance"] = features["ob_depth_ratio"] = features["ob_spread_pct"] = 0
            features["ob_large_orders_bid"] = features["ob_large_orders_ask"] = 0

        # --- НОВОЕ: 14. Funding & OI (3) ---
        meta = market_meta or {}
        features["funding_rate"] = float(meta.get("funding_rate", 0)) * 100  # в %
        features["open_interest_change"] = float(meta.get("oi_change_pct", 0))
        features["long_short_ratio"] = float(meta.get("long_short_ratio", 1.0))

        # --- НОВОЕ: 15. Sentiment (6) ---
        sent = sentiment_data or {}
        features["sentiment_score"] = float(sent.get("sentiment_score", 0))
        features["sentiment_magnitude"] = float(sent.get("sentiment_magnitude", 0))
        features["news_sentiment"] = float(sent.get("news_sentiment", 0))
        features["twitter_sentiment"] = float(sent.get("twitter_sentiment", 0))
        features["news_count_24h"] = float(sent.get("news_count_24h", 0))
        features["sentiment_freshness_min"] = float(sent.get("sentiment_data_freshness_min", 999))

        # --- НОВОЕ: 16. Аномалии и режим (2) ---
        features["anomaly_score"] = float(market_meta.get("anomaly_score", 0)) if market_meta else 0
        features["regime_id"] = float(regime_id if regime_id is not None else -1)

        # Гарантируем что все фичи есть
        for name in self.FEATURE_NAMES:
            if name not in features:
                features[name] = 0.0
            elif pd.isna(features[name]) or features[name] in (float("inf"), float("-inf")):
                features[name] = 0.0

        return features

    @staticmethod
    def _pct_change(current, previous) -> float:
        try:
            current = float(current)
            previous = float(previous)
            if pd.isna(current) or pd.isna(previous) or previous == 0:
                return 0.0
            return (current - previous) / previous * 100
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _slope(series, periods: int = 5) -> float:
        try:
            if series is None or len(series) < periods + 1:
                return 0.0
            vals = series.iloc[-periods - 1:].dropna()
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals))
            slope, _ = np.polyfit(x, vals.values, 1)
            return float(slope / vals.mean() * 100) if vals.mean() != 0 else 0.0
        except Exception:
            return 0.0

    def to_dataframe_row(self, features: Dict) -> pd.DataFrame:
        """Конвертирует фичи в pandas строку для ML."""
        return pd.DataFrame([{name: features.get(name, 0.0) for name in self.FEATURE_NAMES}])
