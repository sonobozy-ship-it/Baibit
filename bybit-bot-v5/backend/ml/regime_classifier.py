"""
Regime Classifier — кластеризация рыночных режимов.
Использует KMeans на ATR, ADX, BB Width → определяет тип рынка:
0: Низкая волатильность, флэт
1: Средняя волатильность, тренд
2: Высокая волатильность, разворот
и т.д.
"""
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
import pandas_ta as ta

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class RegimeClassifier:
    """Определяет текущий рыночный режим (флэт/тренд/волатильность)."""

    REGIME_NAMES = {
        0: "Тихий флэт",
        1: "Восходящий тренд",
        2: "Нисходящий тренд",
        3: "Высокая волатильность",
    }

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = None
        self.feature_columns = ["atr_pct", "adx", "bb_width", "trend_direction"]

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлекает фичи для кластеризации."""
        result = pd.DataFrame(index=df.index)

        atr = ta.atr(df["high"], df["low"], df["close"], length=14)
        result["atr_pct"] = atr / df["close"] * 100

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        result["adx"] = adx_df["ADX_14"] if (adx_df is not None and "ADX_14" in adx_df.columns) else 20

        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None and "BBU_20_2.0" in bb.columns:
            result["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / df["close"] * 100
        else:
            result["bb_width"] = 0

        ema_21 = ta.ema(df["close"], length=21)
        result["trend_direction"] = (df["close"] - ema_21) / df["close"].replace(0, np.nan) * 100

        return result.dropna()

    def fit(self, df: pd.DataFrame) -> Dict:
        """Обучение KMeans на исторических данных."""
        if not SKLEARN_AVAILABLE:
            return {"success": False, "error": "scikit-learn не установлен"}
        if len(df) < 200:
            return {"success": False, "error": "Мало данных"}

        features = self._extract_features(df)
        if features.empty:
            return {"success": False, "error": "Не удалось извлечь фичи"}

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features[self.feature_columns])

        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.model.fit_predict(X)

        # Анализ кластеров — присваиваем имена по характеристикам
        cluster_stats = {}
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_stats[cluster_id] = {
                "count": int(mask.sum()),
                "avg_atr": float(features[mask]["atr_pct"].mean()),
                "avg_adx": float(features[mask]["adx"].mean()),
                "avg_bb_width": float(features[mask]["bb_width"].mean()),
                "avg_trend": float(features[mask]["trend_direction"].mean()),
            }

        logger.info(f"✅ Regime classifier обучен на {len(features)} свечах, {self.n_clusters} кластеров")
        return {
            "success": True,
            "samples": len(features),
            "n_clusters": self.n_clusters,
            "cluster_stats": cluster_stats,
        }

    def predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """Определить режим для последней свечи."""
        if not self.model or not self.scaler:
            return None
        try:
            features = self._extract_features(df)
            if features.empty:
                return None
            X = self.scaler.transform(features[self.feature_columns].iloc[-1:])
            regime_id = int(self.model.predict(X)[0])
            return {
                "regime_id": regime_id,
                "regime_name": self._auto_name(features.iloc[-1]),
                "features": features.iloc[-1].to_dict(),
            }
        except Exception as e:
            logger.error(f"Regime predict ошибка: {e}")
            return None

    @staticmethod
    def _auto_name(features: pd.Series) -> str:
        """Автоматическое именование режима по фичам."""
        adx = features["adx"]
        atr = features["atr_pct"]
        trend = features["trend_direction"]

        if adx < 20 and atr < 1.5:
            return "🟦 Тихий флэт"
        elif adx > 25 and trend > 0.5:
            return "🟢 Восходящий тренд"
        elif adx > 25 and trend < -0.5:
            return "🔴 Нисходящий тренд"
        elif atr > 3:
            return "🟡 Высокая волатильность"
        else:
            return "⚪️ Переходный режим"
