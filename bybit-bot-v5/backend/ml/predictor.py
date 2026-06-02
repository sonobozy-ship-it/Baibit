"""
ML Predictor — realtime инференс перед открытием сделки.
Загружает обученную модель и говорит брать сигнал или нет.
"""
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MLPredictor:
    """Realtime фильтр сигналов через ML."""

    def __init__(self, default_threshold: float = 0.55):
        """
        default_threshold: минимум P(win) для прохождения фильтра
        """
        self.models: Dict[str, Dict] = {}      # strategy_id -> {model, features, ...}
        self.thresholds: Dict[str, float] = {} # strategy_id -> threshold
        self.default_threshold = default_threshold
        self.enabled = True

    def load_model(self, strategy_id: str, file_path: str) -> bool:
        """Загрузка модели из pickle."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Модель не найдена: {file_path}")
                return False
            with open(path, "rb") as f:
                self.models[strategy_id] = pickle.load(f)
            logger.info(f"✅ Модель для {strategy_id} загружена ({len(self.models[strategy_id]['feature_columns'])} фич)")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {strategy_id}: {e}")
            return False

    def load_all_active_models(self, data_store):
        """Загрузка всех активных моделей из data store."""
        from strategies import ALL_STRATEGIES
        for sid in ALL_STRATEGIES.keys():
            active = data_store.get_active_model(sid)
            if active and active.get("file_path"):
                self.load_model(sid, active["file_path"])

    def predict(self, strategy_id: str, features: Dict) -> Dict:
        """
        Предсказание для одного сигнала.

        Возвращает:
        {
            "available": bool,         # есть ли модель
            "probability": float,       # P(win) 0..1
            "should_take": bool,        # выше threshold?
            "threshold": float,         # текущий threshold
            "model_version": str,
            "top_features": [{name, value, importance}, ...]
        }
        """
        if not self.enabled or strategy_id not in self.models:
            return {
                "available": False,
                "probability": None,
                "should_take": True,    # если нет модели — пропускаем сигнал как обычно
                "threshold": self.default_threshold,
                "model_version": None,
            }

        try:
            model_data = self.models[strategy_id]
            model = model_data["model"]
            feature_cols = model_data["feature_columns"]

            # Подготовка фич
            X = pd.DataFrame([{col: features.get(col, 0.0) for col in feature_cols}])
            X = X.fillna(0)

            proba = float(model.predict_proba(X)[0, 1])
            threshold = self.thresholds.get(strategy_id, self.default_threshold)
            should_take = proba >= threshold

            # Топ-5 фич которые повлияли (на основе общей важности)
            importance = model_data.get("feature_importance", {})
            top_features = [
                {
                    "name": name,
                    "value": round(features.get(name, 0), 3),
                    "importance": round(imp, 3),
                }
                for name, imp in list(importance.items())[:5]
            ]

            return {
                "available": True,
                "probability": round(proba, 4),
                "should_take": should_take,
                "threshold": threshold,
                "model_version": model_data.get("strategy_id", "") + "_" + model_data.get("trained_at", "")[:10],
                "top_features": top_features,
            }
        except Exception as e:
            logger.exception(f"Ошибка предсказания для {strategy_id}: {e}")
            return {
                "available": False,
                "probability": None,
                "should_take": True,
                "threshold": self.default_threshold,
                "error": str(e),
            }

    def set_threshold(self, strategy_id: str, threshold: float):
        """Изменить threshold для конкретной стратегии."""
        self.thresholds[strategy_id] = max(0.0, min(1.0, threshold))
        logger.info(f"Threshold {strategy_id} = {self.thresholds[strategy_id]}")

    def get_status(self) -> Dict:
        """Статус всех загруженных моделей."""
        return {
            "enabled": self.enabled,
            "default_threshold": self.default_threshold,
            "models_loaded": list(self.models.keys()),
            "thresholds": dict(self.thresholds),
            "model_details": {
                sid: {
                    "trained_at": data.get("trained_at"),
                    "samples": data.get("samples"),
                    "metrics": data.get("metrics", {}),
                    "feature_count": len(data.get("feature_columns", [])),
                }
                for sid, data in self.models.items()
            },
        }
