"""
ML Trainer — обучение модели для фильтра сигналов.
XGBoost + калибровка вероятностей + walk-forward валидация.
"""
import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Опциональные импорты — XGBoost
try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    from sklearn.calibration import CalibratedClassifierCV
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn / xgboost не установлены. ML отключён.")


class MLTrainer:
    """Тренировка XGBoost для предсказания P(win) каждой сделки."""

    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples_for_training = 300  # минимум для обучения

    def train(
        self,
        training_data: pd.DataFrame,
        strategy_id: str,
        feature_columns: List[str],
        target_column: str = "label",
        n_splits: int = 5,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Обучение модели с walk-forward валидацией.

        training_data: DataFrame с фичами + label
        feature_columns: список фич для обучения
        """
        if not ML_AVAILABLE:
            return {"success": False, "error": "scikit-learn/xgboost не установлены"}

        if len(training_data) < self.min_samples_for_training:
            return {
                "success": False,
                "error": f"Недостаточно данных: {len(training_data)} < {self.min_samples_for_training}"
            }

        # Сортируем по timestamp для time-series split
        if "timestamp" in training_data.columns:
            training_data = training_data.sort_values("timestamp").reset_index(drop=True)

        # Только нужные фичи
        available_features = [f for f in feature_columns if f in training_data.columns]
        if len(available_features) < 10:
            return {"success": False, "error": f"Слишком мало фич: {len(available_features)}"}

        X = training_data[available_features].fillna(0)
        y = training_data[target_column].astype(int)

        # Проверка баланса классов
        wr = y.mean()
        if wr < 0.1 or wr > 0.9:
            logger.warning(f"⚠️ Сильный дисбаланс классов (WR={wr:.2f})")

        # Walk-forward CV — минимум 3 фолда, 1 фолд на каждые 100 примеров
        tscv = TimeSeriesSplit(n_splits=min(n_splits, max(3, len(X) // 100)))

        cv_scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

        hyperparams = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": (1 - wr) / wr if wr > 0 else 1,
            "eval_metric": "logloss",
            "random_state": 42,
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            sw_fold = sample_weight[train_idx] if sample_weight is not None else None
            model = xgb.XGBClassifier(**hyperparams)
            model.fit(X_train, y_train, sample_weight=sw_fold, verbose=False)

            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            cv_scores["accuracy"].append(accuracy_score(y_test, preds))
            cv_scores["precision"].append(precision_score(y_test, preds, zero_division=0))
            cv_scores["recall"].append(recall_score(y_test, preds, zero_division=0))
            cv_scores["f1"].append(f1_score(y_test, preds, zero_division=0))
            try:
                cv_scores["roc_auc"].append(roc_auc_score(y_test, proba))
            except ValueError:
                pass

        # Финальная модель на всех данных + калибровка
        final_model = xgb.XGBClassifier(**hyperparams)
        final_model.fit(X, y, sample_weight=sample_weight, verbose=False)

        # Калибровка вероятностей (важно для разумных threshold'ов)
        try:
            calibrated = CalibratedClassifierCV(final_model, cv=3, method="isotonic")
            calibrated.fit(X, y)
            model_to_save = calibrated
        except Exception as e:
            logger.warning(f"Калибровка не удалась: {e}, использую raw модель")
            model_to_save = final_model

        # Feature importance
        importance = dict(zip(available_features, final_model.feature_importances_.tolist()))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # Метрики
        metrics = {k: round(float(np.mean(v)), 4) if v else 0 for k, v in cv_scores.items()}

        # Сохранение
        version = f"{strategy_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        file_path = self.models_dir / f"{version}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump({
                "model": model_to_save,
                "feature_columns": available_features,
                "trained_at": datetime.utcnow().isoformat(),
                "strategy_id": strategy_id,
                "samples": len(X),
                "metrics": metrics,
                "hyperparams": hyperparams,
                "feature_importance": importance,
            }, f)

        logger.info(f"✅ Модель {version} обучена. F1={metrics['f1']:.3f}, ROC_AUC={metrics['roc_auc']:.3f}")

        return {
            "success": True,
            "version": version,
            "file_path": str(file_path),
            "metrics": metrics,
            "feature_importance": importance,
            "hyperparams": hyperparams,
            "samples": len(X),
        }

    def train_all_strategies(
        self,
        get_training_data_fn,
        strategy_ids: List[str],
        feature_columns: List[str],
    ) -> Dict[str, Dict]:
        """
        Обучить модели для всех стратегий.
        get_training_data_fn(strategy_id) -> DataFrame
        """
        results = {}
        for sid in strategy_ids:
            data = get_training_data_fn(sid)
            if data.empty or len(data) < self.min_samples_for_training:
                results[sid] = {"success": False, "error": "недостаточно данных"}
                continue
            results[sid] = self.train(data, sid, feature_columns)
        return results
