"""
Anomaly Detection (Isolation Forest) + Ensemble Models.
Защита от аномальных рыночных условий и улучшение качества предсказаний.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False
    logger.warning("XGBoost недоступен. Модель отключена.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False
    logger.warning("LightGBM недоступен (возможно, нет libomp). Модель отключена.")


class AnomalyDetector:
    """
    Isolation Forest для детекции аномальных рыночных условий.
    Если score < threshold — это outlier, бот ставит торговлю на паузу.
    """

    def __init__(self, contamination: float = 0.05):
        """
        contamination: ожидаемая доля аномалий (5% = consrvative).
        """
        self.contamination = contamination
        self.model = None
        self.scaler = None

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Dict:
        if not SKLEARN_AVAILABLE:
            return {"success": False, "error": "sklearn недоступен"}
        if len(df) < 100:
            return {"success": False, "error": "Мало данных"}

        cols = feature_columns or [c for c in df.columns if df[c].dtype in (np.float64, np.int64)]
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 3:
            return {"success": False, "error": "Мало числовых фич"}

        X = df[cols].fillna(0).values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples="auto",
        )
        self.model.fit(X_scaled)
        self.feature_columns = cols

        scores = self.model.score_samples(X_scaled)
        return {
            "success": True,
            "samples": len(X),
            "features_used": len(cols),
            "score_distribution": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            },
        }

    def score(self, features: Dict) -> float:
        """
        Anomaly score: чем НИЖЕ — тем более аномально.
        Норма около -0.5, аномалии около -0.7 и ниже.
        """
        if not self.model:
            return 0.0
        try:
            X = pd.DataFrame([{c: features.get(c, 0) for c in self.feature_columns}]).fillna(0).values
            X_scaled = self.scaler.transform(X)
            return float(self.model.score_samples(X_scaled)[0])
        except Exception as e:
            logger.warning(f"Anomaly score: {e}")
            return 0.0

    def is_anomaly(self, features: Dict, threshold: float = -0.65) -> bool:
        return self.score(features) < threshold

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "features": self.feature_columns}, f)

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_columns = data["features"]
            return True
        except Exception as e:
            logger.error(f"Anomaly load: {e}")
            return False


class EnsembleTrainer:
    """
    Ансамбль моделей: XGBoost + LightGBM + LogisticRegression.
    Voting (soft) — усредняет вероятности, обычно даёт +3-5% к точности.
    """

    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_ensemble(
        self,
        training_data: pd.DataFrame,
        strategy_id: str,
        feature_columns: List[str],
        target_column: str = "label",
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict:
        if not SKLEARN_AVAILABLE:
            return {"success": False, "error": "sklearn недоступен"}

        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError:
            return {"success": False, "error": "sklearn metrics недоступны"}

        if len(training_data) < 300:
            return {"success": False, "error": f"Мало данных: {len(training_data)} (нужно ≥300)"}

        if "timestamp" in training_data.columns:
            training_data = training_data.sort_values("timestamp").reset_index(drop=True)
        avail = [f for f in feature_columns if f in training_data.columns]
        X = training_data[avail].fillna(0)
        y = training_data[target_column].astype(int)

        wr = y.mean()
        scale_pos = (1 - wr) / wr if 0 < wr < 1 else 1.0

        # Базовые модели
        estimators = []
        if XGB_AVAILABLE and xgb is not None:
            estimators.append(("xgb", xgb.XGBClassifier(
                max_depth=4, learning_rate=0.05, n_estimators=150,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos, eval_metric="logloss",
                random_state=42, verbosity=0,
            )))
        if LGB_AVAILABLE and lgb is not None:
            estimators.append(("lgb", lgb.LGBMClassifier(
                max_depth=4, learning_rate=0.05, n_estimators=150,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                is_unbalance=True, random_state=42, verbose=-1,
            )))
        estimators.append(("logreg", LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42,
        )))

        if len(estimators) < 2:
            return {"success": False, "error": "Недостаточно ML библиотек"}

        # Walk-forward
        n_splits = min(5, max(2, len(X) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv = {"accuracy": [], "f1": [], "roc_auc": []}
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            try:
                from sklearn.base import clone as _clone
                ensemble = VotingClassifier(
                    estimators=[(n, _clone(m)) for n, m in estimators],
                    voting="soft",
                )
                sw_fold = sample_weight[train_idx] if sample_weight is not None else None
                ensemble.fit(X_train, y_train, sample_weight=sw_fold)
                preds = ensemble.predict(X_test)
                proba = ensemble.predict_proba(X_test)[:, 1]
                cv["accuracy"].append(accuracy_score(y_test, preds))
                cv["f1"].append(f1_score(y_test, preds, zero_division=0))
                try:
                    cv["roc_auc"].append(roc_auc_score(y_test, proba))
                except ValueError:
                    pass
            except Exception as e:
                logger.warning(f"CV fold ошибка: {e}")

        # Финальная модель
        final = VotingClassifier(estimators=estimators, voting="soft")
        final.fit(X, y, sample_weight=sample_weight)
        # Калибровка
        try:
            calibrated = CalibratedClassifierCV(final, cv=3, method="isotonic")
            calibrated.fit(X, y)
            model_to_save = calibrated
        except Exception:
            model_to_save = final

        # Feature importance — берём из XGBoost
        importance = {}
        try:
            xgb_idx = next((i for i, (n, _) in enumerate(estimators) if n == "xgb"), None)
            if xgb_idx is not None:
                fitted_xgb = final.named_estimators_["xgb"]
                importance = dict(zip(avail, fitted_xgb.feature_importances_.tolist()))
                importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.warning(f"Importance: {e}")

        metrics = {k: round(float(np.mean(v)), 4) if v else 0 for k, v in cv.items()}
        version = f"ensemble_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        path = self.models_dir / f"{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": model_to_save,
                "feature_columns": avail,
                "model_type": "Ensemble",
                "models": [n for n, _ in estimators],
                "trained_at": datetime.utcnow().isoformat(),
                "strategy_id": strategy_id,
                "samples": len(X),
                "metrics": metrics,
                "feature_importance": importance,
            }, f)

        logger.info(
            f"✅ Ensemble {version}: F1={metrics['f1']:.3f}, AUC={metrics.get('roc_auc', 0):.3f}"
        )
        return {
            "success": True,
            "version": version,
            "model_type": "Ensemble",
            "models_used": [n for n, _ in estimators],
            "file_path": str(path),
            "metrics": metrics,
            "feature_importance": importance,
            "samples": len(X),
        }
