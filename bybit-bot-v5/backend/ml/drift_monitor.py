"""
Concept Drift Detector.
Следит за качеством ML-предсказаний в реальном времени.
Если модель начала ошибаться чаще обычного — деактивирует её.
"""
import logging
from collections import deque
from typing import Dict, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Rolling-метрики качества предсказаний.

    Идея: если модель обещала P(win)=0.7, а WR факт = 0.4 — она устарела.
    """

    def __init__(
        self,
        window_size: int = 30,           # последние N сделок для расчёта
        min_samples: int = 15,           # минимум для срабатывания
        accuracy_threshold: float = 0.45, # ниже этого = drift
        brier_threshold: float = 0.30,    # Brier score (калибровка)
    ):
        self.window_size = window_size
        self.min_samples = min_samples
        self.accuracy_threshold = accuracy_threshold
        self.brier_threshold = brier_threshold

        # Rolling истории per-strategy
        self.predictions: Dict[str, deque] = {}
        self.outcomes: Dict[str, deque] = {}
        self.alert_history: Dict[str, datetime] = {}

    def record(self, strategy_id: str, predicted_proba: float, actual_outcome: int):
        """
        Записать предсказание и факт.
        actual_outcome: 1 (win) или 0 (loss).
        """
        if strategy_id not in self.predictions:
            self.predictions[strategy_id] = deque(maxlen=self.window_size)
            self.outcomes[strategy_id] = deque(maxlen=self.window_size)

        self.predictions[strategy_id].append(float(predicted_proba))
        self.outcomes[strategy_id].append(int(actual_outcome))

    def calculate_metrics(self, strategy_id: str) -> Optional[Dict]:
        """Вычислить текущие метрики качества."""
        if strategy_id not in self.predictions:
            return None
        preds = list(self.predictions[strategy_id])
        outs = list(self.outcomes[strategy_id])

        if len(preds) < self.min_samples:
            return {
                "ready": False,
                "samples": len(preds),
                "needed": self.min_samples,
            }

        preds_arr = np.array(preds)
        outs_arr = np.array(outs)

        # Accuracy при threshold=0.5
        binary_preds = (preds_arr >= 0.5).astype(int)
        accuracy = (binary_preds == outs_arr).mean()

        # Brier score (квадратичная ошибка вероятности)
        brier = ((preds_arr - outs_arr) ** 2).mean()

        # Calibration: средняя предсказанная вероятность vs реальная WR
        mean_pred = preds_arr.mean()
        actual_wr = outs_arr.mean()
        calibration_gap = abs(mean_pred - actual_wr)

        # Drift detected?
        drift = accuracy < self.accuracy_threshold or brier > self.brier_threshold

        return {
            "ready": True,
            "samples": len(preds),
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 4),
            "mean_predicted_proba": round(mean_pred, 4),
            "actual_win_rate": round(actual_wr, 4),
            "calibration_gap": round(calibration_gap, 4),
            "drift_detected": drift,
            "drift_reason": (
                f"accuracy {accuracy:.2f} < {self.accuracy_threshold}"
                if accuracy < self.accuracy_threshold
                else f"brier {brier:.2f} > {self.brier_threshold}"
                if brier > self.brier_threshold else None
            ),
        }

    def should_disable_model(self, strategy_id: str) -> Dict:
        """Решение: отключать ли модель."""
        metrics = self.calculate_metrics(strategy_id)
        if not metrics or not metrics.get("ready"):
            return {"disable": False, "reason": "Недостаточно данных"}

        if metrics["drift_detected"]:
            # Cooldown: один alert в час
            last_alert = self.alert_history.get(strategy_id)
            if last_alert and (datetime.utcnow() - last_alert) < timedelta(hours=1):
                return {"disable": False, "reason": "alert cooldown"}
            self.alert_history[strategy_id] = datetime.utcnow()
            return {
                "disable": True,
                "reason": f"Concept drift: {metrics['drift_reason']}",
                "metrics": metrics,
            }
        return {"disable": False, "reason": "OK", "metrics": metrics}

    def get_all_metrics(self) -> Dict[str, Dict]:
        return {sid: self.calculate_metrics(sid) for sid in self.predictions.keys()}


class ThresholdOptimizer:
    """
    Auto-tuning ML threshold per стратегия.
    Ищет оптимум на out-of-sample данных.
    """

    @staticmethod
    def find_optimal_threshold(
        predictions: list,   # [proba, ...]
        outcomes: list,      # [0/1, ...]
        objective: str = "f1_weighted_by_count",
    ) -> Dict:
        """
        Перебирает threshold от 0.3 до 0.8 с шагом 0.025 и находит оптимум.

        objective:
        - f1_weighted_by_count: F1 * количество сигналов
        - precision_only: чистая precision (мало сделок, но точные)
        - sharpe_proxy: WR * sqrt(N)
        """
        if len(predictions) < 30:
            return {"threshold": 0.55, "reason": "Мало данных"}

        preds = np.array(predictions)
        outs = np.array(outcomes)

        best_score = -float("inf")
        best_threshold = 0.55
        all_results = []

        for thr in np.arange(0.30, 0.85, 0.025):
            mask = preds >= thr
            n_taken = mask.sum()
            if n_taken < 5:
                continue
            wins = outs[mask].sum()
            precision = wins / n_taken
            recall_total = wins / outs.sum() if outs.sum() > 0 else 0
            f1 = 2 * (precision * recall_total) / (precision + recall_total) if (precision + recall_total) > 0 else 0

            if objective == "f1_weighted_by_count":
                score = f1 * np.log1p(n_taken)
            elif objective == "precision_only":
                score = precision if n_taken >= 10 else 0
            elif objective == "sharpe_proxy":
                score = precision * np.sqrt(n_taken)
            else:
                score = f1

            all_results.append({
                "threshold": round(float(thr), 3),
                "n_signals": int(n_taken),
                "wins": int(wins),
                "wr": round(float(precision), 3),
                "f1": round(float(f1), 3),
                "score": round(float(score), 4),
            })

            if score > best_score:
                best_score = score
                best_threshold = float(thr)

        return {
            "threshold": round(best_threshold, 3),
            "best_score": round(best_score, 4),
            "objective": objective,
            "samples": len(predictions),
            "all_results": all_results,
        }
