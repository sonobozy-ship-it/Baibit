"""
Continuous ML Trainer — автоматическое переобучение моделей по накоплению данных.
Запускается как фоновая задача каждые 30 минут.
Переобучает когда: >=300 размеченных сделок впервые, или +50 новых с последнего обучения.
После обучения: авто-тюн threshold + сброс drift-счётчика.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ContinuousTrainer:
    """Авто-переобучение ML-моделей при накоплении новых данных."""

    MIN_SAMPLES = 300        # минимум для первого обучения
    RETRAIN_DELTA = 50       # новых размеченных сделок для переобучения
    MAX_WINDOW = 5000        # обучаем на последних N примерах (скользящее окно)

    def __init__(
        self,
        ml_store,
        ml_trainer,
        ml_ensemble_trainer,
        ml_predictor,
        feature_extractor,
        drift_monitor,
        db_pool,
    ):
        self.ml_store = ml_store
        self.ml_trainer = ml_trainer
        self.ml_ensemble_trainer = ml_ensemble_trainer
        self.ml_predictor = ml_predictor
        self.feature_extractor = feature_extractor
        self.drift_monitor = drift_monitor
        self.db_pool = db_pool

        self._last_train_counts: Dict[str, int] = {}
        self._last_train_at: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    def _get_labeled_counts(self) -> Dict[str, int]:
        stats = self.ml_store.get_ml_stats()
        return {s["strategy_id"]: s["labeled"] for s in stats["per_strategy"]}

    async def maybe_retrain(self, strategy_ids: List[str]) -> Dict:
        """
        Проверяет нужно ли переобучение.
        Вызывается из trading loop или фонового авто-трейн цикла.
        """
        if self._lock.locked():
            return {}

        counts = self._get_labeled_counts()
        to_train = []

        for sid in strategy_ids:
            labeled = counts.get(sid, 0)
            last = self._last_train_counts.get(sid, 0)

            first_time = labeled >= self.MIN_SAMPLES and last == 0
            incremental = labeled >= self.MIN_SAMPLES and (labeled - last) >= self.RETRAIN_DELTA

            if first_time or incremental:
                reason = "первое обучение" if first_time else f"+{labeled - last} новых"
                logger.info(f"[AutoTrain] {sid}: {reason} (всего={labeled})")
                to_train.append((sid, labeled))

        if not to_train:
            return {}

        results = {}
        async with self._lock:
            for sid, labeled in to_train:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._retrain_sync, sid, labeled
                )
                results[sid] = result

        return results

    def _retrain_sync(self, strategy_id: str, current_count: int) -> Dict:
        """Синхронное переобучение (выполняется в thread executor)."""
        try:
            feature_columns = self.feature_extractor.FEATURE_NAMES

            training_data = self.ml_store.get_training_data(
                strategy_id=strategy_id,
                min_samples=self.MIN_SAMPLES,
                only_taken_trades=False,
            )

            if training_data.empty or len(training_data) < self.MIN_SAMPLES:
                return {
                    "success": False,
                    "error": f"Мало данных: {len(training_data)}/{self.MIN_SAMPLES}",
                }

            # Скользящее окно: берём последние MAX_WINDOW примеров
            if len(training_data) > self.MAX_WINDOW:
                training_data = training_data.sort_values("timestamp").tail(self.MAX_WINDOW).reset_index(drop=True)

            n = len(training_data)
            logger.info(f"[AutoTrain] {strategy_id}: обучаю на {n} примерах...")

            # Пробуем ансамбль первым — лучше по качеству
            result = self.ml_ensemble_trainer.train_ensemble(
                training_data, strategy_id, feature_columns
            )

            # Fallback на XGBoost если ансамбль не удался
            if not result.get("success"):
                result = self.ml_trainer.train(training_data, strategy_id, feature_columns)

            if not result.get("success"):
                return result

            self.ml_store.register_model(
                version=result["version"],
                strategy_id=strategy_id,
                model_type=result.get("model_type", "XGBoost"),
                metrics=result["metrics"],
                hyperparams=result.get("hyperparams", {}),
                feature_importance=result.get("feature_importance", {}),
                file_path=result["file_path"],
                samples_count=result["samples"],
                set_active=True,
            )
            self.ml_predictor.load_model(strategy_id, result["file_path"])
            self._last_train_counts[strategy_id] = current_count
            self._last_train_at[strategy_id] = datetime.utcnow().isoformat()

            # Авто-тюн threshold на последних данных
            self._auto_tune_threshold(strategy_id)

            # Сброс drift-счётчика — модель новая, старые ошибки не считаются
            if strategy_id in self.drift_monitor.predictions:
                self.drift_monitor.predictions[strategy_id].clear()
                self.drift_monitor.outcomes[strategy_id].clear()

            m = result["metrics"]
            logger.info(
                f"[AutoTrain] {strategy_id}: готово! "
                f"F1={m.get('f1', 0):.3f}, AUC={m.get('roc_auc', 0):.3f}, "
                f"примеров={result['samples']}"
            )
            return result

        except Exception as e:
            logger.exception(f"[AutoTrain] {strategy_id} ошибка: {e}")
            return {"success": False, "error": str(e)}

    def _auto_tune_threshold(self, strategy_id: str):
        """Авто-подбор threshold после переобучения."""
        try:
            from .drift_monitor import ThresholdOptimizer

            with self.db_pool.connection() as conn:
                rows = conn.execute("""
                    SELECT ml_prediction, outcome
                    FROM signal_snapshots
                    WHERE strategy_id = ?
                      AND ml_prediction IS NOT NULL
                      AND outcome IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 500
                """, (strategy_id,)).fetchall()

            if len(rows) < 30:
                return

            preds = [r["ml_prediction"] for r in rows]
            outs = [1 if r["outcome"] == "win" else 0 for r in rows]
            res = ThresholdOptimizer.find_optimal_threshold(
                preds, outs, objective="f1_weighted_by_count"
            )
            self.ml_predictor.set_threshold(strategy_id, res["threshold"])
            logger.info(
                f"[AutoTrain] {strategy_id}: threshold → {res['threshold']:.3f} "
                f"(score={res['best_score']:.3f})"
            )
        except Exception as e:
            logger.warning(f"[AutoTrain] threshold tune {strategy_id}: {e}")

    def get_status(self) -> Dict:
        counts = self._get_labeled_counts()
        all_sids = set(list(counts.keys()) + list(self._last_train_counts.keys()))
        return {
            "min_samples": self.MIN_SAMPLES,
            "retrain_delta": self.RETRAIN_DELTA,
            "max_window": self.MAX_WINDOW,
            "is_training": self._lock.locked(),
            "per_strategy": {
                sid: {
                    "labeled": counts.get(sid, 0),
                    "last_train_count": self._last_train_counts.get(sid, 0),
                    "new_since_train": counts.get(sid, 0) - self._last_train_counts.get(sid, 0),
                    "last_trained_at": self._last_train_at.get(sid),
                    "needs_train": (
                        counts.get(sid, 0) >= self.MIN_SAMPLES
                        and (
                            self._last_train_counts.get(sid, 0) == 0
                            or counts.get(sid, 0) - self._last_train_counts.get(sid, 0) >= self.RETRAIN_DELTA
                        )
                    ),
                }
                for sid in all_sids
            },
        }
