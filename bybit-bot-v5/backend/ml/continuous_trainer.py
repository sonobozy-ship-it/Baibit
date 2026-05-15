"""
Continuous ML Trainer — постепенное обучение до 20 000 сделок.

Логика накопления:
  MIN_SAMPLES    = 300   — минимум для первого обучения
  INCREMENTAL_D  = 50    — +50 новых → быстрое обучение на последних 2 000
  FULL_RETRAIN_D = 500   — +500 новых → полное переобучение на всех 20 000
  MAX_WINDOW     = 20000 — максимальное скользящее окно памяти

Веса примеров:
  Новые сделки важнее старых. Применяем экспоненциальный decay:
  w(i) = exp(linspace(-WEIGHT_DECAY, 0, n))
  → самая старая сделка имеет вес exp(-WEIGHT_DECAY), самая новая — 1.0
  WEIGHT_DECAY = 2.0 даёт соотношение ~7.4:1 между новой и старой
"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ContinuousTrainer:
    """Постепенное ML-обучение с памятью до 20 000 сделок на стратегию."""

    MIN_SAMPLES     = 300     # минимум для первого обучения
    INCREMENTAL_D   = 50      # +50 новых → быстрое обучение (окно 2 000)
    FULL_RETRAIN_D  = 500     # +500 новых → полное обучение (окно 20 000)
    MAX_WINDOW      = 20_000  # максимум хранимых примеров
    FAST_WINDOW     = 2_000   # окно для быстрого инкрементального обучения
    WEIGHT_DECAY    = 2.0     # насколько старые сделки менее важны

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
        self.ml_store            = ml_store
        self.ml_trainer          = ml_trainer
        self.ml_ensemble_trainer = ml_ensemble_trainer
        self.ml_predictor        = ml_predictor
        self.feature_extractor   = feature_extractor
        self.drift_monitor       = drift_monitor
        self.db_pool             = db_pool

        self._last_train_counts: Dict[str, int] = {}
        self._last_train_at:     Dict[str, str]  = {}
        self._lock = asyncio.Lock()

    # ── вспомогательные ──────────────────────────────────────

    def _get_labeled_counts(self) -> Dict[str, int]:
        stats = self.ml_store.get_ml_stats()
        return {s["strategy_id"]: s["labeled"] for s in stats["per_strategy"]}

    @staticmethod
    def _compute_sample_weights(n: int, decay: float = 2.0) -> np.ndarray:
        """
        Экспоненциальные веса: новейший пример → 1.0, самый старый → exp(-decay).
        Нормализованы так, что среднее = 1.0 (не меняет масштаб градиента).
        """
        if n == 0:
            return np.array([])
        w = np.exp(np.linspace(-decay, 0.0, n))
        return w / w.mean()

    # ── публичный API ─────────────────────────────────────────

    async def maybe_retrain(self, strategy_ids: List[str]) -> Dict:
        """
        Проверяет нужно ли переобучение для каждой стратегии.
        Вызывается из trading loop или фонового авто-трейн цикла.
        """
        if self._lock.locked():
            return {}

        counts = self._get_labeled_counts()
        to_train: List[tuple] = []   # (sid, labeled, mode)

        for sid in strategy_ids:
            labeled = counts.get(sid, 0)
            last    = self._last_train_counts.get(sid, 0)
            delta   = labeled - last

            if labeled < self.MIN_SAMPLES:
                continue

            if last == 0:
                # первое обучение
                to_train.append((sid, labeled, "full"))
            elif delta >= self.FULL_RETRAIN_D:
                # накопилось много — полное переобучение на 20k
                to_train.append((sid, labeled, "full"))
            elif delta >= self.INCREMENTAL_D:
                # небольшой прирост — быстрое обновление на 2k
                to_train.append((sid, labeled, "incremental"))

        if not to_train:
            return {}

        results: Dict = {}
        async with self._lock:
            for sid, labeled, mode in to_train:
                reason = {
                    "full":        f"полное (всего={labeled}, окно={self.MAX_WINDOW})",
                    "incremental": f"инкрементальное (всего={labeled}, окно={self.FAST_WINDOW})",
                }[mode]
                logger.info(f"[AutoTrain] {sid}: {reason}")
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._retrain_sync, sid, labeled, mode
                )
                results[sid] = result

        return results

    # ── внутреннее обучение ───────────────────────────────────

    def _retrain_sync(self, strategy_id: str, current_count: int, mode: str = "full") -> Dict:
        """Синхронное переобучение (выполняется в thread executor)."""
        try:
            feature_columns = self.feature_extractor.FEATURE_NAMES
            window = self.MAX_WINDOW if mode == "full" else self.FAST_WINDOW

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

            # Скользящее окно — берём последние N
            if len(training_data) > window:
                training_data = (
                    training_data
                    .sort_values("timestamp")
                    .tail(window)
                    .reset_index(drop=True)
                )

            n = len(training_data)
            # Веса: новые сделки важнее — экспоненциальный decay
            sample_weight = self._compute_sample_weights(n, self.WEIGHT_DECAY)

            logger.info(
                f"[AutoTrain] {strategy_id}: {mode} обучение на {n} примерах "
                f"(вес новых / старых ≈ {np.exp(self.WEIGHT_DECAY):.1f}x)"
            )

            # Ансамбль XGB+LGB+LogReg — лучше по качеству
            result = self.ml_ensemble_trainer.train_ensemble(
                training_data, strategy_id, feature_columns,
                sample_weight=sample_weight,
            )

            # Fallback: только XGBoost
            if not result.get("success"):
                result = self.ml_trainer.train(
                    training_data, strategy_id, feature_columns,
                    sample_weight=sample_weight,
                )

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
            self._last_train_at[strategy_id]     = datetime.utcnow().isoformat()

            self._auto_tune_threshold(strategy_id)

            # Сброс drift-счётчика — новая модель, история ошибок не актуальна
            if strategy_id in self.drift_monitor.predictions:
                self.drift_monitor.predictions[strategy_id].clear()
                self.drift_monitor.outcomes[strategy_id].clear()

            m = result["metrics"]
            logger.info(
                f"[AutoTrain] {strategy_id}: готово! "
                f"F1={m.get('f1', 0):.3f}, AUC={m.get('roc_auc', 0):.3f}, "
                f"примеров={result['samples']}, режим={mode}"
            )
            return {**result, "mode": mode, "window": window}

        except Exception as e:
            logger.exception(f"[AutoTrain] {strategy_id} ошибка: {e}")
            return {"success": False, "error": str(e)}

    def _auto_tune_threshold(self, strategy_id: str):
        """Авто-подбор threshold после переобучения (последние 500 сделок)."""
        try:
            from .drift_monitor import ThresholdOptimizer

            sql = self.db_pool.adapt("""
                SELECT ml_prediction, outcome
                FROM signal_snapshots
                WHERE strategy_id = ?
                  AND ml_prediction IS NOT NULL
                  AND outcome IS NOT NULL
                ORDER BY id DESC
                LIMIT 500
            """)
            with self.db_pool.connection() as conn:
                if self.db_pool.is_mysql:
                    with conn.cursor() as c:
                        c.execute(sql, (strategy_id,))
                        rows = c.fetchall()
                else:
                    import sqlite3
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(sql, (strategy_id,)).fetchall()

            if len(rows) < 30:
                return

            preds = [float(r["ml_prediction"]) for r in rows]
            outs  = [1 if r["outcome"] == "win" else 0 for r in rows]
            res   = ThresholdOptimizer.find_optimal_threshold(
                preds, outs, objective="f1_weighted_by_count"
            )
            self.ml_predictor.set_threshold(strategy_id, res["threshold"])
            logger.info(
                f"[AutoTrain] {strategy_id}: threshold → {res['threshold']:.3f} "
                f"(score={res['best_score']:.3f})"
            )
        except Exception as e:
            logger.warning(f"[AutoTrain] threshold tune {strategy_id}: {e}")

    # ── статус ────────────────────────────────────────────────

    def get_status(self) -> Dict:
        counts  = self._get_labeled_counts()
        all_sids = set(list(counts.keys()) + list(self._last_train_counts.keys()))
        return {
            "min_samples":      self.MIN_SAMPLES,
            "incremental_delta": self.INCREMENTAL_D,
            "full_retrain_delta": self.FULL_RETRAIN_D,
            "max_window":       self.MAX_WINDOW,
            "fast_window":      self.FAST_WINDOW,
            "weight_decay":     self.WEIGHT_DECAY,
            "is_training":      self._lock.locked(),
            "per_strategy": {
                sid: {
                    "labeled":          counts.get(sid, 0),
                    "last_train_count": self._last_train_counts.get(sid, 0),
                    "new_since_train":  counts.get(sid, 0) - self._last_train_counts.get(sid, 0),
                    "last_trained_at":  self._last_train_at.get(sid),
                    "next_mode": (
                        "full"
                        if counts.get(sid, 0) - self._last_train_counts.get(sid, 0) >= self.FULL_RETRAIN_D
                        else "incremental"
                        if counts.get(sid, 0) - self._last_train_counts.get(sid, 0) >= self.INCREMENTAL_D
                        else "waiting"
                    ),
                }
                for sid in all_sids
            },
        }
