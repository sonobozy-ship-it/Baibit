"""
Auto Optimizer — автоматическая оптимизация параметров стратегий.
Bayesian optimization через Optuna (намного эффективнее grid search).
"""
import logging
from typing import Dict, Type, Optional
from datetime import datetime
import uuid
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class AutoOptimizer:
    """Bayesian оптимизация параметров стратегии."""

    def __init__(self, backtester, data_store):
        self.backtester = backtester
        self.data_store = data_store

    def optimize(
        self,
        strategy_class: Type,
        df: pd.DataFrame,
        symbol: str,
        n_trials: int = 50,
        objective: str = "sharpe",  # sharpe, profit_factor, roi, total_pnl
    ) -> Dict:
        """
        Оптимизация параметров стратегии.

        Параметры для тюнинга:
        - stop_loss_pct: 0.5 — 3.0
        - take_profit_pct: 1.0 — 6.0
        - breakeven_pct: 0.3 — 1.5
        - trailing_stop_pct: 0.2 — 1.5
        """
        if not OPTUNA_AVAILABLE:
            return {"success": False, "error": "Optuna не установлен (pip install optuna)"}
        if len(df) < 300:
            return {"success": False, "error": "Мало данных для оптимизации"}

        run_id = str(uuid.uuid4())[:8]
        logger.info(f"🔬 Запуск оптимизации {run_id} для {strategy_class.ID} ({n_trials} trials)")

        def trial_fn(trial):
            params = {
                "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.5, 3.0, step=0.1),
                "take_profit_pct": trial.suggest_float("take_profit_pct", 1.0, 6.0, step=0.1),
                "breakeven_pct": trial.suggest_float("breakeven_pct", 0.3, 1.5, step=0.1),
                "trailing_stop_pct": trial.suggest_float("trailing_stop_pct", 0.2, 1.5, step=0.1),
            }
            # RR должен быть > 1.2
            if params["take_profit_pct"] / params["stop_loss_pct"] < 1.2:
                return -999

            try:
                result = self.backtester.run(strategy_class, df.copy(), symbol, **params)
                metrics = result.calculate_metrics()
                if metrics["trades"] < 10:
                    return -999  # слишком мало сделок

                if objective == "sharpe":
                    return metrics["sharpe"]
                elif objective == "profit_factor":
                    return metrics["profit_factor"] if metrics["profit_factor"] != float("inf") else 0
                elif objective == "roi":
                    return metrics["roi_pct"]
                else:
                    return metrics["total_pnl"]
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -999

        study = optuna.create_study(direction="maximize", study_name=f"opt_{run_id}")
        study.optimize(trial_fn, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score = study.best_value

        # Сохранение в БД
        try:
            import sqlite3
            conn = sqlite3.connect(self.data_store.db_path)
            conn.execute("""
                INSERT INTO optimization_runs (
                    run_id, strategy_id, symbol, started_at, completed_at,
                    best_params_json, best_score, method, trials
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, strategy_class.ID, symbol,
                datetime.utcnow().isoformat(), datetime.utcnow().isoformat(),
                str(best_params), best_score, "optuna_bayesian", n_trials,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Сохранение оптимизации: {e}")

        logger.info(f"✅ Оптимизация завершена. Best {objective}={best_score:.3f}")

        # Сравнение с дефолтными параметрами
        default_result = self.backtester.run(strategy_class, df.copy(), symbol)
        default_metrics = default_result.calculate_metrics()
        optimized_result = self.backtester.run(strategy_class, df.copy(), symbol, **best_params)
        optimized_metrics = optimized_result.calculate_metrics()

        return {
            "success": True,
            "run_id": run_id,
            "strategy_id": strategy_class.ID,
            "objective": objective,
            "best_score": round(best_score, 3),
            "best_params": best_params,
            "n_trials": n_trials,
            "default_metrics": default_metrics,
            "optimized_metrics": optimized_metrics,
            "improvement_pct": round(
                (optimized_metrics["total_pnl"] - default_metrics["total_pnl"]) /
                abs(default_metrics["total_pnl"]) * 100 if default_metrics["total_pnl"] != 0 else 0,
                2
            ),
        }
