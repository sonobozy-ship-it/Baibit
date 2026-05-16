"""
Главный оркестратор бота.
Запускает FastAPI сервер с REST + WebSocket API для UI.
Координирует все стратегии, риск-менеджер, журнал, AI, Telegram.
"""
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

# Локальные модули
from bybit_client import BybitClient
from risk_manager import RiskManager
from backtester import Backtester
from telegram_notifier import TelegramNotifier
from trade_journal import TradeJournal
from correlation_filter import CorrelationFilter
from ai_analyzer import AIAnalyzer
from paper_trader import PaperTrader
from strategies import ALL_STRATEGIES
from strategies.fusion import StrategyFusion, SignalBuffer, FusedSignal
from strategies.scalper_pro import ScalperProStrategy, ScalperTrendContext, SCALP_SYMBOLS

# ML модули
from ml import (
    FeatureExtractor, MLDataStore, MLTrainer, MLPredictor,
    RegimeClassifier, AutoOptimizer, ML_AVAILABLE, OPTUNA_AVAILABLE,
    DriftMonitor, ThresholdOptimizer, AnomalyDetector, EnsembleTrainer,
    ContinuousTrainer,
)
# News & Sentiment
from news import NewsManager, background_news_loop
# Position calculation
from position_calc import (
    calculate_pnl, calculate_r_multiple,
    fixed_risk_position_size, kelly_position_size, adaptive_sl_tp,
)
from db_pool import DBPool
from boost_mode import BoostManager, BoostCalculator

# ============================================================
# Загрузка конфига
# ============================================================
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
Path("logs").mkdir(exist_ok=True)


# ============================================================
# Глобальное состояние
# ============================================================
class BotState:
    def __init__(self):
        self.bybit: Optional[BybitClient] = None
        self.risk_manager = RiskManager(
            daily_max_loss_pct=float(os.getenv("DAILY_MAX_LOSS_PCT", "5.0")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "4")),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "1.0")),
            cooldown_after_loss_min=int(os.getenv("COOLDOWN_AFTER_LOSS_MIN", "15")),
        )
        self.journal = TradeJournal()
        self.correlation = CorrelationFilter()
        self.ai = AIAnalyzer()
        self.paper = PaperTrader()
        self.telegram = TelegramNotifier(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
        self.backtester = Backtester()

        # ============== ML инфраструктура ==============
        self.ml_store = MLDataStore()
        self.feature_extractor = FeatureExtractor()
        self.ml_predictor = MLPredictor(default_threshold=float(os.getenv("ML_THRESHOLD", "0.55")))
        self.ml_trainer = MLTrainer()
        self.ml_ensemble_trainer = EnsembleTrainer()
        self.ml_optimizer = AutoOptimizer(self.backtester, self.ml_store)
        self.regime_classifier = RegimeClassifier()
        self.drift_monitor = DriftMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.ml_enabled = os.getenv("ML_ENABLED", "true").lower() == "true"
        self.ml_filter_mode = os.getenv("ML_FILTER_MODE", "advisory")  # advisory | strict | off
        self.use_ensemble = os.getenv("ML_USE_ENSEMBLE", "true").lower() == "true"
        self.use_kelly = os.getenv("USE_KELLY", "false").lower() == "true"
        self.use_adaptive_sl = os.getenv("USE_ADAPTIVE_SL", "true").lower() == "true"
        self.snapshot_signal_id_map: Dict[str, int] = {}

        # ============== News & Sentiment ==============
        self.db_pool = DBPool("data/ml_data.db")
        self.news_manager = NewsManager(
            db_pool=self.db_pool,
            cryptopanic_key=os.getenv("CRYPTOPANIC_API_KEY"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.news_enabled = os.getenv("NEWS_ENABLED", "true").lower() == "true"

        # Locks для предотвращения race conditions
        self.trading_loop_lock = asyncio.Lock()
        self.trading_loop_task: Optional[asyncio.Task] = None
        self.news_loop_task: Optional[asyncio.Task] = None
        self.auto_train_task: Optional[asyncio.Task] = None

        # Continuous trainer — инициализируется после init_strategies
        self.continuous_trainer: Optional[ContinuousTrainer] = None

        # Strategy Fusion Engine
        self.signal_buffer  = SignalBuffer()
        self.strategy_fusion = StrategyFusion()

        # Boost Mode — разгон депозита
        self.boost = BoostManager()

        # Флаг скальпинг-режима (управляется через /api/boost/scalp/activate)
        self.scalp_active: bool = False

        # Кэш старших таймфреймов для MTF-фильтра ScalperPro
        # {symbol: (DataFrame, updated_at)}
        self.h1_cache:  Dict[str, tuple] = {}
        self.m15_cache: Dict[str, tuple] = {}

        # Активные стратегии (создаются при старте)
        self.strategies: Dict[str, object] = {}
        self.bot_running = False
        self.paper_mode = False
        self.ws_clients: List[WebSocket] = []
        self.tickers: Dict[str, Dict] = {}

state = BotState()


# ============================================================
# Инициализация стратегий
# ============================================================
def init_strategies():
    """Создание всех 9 стратегий с дефолтными символами."""
    symbol_map = {
        "S1": "BTCUSDT",
        "S2": "ETHUSDT",
        "S3": "SOLUSDT",
        "S4": "BNBUSDT",
        "S5": "DOGEUSDT",
        "S6": "XRPUSDT",
        "S7": "BTCUSDT",
        "S8": "ETHUSDT",    # Trend Momentum
        "S9": "BTCUSDT",    # Trend + Fibonacci
    }
    for sid, cls in ALL_STRATEGIES.items():
        state.strategies[sid] = cls(symbol=symbol_map.get(sid, "BTCUSDT"))
    logger.info(f"Инициализированы стратегии: {list(state.strategies.keys())}")

    # Загрузка активных ML моделей (если есть)
    if state.ml_enabled and ML_AVAILABLE:
        state.ml_predictor.load_all_active_models(state.ml_store)
        loaded = len(state.ml_predictor.models)
        if loaded:
            logger.info(f"🤖 Загружено {loaded} ML моделей")
        else:
            logger.info("🤖 ML модели не найдены — будет собирать данные для будущего обучения")

    # Инициализация continuous trainer
    state.continuous_trainer = ContinuousTrainer(
        ml_store=state.ml_store,
        ml_trainer=state.ml_trainer,
        ml_ensemble_trainer=state.ml_ensemble_trainer,
        ml_predictor=state.ml_predictor,
        feature_extractor=state.feature_extractor,
        drift_monitor=state.drift_monitor,
        db_pool=state.db_pool,
    )
    logger.info("🔄 ContinuousTrainer инициализирован (порог: 300 сделок)")


def activate_scalp_mode():
    """
    Создаёт экземпляры ScalperPro для каждого символа из SCALP_SYMBOLS
    и регистрирует их в state.strategies как SC_XXX.
    Вызывается при старте boost-режима с mode='scalp' или вручную.
    """
    added = []
    for sym in SCALP_SYMBOLS:
        sid = f"SC_{sym[:3]}"
        if sid not in state.strategies:
            strat = ScalperProStrategy(symbol=sym)
            state.strategies[sid] = strat
            added.append(sid)
        else:
            state.strategies[sid].enabled = True
    state.scalp_active = True
    logger.info(f"⚡ Scalp Mode: добавлено {len(added)} скальперов → {added}")
    return added


def deactivate_scalp_mode():
    """Отключает все SC_* стратегии (не удаляет — можно переключить)."""
    deactivated = []
    for sid, strat in state.strategies.items():
        if sid.startswith("SC_"):
            strat.enabled = False
            deactivated.append(sid)
    state.scalp_active = False
    logger.info(f"⚡ Scalp Mode отключён: {deactivated}")
    return deactivated


def _get_h1_cached(symbol: str) -> Optional[object]:
    """H1 свечи с TTL-кэшем 60 минут (не дёргаем API каждые 5 сек)."""
    if not state.bybit:
        return None
    cached = state.h1_cache.get(symbol)
    if cached and (datetime.utcnow() - cached[1]).total_seconds() < 3600:
        return cached[0]
    try:
        df = state.bybit.get_klines(symbol, "60", limit=100)
        if df is not None and not df.empty:
            state.h1_cache[symbol] = (df, datetime.utcnow())
            return df
    except Exception as e:
        logger.debug(f"H1 cache {symbol}: {e}")
    return None


def _get_m15_cached(symbol: str) -> Optional[object]:
    """15m свечи с TTL-кэшем 15 минут."""
    if not state.bybit:
        return None
    cached = state.m15_cache.get(symbol)
    if cached and (datetime.utcnow() - cached[1]).total_seconds() < 900:
        return cached[0]
    try:
        df = state.bybit.get_klines(symbol, "15", limit=100)
        if df is not None and not df.empty:
            state.m15_cache[symbol] = (df, datetime.utcnow())
            return df
    except Exception as e:
        logger.debug(f"15m cache {symbol}: {e}")
    return None


# ============================================================
# Авто-обучение (фоновая задача)
# ============================================================
async def auto_train_loop():
    """Проверяет и запускает переобучение каждые 30 минут."""
    logger.info("🔄 Auto-train loop запущен (интервал: 30 мин, порог: 300 сделок)")
    while True:
        await asyncio.sleep(30 * 60)
        if not state.ml_enabled or not ML_AVAILABLE or not state.continuous_trainer:
            continue
        try:
            results = await state.continuous_trainer.maybe_retrain(list(state.strategies.keys()))
            for sid, r in results.items():
                if isinstance(r, dict) and r.get("success"):
                    m = r.get("metrics", {})
                    await broadcast_log(
                        f"🔄 AutoTrain {sid}: {r.get('samples', 0)} примеров, "
                        f"F1={m.get('f1', 0):.3f}, AUC={m.get('roc_auc', 0):.3f}",
                        "info",
                    )
                    asyncio.create_task(state.telegram.send(
                        f"🔄 <b>AutoTrain</b> {sid}\n"
                        f"Примеров: {r.get('samples', 0)}\n"
                        f"F1: {m.get('f1', 0):.3f} | AUC: {m.get('roc_auc', 0):.3f}"
                    ))
                elif isinstance(r, dict) and not r.get("success"):
                    logger.info(f"[AutoTrain] {sid}: {r.get('error', 'unknown')}")
        except Exception as e:
            logger.error(f"auto_train_loop error: {e}")


# ============================================================
# Fusion — исполнение объединённого сигнала
# ============================================================
async def _execute_fusion_signal(
    fused: FusedSignal,
    balance: float,
    sentiment_features: dict,
):
    """
    Исполняет FusedSignal: проверки риска, ML-снапшот, открытие позиции.
    Использует size_multiplier для увеличения размера позиции.
    """
    sym = fused.symbol
    sid = "FUSION"

    # Проверяем, нет ли уже открытой позиции на этом символе через любую стратегию
    for strat in state.strategies.values():
        if strat.symbol == sym and strat.current_position:
            logger.debug(f"[Fusion] {sym}: уже открыта позиция через {strat.ID}, пропускаем fusion")
            return

    # Риск-менеджер
    check = state.risk_manager.can_open_trade(sid, balance)
    if not check["allowed"]:
        logger.info(f"[Fusion] ❌ {check['reason']}")
        return

    # Корреляция
    open_pos = [
        {"symbol": s.symbol, "side": s.current_position["side"]}
        for s in state.strategies.values() if s.current_position
    ]
    corr = state.correlation.can_open(sym, fused.action, open_pos)
    if not corr["allowed"]:
        logger.info(f"[Fusion] ❌ {corr['reason']}")
        return

    # ML snapshot + prediction
    ml_prediction = None
    fusion_snapshot_id = None
    if state.ml_enabled:
        try:
            df = state.bybit.get_klines(sym, "60", limit=250) if state.bybit else None
            if df is not None and not df.empty:
                orderbook   = state.bybit.get_orderbook(sym, limit=25)
                market_meta = state.bybit.get_market_meta(sym)
                features = state.feature_extractor.extract(
                    df=df,
                    signal_data={
                        "action":      fused.action,
                        "confidence":  fused.confidence,
                        "entry_price": fused.entry_price,
                        "stop_loss":   fused.stop_loss,
                        "take_profit": fused.take_profit,
                    },
                    strategy_stats={"rolling_wr_20": 0.5, "rolling_pnl_20": 0,
                                    "consecutive_losses": 0, "trades": 0},
                    sentiment_data=sentiment_features,
                    orderbook_data=orderbook,
                    market_meta=market_meta,
                    regime_id=None,
                )
                # Добавляем fusion-специфичные фичи
                features = state.strategy_fusion.get_fusion_features(fused, features)

                ml_prediction = state.ml_predictor.predict(sid, features)
                fusion_snapshot_id = state.ml_store.save_signal_snapshot({
                    "timestamp":        datetime.utcnow().isoformat(),
                    "strategy_id":      sid,
                    "symbol":           sym,
                    "timeframe":        "60",
                    "action":           fused.action,
                    "entry_price":      fused.entry_price,
                    "stop_loss":        fused.stop_loss,
                    "take_profit":      fused.take_profit,
                    "features":         features,
                    "ml_prediction":    ml_prediction.get("probability") if ml_prediction else None,
                    "ml_confidence":    fused.fusion_score,
                    "ml_model_version": ml_prediction.get("model_version") if ml_prediction else None,
                    "trade_taken":      False,
                })

                # В strict режиме ML может отклонить
                if (ml_prediction and ml_prediction.get("available")
                        and state.ml_filter_mode == "strict"
                        and not ml_prediction["should_take"]):
                    logger.info(
                        f"[Fusion] ML отверг (P={ml_prediction['probability']:.2f})"
                    )
                    return
        except Exception as e:
            logger.warning(f"[Fusion] ML pass error: {e}")

    # Размер позиции с multiplier
    base_qty = state.risk_manager.calculate_position_size(
        balance=balance,
        entry_price=fused.entry_price,
        stop_loss_price=fused.stop_loss,
        leverage=3,
    )
    qty = round(base_qty * fused.size_multiplier, 6)
    if qty <= 0:
        return

    # Открытие позиции
    log_msg = (
        f"[Fusion] {fused.action} {sym} "
        f"[{'+'.join(fused.source_strategies)}] "
        f"score={fused.fusion_score:.3f} "
        f"size×{fused.size_multiplier}"
    )

    if state.paper_mode:
        from strategies.base import TradingSignal as _TS
        state.paper.open_position(fused, sid, qty, 3)
        await broadcast_log(f"📄 FUSION {fused.action} {sym} (paper) {log_msg}")
    else:
        if not state.bybit:
            return
        lev_check = state.risk_manager.check_leverage(sid, 3, balance)
        result = state.bybit.place_order(
            symbol=sym,
            side="Buy" if fused.action == "BUY" else "Sell",
            qty=qty,
            stop_loss=fused.stop_loss,
            take_profit=fused.take_profit,
            leverage=lev_check["effective_leverage"],
        )
        if result["success"]:
            notional = qty * fused.entry_price
            state.risk_manager.register_position_open(sid, notional)

            trade_id = state.journal.log_trade({
                "strategy_id":   sid,
                "strategy_name": fused.reason[:64],
                "symbol":        sym,
                "side":          fused.action,
                "entry_price":   fused.entry_price,
                "qty":           qty,
                "leverage":      lev_check["effective_leverage"],
                "stop_loss":     fused.stop_loss,
                "take_profit":   fused.take_profit,
                "filters_passed": {
                    "fusion_score":     fused.fusion_score,
                    "strategies":       fused.source_strategies,
                    "diversity":        fused.diversity_score,
                },
                "opened_at": datetime.utcnow().isoformat(),
            })

            if fusion_snapshot_id:
                state.snapshot_signal_id_map[sym + "_FUSION"] = fusion_snapshot_id
                with state.db_pool.cursor() as c:
                    c.execute(
                        state.db_pool.adapt(
                            "UPDATE signal_snapshots SET trade_taken=1, trade_id=? WHERE id=?"
                        ),
                        (trade_id, fusion_snapshot_id),
                    )

            # Инвалидируем сигналы использованных стратегий — не открываем дубли
            for src_sid in fused.source_strategies:
                state.signal_buffer.invalidate(src_sid)

            asyncio.create_task(state.telegram.notify_trade_open(
                sid, sym, fused.action,
                fused.entry_price, fused.stop_loss, fused.take_profit,
                fused.reason,
            ))
            await broadcast_log(
                f"🔥 FUSION {fused.action} {sym} @ {fused.entry_price} "
                f"[{'+'.join(fused.source_strategies)}] ×{fused.size_multiplier}"
            )
        else:
            logger.error(f"[Fusion] Ошибка ордера: {result.get('error')}")


# ============================================================
# Главный торговый цикл
# ============================================================
async def trading_loop():
    """
    Основной цикл с lock для предотвращения race conditions.
    """
    # Lock: не даём запустить дважды
    if state.trading_loop_lock.locked():
        logger.warning("Trading loop уже запущен, отказываюсь дублировать")
        return

    async with state.trading_loop_lock:
        logger.info("🚀 Торговый цикл запущен")
        while state.bot_running:
            try:
                if not state.bybit:
                    await asyncio.sleep(2)
                    continue

                balance = state.bybit.get_balance("USDT") if not state.paper_mode else state.paper.balance
                state.risk_manager.check_daily_reset(balance)

                # Получаем актуальный sentiment с проверкой свежести (не старше 30 мин)
                sentiment_features = {}
                if state.news_enabled:
                    try:
                        last_upd = getattr(state.news_manager, "last_fetch", None)
                        if last_upd is None:
                            sentiment_features = state.news_manager.get_sentiment_features()
                        else:
                            age_min = (datetime.utcnow() - last_upd).total_seconds() / 60
                            if age_min <= 30:
                                sentiment_features = state.news_manager.get_sentiment_features()
                            else:
                                logger.debug(f"Sentiment устарел ({age_min:.0f} мин > 30), пропускаем")
                    except Exception:
                        sentiment_features = state.news_manager.get_sentiment_features()

                klines_data = {}
                current_regime_name = None   # инициализируем до цикла (используется в fusion)
                current_regime_id   = None

                for sid, strat in state.strategies.items():
                    if not strat.enabled or strat.auto_disabled:
                        continue

                    # Boost-режим: пропускаем стратегии вне разрешённого списка
                    if not state.boost.strategy_allowed(sid):
                        continue

                    df = state.bybit.get_klines(strat.symbol, strat.timeframe, limit=250)
                    if df.empty:
                        continue
                    klines_data[strat.symbol] = df

                    # Определяем рыночный режим один раз per strategy (используется ниже для фильтрации и ML)
                    current_regime_id = None
                    current_regime_name = None
                    if state.ml_enabled and state.regime_classifier.model:
                        try:
                            _rr = state.regime_classifier.predict(df)
                            if _rr:
                                current_regime_id = _rr.get("regime_id")
                                current_regime_name = _rr.get("regime_name", "")
                        except Exception:
                            pass

                    # Soft-фильтр по режиму: пропускаем стратегию если режим не подходит
                    if (current_regime_name and strat.REGIME_PREFERENCE
                            and current_regime_name not in strat.REGIME_PREFERENCE):
                        logger.debug(
                            f"{sid}: режим '{current_regime_name}' не подходит "
                            f"(предпочтение: {strat.REGIME_PREFERENCE})"
                        )
                        continue

                    # Если позиция уже открыта — проверка breakeven/trailing
                    if strat.current_position:
                        current_price = float(df.iloc[-1]["close"])
                        new_sl = strat.check_breakeven(current_price)
                        if new_sl and not state.paper_mode:
                            state.bybit.update_stop_loss(strat.symbol, new_sl)
                            await broadcast_log(f"🛡 {sid} перенос SL в безубыток @ {new_sl:.4f}")

                        trail_sl = strat.check_trailing_stop(current_price)
                        if trail_sl and not state.paper_mode:
                            state.bybit.update_stop_loss(strat.symbol, trail_sl)
                        continue

                    # Поиск сигнала
                    # SC_* (ScalperPro) получают H1 + 15m для MTF-фильтра
                    if sid.startswith("SC_"):
                        signal = strat.analyze(
                            df,
                            df_h1  = _get_h1_cached(strat.symbol),
                            df_m15 = _get_m15_cached(strat.symbol) if state.scalp_active else None,
                        )
                    else:
                        signal = strat.analyze(df)
                    if not signal or signal.action not in ("BUY", "SELL"):
                        continue

                    # Добавляем в буфер для fusion-анализа
                    state.signal_buffer.update(sid, signal, strat.timeframe)

                    # Адаптивный SL/TP по ATR (опционально перезаписывает)
                    if state.use_adaptive_sl:
                        try:
                            import pandas_ta as ta
                            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
                            atr = float(atr_series.iloc[-1])
                            if atr > 0:
                                adaptive = adaptive_sl_tp(
                                    entry=signal.entry_price, atr=atr,
                                    side=signal.action,
                                    atr_multiplier_sl=1.5,
                                    rr_target=strat.take_profit_pct / strat.stop_loss_pct,
                                )
                                signal.stop_loss = adaptive["stop_loss"]
                                signal.take_profit = adaptive["take_profit"]
                        except Exception as e:
                            logger.debug(f"Adaptive SL skip: {e}")

                    # Проверка риск-менеджера
                    check = state.risk_manager.can_open_trade(sid, balance)
                    if not check["allowed"]:
                        logger.info(f"{sid}: ❌ {check['reason']}")
                        continue

                    # Boost-режим: дополнительные проверки фазы
                    if state.boost.is_active:
                        boost_check = state.boost.can_open_trade(balance)
                        if not boost_check["allowed"]:
                            logger.info(f"{sid}: 🚫 Boost: {boost_check['reason']}")
                            continue

                    # Проверка плеча и notional экспозиции
                    lev_check = state.risk_manager.check_leverage(sid, strat.leverage, balance)
                    if not lev_check["allowed"]:
                        logger.info(f"{sid}: ❌ {lev_check['reason']}")
                        continue
                    effective_leverage = lev_check["effective_leverage"]

                    # Корреляция
                    open_positions = [
                        {"symbol": s.symbol, "side": s.current_position["side"]}
                        for s in state.strategies.values() if s.current_position
                    ]
                    corr_check = state.correlation.can_open(strat.symbol, signal.action, open_positions)
                    if not corr_check["allowed"]:
                        logger.info(f"{sid}: ❌ {corr_check['reason']}")
                        continue

                    # AI quick check
                    ai_score = None
                    if state.ai.enabled:
                        ai_result = state.ai.quick_signal_check({
                            "strategy_name": strat.NAME,
                            "symbol": strat.symbol,
                            "action": signal.action,
                            "entry_price": signal.entry_price,
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "filters_passed": signal.filters_passed,
                            "market_context": (
                                f"Sentiment={sentiment_features.get('sentiment_score', 0):+.2f}, "
                                f"News={sentiment_features.get('news_count_24h', 0)}"
                            ),
                        })
                        ai_score = ai_result.get("score", 5)
                        if ai_score < 5:
                            logger.info(f"{sid}: AI отверг (score={ai_score})")
                            continue

                    # ============== ML ФИЛЬТР v2 (с sentiment + orderbook) ==============
                    ml_prediction = None
                    snapshot_id   = None   # явная инициализация — убираем 'in locals()' антипаттерн
                    features = {}
                    if state.ml_enabled:
                        # Order book
                        orderbook = state.bybit.get_orderbook(strat.symbol, limit=25)
                        # Market meta (funding, OI)
                        market_meta = state.bybit.get_market_meta(strat.symbol)
                        # Рыночный режим уже вычислен выше (current_regime_id)
                        regime_id = current_regime_id

                        # Извлечение всех 60+ фич
                        features = state.feature_extractor.extract(
                            df=df,
                            signal_data={
                                "action": signal.action,
                                "confidence": signal.confidence,
                                "entry_price": signal.entry_price,
                                "stop_loss": signal.stop_loss,
                                "take_profit": signal.take_profit,
                            },
                            strategy_stats=strat.get_rolling_stats(),
                            sentiment_data=sentiment_features,
                            orderbook_data=orderbook,
                            market_meta=market_meta,
                            regime_id=regime_id,
                        )

                        # Anomaly detection
                        if state.anomaly_detector.model:
                            anom_score = state.anomaly_detector.score(features)
                            features["anomaly_score"] = anom_score
                            if state.anomaly_detector.is_anomaly(features, threshold=-0.65):
                                logger.warning(f"{sid}: 🚨 АНОМАЛИЯ (score={anom_score:.2f}) — пропускаем")
                                await broadcast_log(f"🚨 {sid} аномальные условия рынка, skip", "warn")
                                continue

                        # ML prediction
                        ml_prediction = state.ml_predictor.predict(sid, features)

                        # Сохраняем snapshot
                        snapshot_id = state.ml_store.save_signal_snapshot({
                            "timestamp": datetime.utcnow().isoformat(),
                            "strategy_id": sid,
                            "symbol": signal.symbol,
                            "timeframe": strat.timeframe,
                            "action": signal.action,
                            "entry_price": signal.entry_price,
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "features": features,
                            "ml_prediction": ml_prediction.get("probability"),
                            "ml_confidence": ml_prediction.get("probability"),
                            "ml_model_version": ml_prediction.get("model_version"),
                            "trade_taken": False,
                        })

                        # Решение
                        if ml_prediction["available"] and state.ml_filter_mode == "strict":
                            if not ml_prediction["should_take"]:
                                logger.info(
                                    f"{sid}: ❌ ML отверг (P={ml_prediction['probability']:.2f})"
                                )
                                await broadcast_log(
                                    f"🤖 {sid} ML отверг сигнал (P={ml_prediction['probability']:.2f})",
                                    "warn",
                                )
                                continue
                        elif ml_prediction["available"]:
                            emoji = "✅" if ml_prediction["should_take"] else "⚠️"
                            await broadcast_log(
                                f"{emoji} {sid} ML P(win)={ml_prediction['probability']:.2f} "
                                f"| Sentiment={sentiment_features.get('sentiment_score', 0):+.2f}",
                                "info",
                            )

                    # Boost-режим: переопределить SL/TP и leverage
                    boost_params = state.boost.get_risk_params()
                    if boost_params:
                        signal.stop_loss, signal.take_profit = state.boost.apply_sl_tp(
                            entry=signal.entry_price,
                            side=signal.action,
                            original_sl=signal.stop_loss,
                            original_tp=signal.take_profit,
                        )
                        strat.leverage = boost_params["leverage"]

                    # ============== Position Sizing ==============
                    if state.boost.is_active:
                        qty = state.boost.calculate_qty(balance, signal.entry_price, strat.leverage)
                    elif state.use_kelly and ml_prediction and ml_prediction.get("available"):
                        kelly_res = kelly_position_size(
                            balance=balance,
                            entry=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            win_probability=ml_prediction["probability"],
                            side=signal.action,
                            kelly_fraction=0.25,
                            max_risk_pct=state.risk_manager.risk_per_trade_pct * 2,
                        )
                        qty = kelly_res["qty"]
                        if qty <= 0:
                            logger.info(f"{sid}: Kelly = 0 (нет преимущества)")
                            continue
                        logger.info(f"{sid}: 📐 Kelly size: qty={qty}, risk={kelly_res['risk_pct']}%")
                    else:
                        qty = state.risk_manager.calculate_position_size(
                            balance=balance,
                            entry_price=signal.entry_price,
                            stop_loss_price=signal.stop_loss,
                            leverage=strat.leverage,
                        )

                    # ============== Открытие позиции ==============
                    if state.paper_mode:
                        state.paper.open_position(signal, sid, qty, strat.leverage)
                    else:
                        result = state.bybit.place_order(
                            symbol=signal.symbol,
                            side="Buy" if signal.action == "BUY" else "Sell",
                            qty=qty,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            leverage=effective_leverage,
                        )
                        if result["success"]:
                            strat.register_position(
                                "Buy" if signal.action == "BUY" else "Sell",
                                signal.entry_price, signal.stop_loss, signal.take_profit,
                            )
                            strat.current_position["qty"] = qty
                            notional = qty * signal.entry_price
                            state.risk_manager.register_position_open(sid, notional)

                            trade_id = state.journal.log_trade({
                                "strategy_id": sid,
                                "strategy_name": strat.NAME,
                                "symbol": signal.symbol,
                                "side": signal.action,
                                "entry_price": signal.entry_price,
                                "qty": qty,
                                "leverage": strat.leverage,
                                "stop_loss": signal.stop_loss,
                                "take_profit": signal.take_profit,
                                "filters_passed": signal.filters_passed,
                                "ai_score": ai_score,
                                "opened_at": datetime.utcnow().isoformat(),
                            })

                            if state.ml_enabled and snapshot_id:
                                state.snapshot_signal_id_map[signal.symbol] = snapshot_id
                                with state.db_pool.cursor() as c:
                                    c.execute(
                                        state.db_pool.adapt(
                                            "UPDATE signal_snapshots SET trade_taken=1, trade_id=? WHERE id=?"
                                        ),
                                        (trade_id, snapshot_id),
                                    )

                            # Fire-and-forget Telegram
                            asyncio.create_task(state.telegram.notify_trade_open(
                                sid, signal.symbol, signal.action,
                                signal.entry_price, signal.stop_loss, signal.take_profit,
                                signal.reason,
                            ))
                            await broadcast_log(f"🟢 {sid} {signal.action} {signal.symbol} @ {signal.entry_price}")

                # ============================================================
                # FUSION PASS — объединение согласных стратегий
                # ============================================================
                try:
                    fused = state.strategy_fusion.evaluate(
                        state.signal_buffer,
                        regime=current_regime_name,
                    )
                    if fused:
                        await _execute_fusion_signal(fused, balance, sentiment_features)
                except Exception as _fe:
                    logger.error(f"Fusion pass error: {_fe}")

                # Обновление корреляций
                if klines_data and len(klines_data) >= 2:
                    state.correlation.calculate_correlation(klines_data)

                # ============== Детекция закрытия позиций ==============
                if state.bybit and not state.paper_mode:
                    try:
                        real_positions = state.bybit.get_positions()
                        real_symbols = {p["symbol"] for p in real_positions}

                        for sid, strat in state.strategies.items():
                            if not strat.current_position:
                                continue
                            if strat.symbol not in real_symbols:
                                ticker = state.bybit.get_ticker(strat.symbol)
                                if not ticker:
                                    continue
                                exit_price = ticker["price"]
                                pos = strat.current_position
                                entry = pos["entry"]
                                side = pos["side"]
                                sl = pos["sl"]
                                tp = pos["tp"]
                                qty = pos.get("qty", 1.0)

                                # Корректный PnL
                                close_result = strat.close_position(exit_price, qty=qty)
                                pnl_usd = close_result["pnl_usd"]
                                r_multiple = close_result["r_multiple"]

                                state.risk_manager.register_trade_result(sid, pnl_usd)
                                state.risk_manager.register_position_close(sid)

                                exit_reason = "TP" if pnl_usd > 0 else "SL"

                                # ML labelling
                                if state.ml_enabled and strat.symbol in state.snapshot_signal_id_map:
                                    snapshot_id = state.snapshot_signal_id_map.pop(strat.symbol)
                                    outcome = "win" if pnl_usd > 0 else "loss"
                                    state.ml_store.update_signal_outcome(
                                        snapshot_id=snapshot_id, outcome=outcome,
                                        pnl_r=r_multiple, exit_reason=exit_reason,
                                        duration_min=0,
                                    )
                                    # Drift monitor
                                    with state.db_pool.connection() as conn:
                                        row = conn.execute(
                                            "SELECT ml_prediction FROM signal_snapshots WHERE id = ?",
                                            (snapshot_id,),
                                        ).fetchone()
                                        if row and row["ml_prediction"] is not None:
                                            state.drift_monitor.record(
                                                sid, row["ml_prediction"], 1 if outcome == "win" else 0,
                                            )
                                            # Проверка drift
                                            drift = state.drift_monitor.should_disable_model(sid)
                                            if drift["disable"]:
                                                logger.critical(f"🚨 DRIFT detected {sid}: {drift['reason']}")
                                                # Деактивируем модель
                                                state.ml_predictor.models.pop(sid, None)
                                                asyncio.create_task(state.telegram.send(
                                                    f"🚨 <b>ML DRIFT</b> {sid}\n{drift['reason']}\nМодель отключена"
                                                ))

                                    await broadcast_log(
                                        f"🎯 {sid} → {outcome.upper()} R={r_multiple:+.2f}",
                                        "ok" if outcome == "win" else "warn",
                                    )

                                asyncio.create_task(state.telegram.notify_trade_close(
                                    sid, strat.symbol, pnl_usd, exit_reason
                                ))
                                # Boost: регистрируем результат, обновляем фазу
                                new_balance = state.bybit.get_balance("USDT")
                                state.boost.register_trade(pnl_usd, new_balance)

                                # Проверяем нужно ли переобучение ML
                                if state.ml_enabled and ML_AVAILABLE and state.continuous_trainer:
                                    asyncio.create_task(
                                        state.continuous_trainer.maybe_retrain(list(state.strategies.keys()))
                                    )
                                await broadcast_log(
                                    f"{'✅' if pnl_usd > 0 else '❌'} {sid} CLOSED {strat.symbol} → {pnl_usd:+.2f} USDT",
                                    "ok" if pnl_usd > 0 else "warn",
                                )
                    except Exception as e:
                        logger.error(f"Position close detection: {e}")

                # Авто-отключение
                for sid, strat in state.strategies.items():
                    if strat.auto_disabled or not strat.enabled:
                        continue
                    if strat.consecutive_losses >= state.risk_manager.max_consecutive_losses:
                        strat.auto_disabled = True
                        asyncio.create_task(state.telegram.notify_strategy_disabled(
                            sid, strat.NAME, f"Серия {strat.consecutive_losses} убытков"
                        ))
                        await broadcast_log(f"⚠️ {sid} АВТО-ОТКЛЮЧЕНА")

                await broadcast_state()
                # Скальпинг: цикл каждые 5 сек; обычный режим: 10 сек
                await asyncio.sleep(5 if state.scalp_active else 10)

            except Exception as e:
                logger.exception(f"Ошибка в торговом цикле: {e}")
                await asyncio.sleep(5)

        logger.info("🛑 Торговый цикл остановлен")

# ============================================================
# WebSocket broadcasting
# ============================================================
async def broadcast_state():
    """Отправка состояния всем подключённым UI."""
    if not state.ws_clients:
        return
    payload = {
        "type": "state_update",
        "strategies": [s.to_dict() for s in state.strategies.values()],
        "risk_status": state.risk_manager.get_status(),
        "bot_running": state.bot_running,
        "paper_mode": state.paper_mode,
        "paper_stats": state.paper.get_stats() if state.paper_mode else None,
        "correlations": {f"{k[0]}-{k[1]}": v for k, v in state.correlation.correlations.items()},
        "ml_status": {
            "enabled": state.ml_enabled,
            "filter_mode": state.ml_filter_mode,
            "models_loaded": list(state.ml_predictor.models.keys()),
            "data_stats": state.ml_store.get_ml_stats() if state.ml_enabled else None,
        },
        "sentiment": state.news_manager.get_current_sentiment() if state.news_enabled else None,
        "drift_status": state.drift_monitor.get_all_metrics(),
        "boost_status": state.boost.get_status() if state.boost.is_active else {"active": False},
        "timestamp": datetime.utcnow().isoformat(),
    }
    disconnected = []
    for ws in state.ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        state.ws_clients.remove(ws)


async def broadcast_log(message: str, level: str = "info"):
    """Отправка лога в UI."""
    payload = {
        "type": "log",
        "message": message,
        "level": level,
        "timestamp": datetime.utcnow().isoformat(),
    }
    for ws in state.ws_clients[:]:
        try:
            await ws.send_json(payload)
        except Exception:
            pass


# ============================================================
# FastAPI приложение
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_strategies()

    # Загрузка anomaly detector если есть
    from pathlib import Path as _P
    if _P("data/models/anomaly_detector.pkl").exists():
        if state.anomaly_detector.load("data/models/anomaly_detector.pkl"):
            logger.info("🛡 Anomaly detector загружен")

    # News loop стартует автоматически (даже если бот не запущен — sentiment-данные нужны заранее)
    if state.news_enabled:
        state.news_loop_task = asyncio.create_task(
            background_news_loop(state.news_manager, interval_min=15)
        )
        logger.info("📡 News loop запущен")

    # Auto-train loop
    if state.ml_enabled and ML_AVAILABLE:
        state.auto_train_task = asyncio.create_task(auto_train_loop())
        logger.info("🔄 Auto-train loop запущен")

    logger.info("✅ Бэкенд готов")
    yield
    state.bot_running = False
    if state.trading_loop_task:
        state.trading_loop_task.cancel()
    if state.news_loop_task:
        state.news_loop_task.cancel()
    if state.auto_train_task:
        state.auto_train_task.cancel()
    await state.news_manager.close()
    logger.info("🛑 Завершение работы")


app = FastAPI(title="Bybit Autotrader Pro", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Pydantic модели ==========
class APIConnectRequest(BaseModel):
    api_key: str
    api_secret: str
    testnet: bool = True


class BotControlRequest(BaseModel):
    action: str  # "start" | "stop" | "paper_on" | "paper_off"


class StrategyToggleRequest(BaseModel):
    strategy_id: str
    enabled: bool


class StrategyUpdateRequest(BaseModel):
    strategy_id: str
    symbol: Optional[str] = None
    leverage: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    breakeven_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None


class BacktestRequest(BaseModel):
    strategy_id: str
    symbol: str = "BTCUSDT"
    interval: str = "60"
    days: int = 30
    initial_balance: float = 1000.0


# ========== ENDPOINTS ==========
@app.get("/")
async def root():
    return {"status": "ok", "bot_running": state.bot_running, "strategies": len(state.strategies)}


@app.post("/api/connect")
async def connect(req: APIConnectRequest):
    try:
        state.bybit = BybitClient(req.api_key, req.api_secret, req.testnet)
        balance = state.bybit.get_balance()
        return {"success": True, "balance": balance, "testnet": req.testnet}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/bot/control")
async def bot_control(req: BotControlRequest):
    if req.action == "start":
        if not state.bybit and not state.paper_mode:
            raise HTTPException(400, "Сначала подключите Bybit API или включите Paper Mode")
        if state.trading_loop_task and not state.trading_loop_task.done():
            return {"success": True, "running": True, "message": "Уже запущен"}
        state.bot_running = True
        state.trading_loop_task = asyncio.create_task(trading_loop())
        # Запускаем news loop если ещё не запущен
        if state.news_enabled and (not state.news_loop_task or state.news_loop_task.done()):
            state.news_loop_task = asyncio.create_task(
                background_news_loop(state.news_manager, interval_min=15)
            )
        return {"success": True, "running": True}
    elif req.action == "stop":
        state.bot_running = False
        if state.trading_loop_task:
            try:
                await asyncio.wait_for(state.trading_loop_task, timeout=15)
            except asyncio.TimeoutError:
                state.trading_loop_task.cancel()
        return {"success": True, "running": False}
    elif req.action == "paper_on":
        state.paper_mode = True
        return {"success": True, "paper_mode": True}
    elif req.action == "paper_off":
        state.paper_mode = False
        return {"success": True, "paper_mode": False}
    elif req.action == "kill_switch_reset":
        state.risk_manager.manual_reset_kill_switch()
        return {"success": True}
    raise HTTPException(400, "Unknown action")


@app.get("/api/strategies")
async def list_strategies():
    return [s.to_dict() for s in state.strategies.values()]


@app.post("/api/strategies/toggle")
async def toggle_strategy(req: StrategyToggleRequest):
    strat = state.strategies.get(req.strategy_id)
    if not strat:
        raise HTTPException(404, "Strategy not found")
    strat.enabled = req.enabled
    if req.enabled:
        strat.auto_disabled = False
    return strat.to_dict()


@app.post("/api/strategies/update")
async def update_strategy(req: StrategyUpdateRequest):
    strat = state.strategies.get(req.strategy_id)
    if not strat:
        raise HTTPException(404, "Strategy not found")
    if req.symbol is not None:
        strat.symbol = req.symbol
    if req.leverage is not None:
        strat.leverage = req.leverage
    if req.stop_loss_pct is not None:
        strat.stop_loss_pct = req.stop_loss_pct
    if req.take_profit_pct is not None:
        strat.take_profit_pct = req.take_profit_pct
    if req.breakeven_pct is not None:
        strat.breakeven_pct = req.breakeven_pct
    if req.trailing_stop_pct is not None:
        strat.trailing_stop_pct = req.trailing_stop_pct
    return strat.to_dict()


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    if not state.bybit:
        raise HTTPException(400, "Сначала подключите Bybit API для получения данных")
    cls = ALL_STRATEGIES.get(req.strategy_id)
    if not cls:
        raise HTTPException(404, "Strategy not found")
    df = state.bybit.get_klines(req.symbol, req.interval, limit=min(1000, req.days * 24))
    if df.empty:
        raise HTTPException(400, "Не удалось получить данные")
    bt = Backtester(initial_balance=req.initial_balance)
    result = bt.run(cls, df, req.symbol)
    return {
        "metrics": result.calculate_metrics(),
        "equity_curve": result.equity_curve,
        "trades_count": len(result.trades),
        "first_10_trades": result.trades[:10],
        "last_10_trades": result.trades[-10:],
    }


@app.get("/api/journal/stats")
async def journal_stats():
    return state.journal.get_stats_by_strategy()


@app.get("/api/journal/trades")
async def journal_trades(strategy_id: Optional[str] = None, limit: int = 100):
    return state.journal.get_trades(strategy_id=strategy_id, limit=limit)


@app.get("/api/journal/export/csv")
async def export_csv():
    path = state.journal.export_to_csv()
    return FileResponse(path, filename="trades.csv")


@app.get("/api/journal/export/excel")
async def export_excel():
    path = state.journal.export_to_excel()
    return FileResponse(path, filename="trades.xlsx")


@app.get("/api/journal/heatmap")
async def heatmap():
    return state.journal.get_heatmap_data()


@app.get("/api/ai/analyze/{strategy_id}")
async def ai_analyze(strategy_id: str):
    strat = state.strategies.get(strategy_id)
    if not strat:
        raise HTTPException(404)
    trades = state.journal.get_trades(strategy_id=strategy_id, limit=100)
    stats = next(
        (s for s in state.journal.get_stats_by_strategy() if s["strategy_id"] == strategy_id),
        {},
    )
    result = state.ai.analyze_strategy_performance({
        "id": strategy_id,
        "name": strat.NAME,
        "symbol": strat.symbol,
        "trades": trades,
        "stats": stats,
    })
    return result


@app.get("/api/risk/status")
async def risk_status():
    return state.risk_manager.get_status()


@app.get("/api/correlations")
async def correlations():
    return {f"{k[0]}-{k[1]}": v for k, v in state.correlation.correlations.items()}


@app.get("/api/paper/stats")
async def paper_stats():
    return state.paper.get_stats()


@app.post("/api/bot/emergency_close")
async def emergency_close():
    """Принудительное закрытие ВСЕХ позиций + стоп бота."""
    if not state.bybit and not state.paper_mode:
        raise HTTPException(400, "Нет подключения к Bybit")

    closed = []
    errors = []

    if state.paper_mode:
        for sid, strat in state.strategies.items():
            if strat.current_position:
                try:
                    ticker = state.paper.positions.get(sid, {})
                    exit_price = ticker.get("entry", 0) if ticker else 0
                    strat.current_position = None
                    closed.append(sid)
                except Exception as e:
                    errors.append(f"{sid}: {e}")
    else:
        positions = state.bybit.get_positions() if state.bybit else []
        for pos in positions:
            symbol = pos.get("symbol", "")
            side = "Sell" if pos.get("side") == "Buy" else "Buy"
            qty = pos.get("size", "0")
            try:
                result = state.bybit.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=qty,
                    reduceOnly=True,
                )
                if result.get("retCode") == 0:
                    closed.append(symbol)
                else:
                    errors.append(f"{symbol}: {result.get('retMsg')}")
            except Exception as e:
                errors.append(f"{symbol}: {e}")

        # Сбрасываем внутреннее состояние стратегий
        for strat in state.strategies.values():
            strat.current_position = None

    state.bot_running = False
    state.risk_manager.open_positions_count = 0
    state.risk_manager._open_notional.clear()

    await broadcast_log(f"🚨 EMERGENCY CLOSE: закрыто {len(closed)} позиций", "warn")
    asyncio.create_task(state.telegram.send(
        f"🚨 <b>EMERGENCY CLOSE</b>\nЗакрыто: {len(closed)}\nОшибки: {len(errors)}"
    ))

    return {"closed": closed, "errors": errors, "bot_stopped": True}


# ============================================================
# ML ENDPOINTS
# ============================================================
@app.get("/api/ml/status")
async def ml_status():
    """Полный статус ML-системы."""
    return {
        "enabled": state.ml_enabled,
        "filter_mode": state.ml_filter_mode,
        "ml_available": ML_AVAILABLE,
        "optuna_available": OPTUNA_AVAILABLE,
        "data_stats": state.ml_store.get_ml_stats(),
        "predictor_status": state.ml_predictor.get_status(),
    }


class MLConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    filter_mode: Optional[str] = None  # "advisory" | "strict" | "off"
    default_threshold: Optional[float] = None


@app.post("/api/ml/config")
async def ml_config(req: MLConfigRequest):
    if req.enabled is not None:
        state.ml_enabled = req.enabled
    if req.filter_mode in ("advisory", "strict", "off"):
        state.ml_filter_mode = req.filter_mode
    if req.default_threshold is not None:
        state.ml_predictor.default_threshold = max(0, min(1, req.default_threshold))
    return {
        "success": True,
        "enabled": state.ml_enabled,
        "filter_mode": state.ml_filter_mode,
        "default_threshold": state.ml_predictor.default_threshold,
    }


class MLThresholdRequest(BaseModel):
    strategy_id: str
    threshold: float


@app.post("/api/ml/threshold")
async def ml_set_threshold(req: MLThresholdRequest):
    state.ml_predictor.set_threshold(req.strategy_id, req.threshold)
    return {"success": True, "strategy_id": req.strategy_id, "threshold": req.threshold}


class MLTrainRequest(BaseModel):
    strategy_id: Optional[str] = None  # None = все стратегии
    min_samples: int = 100


@app.post("/api/ml/train")
async def ml_train(req: MLTrainRequest):
    """Запуск обучения модели (синхронно, может занять время)."""
    if not ML_AVAILABLE:
        raise HTTPException(400, "scikit-learn/xgboost не установлены")

    feature_columns = state.feature_extractor.FEATURE_NAMES

    if req.strategy_id:
        # Обучаем одну стратегию
        training_data = state.ml_store.get_training_data(
            strategy_id=req.strategy_id,
            min_samples=req.min_samples,
            only_taken_trades=False,  # учим на всех сигналах, включая не взятые
        )
        if training_data.empty:
            return {
                "success": False,
                "error": f"Мало данных для {req.strategy_id}. Нужно ≥{req.min_samples} размеченных сигналов."
            }

        result = state.ml_trainer.train(training_data, req.strategy_id, feature_columns)
        if result.get("success"):
            state.ml_store.register_model(
                version=result["version"],
                strategy_id=req.strategy_id,
                model_type="XGBoost",
                metrics=result["metrics"],
                hyperparams=result["hyperparams"],
                feature_importance=result["feature_importance"],
                file_path=result["file_path"],
                samples_count=result["samples"],
                set_active=True,
            )
            state.ml_predictor.load_model(req.strategy_id, result["file_path"])
        return result

    # Обучаем все стратегии
    results = {}
    for sid in state.strategies.keys():
        data = state.ml_store.get_training_data(strategy_id=sid, min_samples=req.min_samples)
        if data.empty or len(data) < req.min_samples:
            results[sid] = {"success": False, "error": f"Мало данных ({len(data)})"}
            continue
        r = state.ml_trainer.train(data, sid, feature_columns)
        results[sid] = r
        if r.get("success"):
            state.ml_store.register_model(
                version=r["version"], strategy_id=sid, model_type="XGBoost",
                metrics=r["metrics"], hyperparams=r["hyperparams"],
                feature_importance=r["feature_importance"], file_path=r["file_path"],
                samples_count=r["samples"], set_active=True,
            )
            state.ml_predictor.load_model(sid, r["file_path"])
    return results


@app.get("/api/ml/models")
async def ml_list_models(strategy_id: Optional[str] = None):
    return state.ml_store.list_models(strategy_id)


@app.get("/api/ml/feature-importance/{strategy_id}")
async def ml_feature_importance(strategy_id: str):
    active = state.ml_store.get_active_model(strategy_id)
    if not active:
        return {"available": False}
    import json
    try:
        importance = json.loads(active.get("feature_importance_json", "{}"))
        return {
            "available": True,
            "strategy_id": strategy_id,
            "version": active["version"],
            "metrics": {
                "accuracy": active.get("accuracy"),
                "precision": active.get("precision_val"),
                "recall": active.get("recall_val"),
                "f1": active.get("f1_score"),
                "roc_auc": active.get("roc_auc"),
            },
            "samples_count": active.get("samples_count"),
            "top_features": list(importance.items())[:15],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


class MLOptimizeRequest(BaseModel):
    strategy_id: str
    symbol: str = "BTCUSDT"
    interval: str = "60"
    days: int = 60
    n_trials: int = 50
    objective: str = "sharpe"


@app.post("/api/ml/optimize")
async def ml_optimize(req: MLOptimizeRequest):
    """Bayesian оптимизация параметров стратегии."""
    if not OPTUNA_AVAILABLE:
        raise HTTPException(400, "Optuna не установлен (pip install optuna)")
    if not state.bybit:
        raise HTTPException(400, "Сначала подключите Bybit для получения данных")

    cls = ALL_STRATEGIES.get(req.strategy_id)
    if not cls:
        raise HTTPException(404, "Strategy not found")

    df = state.bybit.get_klines(req.symbol, req.interval, limit=min(1000, req.days * 24))
    if df.empty:
        raise HTTPException(400, "Не удалось получить данные")

    result = state.ml_optimizer.optimize(
        strategy_class=cls, df=df, symbol=req.symbol,
        n_trials=req.n_trials, objective=req.objective,
    )
    return result


class RegimeFitRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "60"
    days: int = 90
    n_clusters: int = 4


@app.post("/api/ml/regime/fit")
async def ml_regime_fit(req: RegimeFitRequest):
    """Обучить классификатор рыночных режимов."""
    if not state.bybit:
        raise HTTPException(400, "Сначала подключите Bybit")
    df = state.bybit.get_klines(req.symbol, req.interval, limit=min(1000, req.days * 24))
    if df.empty:
        raise HTTPException(400, "Нет данных")
    state.regime_classifier = RegimeClassifier(n_clusters=req.n_clusters)
    return state.regime_classifier.fit(df)


@app.get("/api/ml/regime/current/{symbol}")
async def ml_regime_current(symbol: str):
    """Определить текущий рыночный режим."""
    if not state.bybit or not state.regime_classifier.model:
        return {"available": False, "reason": "Сначала обучите классификатор"}
    df = state.bybit.get_klines(symbol, "60", limit=100)
    if df.empty:
        return {"available": False}
    res = state.regime_classifier.predict(df)
    return {"available": True, "result": res} if res else {"available": False}


@app.get("/api/ml/signals/recent")
async def ml_recent_signals(strategy_id: Optional[str] = None, limit: int = 50):
    """Последние сигналы со снимками фич."""
    query = """
        SELECT id, timestamp, strategy_id, symbol, action,
               entry_price, ml_prediction, ml_confidence,
               trade_taken, outcome, pnl_r, exit_reason
        FROM signal_snapshots
    """
    params: list = []
    if strategy_id:
        query += " WHERE strategy_id = ?"
        params.append(strategy_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    pool = state.ml_store.pool
    with pool.connection() as conn:
        if pool.is_mysql:
            with conn.cursor() as c:
                c.execute(pool.adapt(query), params)
                return list(c.fetchall())
        else:
            import sqlite3
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(pool.adapt(query), params).fetchall()]


# ============================================================
# NEWS & SENTIMENT ENDPOINTS
# ============================================================
@app.post("/api/news/refresh")
async def news_refresh():
    """Принудительный сбор новостей."""
    if not state.news_enabled:
        raise HTTPException(400, "News отключены в .env")
    result = await state.news_manager.collect_all()
    return result


@app.get("/api/news/sentiment")
async def news_sentiment():
    """Текущий sentiment."""
    return {
        "current": state.news_manager.get_current_sentiment(),
        "features": state.news_manager.get_sentiment_features(),
        "deep_analysis": state.news_manager.cached_deep,
    }


@app.get("/api/news/items")
async def news_items(limit: int = 50, source: Optional[str] = None):
    """Список последних новостей."""
    return state.news_manager.get_recent_news(limit=limit, source_filter=source)


@app.get("/api/news/history")
async def news_history(hours: int = 24):
    """История sentiment за последние N часов."""
    return state.news_manager.get_sentiment_history(hours=hours)


@app.post("/api/news/deep-analysis")
async def news_deep_analysis():
    """Запуск AI-анализа через Claude."""
    return await state.news_manager.deep_analysis()


# ============================================================
# DRIFT MONITOR ENDPOINTS
# ============================================================
@app.get("/api/ml/drift")
async def ml_drift_status(strategy_id: Optional[str] = None):
    """Статус concept drift."""
    if strategy_id:
        return state.drift_monitor.calculate_metrics(strategy_id) or {"available": False}
    return state.drift_monitor.get_all_metrics()


class ThresholdAutoTuneRequest(BaseModel):
    strategy_id: str
    objective: str = "f1_weighted_by_count"


@app.post("/api/ml/threshold/auto-tune")
async def ml_threshold_auto_tune(req: ThresholdAutoTuneRequest):
    """Авто-подбор optimal threshold per strategy."""
    with state.db_pool.connection() as conn:
        rows = conn.execute("""
            SELECT ml_prediction, outcome FROM signal_snapshots
            WHERE strategy_id = ? AND ml_prediction IS NOT NULL
            AND outcome IS NOT NULL
        """, (req.strategy_id,)).fetchall()
    if len(rows) < 30:
        return {"success": False, "error": f"Мало данных: {len(rows)}"}
    preds = [r["ml_prediction"] for r in rows]
    outs = [1 if r["outcome"] == "win" else 0 for r in rows]
    result = ThresholdOptimizer.find_optimal_threshold(preds, outs, objective=req.objective)
    # Применяем
    state.ml_predictor.set_threshold(req.strategy_id, result["threshold"])
    result["success"] = True
    result["applied"] = True
    return result


# ============================================================
# ENSEMBLE & ANOMALY ENDPOINTS
# ============================================================
class EnsembleTrainRequest(BaseModel):
    strategy_id: str
    min_samples: int = 100


@app.post("/api/ml/train-ensemble")
async def ml_train_ensemble(req: EnsembleTrainRequest):
    """Обучить ансамблевую модель."""
    training_data = state.ml_store.get_training_data(
        strategy_id=req.strategy_id, min_samples=req.min_samples,
    )
    if training_data.empty:
        return {"success": False, "error": "Мало данных"}

    feature_columns = state.feature_extractor.FEATURE_NAMES
    result = state.ml_ensemble_trainer.train_ensemble(
        training_data, req.strategy_id, feature_columns,
    )
    if result.get("success"):
        state.ml_store.register_model(
            version=result["version"], strategy_id=req.strategy_id,
            model_type=result["model_type"],
            metrics=result["metrics"],
            hyperparams={"models": result["models_used"]},
            feature_importance=result["feature_importance"],
            file_path=result["file_path"],
            samples_count=result["samples"],
            set_active=True,
        )
        state.ml_predictor.load_model(req.strategy_id, result["file_path"])
    return result


@app.get("/api/ml/auto-train/status")
async def ml_auto_train_status():
    """Статус системы авто-переобучения."""
    if not state.continuous_trainer:
        return {"available": False}
    return {"available": True, **state.continuous_trainer.get_status()}


@app.post("/api/ml/auto-train/trigger")
async def ml_auto_train_trigger():
    """Принудительный запуск авто-переобучения (не дожидаясь таймера)."""
    if not state.continuous_trainer or not ML_AVAILABLE:
        raise HTTPException(400, "ContinuousTrainer недоступен")
    all_ids = list(state.strategies.keys()) + ["FUSION"]
    results = await state.continuous_trainer.maybe_retrain(all_ids)
    return {"triggered": True, "results": results}


@app.get("/api/fusion/status")
async def fusion_status():
    """Статус Strategy Fusion Engine: буфер сигналов, последние fusion-сделки."""
    fresh = state.signal_buffer.fresh()
    return {
        "buffer_size":   len(fresh),
        "fresh_signals": {
            sid: {
                "action":     sig.action,
                "symbol":     sig.symbol,
                "confidence": sig.confidence,
            }
            for sid, sig in fresh.items()
        },
        "min_confluence":   StrategyFusion.MIN_CONFLUENCE,
        "min_fusion_score": StrategyFusion.MIN_FUSION_SCORE,
        "cooldown_minutes": StrategyFusion.COOLDOWN_MINUTES,
    }


# ============================================================
# BOOST MODE ENDPOINTS
# ============================================================
class BoostStartRequest(BaseModel):
    initial_balance: float
    target_balance: float
    deadline_days: int = 21
    mode: str = "moderate"   # "safe" | "moderate" | "aggressive"


@app.post("/api/boost/start")
async def boost_start(req: BoostStartRequest):
    """Запустить сессию разгона депозита."""
    if req.initial_balance <= 0 or req.target_balance <= req.initial_balance:
        raise HTTPException(400, "Некорректные параметры: target должен быть > initial")
    if req.mode not in ("safe", "moderate", "aggressive"):
        raise HTTPException(400, "mode должен быть: safe | moderate | aggressive")
    result = state.boost.start(req.initial_balance, req.target_balance, req.deadline_days, req.mode)
    if result.get("success"):
        scalp_info = ""
        if req.mode == "scalp":
            added = activate_scalp_mode()
            scalp_info = f"\n⚡ Скальп-стратегии: {len(added)} символов"
        asyncio.create_task(state.telegram.send(
            f"🚀 <b>Boost Mode запущен</b> ({req.mode})\n"
            f"${req.initial_balance:.2f} → ${req.target_balance:.2f} за {req.deadline_days} дн.\n"
            f"Требуется: {result['analysis']['required_daily_pct']}%/день{scalp_info}"
        ))
    return result


@app.get("/api/boost/status")
async def boost_status():
    """Текущее состояние boost-сессии."""
    return state.boost.get_status()


@app.post("/api/boost/scalp/activate")
async def boost_scalp_activate():
    """
    Активирует скальпинг-режим: создаёт ScalperPro для 10 символов.
    Цель: 30-50 прибыльных сделок в день.
    """
    added = activate_scalp_mode()
    return {
        "success": True,
        "scalp_active": True,
        "strategies_added": added,
        "total_scalpers": len(added),
        "symbols": SCALP_SYMBOLS,
        "timeframe": "3m",
        "expected_signals_per_day": f"{len(added) * 5}–{len(added) * 8}",
        "expected_profitable_at_60wr": f"{int(len(added) * 5 * 0.60)}–{int(len(added) * 8 * 0.60)}",
    }


@app.post("/api/boost/scalp/deactivate")
async def boost_scalp_deactivate():
    """Деактивирует все скальп-стратегии."""
    deactivated = deactivate_scalp_mode()
    return {"success": True, "scalp_active": False, "deactivated": deactivated}


@app.get("/api/boost/scalp/status")
async def boost_scalp_status():
    """Статус скальп-стратегий: сколько сигналов, сколько сделок."""
    scalpers = {
        sid: {
            "symbol":    strat.symbol,
            "enabled":   strat.enabled,
            "trades":    strat.trades,
            "wins":      strat.wins,
            "losses":    strat.losses,
            "win_rate":  round(strat.wins / strat.trades * 100, 1) if strat.trades else 0,
            "pnl":       round(strat.pnl, 4),
            "position":  bool(strat.current_position),
        }
        for sid, strat in state.strategies.items()
        if sid.startswith("SC_")
    }
    total_trades   = sum(s["trades"] for s in scalpers.values())
    total_wins     = sum(s["wins"]   for s in scalpers.values())
    # Текущий H1 тренд для каждого символа
    h1_trends = {}
    for sym in SCALP_SYMBOLS:
        df_h1 = state.h1_cache.get(sym, (None,))[0]
        h1_trends[sym] = ScalperTrendContext.compute(df_h1, "H1")

    return {
        "scalp_active":     state.scalp_active,
        "scalpers":         scalpers,
        "total_scalpers":   len(scalpers),
        "total_trades":     total_trades,
        "total_wins":       total_wins,
        "overall_win_rate": round(total_wins / total_trades * 100, 1) if total_trades else 0,
        "loop_interval_sec": 5 if state.scalp_active else 10,
        "h1_trends":        h1_trends,
        "mtf_mode":         "H1+15m" if state.scalp_active else "H1-only",
    }


@app.post("/api/boost/stop")
async def boost_stop(reason: str = "Ручная остановка"):
    """Остановить boost-сессию."""
    result = state.boost.stop(reason)
    if result.get("success"):
        if state.scalp_active:
            deactivate_scalp_mode()
        asyncio.create_task(state.telegram.send(
            f"🛑 <b>Boost Mode остановлен</b>\n{reason}"
        ))
    return result


class BoostAnalyzeRequest(BaseModel):
    initial: float = 10.0
    target: float = 100.0
    days: int = 21
    mode: str = "moderate"   # "safe" | "moderate" | "aggressive"


@app.post("/api/boost/analyze")
async def boost_analyze(req: BoostAnalyzeRequest):
    """
    Математический анализ плана разгона с Monte Carlo симуляцией.
    Возвращает сценарии + prob_targets: максимальный баланс при 70%/75%/80% вероятности.
    """
    if req.initial <= 0 or req.target <= req.initial or req.days <= 0:
        raise HTTPException(400, "Некорректные параметры")
    if req.mode not in ("safe", "moderate", "aggressive"):
        raise HTTPException(400, "mode должен быть: safe | moderate | aggressive")
    return BoostCalculator.full_analysis(req.initial, req.target, req.days, req.mode)


@app.post("/api/ml/anomaly/fit")
async def ml_anomaly_fit():
    """Обучить anomaly detector на исторических сигналах."""
    # Берём все собранные signal_snapshots
    with state.db_pool.connection() as conn:
        import pandas as pd, json
        rows = conn.execute("SELECT features_json FROM signal_snapshots LIMIT 5000").fetchall()
    if len(rows) < 100:
        return {"success": False, "error": f"Мало данных: {len(rows)} (нужно ≥100)"}
    features_list = []
    for r in rows:
        try:
            features_list.append(json.loads(r["features_json"]))
        except Exception:
            continue
    df = pd.DataFrame(features_list)
    result = state.anomaly_detector.fit(df, feature_columns=state.feature_extractor.FEATURE_NAMES)
    if result.get("success"):
        state.anomaly_detector.save("data/models/anomaly_detector.pkl")
    return result


# ========== WebSocket ==========
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.ws_clients.append(ws)
    logger.info(f"WS connected. Total: {len(state.ws_clients)}")
    try:
        await broadcast_state()
        while True:
            data = await ws.receive_text()
    except WebSocketDisconnect:
        state.ws_clients.remove(ws)
        logger.info(f"WS disconnected. Total: {len(state.ws_clients)}")


# ============================================================
# ПОЛНЫЙ СТАТУС — один эндпоинт, всё состояние бота
# Удобен для мониторинга снаружи (Claude, скрипты, дашборд)
# ============================================================
@app.get("/api/status/full")
async def full_status():
    """
    Единый дашборд: баланс, позиции, сделки за сегодня,
    статус стратегий, boost, скальп, ML, риск.
    """
    now = datetime.utcnow()

    # ── Баланс ────────────────────────────────────────────────
    balance_info: Dict = {}
    if state.bybit and not state.paper_mode:
        try:
            bal = state.bybit.get_balance("USDT")
            balance_info = {"usdt": round(bal, 4), "source": "bybit"}
        except Exception as e:
            balance_info = {"error": str(e)}
    elif state.paper_mode:
        ps = state.paper.get_stats()
        balance_info = {
            "usdt":       round(ps.get("current_balance", 0), 4),
            "source":     "paper",
            "paper_pnl":  round(ps.get("total_pnl", 0), 4),
            "paper_trades": ps.get("total_trades", 0),
        }
    else:
        balance_info = {"usdt": None, "source": "not_connected"}

    # ── Открытые позиции ──────────────────────────────────────
    open_positions = []
    for sid, strat in state.strategies.items():
        if strat.current_position:
            pos = strat.current_position
            open_positions.append({
                "strategy":  sid,
                "symbol":    strat.symbol,
                "side":      pos.get("side"),
                "entry":     pos.get("entry"),
                "sl":        pos.get("sl"),
                "tp":        pos.get("tp"),
                "qty":       pos.get("qty"),
                "be_moved":  pos.get("be_moved", False),
            })

    # ── Сделки за сегодня ─────────────────────────────────────
    today = now.date().isoformat()
    today_trades = state.journal.get_trades(start_date=today, limit=200)
    today_pnl    = round(sum(t.get("pnl_usd", 0) or 0 for t in today_trades), 4)
    today_wins   = sum(1 for t in today_trades if (t.get("pnl_usd") or 0) > 0)
    today_losses = sum(1 for t in today_trades if (t.get("pnl_usd") or 0) < 0)

    # ── Стратегии ─────────────────────────────────────────────
    strategies_summary = []
    for sid, strat in state.strategies.items():
        strategies_summary.append({
            "id":        sid,
            "name":      strat.NAME,
            "symbol":    strat.symbol,
            "timeframe": strat.timeframe,
            "enabled":   strat.enabled and not strat.auto_disabled,
            "trades":    strat.trades,
            "wins":      strat.wins,
            "wr_pct":    round(strat.wins / strat.trades * 100, 1) if strat.trades else 0,
            "pnl":       round(strat.pnl, 4),
            "position":  bool(strat.current_position),
            "consec_losses": strat.consecutive_losses,
        })

    # ── Последние 5 сделок ────────────────────────────────────
    recent_trades = state.journal.get_trades(limit=5)

    # ── Boost ─────────────────────────────────────────────────
    boost_info = state.boost.get_status()

    # ── Скальп ────────────────────────────────────────────────
    scalp_strats = {
        sid: {
            "symbol":   strat.symbol,
            "enabled":  strat.enabled,
            "trades":   strat.trades,
            "wins":     strat.wins,
            "wr_pct":   round(strat.wins / strat.trades * 100, 1) if strat.trades else 0,
            "position": bool(strat.current_position),
        }
        for sid, strat in state.strategies.items()
        if sid.startswith("SC_")
    }

    # ── ML ────────────────────────────────────────────────────
    ml_info = {
        "enabled":      state.ml_enabled,
        "filter_mode":  state.ml_filter_mode,
        "models_loaded": list(state.ml_predictor.models.keys()),
    }

    return {
        "timestamp":      now.isoformat(),
        "bot_running":    state.bot_running,
        "paper_mode":     state.paper_mode,
        "balance":        balance_info,
        "open_positions": open_positions,
        "open_count":     len(open_positions),
        "today": {
            "date":    today,
            "trades":  len(today_trades),
            "wins":    today_wins,
            "losses":  today_losses,
            "pnl_usd": today_pnl,
            "wr_pct":  round(today_wins / len(today_trades) * 100, 1) if today_trades else 0,
        },
        "recent_trades":  recent_trades,
        "strategies":     strategies_summary,
        "scalp_active":   state.scalp_active,
        "scalpers":       scalp_strats,
        "boost":          boost_info,
        "ml":             ml_info,
        "risk":           state.risk_manager.get_status(),
        "sentiment":      state.news_manager.get_current_sentiment() if state.news_enabled else None,
    }


@app.post("/api/command")
async def bot_command(body: dict):
    """
    Универсальный командный эндпоинт.
    Принимает JSON: {"cmd": "balance|positions|trades|boost_start|scalp_on|stop", ...}
    Удобен для вызова из Claude или внешних скриптов.
    """
    cmd = body.get("cmd", "").lower()

    if cmd == "balance":
        if state.paper_mode:
            return {"balance": state.paper.get_stats().get("current_balance"), "mode": "paper"}
        if state.bybit:
            return {"balance": state.bybit.get_balance("USDT"), "mode": "live"}
        return {"balance": None, "mode": "disconnected"}

    elif cmd == "positions":
        positions = [
            {"strategy": sid, "symbol": strat.symbol, **strat.current_position}
            for sid, strat in state.strategies.items()
            if strat.current_position
        ]
        return {"open_positions": positions, "count": len(positions)}

    elif cmd == "trades":
        limit = int(body.get("limit", 10))
        trades = state.journal.get_trades(limit=limit)
        total_pnl = round(sum(t.get("pnl_usd", 0) or 0 for t in trades), 4)
        return {"trades": trades, "count": len(trades), "total_pnl": total_pnl}

    elif cmd == "pnl_today":
        today = datetime.utcnow().date().isoformat()
        trades = state.journal.get_trades(start_date=today, limit=500)
        wins   = [t for t in trades if (t.get("pnl_usd") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl_usd") or 0) < 0]
        return {
            "date":    today,
            "trades":  len(trades),
            "wins":    len(wins),
            "losses":  len(losses),
            "pnl_usd": round(sum(t.get("pnl_usd", 0) or 0 for t in trades), 4),
            "wr_pct":  round(len(wins) / len(trades) * 100, 1) if trades else 0,
        }

    elif cmd == "scalp_on":
        added = activate_scalp_mode()
        return {"scalp_active": True, "strategies": added}

    elif cmd == "scalp_off":
        disabled = deactivate_scalp_mode()
        return {"scalp_active": False, "disabled": disabled}

    elif cmd == "start":
        if not state.bot_running:
            state.bot_running = True
            state.trading_loop_task = asyncio.create_task(trading_loop())
        return {"bot_running": state.bot_running}

    elif cmd == "stop":
        state.bot_running = False
        return {"bot_running": False}

    elif cmd == "paper_on":
        state.paper_mode = True
        return {"paper_mode": True}

    elif cmd == "paper_off":
        state.paper_mode = False
        return {"paper_mode": False}

    elif cmd == "boost_start":
        result = state.boost.start(
            initial_balance=float(body.get("initial", 10)),
            target_balance=float(body.get("target", 50)),
            deadline_days=int(body.get("days", 21)),
            mode=body.get("mode", "moderate"),
        )
        if result.get("success") and body.get("mode") == "scalp":
            activate_scalp_mode()
        return result

    elif cmd == "boost_stop":
        return state.boost.stop(body.get("reason", "Команда остановки"))

    elif cmd == "status":
        # Компактный статус
        bal = None
        if state.paper_mode:
            bal = state.paper.get_stats().get("current_balance")
        elif state.bybit:
            try:
                bal = state.bybit.get_balance("USDT")
            except Exception:
                pass
        return {
            "bot_running":  state.bot_running,
            "paper_mode":   state.paper_mode,
            "balance_usdt": bal,
            "open_positions": sum(1 for s in state.strategies.values() if s.current_position),
            "scalp_active": state.scalp_active,
            "boost_active": state.boost.is_active,
        }

    return {"error": f"Неизвестная команда: '{cmd}'", "available": [
        "balance", "positions", "trades", "pnl_today",
        "scalp_on", "scalp_off", "start", "stop",
        "paper_on", "paper_off", "boost_start", "boost_stop", "status",
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
