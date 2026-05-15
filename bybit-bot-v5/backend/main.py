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

# ML модули
from ml import (
    FeatureExtractor, MLDataStore, MLTrainer, MLPredictor,
    RegimeClassifier, AutoOptimizer, ML_AVAILABLE, OPTUNA_AVAILABLE,
    DriftMonitor, ThresholdOptimizer, AnomalyDetector, EnsembleTrainer,
)
# News & Sentiment
from news import NewsManager, background_news_loop
# Position calculation
from position_calc import (
    calculate_pnl, calculate_r_multiple,
    fixed_risk_position_size, kelly_position_size, adaptive_sl_tp,
)
from db_pool import DBPool

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
    """Создание всех 7 стратегий с дефолтными символами."""
    symbol_map = {
        "S1": "BTCUSDT",
        "S2": "ETHUSDT",
        "S3": "SOLUSDT",
        "S4": "BNBUSDT",
        "S5": "DOGEUSDT",
        "S6": "XRPUSDT",
        "S7": "BTCUSDT",
    }
    for sid, cls in ALL_STRATEGIES.items():
        state.strategies[sid] = cls(symbol=symbol_map[sid])
    logger.info(f"Инициализированы стратегии: {list(state.strategies.keys())}")

    # Загрузка активных ML моделей (если есть)
    if state.ml_enabled and ML_AVAILABLE:
        state.ml_predictor.load_all_active_models(state.ml_store)
        loaded = len(state.ml_predictor.models)
        if loaded:
            logger.info(f"🤖 Загружено {loaded} ML моделей")
        else:
            logger.info("🤖 ML модели не найдены — будет собирать данные для будущего обучения")


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

                # Получаем актуальный sentiment (cached, не блокирует)
                sentiment_features = state.news_manager.get_sentiment_features() if state.news_enabled else {}

                klines_data = {}

                for sid, strat in state.strategies.items():
                    if not strat.enabled or strat.auto_disabled:
                        continue

                    df = state.bybit.get_klines(strat.symbol, strat.timeframe, limit=250)
                    if df.empty:
                        continue
                    klines_data[strat.symbol] = df

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
                    signal = strat.analyze(df)
                    if not signal or signal.action not in ("BUY", "SELL"):
                        continue

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
                    features = {}
                    if state.ml_enabled:
                        # Order book
                        orderbook = state.bybit.get_orderbook(strat.symbol, limit=25)
                        # Market meta (funding, OI)
                        market_meta = state.bybit.get_market_meta(strat.symbol)
                        # Текущий рыночный режим
                        regime_id = None
                        if state.regime_classifier.model:
                            regime_result = state.regime_classifier.predict(df)
                            if regime_result:
                                regime_id = regime_result["regime_id"]

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

                    # ============== Position Sizing ==============
                    if state.use_kelly and ml_prediction and ml_prediction.get("available"):
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
                            leverage=strat.leverage,
                        )
                        if result["success"]:
                            strat.register_position(
                                "Buy" if signal.action == "BUY" else "Sell",
                                signal.entry_price, signal.stop_loss, signal.take_profit,
                            )
                            strat.current_position["qty"] = qty
                            state.risk_manager.register_position_open()

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

                            if state.ml_enabled and 'snapshot_id' in locals() and snapshot_id:
                                state.snapshot_signal_id_map[signal.symbol] = snapshot_id
                                with state.db_pool.cursor() as c:
                                    c.execute(
                                        "UPDATE signal_snapshots SET trade_taken=1, trade_id=? WHERE id=?",
                                        (trade_id, snapshot_id),
                                    )

                            # Fire-and-forget Telegram
                            asyncio.create_task(state.telegram.notify_trade_open(
                                sid, signal.symbol, signal.action,
                                signal.entry_price, signal.stop_loss, signal.take_profit,
                                signal.reason,
                            ))
                            await broadcast_log(f"🟢 {sid} {signal.action} {signal.symbol} @ {signal.entry_price}")

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
                                state.risk_manager.register_position_close()

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
                await asyncio.sleep(10)

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

    logger.info("✅ Бэкенд готов")
    yield
    state.bot_running = False
    if state.trading_loop_task:
        state.trading_loop_task.cancel()
    if state.news_loop_task:
        state.news_loop_task.cancel()
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
    import sqlite3, json
    conn = sqlite3.connect(state.ml_store.db_path)
    conn.row_factory = sqlite3.Row
    query = """
        SELECT id, timestamp, strategy_id, symbol, action,
               entry_price, ml_prediction, ml_confidence,
               trade_taken, outcome, pnl_r, exit_reason
        FROM signal_snapshots
    """
    params = []
    if strategy_id:
        query += " WHERE strategy_id = ?"
        params.append(strategy_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()
    return rows


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
        # Сразу шлём текущее состояние
        await broadcast_state()
        while True:
            data = await ws.receive_text()
            # Можно обрабатывать команды от клиента
    except WebSocketDisconnect:
        state.ws_clients.remove(ws)
        logger.info(f"WS disconnected. Total: {len(state.ws_clients)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
