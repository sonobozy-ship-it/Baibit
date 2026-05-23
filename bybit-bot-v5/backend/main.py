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
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

# Локальные модули
import config as cfg
from bybit_client import BybitClient
from risk_manager import RiskManager
from backtester import Backtester
from telegram_notifier import TelegramNotifier
from telegram_commander import TelegramCommander
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
from adaptive_params import AdaptiveParamManager
from claude_orchestrator import ClaudeOrchestrator
from strategy_advisor import StrategyAdvisor
from position_monitor import PositionMonitor
from global_trade_guard import GlobalTradeGuard
from strategies.tp_normalizer import normalize_take_profit
from strategies.entry_filter import validate_entry_confirmation
from strategies.time_rate import TimeRateManager
from strategies.market_filters import calc_quality_score

# ============================================================
# Загрузка конфига
# ============================================================
load_dotenv()
Path("logs").mkdir(exist_ok=True)  # должно быть ДО FileHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# Глобальное состояние
# ============================================================
class BotState:
    def __init__(self):
        self.bybit: Optional[BybitClient] = None
        self.risk_manager = RiskManager(
            daily_max_loss_pct=float(os.getenv("DAILY_MAX_LOSS_PCT", "5.0")),
            daily_pause_loss_pct=float(os.getenv("DAILY_PAUSE_LOSS_PCT", "3.0")),
            daily_pause_hours=float(os.getenv("DAILY_PAUSE_HOURS", "1.0")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "20")),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "1.0")),
            cooldown_after_loss_min=int(os.getenv("COOLDOWN_AFTER_LOSS_MIN", "15")),
            max_daily_trades=int(os.getenv("MAX_DAILY_TRADES", "0")),          # 0 = без лимита
            max_daily_losses=int(os.getenv("MAX_DAILY_LOSSES", "0")),           # 0 = глобальный стоп выключен
            max_strategy_daily_losses=int(os.getenv("MAX_STRATEGY_DAILY_LOSSES", "5")),   # лимит убытков на стратегию (5 пока на обучении)
        )
        self.journal = TradeJournal()
        self.correlation = CorrelationFilter()
        self.ai = AIAnalyzer()
        # Ранний TP: закрыть когда цена прошла X% пути к TP (0 = выключено)
        self.early_tp_enabled = os.getenv("EARLY_TP_ENABLED", "false").lower() in ("1", "true", "yes", "on")
        self.early_tp_pct = float(os.getenv("EARLY_TP_PCT", "85"))
        self.paper = PaperTrader(
            initial_balance=float(os.getenv("PAPER_INITIAL_BALANCE", "1000")),
        )
        _tg_token   = os.getenv("TELEGRAM_BOT_TOKEN", "") or os.getenv("TELEGRAM_TOKEN", "")
        _tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.telegram = TelegramNotifier(
            bot_token=_tg_token,
            chat_id=_tg_chat_id,
        )
        self.commander: Optional[TelegramCommander] = None
        self.commander_task: Optional[asyncio.Task] = None
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
            openai_key=os.getenv("OPENAI_API_KEY"),
        )
        self.news_enabled = os.getenv("NEWS_ENABLED", "true").lower() == "true"

        # ============== Strategy Advisor ==============
        self.advisor = StrategyAdvisor(
            db_pool=self.db_pool,
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_key=os.getenv("OPENAI_API_KEY", ""),
        )

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

        # Адаптивные параметры стратегий (самообучение по реальным сделкам)
        self.adaptive = AdaptiveParamManager(window=int(os.getenv("ADAPTIVE_WINDOW", "30")))

        # Мониторинг позиций и умное закрытие
        self.position_monitor = PositionMonitor()

        # Единый защитный слой перед открытием сделок
        self.trade_guard = GlobalTradeGuard()

        # TimeRate — кулдауны и anti-overtrade для скальперов
        self.time_rate = TimeRateManager()
        self.max_parallel_scalps     = int(os.getenv("MAX_PARALLEL_SCALPS", "2"))
        self.scalper_priority_weight = float(os.getenv("SCALPER_PRIORITY_WEIGHT", "1.8"))

        # Режим сбора данных: убирает все лимиты (лосс-лимиты, кулдауны, Guard-блоки)
        self.training_mode: bool = False

        # Кэш старших таймфреймов для MTF-фильтра ScalperPro
        # {symbol: (DataFrame, updated_at)}
        self.h1_cache:  Dict[str, tuple] = {}
        self.m15_cache: Dict[str, tuple] = {}

        # Claude AI Orchestrator — анализирует состояние и выдаёт команды
        self.orchestrator: Optional[ClaudeOrchestrator] = None
        self.orchestrator_task: Optional[asyncio.Task] = None

        # Активные стратегии (создаются при старте)
        self.strategies: Dict[str, object] = {}
        self.bot_running = False
        self.paper_mode = False
        self.trading_mode: str = "PAPER"
        self.ws_clients: List[WebSocket] = []
        self.tickers: Dict[str, Dict] = {}

state = BotState()


# ============================================================
# Инициализация стратегий
# ============================================================
def init_strategies():
    """Создание всех 9 стратегий с дефолтными символами."""
    symbol_map = {
        "S1":  "BTCUSDT",
        "S2":  "ETHUSDT",
        "S3":  "SOLUSDT",
        "S4":  "BNBUSDT",
        "S5":  "DOGEUSDT",
        "S6":  "XRPUSDT",
        "S7":  "UNIUSDT",         # MultiConfirm (отключена) — LINKUSDT теперь у S10
        "S8":  "DOTUSDT",
        "S9":  "NEARUSDT",
        "S10": "LINKUSDT",        # ScalperPro 3m — S7 отключена, символ свободен
        "S11": "AVAXUSDT",        # DragonflyGold
        "S12": "ADAUSDT",
        "S13": "1000PEPEUSDT",
        "S14": "WIFUSDT",
        "S15": "OPUSDT",          # Aggressive Momentum AI (1m, L2 token)
    }
    for sid, cls in ALL_STRATEGIES.items():
        state.strategies[sid] = cls(symbol=symbol_map.get(sid, "BTCUSDT"))
    logger.info(f"Инициализированы стратегии: {list(state.strategies.keys())}")

    # Регистрируем все стратегии в адаптивном менеджере
    for sid, strat in state.strategies.items():
        base_conf = getattr(strat, "edge_wr_target", 0.60)
        state.adaptive.register(sid, base_confidence=base_conf)
    state.adaptive.register("FUSION", base_confidence=0.65)
    logger.info("🧠 AdaptiveParamManager инициализирован")

    # Загрузка активных ML моделей (если есть)
    if state.ml_enabled and ML_AVAILABLE:
        state.ml_predictor.load_all_active_models(state.ml_store)
        loaded = len(state.ml_predictor.models)
        if loaded:
            logger.info(f"🤖 Загружено {loaded} ML моделей")
        else:
            logger.info("🤖 ML модели не найдены — будет собирать данные для будущего обучения")

    # Восстановление открытых бумажных позиций после перезапуска
    _restore_paper_positions()

    # Восстановление счётчиков стратегий из БД (raw cursor — без pandas, надёжно)
    restored, skipped = [], []
    for sid, strat in state.strategies.items():
        st = state.journal.restore_strategy_stats(sid)
        if st:
            strat.trades             = st["trades"]
            strat.wins               = st["wins"]
            strat.losses             = st["losses"]
            strat.pnl                = st["total_pnl"]
            strat.history            = st["history"]
            strat.consecutive_losses = st["consecutive_losses"]
            restored.append(f"{sid}:{st['trades']}сд WR={round(st['wins']/st['trades']*100) if st['trades'] else 0}%")
        else:
            skipped.append(sid)
    if restored:
        logger.info(f"[StatRestore] ✅ Восстановлено: {', '.join(restored)}")
    if skipped:
        logger.info(f"[StatRestore] Нет истории в БД: {', '.join(skipped)}")


def _restore_paper_positions():
    """Восстанавливает current_position стратегий из сохранённого paper state."""
    if not state.paper.positions:
        return
    # Строим карту strategy_id -> symbol из paper позиций
    sid_map = {pos["strategy_id"]: sym for sym, pos in state.paper.positions.items()}
    for sid, strat in state.strategies.items():
        if sid not in sid_map:
            continue
        sym = sid_map[sid]
        pos = state.paper.positions.get(sym)
        if not pos:
            continue
        _jtid = pos.get("journal_trade_id")
        strat.current_position = {
            "side":             pos["side"],
            "entry":            pos["entry_price"],
            "sl":               pos["stop_loss"],
            "tp":               pos["take_profit"],
            "initial_sl":       pos.get("stop_loss"),
            "be_moved":         pos.get("be_moved", False),
            "qty":              pos.get("qty", 0),
            "leverage":         pos.get("leverage", 1),
            "opened_at":        pos.get("opened_at"),
            "journal_trade_id": _jtid,
        }
        # Если journal_trade_id не сохранён — ищем незакрытую запись в БД
        if _jtid is None:
            try:
                open_trade = state.journal.get_open_trade(sid, sym)
                if open_trade:
                    recovered_id = open_trade["id"]
                    strat.current_position["journal_trade_id"] = recovered_id
                    state.paper.set_journal_id(sym, recovered_id)
                    logger.info(
                        f"[PaperRestore] {sid} {sym}: journal_trade_id={recovered_id} "
                        f"восстановлен из БД"
                    )
            except Exception as e:
                logger.warning(f"[PaperRestore] {sid} {sym}: не удалось восстановить journal_trade_id: {e}")
        state.risk_manager.register_position_open(sid, pos.get("qty", 0) * pos.get("entry_price", 0))
        logger.info(f"[PaperRestore] {sid} {pos['side']} {sym} @ {pos['entry_price']} восстановлен")

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


def _scalp_sid(symbol: str) -> str:
    """SC_BTC из BTCUSDT, SC_SHIB из 1000SHIBUSDT и т.д."""
    base = symbol.replace("USDT", "")
    if base.startswith("1000"):
        base = base[4:]   # 1000SHIB → SHIB
    return f"SC_{base[:4]}"


def activate_scalp_mode(symbols: List[str] = None):
    """
    Создаёт ScalperPro для каждого символа и регистрирует как SC_XXX.
    symbols — явный список; если None, авто-выбирает топ по объёму с Bybit
    (или SCALP_SYMBOLS как резервный вариант).
    """
    # Определяем список символов
    if symbols:
        chosen = symbols
    else:
        # Символы уже занятые основными стратегиями
        used = {strat.symbol for sid, strat in state.strategies.items()
                if not sid.startswith("SC_")}
        top_n = int(os.getenv("SCALP_TOP_N", "15"))
        if state.bybit:
            chosen = state.bybit.get_top_usdt_symbols(top_n=top_n, exclude=used)
        else:
            chosen = []
        if not chosen:
            # Резервный список (уже без конфликтов с S1-S14)
            chosen = SCALP_SYMBOLS

    added = []
    for sym in chosen:
        sid = _scalp_sid(sym)
        if sid not in state.strategies:
            strat = ScalperProStrategy(symbol=sym)
            # Восстанавливаем историю из БД при первом создании скальпера
            _sc_st = state.journal.restore_strategy_stats(sid)
            if _sc_st:
                strat.trades             = _sc_st["trades"]
                strat.wins               = _sc_st["wins"]
                strat.losses             = _sc_st["losses"]
                strat.pnl                = _sc_st["total_pnl"]
                strat.history            = _sc_st["history"]
                strat.consecutive_losses = _sc_st["consecutive_losses"]
            state.strategies[sid] = strat
            added.append(sid)
        else:
            state.strategies[sid].symbol           = sym
            state.strategies[sid].enabled          = True
            state.strategies[sid].auto_disabled    = False   # сброс блокировки
            state.strategies[sid].consecutive_losses = 0     # сброс серии убытков
    state.scalp_active = True
    logger.info(f"⚡ Scalp Mode: {len(added)} скальперов → {added}")
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


async def auto_select_symbols():
    """
    Авто-назначает символы ТОЛЬКО скальперам SC_* по топу объёма с Bybit.
    Основные стратегии S1-S14 НИКОГДА не трогаются — у них параметры
    заточены под конкретные рынки (S1=BTC-тренд, S5=DOGE-волатильность и т.д.).
    Если AUTO_SELECT_SYMBOLS=false — пропускаем.
    """
    if os.getenv("AUTO_SELECT_SYMBOLS", "true").lower() == "false":
        return
    if not state.bybit:
        return
    if not state.scalp_active:
        return

    n_scalp = int(os.getenv("SCALP_TOP_N", "15"))

    # Символы занятые основными стратегиями — скальперы не берут их
    used = {strat.symbol for sid, strat in state.strategies.items()
            if not sid.startswith("SC_")}

    scalp_pool = state.bybit.get_top_usdt_symbols(top_n=n_scalp + 10, exclude=used)
    scalp_pool = scalp_pool[:n_scalp]

    if not scalp_pool:
        logger.warning("[SymbolSelect] Bybit вернул пустой список, оставляем текущие скальперы")
        return

    # Пересоздаём SC_* без открытых позиций
    # Также удаляем SC_* которые торгуют заблокированными монетами
    blocklist = getattr(state.bybit, "_NON_CRYPTO", frozenset())
    to_remove = [
        sid for sid, strat in list(state.strategies.items())
        if sid.startswith("SC_") and not strat.current_position
        and (strat.symbol not in scalp_pool or strat.symbol in blocklist)
    ]
    for sid in to_remove:
        del state.strategies[sid]

    activate_scalp_mode(symbols=scalp_pool)
    logger.info(f"[SymbolSelect] Скальперы обновлены: {scalp_pool[:5]}... ({len(scalp_pool)} пар)")


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


def _execute_bot_command(cmd_dict: Dict) -> Dict:
    """Маппинг команд Claude-оркестратора → внутренние действия бота."""
    cmd = cmd_dict.get("cmd", "")

    if cmd == "start":
        if not state.bot_running:
            state.bot_running = True
            # create_task требует запущенного event loop; вызывается из asyncio-контекста через orchestrator
            try:
                loop = asyncio.get_running_loop()
                if not state.trading_loop_lock.locked():
                    state.trading_loop_task = loop.create_task(trading_loop())
            except RuntimeError:
                logger.error("[Orchestrator] create_task невозможен — нет running event loop")
        return {"bot_running": state.bot_running}

    if cmd == "stop":
        state.bot_running = False
        return {"bot_running": False, "reason": cmd_dict.get("reason", "")}

    if cmd == "paper_on":
        state.paper_mode = True
        return {"paper_mode": True}

    if cmd == "paper_off":
        state.paper_mode = False
        return {"paper_mode": False}

    if cmd == "scalp_on":
        added = activate_scalp_mode()
        return {"scalp_active": True, "strategies": added}

    if cmd == "scalp_off":
        disabled = deactivate_scalp_mode()
        return {"scalp_active": False, "disabled": disabled}

    if cmd == "boost_start":
        result = state.boost.start(
            initial_balance=float(cmd_dict.get("initial", 10)),
            target_balance=float(cmd_dict.get("target", 50)),
            deadline_days=int(cmd_dict.get("days", 21)),
            mode=cmd_dict.get("mode", "moderate"),
        )
        if result.get("success") and cmd_dict.get("mode") == "scalp":
            activate_scalp_mode()
        return result

    if cmd == "boost_stop":
        result = state.boost.stop(cmd_dict.get("reason", "Оркестратор"))
        if result.get("success") and state.scalp_active:
            deactivate_scalp_mode()
        return result

    if cmd == "set_leverage":
        sid = cmd_dict.get("sid", "")
        strat = state.strategies.get(sid)
        if not strat:
            return {"error": f"Стратегия {sid} не найдена"}
        strat.leverage = int(cmd_dict.get("value", strat.leverage))
        return {"sid": sid, "leverage": strat.leverage}

    if cmd == "disable_strategy":
        sid = cmd_dict.get("sid", "")
        strat = state.strategies.get(sid)
        if not strat:
            return {"error": f"Стратегия {sid} не найдена"}
        strat.enabled = False
        return {"sid": sid, "enabled": False}

    if cmd == "enable_strategy":
        sid = cmd_dict.get("sid", "")
        strat = state.strategies.get(sid)
        if not strat:
            return {"error": f"Стратегия {sid} не найдена"}
        strat.enabled = True
        strat.auto_disabled = False
        return {"sid": sid, "enabled": True}

    if cmd == "set_ml_mode":
        mode = cmd_dict.get("mode", "advisory")
        if mode in ("advisory", "strict", "off"):
            state.ml_filter_mode = mode
        return {"ml_filter_mode": state.ml_filter_mode}

    return {"error": f"Неизвестная команда: {cmd}"}


# ============================================================
# Claude Orchestrator — фоновый цикл
# ============================================================
async def orchestrator_loop():
    """Каждую минуту проверяет расписание; запускает цикл оркестратора когда пора."""
    logger.info("[Orchestrator] Фоновый цикл запущен")
    while True:
        await asyncio.sleep(60)
        if not state.orchestrator or not state.orchestrator.enabled:
            continue
        if not state.orchestrator.is_due:
            continue
        # В режиме обучения оркестратор только оповещает, не вмешивается
        if state.training_mode:
            logger.debug("[Orchestrator] Режим обучения — автодействия пропущены")
            continue
        try:
            full = await full_status()
            result = await state.orchestrator.run_cycle(full)
            if result:
                level = "warn" if result.risk_level in ("high", "critical") else "info"
                cmds  = [d.cmd for d in result.decisions if d.cmd != "wait"]
                msg   = f"🤖 Orchestrator [{result.risk_level}]: {result.analysis[:80]}"
                if cmds:
                    msg += f" | cmds: {', '.join(cmds)}"
                await broadcast_log(msg, level)
        except Exception as e:
            logger.error(f"[Orchestrator] Loop error: {e}")


# ============================================================
# Авто-обучение (фоновая задача)
# ============================================================
async def symbol_refresh_loop():
    """Обновляет символы стратегий каждые 4 часа по топу объёма."""
    while True:
        await asyncio.sleep(4 * 3600)
        try:
            await auto_select_symbols()
        except Exception as e:
            logger.warning(f"[SymbolRefresh] Ошибка: {e}")


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
# AI-советы по стратегии (раз в 10 сделок или по запросу)
# ============================================================
async def _auto_advisor_run():
    """Автоматический запуск AI-советника по всем стратегиям."""
    if not state.advisor.enabled:
        return
    try:
        journal_stats = state.journal.get_stats_by_strategy()
        market_ctx = {
            "balance": state.paper.balance if state.paper_mode else 0,
            "open_positions": state.risk_manager.open_positions_count,
            "regime": "unknown",
        }
        result = await state.advisor.analyze_all(
            journal_stats=journal_stats,
            strategies=state.strategies,
            market_context=market_ctx,
            triggered_by="auto",
        )
        if result.get("available"):
            brief = StrategyAdvisor.format_telegram(result, brief=True)
            asyncio.create_task(state.telegram.send(
                f"🤖 <b>Авто-анализ стратегий</b>\n\n{brief}"
            ))
    except Exception as e:
        logger.debug(f"_auto_advisor_run: {e}")


async def _auto_ai_improve(sid: str, strat_name: str, strat_symbol: str, trades_done: int):
    """Запускает AI-анализ стратегии и отправляет советы в Telegram."""
    if not state.ai.enabled:
        return
    try:
        trades_hist = state.journal.get_trades(strategy_id=sid, limit=50)
        stats_list  = state.journal.get_stats_by_strategy()
        stats       = next((x for x in stats_list if x["strategy_id"] == sid), {})

        # Если журнал пустой — подтягиваем живые данные из памяти стратегии
        strat_obj = state.strategies.get(sid)
        if strat_obj and (not stats or stats.get("trades", 0) == 0):
            pnls = strat_obj.history[-trades_done:] if strat_obj.history else []
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            stats = {
                "strategy_id":   sid,
                "trades":        strat_obj.trades,
                "win_rate":      strat_obj.win_rate,
                "total_pnl":     round(strat_obj.pnl, 4),
                "avg_win":       round(sum(wins) / len(wins), 4) if wins else 0,
                "avg_loss":      round(sum(losses) / len(losses), 4) if losses else 0,
                "profit_factor": round(abs(sum(wins) / sum(losses)), 3) if losses and sum(losses) != 0 else 0,
                "max_drawdown_pct": 0,
                "longest_loss_streak": strat_obj.consecutive_losses,
            }

        result = state.ai.analyze_strategy_performance({
            "id":     sid,
            "name":   strat_name,
            "symbol": strat_symbol,
            "trades": trades_hist,
            "stats":  stats,
        })
        if result.get("available"):
            analysis = result.get("analysis", "")[:2000]
            wr = stats.get("win_rate", 0)
            pnl = stats.get("total_pnl", 0)
            asyncio.create_task(state.telegram.send(
                f"🤖 <b>AI-советы: {sid} ({strat_name})</b>\n"
                f"Сделок: {stats.get('trades', trades_done)} | WR: {wr:.0f}% | PnL: {pnl:+.2f}$\n\n"
                f"{analysis}"
            ))
    except Exception as e:
        logger.debug(f"_auto_ai_improve {sid}: {e}")


# ============================================================
# Ранний выход из позиции (Early TP)
# ============================================================
async def _close_position_early(
    sid: str,
    strat,
    exit_price: float,
    klines_data: dict,
    reason: str = "",
):
    """Закрывает позицию досрочно: по EarlyTP или по таймауту MaxHold."""
    pos = strat.current_position
    if not pos:
        return

    qty              = pos.get("qty", 1.0)
    side             = pos["side"]
    entry            = pos["entry"]
    tp               = pos["tp"]
    opened_at        = pos.get("opened_at")
    _early_jtid      = pos.get("journal_trade_id")
    if hasattr(opened_at, "isoformat"):
        opened_at = opened_at.isoformat()

    monitor_labels = {
        "tp_progress":    "TPProgress",
        "near_tp":        "NearTP",
        "trailing_profit":"TrailingProfit",
        "profit_return":  "ProfitReturn",
        "timeout":        "Timeout",
    }

    if reason == "MaxHold":
        held_min = 0.0
        if opened_at:
            try:
                import pandas as _pd
                oa = _pd.Timestamp(opened_at)
                held_min = (_pd.Timestamp.utcnow() - oa).total_seconds() / 60.0
            except Exception:
                pass
        reason_str = f"MaxHold {held_min:.0f}m"
        tg_msg = (
            f"⏱ <b>Таймаут скальпа: {sid} {strat.symbol}</b>\n"
            f"Позиция {held_min:.0f} мин → принудительное закрытие\n"
            f"Выход: {exit_price:.6f} | PnL: <b>{{pnl:+.2f}} USDT</b>"
        )
    elif reason in monitor_labels:
        reason_str = monitor_labels[reason]
        tg_msg = (
            f"📊 <b>Умное закрытие: {sid} {strat.symbol}</b>\n"
            f"Причина: {reason_str}\n"
            f"Выход: {exit_price:.6f} | PnL: <b>{{pnl:+.2f}} USDT</b>"
        )
    else:
        # Empty reason — called from check_early_tp path
        tp_dist  = (tp - entry) if side == "Buy" else (entry - tp)
        progress = ((exit_price - entry) / tp_dist * 100) if (side == "Buy" and tp_dist > 0) \
                   else ((entry - exit_price) / tp_dist * 100) if tp_dist > 0 else 0
        reason_str = f"EarlyTP {progress:.0f}%"
        tg_msg = (
            f"💰 <b>Ранний TP: {sid} {strat.symbol}</b>\n"
            f"Закрыт на <b>{progress:.0f}%</b> пути к TP\n"
            f"Выход: {exit_price:.6f} | PnL: <b>{{pnl:+.2f}} USDT</b>"
        )

    if state.paper_mode:
        closed    = state.paper._close_position(strat.symbol, exit_price, reason_str)
        pnl       = closed.get("pnl_usd", 0)
        close_res = strat.close_position(exit_price, qty=qty)
    elif state.bybit:
        close_side = "Sell" if side == "Buy" else "Buy"
        result     = state.bybit.place_order(
            symbol=strat.symbol,
            side=close_side,
            qty=qty,
            reduce_only=True,
        )
        if not result.get("success"):
            logger.warning(f"[{reason_str}] {sid} place_order failed: {result}")
            return
        close_res = strat.close_position(exit_price, qty=qty)
        pnl       = close_res.get("pnl_usd", 0)
    else:
        return

    state.risk_manager.register_trade_result(sid, pnl)
    state.risk_manager.register_position_close(sid)
    _early_r = close_res.get("r_multiple", 0)
    state.adaptive.record(sid, _early_r)
    if sid.startswith("SC_") or sid in ("S5", "S10", "S15"):
        state.time_rate.register_close(sid, strat.symbol, pnl)
    if pnl > 0:
        state.trade_guard.record_win(sid)
    else:
        state.trade_guard.record_loss(sid)

    if _early_jtid:
        state.journal.update_trade_close(
            _early_jtid,
            exit_price=exit_price,
            pnl_usd=pnl,
            pnl_pct=close_res.get("pnl_pct", 0),
            exit_reason=reason_str,
            r_multiple=_early_r,
        )

    if strat.trades > 0 and strat.trades % 10 == 0:
        asyncio.create_task(_auto_ai_improve(sid, strat.NAME, strat.symbol, strat.trades))

    asyncio.create_task(state.telegram.notify_trade_close(
        sid, strat.symbol, pnl, reason_str,
        leverage=pos.get("leverage", 1),
        df=klines_data.get(strat.symbol),
        entry=entry,
        side=side,
        sl=pos.get("sl"),
        tp=tp,
        exit_price=exit_price,
        opened_at=opened_at,
    ))
    asyncio.create_task(state.telegram.send(tg_msg.format(pnl=pnl)))
    await broadcast_log(
        f"{'⏱' if reason == 'MaxHold' else '💰'} {reason_str} {sid} {strat.symbol} → {pnl:+.2f} USDT",
        "ok" if pnl > 0 else "warn",
    )


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
    # Snapshot стратегий под локом чтобы избежать race condition
    async with state.trading_loop_lock:
        existing = any(
            s.symbol == sym and s.current_position
            for s in state.strategies.values()
        )
    if existing:
        logger.debug(f"[Fusion] {sym}: уже открыта позиция, пропускаем fusion")
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

    # Глобальный R:R фильтр: минимум 2.0
    _fusion_rr = GlobalTradeGuard._calc_rr(fused.action, fused.entry_price, fused.stop_loss, fused.take_profit)
    if _fusion_rr is None or _fusion_rr < 2.0:
        logger.info(
            f"[Fusion] ❌ R:R {_fusion_rr} < 2.0 "
            f"({sym} TP={fused.take_profit} SL={fused.stop_loss})"
        )
        return

    # Свечи для ML и графика
    chart_df = None
    if state.bybit:
        try:
            _cdf = state.bybit.get_klines(sym, "60", limit=250)
            if _cdf is not None and not _cdf.empty:
                chart_df = _cdf
        except Exception:
            pass

    # ML snapshot + prediction
    ml_prediction = None
    fusion_snapshot_id = None
    if state.ml_enabled:
        try:
            df = chart_df
            if df is not None and not df.empty:
                orderbook   = state.bybit.get_orderbook(sym, limit=25) if state.bybit else {}
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
        min_notional=state.risk_manager.min_trade_usdt,
    )
    qty = round(base_qty * fused.size_multiplier, 6)
    if qty <= 0:
        return

    # Кап риска в USD на сделку
    if cfg.MAX_RISK_PER_TRADE_USDT > 0 and fused.entry_price and fused.stop_loss:
        _sl_dist = abs(fused.entry_price - fused.stop_loss)
        if _sl_dist > 0 and qty * _sl_dist > cfg.MAX_RISK_PER_TRADE_USDT:
            qty = round(cfg.MAX_RISK_PER_TRADE_USDT / _sl_dist, 6)
            logger.info(
                f"[Fusion] 💰 Риск скейлирован до ${cfg.MAX_RISK_PER_TRADE_USDT}: "
                f"qty={qty} ({sym})"
            )

    # GlobalTradeGuard проверяет Fusion-сигнал
    _fusion_guard = state.trade_guard.check(
        strategy_id=sid,
        signal_action=fused.action,
        entry_price=fused.entry_price,
        stop_loss=fused.stop_loss,
        take_profit=fused.take_profit,
        qty=qty,
        leverage=3,
        balance=balance,
        df=chart_df,
        df_h1=chart_df,
        ml_probability=(ml_prediction.get("probability") if ml_prediction and ml_prediction.get("available") else None),
        signal_confidence=fused.confidence,
        scalp_mode=False,
    )
    if not _fusion_guard.approved:
        logger.info(f"[Fusion] 🛡 Guard BLOCKED {fused.action} {sym}: {_fusion_guard.blocked_by}")
        return

    # Открытие позиции
    log_msg = (
        f"[Fusion] {fused.action} {sym} "
        f"[{'+'.join(fused.source_strategies)}] "
        f"score={fused.fusion_score:.3f} "
        f"size×{fused.size_multiplier}"
    )

    if state.paper_mode:
        _fusion_lev = min(3, state.risk_manager.max_leverage_cap)
        _fp_result = state.paper.open_position(fused, sid, qty, _fusion_lev)
        if not _fp_result.get("success"):
            logger.warning(f"[Fusion PAPER] отказ: {_fp_result.get('reason')}")
            return
        asyncio.create_task(state.telegram.notify_trade_open(
            sid, sym, fused.action,
            fused.entry_price, fused.stop_loss, fused.take_profit,
            fused.reason, qty=qty, leverage=_fusion_lev, df=chart_df,
            timeframe="15",
        ))
        await broadcast_log(f"📄 FUSION {fused.action} {sym} (paper) {log_msg}")
    else:
        if not state.bybit:
            return
        lev_check = state.risk_manager.check_leverage(sid, 3, balance)
        if not lev_check["allowed"]:
            logger.info(f"[Fusion] ❌ {lev_check['reason']}")
            return
        result = state.bybit.place_order(
            symbol=sym,
            side="Buy" if fused.action == "BUY" else "Sell",
            qty=qty,
            stop_loss=fused.stop_loss,
            take_profit=fused.take_profit,
            leverage=lev_check["effective_leverage"],
        )
        if result.get("success"):
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
                "initial_sl":    fused.stop_loss,
                "initial_tp":    fused.take_profit,
                "signal_reason": fused.reason,
                "filters_passed": {
                    "fusion_score":     fused.fusion_score,
                    "strategies":       fused.source_strategies,
                    "diversity":        fused.diversity_score,
                },
                "opened_at": datetime.utcnow().isoformat(),
            })

            if fusion_snapshot_id:
                # Храним под ключом символа (без _FUSION) для совместимости с closing-loop
                state.snapshot_signal_id_map[sym] = fusion_snapshot_id
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
                qty=qty, leverage=lev_check["effective_leverage"], df=chart_df,
                timeframe="15",
            ))
            await broadcast_log(
                f"🔥 FUSION {fused.action} {sym} @ {fused.entry_price} "
                f"[{'+'.join(fused.source_strategies)}] ×{fused.size_multiplier}"
            )
        else:
            logger.error(f"[Fusion] Ошибка ордера: {result.get('error')}")


def _log_trade_attempt(mode: str, strategy_id: str, symbol: str, side: str,
                        entry: float, sl: float, tp: float, qty: float,
                        leverage: int, balance: float, reason: str):
    notional = qty * entry
    margin = notional / leverage if leverage else notional
    risk_usd = abs(entry - sl) * qty if sl else 0
    risk_pct = risk_usd / balance * 100 if balance else 0
    logger.info(
        f"[TRADE:{mode}] {strategy_id} {symbol} {side} | "
        f"entry={entry:.6g} SL={sl:.6g} TP={tp:.6g} | "
        f"qty={qty} notional={notional:.2f}$ margin={margin:.2f}$ "
        f"lev={leverage}x | "
        f"risk={risk_usd:.2f}$ ({risk_pct:.2f}%) | "
        f"reason={reason[:120]}"
    )


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
                if not state.bybit and not state.paper_mode:
                    await asyncio.sleep(2)
                    continue

                if state.paper_mode:
                    # Equity = free balance + locked margin (правильный баланс для risk-менеджера)
                    locked = sum(
                        p.get("margin", 0) for p in state.paper.positions.values()
                    )
                    balance = round(state.paper.balance + locked, 4)
                elif state.bybit and not getattr(state.bybit, "public_only", False):
                    loop = asyncio.get_running_loop()
                    balance = await loop.run_in_executor(
                        None, lambda: state.bybit.get_balance("USDT")
                    )
                else:
                    balance = 0
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
                current_regime_name = None
                current_regime_id   = None

                # AI-рекомендации (раз в 30 мин, advisory — не блокируют торговлю)
                if state.news_enabled and (
                    not hasattr(state, "_last_ai_rec")
                    or (datetime.utcnow() - state._last_ai_rec).total_seconds() > 1800
                ):
                    try:
                        portfolio_ctx = {
                            "balance": balance,
                            "open_positions": state.risk_manager.open_positions_count,
                            "daily_pnl": state.risk_manager.daily_pnl,
                            "active_strategies": [
                                sid for sid, s in state.strategies.items() if s.enabled
                            ],
                            "regime": current_regime_name or "unknown",
                        }
                        ai_rec = await state.news_manager.get_trading_recommendations(
                            portfolio_context=portfolio_ctx,
                        )
                        state._last_ai_rec = datetime.utcnow()
                        state._cached_ai_rec = ai_rec
                        if ai_rec.get("available"):
                            action = ai_rec.get("action", "trade")
                            risk   = ai_rec.get("risk_level", "medium")
                            await broadcast_log(
                                f"🧠 AI рекомендация: {action.upper()} | риск={risk} | "
                                f"{ai_rec.get('reasoning', '')[:120]}",
                                "warn" if risk in ("high", "extreme") else "info",
                            )
                            # Критическая ситуация — уведомление в Telegram
                            if action == "stop" or risk == "extreme":
                                asyncio.create_task(state.telegram.send(
                                    f"🚨 <b>AI СИГНАЛ ОПАСНОСТИ</b>\n"
                                    f"Действие: {action.upper()}\n"
                                    f"Риск: {risk}\n"
                                    f"{ai_rec.get('reasoning', '')}"
                                ))
                    except Exception as e:
                        logger.debug(f"AI рекомендации пропущены: {e}")

                for sid, strat in state.strategies.items():
                    # В режиме обучения пропускаем только явно disabled (не auto_disabled)
                    if not strat.enabled:
                        continue
                    if strat.auto_disabled and not state.training_mode:
                        continue

                    # Boost-режим: ограничение списка стратегий (в обучении — все стратегии)
                    if not state.training_mode and not state.boost.strategy_allowed(sid):
                        continue

                    # Нет подключения к бирже — нет рыночных данных
                    if not state.bybit:
                        continue

                    df = state.bybit.get_klines(strat.symbol, strat.timeframe, limit=250)
                    if df is None or df.empty:
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

                    # Soft-фильтр по режиму (в обучении — торгуем при любом режиме)
                    if (not state.training_mode
                            and current_regime_name and strat.REGIME_PREFERENCE
                            and current_regime_name not in strat.REGIME_PREFERENCE):
                        logger.debug(
                            f"{sid}: режим '{current_regime_name}' не подходит "
                            f"(предпочтение: {strat.REGIME_PREFERENCE})"
                        )
                        continue

                    # Если позиция уже открыта — проверка breakeven/trailing/early_tp
                    if strat.current_position:
                        current_price = float(df.iloc[-1]["close"])

                        # Breakeven
                        new_sl = strat.check_breakeven(current_price)
                        if new_sl and not state.paper_mode:
                            state.bybit.update_stop_loss(strat.symbol, new_sl)
                            await broadcast_log(f"🛡 {sid} перенос SL в безубыток @ {new_sl:.4f}")
                            _be_tid = strat.current_position.get("journal_trade_id") if strat.current_position else None
                            if _be_tid:
                                state.journal.update_trade_levels(_be_tid, stop_loss=new_sl, be_triggered=True)

                        # Trailing stop
                        trail_sl = strat.check_trailing_stop(current_price)
                        if trail_sl and not state.paper_mode:
                            state.bybit.update_stop_loss(strat.symbol, trail_sl)
                            _tr_tid = strat.current_position.get("journal_trade_id") if strat.current_position else None
                            if _tr_tid:
                                state.journal.update_trade_levels(_tr_tid, stop_loss=trail_sl, trailing_triggered=True)

                        # PositionMonitor: умное закрытие (TP%, trailing, near_tp, timeout)
                        _monitor_decision = state.position_monitor.check_position(
                            sid, strat.current_position, current_price,
                        )
                        if _monitor_decision:
                            await _close_position_early(
                                sid, strat, current_price, klines_data,
                                reason=_monitor_decision.reason,
                            )
                            state.position_monitor.reset_state(sid)
                            continue

                        # Ранний TP — закрыть если цена прошла early_tp_pct% пути к TP
                        if state.early_tp_enabled:
                            early_exit = strat.check_early_tp(current_price, state.early_tp_pct)
                            if early_exit:
                                await _close_position_early(sid, strat, current_price, klines_data)
                                state.position_monitor.reset_state(sid)

                        # Таймаут позиции (max_hold_minutes) — скальперы закрываются принудительно
                        if strat.check_max_hold():
                            await _close_position_early(sid, strat, current_price, klines_data, reason="MaxHold")
                            state.position_monitor.reset_state(sid)

                        continue

                    # Поиск сигнала
                    # SC_*, S10 и S15 получают MTF данные для trend-фильтра
                    # S5 получает df_h1 для HTF-проверки (EMA50 vs EMA200)
                    if sid.startswith("SC_") or sid in ("S15", "S10"):
                        signal = strat.analyze(
                            df,
                            df_h1  = _get_h1_cached(strat.symbol),
                            df_m15 = _get_m15_cached(strat.symbol),
                        )
                    elif sid == "S5":
                        signal = strat.analyze(
                            df,
                            df_h1 = _get_h1_cached(strat.symbol),
                        )
                    else:
                        signal = strat.analyze(df)
                    if not signal or signal.action not in ("BUY", "SELL"):
                        continue

                    # Скальперы SC_* и S15/S10 — устанавливаем ДО всех фильтров
                    # (нужно знать тип стратегии для выбора порогов R:R и применения стопов)
                    _is_scalper = sid.startswith("SC_")
                    _scalp_like = _is_scalper or sid in ("S15", "S10")

                    _entry_atr: Optional[float] = None  # ATR на момент входа (для журнала)

                    # Адаптивный SL/TP по ATR (множители зависят от режима и статистики)
                    if state.use_adaptive_sl:
                        try:
                            import pandas_ta as ta
                            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
                            if atr_series is None or len(atr_series) == 0:
                                raise ValueError("ATR empty")
                            atr = float(atr_series.iloc[-1])
                            _entry_atr = atr
                            if atr > 0:
                                sl_mult, rr = state.adaptive.atr_params(sid, regime=current_regime_name)
                                adaptive = adaptive_sl_tp(
                                    entry=signal.entry_price, atr=atr,
                                    side=signal.action,
                                    atr_multiplier_sl=sl_mult,
                                    rr_target=rr,
                                )
                                signal.stop_loss = adaptive["stop_loss"]
                                signal.take_profit = adaptive["take_profit"]
                        except Exception as e:
                            logger.debug(f"Adaptive SL skip: {e}")

                    # Глобальный R:R фильтр (после adaptive SL/TP — проверяем финальные значения):
                    #   Свинг/тренд >= 2.0 | Скальперы SC_*/S10/S15 >= 1.2
                    _rr_min_for_sid = 1.2 if _scalp_like else 2.0
                    _s_rr = GlobalTradeGuard._calc_rr(
                        signal.action, signal.entry_price, signal.stop_loss, signal.take_profit
                    )
                    if _s_rr is None or _s_rr < _rr_min_for_sid:
                        logger.info(
                            f"{sid}: ❌ R:R {_s_rr} < {_rr_min_for_sid} "
                            f"(TP={signal.take_profit:.6g} SL={signal.stop_loss:.6g})"
                        )
                        continue

                    # Добавляем в буфер для fusion-анализа — только сигналы прошедшие R:R
                    state.signal_buffer.update(sid, signal, strat.timeframe)

                    # TimeRate / anti-overtrade — только для скальп-стратегий
                    if _scalp_like and not state.training_mode:
                        _tr = state.time_rate.can_trade(sid, strat.symbol)
                        if not _tr["ok"]:
                            logger.info(f"{sid}: ⏳ TimeRate: {_tr['reason']}")
                            continue
                        _ot = state.time_rate.check_overtrade(sid, strat.symbol)
                        if not _ot["ok"]:
                            logger.info(f"{sid}: ⛔ Overtrade: {_ot['reason']}")
                            continue

                    # Проверка риск-менеджера
                    # Скальперы SC_* обходят лимит позиций, но уважают дневные стопы
                    if not state.training_mode:
                        if _is_scalper:
                            # Kill switch и pause проверяются по состоянию,
                            # которое _check_drawdown_limits() установил в начале тика.
                            if state.risk_manager.kill_switch:
                                logger.info(f"{sid}: ❌ KILL SWITCH: {state.risk_manager.kill_switch_reason}")
                                continue
                            _pause = state.risk_manager.daily_pause_until
                            if _pause is not None:
                                _now_ts = datetime.utcnow()
                                if _now_ts < _pause:
                                    _rem = int((_pause - _now_ts).total_seconds() / 60)
                                    logger.info(f"{sid}: ⏸ Дневная пауза, осталось {_rem} мин")
                                    continue
                        else:
                            check = state.risk_manager.can_open_trade(sid, balance)
                            if not check["allowed"]:
                                logger.info(f"{sid}: ❌ {check['reason']}")
                                continue

                    # Boost-режим: дополнительные проверки фазы
                    if state.boost.is_active and not state.training_mode and not _is_scalper:
                        boost_check = state.boost.can_open_trade(balance)
                        if not boost_check["allowed"]:
                            logger.info(f"{sid}: 🚫 Boost: {boost_check['reason']}")
                            continue

                    # Проверка плеча и notional экспозиции (пропускается в режиме обучения и для скальперов)
                    if not state.training_mode:
                        lev_check = state.risk_manager.check_leverage(sid, strat.leverage, balance)
                        if not lev_check["allowed"] and not _is_scalper:
                            logger.info(f"{sid}: ❌ {lev_check['reason']}")
                            continue
                        effective_leverage = lev_check["effective_leverage"]
                    else:
                        effective_leverage = strat.leverage

                    # Корреляция (пропускается в режиме обучения и для скальперов)
                    open_positions = [
                        {"symbol": s.symbol, "side": s.current_position["side"]}
                        for s in state.strategies.values() if s.current_position
                    ]
                    if not state.training_mode and not _is_scalper:
                        corr_check = state.correlation.can_open(strat.symbol, signal.action, open_positions)
                        if not corr_check["allowed"]:
                            logger.info(f"{sid}: ❌ {corr_check['reason']}")
                            continue

                    # ============== AI-АНАЛИЗ сигнала ==============
                    ai_score = None
                    ai_reasoning = ""
                    if state.ai.enabled:
                        risk_pct = state.risk_manager.adaptive_risk_pct(balance)
                        ai_result = await state.ai.analyze_signal_async(
                            signal_data={
                                "strategy_name": strat.NAME,
                                "strategy_id": sid,
                                "symbol": strat.symbol,
                                "action": signal.action,
                                "entry_price": signal.entry_price,
                                "stop_loss": signal.stop_loss,
                                "take_profit": signal.take_profit,
                                "filters_passed": signal.filters_passed,
                                "confidence": signal.confidence,
                                "reason": signal.reason,
                                "balance_usdt": round(balance, 2),
                                "risk_per_trade_usdt": round(balance * risk_pct / 100, 2),
                                "open_positions": state.risk_manager.open_positions_count,
                            },
                            df=df,
                            sentiment=sentiment_features,
                        )
                        ai_score = ai_result.get("score", 5)
                        approved = ai_result.get("approved", True)
                        ai_reasoning = ai_result.get("reasoning", ai_result.get("comment", ""))
                        if not approved and not state.training_mode:
                            # Уведомляем об отклонении и пропускаем сделку
                            asyncio.create_task(state.telegram.send(
                                f"🚫 <b>AI отверг: {signal.action} {strat.symbol}</b>\n"
                                f"Оценка: {ai_score}/10 | {sid}\n"
                                f"<i>{ai_reasoning[:200]}</i>"
                            ))
                            logger.info(f"{sid}: AI отверг (score={ai_score}) — {ai_reasoning}")
                            continue

                    # ============== ML ФИЛЬТР v2 (с sentiment + orderbook) ==============
                    ml_prediction = None
                    snapshot_id   = None   # явная инициализация — убираем 'in locals()' антипаттерн
                    features = {}
                    if state.ml_enabled and state.bybit:
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

                        # Проверяем что фичи реально извлечены (< 50 свечей → {})
                        if not features:
                            logger.debug(f"{sid}: недостаточно свечей для ML-фич, пропуск")
                            ml_prediction = None
                        else:
                            # Anomaly detection — в режиме обучения не блокируем (собираем аномалии тоже)
                            if state.anomaly_detector.model:
                                anom_score = state.anomaly_detector.score(features)
                                features["anomaly_score"] = anom_score
                                if anom_score < -0.65 and not state.training_mode:
                                    logger.warning(f"{sid}: 🚨 АНОМАЛИЯ (score={anom_score:.2f}) — пропускаем")
                                    await broadcast_log(f"🚨 {sid} аномальные условия рынка, skip", "warn")
                                    continue

                        # ML prediction
                        ml_prediction = state.ml_predictor.predict(sid, features) if features else None

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
                            "ml_prediction": ml_prediction.get("probability") if ml_prediction else None,
                            "ml_confidence": ml_prediction.get("probability") if ml_prediction else None,
                            "ml_model_version": ml_prediction.get("model_version") if ml_prediction else None,
                            "trade_taken": False,
                        })

                        # Решение: адаптивный порог уверенности (пропускается в режиме обучения)
                        adaptive_threshold = state.adaptive.confidence_threshold(sid)
                        if (ml_prediction and ml_prediction.get("available")
                                and state.ml_filter_mode == "strict"
                                and not state.training_mode):
                            prob = ml_prediction.get("probability", 0)
                            if prob < adaptive_threshold:
                                logger.info(
                                    f"{sid}: ❌ ML отверг (P={prob:.2f} < адапт.порог {adaptive_threshold:.2f})"
                                )
                                await broadcast_log(
                                    f"🤖 {sid} ML отверг сигнал (P={prob:.2f} < {adaptive_threshold:.2f})",
                                    "warn",
                                )
                                continue
                        elif ml_prediction and ml_prediction.get("available"):
                            emoji = "✅" if ml_prediction.get("should_take") else "⚠️"
                            await broadcast_log(
                                f"{emoji} {sid} ML P(win)={ml_prediction.get('probability', 0):.2f} "
                                f"| Sentiment={sentiment_features.get('sentiment_score', 0):+.2f}",
                                "info",
                            )

                    # ── Quality score для скальп-стратегий ─────────────────────────────
                    if _scalp_like and not state.training_mode:
                        try:
                            _ml_prob   = (ml_prediction.get("probability")
                                          if ml_prediction and ml_prediction.get("available")
                                          else None)
                            _sp_pct    = (signal.filters_passed.get("spread_pct", 0.0)
                                          if signal.filters_passed else 0.0)
                            _atr_q_pct = (_entry_atr / signal.entry_price * 100
                                          if _entry_atr and signal.entry_price else 0.0)
                            _quality   = calc_quality_score(df, signal.action, _sp_pct, _ml_prob, _atr_q_pct)
                            _min_q     = 8.0 if _is_scalper else 7.5
                            if _quality < _min_q:
                                logger.info(f"{sid}: 📊 Quality {_quality:.1f} < {_min_q} → skip")
                                continue
                            if signal.filters_passed is not None:
                                signal.filters_passed["quality_score"] = round(_quality, 2)
                        except Exception as _qe:
                            logger.debug(f"Quality score error: {_qe}")

                    # ── Ограничение параллельных скальперов SC_* ────────────────────────
                    if _is_scalper and not state.training_mode:
                        _active_scalps = sum(
                            1 for _k2, _s2 in state.strategies.items()
                            if _k2.startswith("SC_") and _s2.current_position
                        )
                        if _active_scalps >= state.max_parallel_scalps:
                            logger.info(
                                f"{sid}: 🚫 Лимит параллельных скальпов "
                                f"({_active_scalps}/{state.max_parallel_scalps})"
                            )
                            continue

                    # Boost-режим: переопределить SL/TP и leverage
                    boost_params = state.boost.get_risk_params()
                    effective_boost_leverage = strat.leverage  # дефолт — собственное плечо стратегии
                    if boost_params:
                        signal.stop_loss, signal.take_profit = state.boost.apply_sl_tp(
                            entry=signal.entry_price,
                            side=signal.action,
                            original_sl=signal.stop_loss,
                            original_tp=signal.take_profit,
                        )
                        effective_boost_leverage = boost_params["leverage"]  # не мутируем strat.leverage

                    # ============== Position Sizing ==============
                    if state.boost.is_active:
                        qty = state.boost.calculate_qty(balance, signal.entry_price, effective_boost_leverage)
                    elif state.use_kelly and ml_prediction and ml_prediction.get("available"):
                        kelly_res = kelly_position_size(
                            balance=balance,
                            entry=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            win_probability=ml_prediction.get("probability") if ml_prediction.get("probability") is not None else 0.5,
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
                        # Используем ATR уже вычисленный в блоке adaptive SL/TP
                        _atr_val = _entry_atr or 0.0
                        qty = state.risk_manager.calculate_position_size(
                            balance=balance,
                            entry_price=signal.entry_price,
                            stop_loss_price=signal.stop_loss,
                            leverage=effective_leverage,
                            atr=_atr_val,
                            min_notional=state.risk_manager.min_trade_usdt,
                        )
                        if _atr_val:
                            logger.debug(f"{sid}: ATR={_atr_val:.6f} → qty={qty}")

                        # Кап риска в USD на сделку (cfg.MAX_RISK_PER_TRADE_USDT из config.py)
                        if cfg.MAX_RISK_PER_TRADE_USDT > 0 and signal.entry_price and signal.stop_loss:
                            _sl_d = abs(signal.entry_price - signal.stop_loss)
                            if _sl_d > 0 and qty * _sl_d > cfg.MAX_RISK_PER_TRADE_USDT:
                                qty = round(cfg.MAX_RISK_PER_TRADE_USDT / _sl_d, 6)
                                logger.info(f"{sid}: 💰 Риск скейлирован до ${cfg.MAX_RISK_PER_TRADE_USDT}")

                        # Reversal Engine: уменьшаем объём если size_factor < 1.0
                        if signal.size_factor < 1.0:
                            qty = round(qty * signal.size_factor, 6)
                            logger.info(f"{sid}: 🔄 Reversal объём ×{signal.size_factor:.0%} → qty={qty}")

                    # ML protection: уменьшаем объём при низком WR (≥20 сделок)
                    if strat.trades >= 20:
                        _wr20 = strat.rolling_wr_20
                        if _wr20 < 0.35:
                            qty = round(qty * 0.25, 6)
                            logger.warning(f"{sid}: 🔴 ML protect qty×0.25 (WR={_wr20:.0%})")
                        elif _wr20 < 0.40:
                            qty = round(qty * 0.50, 6)
                            logger.info(f"{sid}: 🟡 ML protect qty×0.50 (WR={_wr20:.0%})")
                        elif _wr20 < 0.45:
                            qty = round(qty * 0.75, 6)
                            logger.info(f"{sid}: 🟡 ML protect qty×0.75 (WR={_wr20:.0%})")

                    # ============== GlobalTradeGuard ==============
                    _guard_h1 = _get_h1_cached(strat.symbol) if state.bybit else None
                    _guard_result = state.trade_guard.check(
                        strategy_id=sid,
                        signal_action=signal.action,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        qty=qty,
                        leverage=effective_leverage,
                        balance=balance,
                        df=df,
                        df_h1=_guard_h1,
                        ml_probability=(
                            ml_prediction.get("probability")
                            if ml_prediction and ml_prediction.get("available") else None
                        ),
                        signal_confidence=signal.confidence,
                        scalp_mode=_scalp_like,
                    )
                    if not _guard_result.approved and not state.training_mode:
                        logger.info(
                            f"{sid}: 🛡 Guard BLOCKED {signal.action} {strat.symbol}: "
                            f"{_guard_result.blocked_by}"
                        )
                        continue

                    # ── Нормализация TP + фильтр качества входа ─────────────
                    _sig_pre_norm = signal   # сохраняем до нормализации
                    signal = normalize_take_profit(signal, df, _scalp_like)
                    if signal is None:
                        if state.training_mode:
                            signal = _sig_pre_norm   # в обучении не блокируем
                        else:
                            logger.info(f"{sid}: 🚫 TP_NORM заблокировал — цена слишком близко к уровню")
                            continue

                    # ── Подтверждение входа: нет ножей, есть разворот ──────────
                    # В режиме обучения пропускаем — собираем любые данные
                    if not state.training_mode:
                        _entry_check = validate_entry_confirmation(df, signal.action, _scalp_like)
                        if not _entry_check["entry_allowed"]:
                            reason_code = _entry_check["entry_block_reason"]
                            candles_dir = _entry_check.get("last_3_candles_direction", "?")
                            logger.info(
                                f"{sid}: 🔪 EntryFilter BLOCKED {signal.action} {strat.symbol} "
                                f"[{reason_code}] свечи={candles_dir} "
                                f"EMA9={_entry_check.get('ema9_position','?')}"
                            )
                            continue
                        # Добавляем в filters_passed для логирования
                        if signal.filters_passed is None:
                            signal.filters_passed = {}
                        signal.filters_passed.update({
                            "entry_allowed":     True,
                            "last_3_candles":    _entry_check.get("last_3_candles_direction", "?"),
                            "ema9_pos":          _entry_check.get("ema9_position", "?"),
                            "reversal_detected": _entry_check.get("reversal_candle_detected", False),
                            "confirm_detected":  _entry_check.get("confirmation_candle_detected", False),
                        })

                    # ============== Открытие позиции ==============
                    _log_trade_attempt(
                        mode=state.trading_mode,
                        strategy_id=sid,
                        symbol=signal.symbol,
                        side=signal.action,
                        entry=signal.entry_price,
                        sl=signal.stop_loss,
                        tp=signal.take_profit,
                        qty=qty,
                        leverage=effective_leverage,
                        balance=balance,
                        reason=getattr(signal, "reason", ""),
                    )
                    if state.paper_mode:
                        paper_result = state.paper.open_position(signal, sid, qty, effective_leverage)
                        if not paper_result.get("success"):
                            logger.warning(f"[PAPER] {sid} отказ: {paper_result.get('reason')}")
                            continue
                        strat.register_position(
                            "Buy" if signal.action == "BUY" else "Sell",
                            signal.entry_price, signal.stop_loss, signal.take_profit,
                        )
                        strat.current_position["qty"] = qty
                        strat.current_position["leverage"] = effective_leverage
                        strat.current_position["opened_at"] = state.paper.positions[signal.symbol]["opened_at"]
                        if _scalp_like:
                            state.time_rate.register_trade_open(sid, strat.symbol)
                        notional = qty * signal.entry_price
                        state.risk_manager.register_position_open(sid, notional)
                        _ptid = state.journal.log_trade({
                            "strategy_id":   sid,
                            "strategy_name": strat.NAME,
                            "symbol":        signal.symbol,
                            "side":          signal.action,
                            "entry_price":   signal.entry_price,
                            "qty":           qty,
                            "leverage":      effective_leverage,
                            "stop_loss":     signal.stop_loss,
                            "take_profit":   signal.take_profit,
                            "initial_sl":    signal.stop_loss,
                            "initial_tp":    signal.take_profit,
                            "atr_at_entry":  _entry_atr,
                            "signal_reason": signal.reason,
                            "filters_passed": signal.filters_passed,
                            "ai_score":      ai_score,
                            "paper_trading": True,
                            "opened_at":     strat.current_position["opened_at"],
                            "market_regime": current_regime_name,
                            "consecutive_losses_at_entry": strat.consecutive_losses,
                        })
                        strat.current_position["journal_trade_id"] = _ptid
                        state.paper.set_journal_id(signal.symbol, _ptid)
                        asyncio.create_task(state.telegram.notify_trade_open(
                            sid, signal.symbol, signal.action,
                            signal.entry_price, signal.stop_loss, signal.take_profit,
                            signal.reason,
                            qty=qty, leverage=effective_leverage, df=df,
                            timeframe=strat.timeframe,
                            ai_score=ai_score, ai_reasoning=ai_reasoning,
                        ))
                    else:
                        result = state.bybit.place_order(
                            symbol=signal.symbol,
                            side="Buy" if signal.action == "BUY" else "Sell",
                            qty=qty,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            leverage=effective_leverage,
                        )
                        if result.get("success"):
                            strat.register_position(
                                "Buy" if signal.action == "BUY" else "Sell",
                                signal.entry_price, signal.stop_loss, signal.take_profit,
                            )
                            strat.current_position["qty"] = qty
                            strat.current_position["leverage"] = effective_leverage
                            if _scalp_like:
                                state.time_rate.register_trade_open(sid, strat.symbol)
                            notional = qty * signal.entry_price
                            state.risk_manager.register_position_open(sid, notional)

                            trade_id = state.journal.log_trade({
                                "strategy_id":   sid,
                                "strategy_name": strat.NAME,
                                "symbol":        signal.symbol,
                                "side":          signal.action,
                                "entry_price":   signal.entry_price,
                                "qty":           qty,
                                "leverage":      effective_leverage,
                                "stop_loss":     signal.stop_loss,
                                "take_profit":   signal.take_profit,
                                "initial_sl":    signal.stop_loss,
                                "initial_tp":    signal.take_profit,
                                "atr_at_entry":  _entry_atr,
                                "signal_reason": signal.reason,
                                "filters_passed": signal.filters_passed,
                                "ai_score":      ai_score,
                                "opened_at":     datetime.utcnow().isoformat(),
                                "market_regime": current_regime_name,
                                "consecutive_losses_at_entry": strat.consecutive_losses,
                            })
                            strat.current_position["journal_trade_id"] = trade_id

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
                                qty=qty, leverage=effective_leverage, df=df,
                                timeframe=strat.timeframe,
                                ai_score=ai_score, ai_reasoning=ai_reasoning,
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
                if state.paper_mode and state.paper.positions:
                    current_prices = {
                        sym: df.iloc[-1]["close"]
                        for sym, df in klines_data.items()
                    }
                    closed_paper = state.paper.check_positions(current_prices)
                    for closed_pos in closed_paper:
                        c_sym = closed_pos.get("symbol") or closed_pos.get("strategy_id", "?")
                        pnl = closed_pos.get("pnl_usd", 0)
                        reason = closed_pos.get("exit_reason", "?")
                        # Находим стратегию по символу и снимаем позицию
                        for _sid, _strat in state.strategies.items():
                            if _strat.symbol == c_sym and _strat.current_position:
                                _opened_at = _strat.current_position.get("opened_at")
                                if hasattr(_opened_at, "isoformat"):
                                    _opened_at = _opened_at.isoformat()
                                _paper_jtid = _strat.current_position.get("journal_trade_id")
                                close_res = _strat.close_position(
                                    closed_pos.get("exit_price", 0),
                                    qty=_strat.current_position.get("qty", 0),
                                )
                                state.risk_manager.register_trade_result(_sid, pnl)
                                state.risk_manager.register_position_close(_sid)
                                if _sid.startswith("SC_") or _sid in ("S5", "S10", "S15"):
                                    state.time_rate.register_close(_sid, _strat.symbol, pnl)
                                _paper_r = close_res.get("r_multiple", 0)
                                state.adaptive.record(_sid, _paper_r)
                                if _paper_jtid:
                                    state.journal.update_trade_close(
                                        _paper_jtid,
                                        exit_price=closed_pos.get("exit_price", 0),
                                        pnl_usd=pnl,
                                        pnl_pct=close_res.get("pnl_pct", 0),
                                        exit_reason=reason,
                                        r_multiple=_paper_r,
                                    )
                                # AI-советы каждые 10 закрытых сделок стратегии
                                if _strat.trades > 0 and _strat.trades % 10 == 0:
                                    asyncio.create_task(_auto_ai_improve(
                                        _sid, _strat.NAME, _strat.symbol, _strat.trades
                                    ))
                                asyncio.create_task(state.telegram.notify_trade_close(
                                    _sid, c_sym, pnl, reason,
                                    leverage=closed_pos.get("leverage", 1),
                                    df=klines_data.get(c_sym),
                                    entry=closed_pos.get("entry_price"),
                                    side=closed_pos.get("side"),
                                    sl=closed_pos.get("stop_loss"),
                                    tp=closed_pos.get("take_profit"),
                                    exit_price=closed_pos.get("exit_price"),
                                    opened_at=_opened_at,
                                ))
                                await broadcast_log(
                                    f"{'✅' if pnl > 0 else '❌'} [PAPER] {_sid} {c_sym} {reason} → {pnl:+.2f} USDT",
                                    "ok" if pnl > 0 else "warn",
                                )
                                # Guard: учитываем результат для кулдауна
                                if pnl > 0:
                                    state.trade_guard.record_win(_sid)
                                else:
                                    state.trade_guard.record_loss(_sid)
                                # Monitor: сбрасываем состояние закрытой позиции
                                state.position_monitor.reset_state(_sid)
                                break

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
                                _real_jtid = pos.get("journal_trade_id")

                                # Корректный PnL
                                close_result = strat.close_position(exit_price, qty=qty)
                                pnl_usd = close_result.get("pnl_usd", 0)
                                r_multiple = close_result.get("r_multiple", 0)

                                state.risk_manager.register_trade_result(sid, pnl_usd)
                                state.risk_manager.register_position_close(sid)
                                if sid.startswith("SC_") or sid in ("S5", "S10", "S15"):
                                    state.time_rate.register_close(sid, strat.symbol, pnl_usd)
                                # Адаптивное самообучение: записываем результат в R-multiple
                                state.adaptive.record(sid, r_multiple)
                                # AI-советы каждые 10 закрытых сделок стратегии
                                if strat.trades > 0 and strat.trades % 10 == 0:
                                    asyncio.create_task(_auto_ai_improve(
                                        sid, strat.NAME, strat.symbol, strat.trades
                                    ))
                                # Strategy Advisor: автоанализ всех стратегий
                                total_trades = state.risk_manager.daily_trades_count
                                if state.advisor.should_auto_run(total_trades):
                                    asyncio.create_task(_auto_advisor_run())

                                exit_reason = "TP" if pnl_usd > 0 else "SL"

                                if _real_jtid:
                                    state.journal.update_trade_close(
                                        _real_jtid,
                                        exit_price=exit_price,
                                        pnl_usd=pnl_usd,
                                        pnl_pct=close_result.get("pnl_pct", 0),
                                        exit_reason=exit_reason,
                                        r_multiple=r_multiple,
                                    )

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
                                    _drift_row = None
                                    try:
                                        _sql = state.db_pool.adapt(
                                            "SELECT ml_prediction FROM signal_snapshots WHERE id = ?"
                                        )
                                        with state.db_pool.connection() as _dc:
                                            if state.db_pool.is_mysql:
                                                with _dc.cursor() as _cc:
                                                    _cc.execute(_sql, (snapshot_id,))
                                                    _drift_row = _cc.fetchone()
                                            else:
                                                import sqlite3 as _sq3
                                                _dc.row_factory = _sq3.Row
                                                _r = _dc.execute(_sql, (snapshot_id,)).fetchone()
                                                _drift_row = dict(_r) if _r else None
                                    except Exception:
                                        pass
                                    if _drift_row and _drift_row.get("ml_prediction") is not None:
                                        state.drift_monitor.record(
                                            sid, _drift_row["ml_prediction"], 1 if outcome == "win" else 0,
                                        )
                                        # Проверка drift
                                        drift = state.drift_monitor.should_disable_model(sid)
                                        if drift["disable"]:
                                            logger.critical(f"🚨 DRIFT detected {sid}: {drift['reason']}")
                                            state.ml_predictor.models.pop(sid, None)
                                            asyncio.create_task(state.telegram.send(
                                                f"🚨 <b>ML DRIFT</b> {sid}\n{drift['reason']}\nМодель отключена"
                                            ))

                                    await broadcast_log(
                                        f"🎯 {sid} → {outcome.upper()} R={r_multiple:+.2f}",
                                        "ok" if outcome == "win" else "warn",
                                    )

                                asyncio.create_task(state.telegram.notify_trade_close(
                                    sid, strat.symbol, pnl_usd, exit_reason,
                                    leverage=pos.get("leverage", 1),
                                    df=klines_data.get(strat.symbol),
                                    entry=pos["entry"], side=pos["side"],
                                    sl=pos["sl"], tp=pos["tp"],
                                    exit_price=exit_price,
                                    opened_at=pos.get("opened_at"),
                                ))
                                # Boost: регистрируем результат, обновляем фазу
                                new_balance = state.bybit.get_balance("USDT")
                                state.boost.register_trade(pnl_usd, new_balance)
                                # Guard: учитываем результат для кулдауна
                                if pnl_usd > 0:
                                    state.trade_guard.record_win(sid)
                                else:
                                    state.trade_guard.record_loss(sid)
                                # Monitor: сбрасываем состояние закрытой позиции
                                state.position_monitor.reset_state(sid)

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

                # Авто-отключение убрано: RiskManager уже блокирует через 12ч cooldown
                # после 3 убытков подряд. Двойной бан (auto_disabled=True) мешал перезапуску SC_*.

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
        "adaptive_params": state.adaptive.status(),
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
# Часовой отчёт + ежедневный итог в Telegram
# ─────────────────────────────────────────────────────────────

async def _hourly_report_loop():
    """Каждый час отправляет полный отчёт в Telegram."""
    await asyncio.sleep(60)
    while True:
        try:
            await _send_hourly_report()
        except Exception as e:
            logger.error(f"[HourlyReport] Ошибка: {e}")
        await asyncio.sleep(3600)


async def _daily_summary_loop():
    """В 00:00 UTC отправляет дневной итог за прошедший день."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            next_midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc,
            )
            wait_sec = (next_midnight - now).total_seconds()
            await asyncio.sleep(wait_sec)
            await _send_daily_summary()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[DailySummary] Ошибка: {e}")
            await asyncio.sleep(3600)


def _get_balance() -> float:
    if state.paper_mode:
        return state.paper.balance
    if state.bybit:
        try:
            return state.bybit.get_balance("USDT")
        except Exception:
            pass
    return 0.0


def _unrealized_pnl(pos: dict, current_price: float) -> float:
    """Примерная нереализованная прибыль позиции."""
    entry = pos.get("entry", 0.0)
    qty   = pos.get("qty", 0.0)
    side  = pos.get("side", "Buy")
    if side == "Buy":
        return (current_price - entry) * qty
    return (entry - current_price) * qty


async def _send_hourly_report():
    """Полный часовой отчёт с дневной статистикой, позициями и кулдаунами."""
    rm      = state.risk_manager
    balance = _get_balance()
    now_utc = datetime.now(timezone.utc)

    # ── Дневные метрики (из risk_manager) ──────────────────────────────────
    day_pnl    = rm.daily_pnl
    day_trades = rm.daily_trades_count
    day_losses = rm.daily_losses_count
    day_wins   = day_trades - day_losses
    day_wr     = (day_wins / day_trades * 100) if day_trades > 0 else 0.0
    day_pct    = (day_pnl / rm.daily_start_balance * 100) if rm.daily_start_balance else 0.0

    day_pnl_str  = f"{'+'if day_pnl>=0 else ''}{day_pnl:.2f} USDT ({day_pct:+.1f}%)"
    day_trade_str = f"{day_trades} ({day_wins}↑ {day_losses}↓) | WR: {day_wr:.1f}%"

    # ── Сессионные метрики (из стратегий) ──────────────────────────────────
    all_pnls   = [round(p, 2) for s in state.strategies.values() for p in s.history]
    sess_pnl   = sum(s.pnl    for s in state.strategies.values())
    sess_trades= sum(s.trades  for s in state.strategies.values())
    sess_wins  = sum(s.wins    for s in state.strategies.values())
    sess_wr    = (sess_wins / sess_trades * 100) if sess_trades > 0 else 0.0
    best_trade = max(all_pnls, default=0.0)
    worst_trade= min(all_pnls, default=0.0)

    # ── Открытые позиции с unrealized PnL ──────────────────────────────────
    open_lines = []
    for sid, strat in sorted(state.strategies.items()):
        pos = strat.current_position
        if not pos:
            continue
        cp = 0.0
        try:
            cp_raw = state.tickers.get(strat.symbol, {})
            cp = float(cp_raw.get("price", pos["entry"]))
        except Exception:
            cp = pos.get("entry", 0.0)
        upnl = _unrealized_pnl(pos, cp)
        sign = "+" if upnl >= 0 else ""
        arrow = "📈" if pos["side"] == "Buy" else "📉"
        open_lines.append(
            f"  {arrow} <code>{sid}</code> {strat.symbol} @ {pos['entry']:.4g} | uPnL: <b>{sign}{upnl:.2f}</b>"
        )

    pos_block = "\n".join(open_lines) if open_lines else "  нет открытых позиций"

    # ── TimeRate кулдауны ──────────────────────────────────────────────────
    tr_status   = state.time_rate.status()
    cool_lines  = []
    for sym, rem in tr_status.get("cooled_symbols", {}).items():
        cool_lines.append(f"  ⏳ {sym}: {rem:.1f} мин (символ)")
    for stid, rem in tr_status.get("blocked_strategies", {}).items():
        cool_lines.append(f"  🚫 {stid}: {rem:.1f} мин (стратегия)")
    cool_block = "\n".join(cool_lines) if cool_lines else "  нет активных кулдаунов"

    # ── Топ-5 стратегий по PnL за сессию ────────────────────────────────────
    strat_stats = [
        (sid, s.pnl, s.trades, s.wins)
        for sid, s in state.strategies.items()
        if s.trades > 0
    ]
    strat_stats.sort(key=lambda x: x[1], reverse=True)
    top_lines = []
    for sid, pnl, tr, wr_n in strat_stats[:5]:
        wr_pct = (wr_n / tr * 100) if tr else 0
        sign   = "+" if pnl >= 0 else ""
        top_lines.append(
            f"  <code>{sid:<8}</code> {sign}{pnl:.2f} USDT | {tr} сд | WR {wr_pct:.0f}%"
        )
    top_block = "\n".join(top_lines) if top_lines else "  нет сделок"

    # ── Статус защиты ──────────────────────────────────────────────────────
    kill_str   = "🔴 KILL SWITCH" if rm.kill_switch else "🟢 активна"
    pause_str  = ""
    if rm.daily_pause_until and datetime.utcnow() < rm.daily_pause_until:
        rem_p = int((rm.daily_pause_until - datetime.utcnow()).total_seconds() / 60)
        pause_str = f" | ⏸ пауза {rem_p} мин"

    mode   = "📄 Paper" if state.paper_mode else "💰 Real"
    status = "🟢 Работает" if state.bot_running else "🔴 Остановлен"

    text = (
        f"⏰ <b>Часовой отчёт</b> — {now_utc.strftime('%H:%M UTC')}\n"
        f"{'─'*30}\n"
        f"{mode} | {status}\n"
        f"💰 Баланс: <b>{balance:.2f} USDT</b>\n\n"
        f"📅 <b>Сегодня:</b>\n"
        f"  Сделок: {day_trade_str}\n"
        f"  PnL: <b>{day_pnl_str}</b>\n"
        f"  Лучшая: <code>{best_trade:+.2f}</code> | Худшая: <code>{worst_trade:+.2f}</code>\n\n"
        f"📊 <b>Сессия всего:</b>\n"
        f"  {sess_trades} сделок | WR {sess_wr:.1f}% | "
        f"PnL <b>{'+'if sess_pnl>=0 else ''}{sess_pnl:.2f} USDT</b>\n\n"
        f"📂 <b>Позиции ({len(open_lines)}):</b>\n{pos_block}\n\n"
        f"⏳ <b>TimeRate:</b>\n{cool_block}\n\n"
        f"🏆 <b>Топ стратегии:</b>\n{top_block}\n\n"
        f"🛡 Защита: {kill_str}{pause_str} | "
        f"Day PnL: {day_pnl:+.2f} USDT | Лимит: -{rm.daily_max_loss_pct}%"
    )
    await state.telegram.send(text)


async def _send_daily_summary():
    """Полный итог за прошедший день — отправляется в 00:00 UTC."""
    rm      = state.risk_manager
    balance = _get_balance()
    now_utc = datetime.now(timezone.utc)

    day_pnl    = rm.daily_pnl
    day_trades = rm.daily_trades_count
    day_losses = rm.daily_losses_count
    day_wins   = day_trades - day_losses
    day_wr     = (day_wins / day_trades * 100) if day_trades > 0 else 0.0
    day_start  = rm.daily_start_balance or balance
    day_pct    = (day_pnl / day_start * 100) if day_start else 0.0

    # Статистика по стратегиям за сессию (лучшая / худшая / топ)
    all_pnls    = [round(p, 2) for s in state.strategies.values() for p in s.history]
    best_trade  = max(all_pnls, default=0.0)
    worst_trade = min(all_pnls, default=0.0)
    sess_pnl    = sum(s.pnl for s in state.strategies.values())

    # Топ и аутсайдеры
    by_pnl = sorted(
        [(sid, s.pnl, s.trades, s.wins)
         for sid, s in state.strategies.items() if s.trades > 0],
        key=lambda x: x[1], reverse=True,
    )
    top_line  = ""
    worst_line = ""
    if by_pnl:
        t = by_pnl[0]
        top_line = (
            f"  🥇 <code>{t[0]}</code>: {t[1]:+.2f} USDT | "
            f"{t[2]} сд | WR {t[3]/t[2]*100:.0f}%"
        )
        w = by_pnl[-1]
        worst_line = (
            f"  ⚠️ <code>{w[0]}</code>: {w[1]:+.2f} USDT | "
            f"{w[2]} сд | WR {w[3]/w[2]*100:.0f}%"
        )

    # Активные кулдауны на начало нового дня
    tr_status  = state.time_rate.status()
    n_cool     = len(tr_status.get("cooled_symbols", {}))
    n_blocked  = len(tr_status.get("blocked_strategies", {}))
    cool_note  = f"{n_cool} символов, {n_blocked} стратегий" if (n_cool or n_blocked) else "нет"

    emoji = "🟢" if day_pnl >= 0 else "🔴"
    text = (
        f"{emoji} <b>Дневной итог — {now_utc.strftime('%d.%m.%Y')}</b>\n"
        f"{'═'*30}\n\n"
        f"💰 Баланс: <b>{balance:.2f} USDT</b>\n"
        f"  Изменение за день: <b>{day_pnl:+.2f} USDT ({day_pct:+.1f}%)</b>\n\n"
        f"📈 <b>Итоги дня:</b>\n"
        f"  Сделок: <b>{day_trades}</b>  ({day_wins}↑ {day_losses}↓)\n"
        f"  Win Rate: <b>{day_wr:.1f}%</b>\n"
        f"  Лучшая: <code>{best_trade:+.2f} USDT</code>\n"
        f"  Худшая: <code>{worst_trade:+.2f} USDT</code>\n\n"
        f"🏆 <b>Рекорды дня:</b>\n"
        f"{top_line or '  нет сделок'}\n"
        f"{worst_line}\n\n"
        f"📊 PnL сессии (всего): <b>{sess_pnl:+.2f} USDT</b>\n"
        f"⏳ Кулдауны на старт: {cool_note}\n\n"
        f"<i>Новый день начат. Удачной торговли! 🚀</i>"
    )
    await state.telegram.send(text)


# ============================================================
# FastAPI приложение
# ── Вспомогательные корутины для Telegram Commander ──────────────────────────
async def _tg_start_bot():
    """Запустить торговый цикл из Telegram команды."""
    if not state.bot_running:
        state.bot_running = True
        if not state.trading_loop_task or state.trading_loop_task.done():
            state.trading_loop_task = asyncio.create_task(trading_loop())

async def _tg_stop_bot():
    """Остановить торговый цикл из Telegram команды."""
    state.bot_running = False
    if state.trading_loop_task and not state.trading_loop_task.done():
        state.trading_loop_task.cancel()


# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_strategies()

    # ── Определение режима торговли ──────────────────────────────
    _trading_mode = os.getenv("TRADING_MODE", "").upper()
    if not _trading_mode:
        # Обратная совместимость
        if os.getenv("PAPER_TRADING", "true").lower() in ("1", "true", "yes"):
            _trading_mode = "PAPER"
        elif os.getenv("BYBIT_TESTNET", "true").lower() in ("1", "true", "yes"):
            _trading_mode = "TESTNET"
        else:
            _trading_mode = "LIVE"
    state.trading_mode = _trading_mode

    # ── Защита от случайной LIVE-торговли ────────────────────────
    if _trading_mode == "LIVE":
        confirm = os.getenv("LIVE_TRADING_CONFIRM", "false").lower() in ("1", "true", "yes")
        understand = os.getenv("I_UNDERSTAND_REAL_MONEY_RISK", "false").lower() in ("1", "true", "yes")
        if not (confirm and understand):
            logger.critical(
                "🚫 LIVE-режим заблокирован! "
                "Установи LIVE_TRADING_CONFIRM=true и I_UNDERSTAND_REAL_MONEY_RISK=true в .env"
            )
            raise SystemExit(
                "\n\n❌ LIVE TRADING ЗАБЛОКИРОВАН\n"
                "Для активации реальной торговли установи в .env:\n"
                "  LIVE_TRADING_CONFIRM=true\n"
                "  I_UNDERSTAND_REAL_MONEY_RISK=true\n"
                "Убедись, что понимаешь риски потери средств!\n"
            )

    if _trading_mode == "DRY_RUN":
        state.paper_mode = True
        logger.info("🔇 DRY_RUN режим: ордера не отправляются, только логи")
    elif _trading_mode == "PAPER":
        state.paper_mode = True
        logger.info("📄 PAPER режим: виртуальная торговля")
    elif _trading_mode == "TESTNET":
        state.paper_mode = False
        logger.info("🧪 TESTNET режим: реальные ордера на тестовой бирже")
    elif _trading_mode == "LIVE":
        state.paper_mode = False
        logger.warning("💰 LIVE режим: РЕАЛЬНЫЕ ДЕНЬГИ!")

    # ── Автоподключение Bybit из переменных окружения ─────────────────────────
    _bybit_key    = os.getenv("BYBIT_API_KEY", "").strip()
    _bybit_secret = os.getenv("BYBIT_API_SECRET", "").strip()
    _bybit_testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    if _bybit_key and _bybit_secret:
        try:
            state.bybit = BybitClient(_bybit_key, _bybit_secret, _bybit_testnet)
            _bal = state.bybit.get_balance("USDT")
            _net = "TESTNET" if _bybit_testnet else "MAINNET"
            logger.info(f"✅ Bybit {_net} подключён автоматически | Баланс: {_bal:.2f} USDT")
        except Exception as _e:
            logger.error(f"❌ Автоподключение Bybit не удалось: {_e}")
            state.bybit = None
    else:
        logger.info("ℹ️ BYBIT_API_KEY не задан — подключите через /api/connect или включите Paper Mode")

    # ── Paper Mode из переменной окружения ────────────────────────────────────
    if os.getenv("PAPER_TRADING", "false").lower() == "true":
        state.paper_mode = True
        logger.info("📄 Paper Trading Mode активирован (PAPER_TRADING=true)")

    # ── Read-only Bybit для рыночных данных (paper mode без API ключей) ───────
    # Свечи, тикеры, ордербук — публичные эндпоинты, не требуют авторизации.
    # Торговые операции в paper mode идут через PaperTrader, а не через Bybit.
    if not state.bybit and state.paper_mode:
        try:
            state.bybit = BybitClient("", "", testnet=False)
            logger.info("📡 Bybit read-only подключён для рыночных данных (paper mode)")
        except Exception as _ro_e:
            logger.warning(f"Bybit read-only init: {_ro_e}")

    # ── Scalp Mode по умолчанию ───────────────────────────────────────────────
    if os.getenv("SCALP_DEFAULT", "true").lower() != "false":
        activate_scalp_mode()
        logger.info("⚡ ScalperPro активирован автоматически (SCALP_DEFAULT=true)")

    # ── Авто-выбор символов по топу объёма (для всех стратегий + скальперов) ──
    asyncio.create_task(auto_select_symbols())
    asyncio.create_task(symbol_refresh_loop())
    logger.info("📊 Авто-выбор символов по объёму запущен (обновление каждые 4ч)")

    # ── Автостарт бота ────────────────────────────────────────────────────────
    if os.getenv("AUTO_START", "false").lower() == "true":
        if state.bybit or state.paper_mode:
            state.bot_running = True
            state.trading_loop_task = asyncio.create_task(trading_loop())
            logger.info("🚀 Бот запущен автоматически (AUTO_START=true)")
        else:
            logger.warning("AUTO_START=true но нет Bybit API и не включён Paper Mode — старт пропущен")

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

    # Multi-provider Orchestrator (Anthropic → OpenAI → Ollama)
    _anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    _openai_key    = os.getenv("OPENAI_API_KEY")
    _ollama_url    = os.getenv("OLLAMA_BASE_URL")
    if _anthropic_key or _openai_key or _ollama_url:
        state.orchestrator = ClaudeOrchestrator(
            api_key=_anthropic_key,
            openai_key=_openai_key,
            ollama_url=_ollama_url,
            execute_fn=_execute_bot_command,
            notify_fn=lambda msg, level: asyncio.create_task(state.telegram.send(msg)),
        )
        state.orchestrator_task = asyncio.create_task(orchestrator_loop())
        logger.info("🤖 Orchestrator запущен")
    else:
        logger.info("🤖 Оркестратор отключён — задайте ANTHROPIC_API_KEY / OPENAI_API_KEY / OLLAMA_BASE_URL")

    # ── Telegram Commander (двустороннее управление) ──────────────────────────
    _tg_token   = os.getenv("TELEGRAM_BOT_TOKEN", "") or os.getenv("TELEGRAM_TOKEN", "")
    _tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if _tg_token and _tg_chat_id:
        state.commander = TelegramCommander(
            bot_token=_tg_token,
            allowed_chat_id=_tg_chat_id,
            state_getter=lambda: state,
            start_fn=_tg_start_bot,
            stop_fn=_tg_stop_bot,
        )
        state.commander_task = asyncio.create_task(state.commander.run())
        logger.info("📱 Telegram Commander запущен (long-polling)")
    else:
        logger.info("📱 Telegram Commander отключён — задайте TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")

    # ── Часовой + дневной отчёт в Telegram ───────────────────────────────────
    if state.telegram.enabled:
        asyncio.create_task(_hourly_report_loop())
        asyncio.create_task(_daily_summary_loop())
        logger.info("📊 Часовой и ежедневный Telegram-отчёт запущен")

    logger.info("✅ Бэкенд готов")
    yield
    state.bot_running = False
    if state.trading_loop_task:
        state.trading_loop_task.cancel()
    if state.news_loop_task:
        state.news_loop_task.cancel()
    if state.auto_train_task:
        state.auto_train_task.cancel()
    if state.orchestrator_task:
        state.orchestrator_task.cancel()
    if state.commander_task:
        state.commander_task.cancel()
    if state.commander:
        await state.commander.close()
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


@app.get("/api/connection/status")
async def connection_status():
    """Статус подключения: Bybit API, AI, Bot."""
    # Bybit
    bybit_ok = False
    bybit_balance = None
    bybit_error = None
    bybit_testnet = None
    if state.bybit:
        try:
            bybit_balance = round(state.bybit.get_balance("USDT"), 4)
            bybit_ok = True
            bybit_testnet = getattr(state.bybit, "testnet", None)
        except Exception as e:
            bybit_error = str(e)

    # AI (Anthropic / OpenAI / Ollama)
    ai_ok = state.ai.enabled
    ai_provider = state.ai._provider.name if ai_ok and state.ai._provider else None
    ai_model = state.ai._provider.model if ai_ok and state.ai._provider else None

    return {
        "bybit": {
            "connected": bybit_ok,
            "testnet": bybit_testnet,
            "balance_usdt": bybit_balance,
            "error": bybit_error,
        },
        "ai": {
            "enabled": ai_ok,
            "provider": ai_provider,
            "model": ai_model,
            "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "ollama_url": os.getenv("OLLAMA_BASE_URL"),
        },
        "bot": {
            "running": state.bot_running,
            "paper_mode": state.paper_mode,
            "strategies": len(state.strategies),
            "news_enabled": state.news_enabled,
            "ml_enabled": state.ml_enabled,
        },
    }


@app.post("/api/connect")
async def connect(req: APIConnectRequest):
    try:
        state.bybit = BybitClient(req.api_key, req.api_secret, req.testnet)
        balance = state.bybit.get_balance("USDT")
        net = "TESTNET" if req.testnet else "MAINNET"
        logger.info(f"✅ Bybit {net} подключён через API | Баланс: {balance:.2f} USDT")
        return {
            "success": True,
            "balance": round(balance, 4),
            "testnet": req.testnet,
            "network": net,
        }
    except Exception as e:
        state.bybit = None
        logger.error(f"❌ Подключение Bybit не удалось: {e}")
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
async def export_csv(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """CSV выгрузка сделок с фильтрами по стратегии, символу, датам."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path = state.journal.export_to_csv(
        output_path=f"logs/trades_{ts}.csv",
        strategy_id=strategy_id,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
    )
    return FileResponse(path, filename=f"trades_{ts}.csv", media_type="text/csv")


@app.get("/api/journal/export/excel")
async def export_excel(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """Excel с листами: All Trades, By Strategy, Heatmap, Equity Curve."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path = state.journal.export_to_excel(
        output_path=f"logs/trades_{ts}.xlsx",
        strategy_id=strategy_id,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
    )
    return FileResponse(
        path, filename=f"trades_{ts}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/journal/export/snapshots")
async def export_snapshots(
    strategy_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """CSV с ML-снапшотами: фичи + исход (outcome, pnl_r) — для внешнего анализа."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path = state.journal.export_snapshots_csv(
        output_path=f"logs/snapshots_{ts}.csv",
        strategy_id=strategy_id,
        from_date=from_date,
        to_date=to_date,
    )
    return FileResponse(path, filename=f"snapshots_{ts}.csv", media_type="text/csv")


@app.get("/api/journal/equity")
async def equity_curve(
    strategy_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """Нарастающий PnL по времени для построения equity-кривой."""
    return state.journal.get_equity_curve(
        strategy_id=strategy_id,
        from_date=from_date,
        to_date=to_date,
    )


@app.get("/api/journal/export/db")
async def export_db():
    """Скачать весь SQLite файл. При MySQL — отдаёт CSV-дамп."""
    db_path = getattr(state.journal.pool, "db_path", None)
    if db_path and Path(db_path).exists():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
        return FileResponse(
            db_path,
            filename=f"baibit_trades_{ts}.db",
            media_type="application/octet-stream",
        )
    path = state.journal.export_to_csv()
    return FileResponse(path, filename="trades_dump.csv", media_type="text/csv")


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


@app.get("/api/guard/status")
async def guard_status():
    """Статус GlobalTradeGuard: серии убытков, кулдауны по стратегиям."""
    return state.trade_guard.get_status()


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
                    paper_pos = state.paper.positions.get(strat.symbol, {})
                    exit_price = strat.current_position.get("entry", paper_pos.get("entry_price", 0))
                    if exit_price and strat.current_position:
                        state.paper._close_position(strat.symbol, exit_price, reason="emergency_close")
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


@app.get("/api/news/recommendations")
async def news_recommendations():
    """Последние AI-рекомендации (кешированные, обновляются раз в 30 мин)."""
    cached = getattr(state, "_cached_ai_rec", None)
    if cached:
        return cached
    return {"available": False, "reason": "Рекомендации ещё не сгенерированы — запустите бота"}


@app.post("/api/news/recommendations/refresh")
async def news_recommendations_refresh():
    """Принудительно обновить AI-рекомендации прямо сейчас."""
    if not state.news_enabled:
        return {"available": False, "reason": "News модуль отключён"}
    portfolio_ctx = {
        "balance": (state.bybit.get_balance("USDT") if state.bybit else 0),
        "open_positions": state.risk_manager.open_positions_count,
        "daily_pnl": state.risk_manager.daily_pnl,
        "active_strategies": [sid for sid, s in state.strategies.items() if s.enabled],
    }
    result = await state.news_manager.get_trading_recommendations(portfolio_context=portfolio_ctx)
    state._cached_ai_rec = result
    state._last_ai_rec = datetime.utcnow()
    return result


# ============================================================
# ADAPTIVE PARAMS ENDPOINTS
# ============================================================
@app.get("/api/adaptive/status")
async def adaptive_status():
    """Текущие адаптивные параметры всех стратегий (порог, Kelly, размер позиции)."""
    return {"status": state.adaptive.status(), "summary": state.adaptive.summary_line()}


@app.post("/api/adaptive/reset/{strategy_id}")
async def adaptive_reset(strategy_id: str):
    """Сброс адаптивной статистики для одной стратегии (возврат к базовым параметрам)."""
    if strategy_id not in state.adaptive._stats:
        return {"ok": False, "error": "Стратегия не найдена"}
    base = state.adaptive._base_conf.get(strategy_id, 0.60)
    state.adaptive._stats[strategy_id]._pnls.clear()
    state.adaptive._conf[strategy_id] = base
    return {"ok": True, "strategy_id": strategy_id, "confidence_reset_to": base}


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
    _pool = state.db_pool
    _q = _pool.adapt("""
        SELECT ml_prediction, outcome FROM signal_snapshots
        WHERE strategy_id = ? AND ml_prediction IS NOT NULL
        AND outcome IS NOT NULL
    """)
    with _pool.connection() as conn:
        if _pool.is_mysql:
            with conn.cursor() as _c:
                _c.execute(_q, (req.strategy_id,))
                rows = _c.fetchall()
        else:
            rows = conn.execute(_q, (req.strategy_id,)).fetchall()
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
    if req.mode not in ("safe", "moderate", "aggressive", "scalp"):
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
    if req.mode not in ("safe", "moderate", "aggressive", "scalp"):
        raise HTTPException(400, "mode должен быть: safe | moderate | aggressive")
    return BoostCalculator.full_analysis(req.initial, req.target, req.days, req.mode)


# ============================================================
# ORCHESTRATOR ENDPOINTS
# ============================================================
@app.get("/api/orchestrator/status")
async def orchestrator_status():
    """Статус Claude-оркестратора: последний цикл, расписание, история."""
    if not state.orchestrator:
        return {"enabled": False, "reason": "ANTHROPIC_API_KEY не настроен"}
    return state.orchestrator.get_status()


@app.post("/api/orchestrator/trigger")
async def orchestrator_trigger():
    """Принудительный запуск цикла оркестратора (не дожидаясь таймера)."""
    if not state.orchestrator or not state.orchestrator.enabled:
        raise HTTPException(400, "Оркестратор недоступен (нет ключа или отключён)")
    full = await full_status()
    result = await state.orchestrator.run_cycle(full)
    if result:
        return result.to_dict()
    return {"error": "Цикл не вернул результат"}


@app.post("/api/orchestrator/enable")
async def orchestrator_enable():
    """Включить оркестратор."""
    if not state.orchestrator:
        raise HTTPException(400, "Оркестратор не инициализирован")
    state.orchestrator.enabled = True
    return {"enabled": True}


@app.post("/api/orchestrator/disable")
async def orchestrator_disable():
    """Выключить оркестратор (приостановить без перезапуска)."""
    if not state.orchestrator:
        raise HTTPException(400, "Оркестратор не инициализирован")
    state.orchestrator.enabled = False
    return {"enabled": False}


@app.post("/api/ml/anomaly/fit")
async def ml_anomaly_fit():
    """Обучить anomaly detector на исторических сигналах."""
    import pandas as pd, json
    pool = state.db_pool
    with pool.connection() as conn:
        if pool.is_mysql:
            with conn.cursor() as _c:
                _c.execute("SELECT features_json FROM signal_snapshots LIMIT 5000")
                rows = _c.fetchall()
        else:
            rows = conn.execute("SELECT features_json FROM signal_snapshots LIMIT 5000").fetchall()
    if len(rows) < 100:
        return {"success": False, "error": f"Мало данных: {len(rows)} (нужно ≥100)"}
    features_list = []
    for r in rows:
        try:
            features_list.append(json.loads(r["features_json"] if isinstance(r, dict) else r[0]))
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
        locked_margin = sum(
            pos.get("margin", pos.get("qty", 0) * pos.get("entry_price", 0) / max(pos.get("leverage", 1), 1))
            for pos in state.paper.positions.values()
        )
        avail = round(ps.get("balance", 0), 4)
        equity = round(avail + locked_margin, 4)
        balance_info = {
            "usdt":           avail,
            "equity_usdt":    equity,
            "locked_margin":  round(locked_margin, 4),
            "note":           "usdt=свободные средства; equity=usdt+заблокированная_маржа (реальный баланс)",
            "source":         "paper",
            "paper_pnl":      round(ps.get("total_pnl", 0), 4),
            "paper_trades":   ps.get("trades", 0),
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
            return {"balance": state.paper.get_stats().get("balance"), "mode": "paper"}
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
            bal = state.paper.get_stats().get("balance")
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
