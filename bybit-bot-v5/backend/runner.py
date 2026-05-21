"""
Автономный запуск бота без веб-сервера.
Только торговый цикл + Telegram управление + часовые отчёты.
Запуск: python runner.py
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Логирование ───────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("runner")

# ── Импорт основных компонентов ───────────────────────────────
from main import (
    state,
    init_strategies,
    trading_loop,
    orchestrator_loop,
    auto_train_loop,
    _hourly_report_loop,
    _send_hourly_report,
    _tg_start_bot,
    _tg_stop_bot,
    _execute_bot_command,
    ML_AVAILABLE,
)
from bybit_client import BybitClient
from telegram_commander import TelegramCommander
from news import background_news_loop
from claude_orchestrator import ClaudeOrchestrator


async def main():
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║       Baibit Trading Bot             ║")
    logger.info("╚══════════════════════════════════════╝")

    init_strategies()

    # ── Bybit ─────────────────────────────────────────────────
    _key    = os.getenv("BYBIT_API_KEY", "").strip()
    _secret = os.getenv("BYBIT_API_SECRET", "").strip()
    _testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

    # Determine trading mode and warn if live without confirmation
    _trading_mode = os.getenv("TRADING_MODE", "paper").lower()
    state.trading_mode = _trading_mode.upper()

    _live_confirmed = os.getenv("LIVE_TRADING_CONFIRMED", "false").lower() in ("1", "true", "yes")
    if _trading_mode == "live" and not _live_confirmed:
        logger.warning(
            "⚠️  TRADING_MODE=live but LIVE_TRADING_CONFIRMED is not set. "
            "Operating in safe paper mode. Set LIVE_TRADING_CONFIRMED=true to enable live trading."
        )
        _trading_mode = "paper"
        state.trading_mode = "PAPER"

    # Public-only mode if no keys provided — avoids fake placeholder keys
    _has_keys = bool(_key and _secret)

    # Always create client — public endpoints (klines, ticker, orderbook)
    # work without API keys. Keys are only needed for real orders.
    try:
        if _has_keys:
            state.bybit = BybitClient(_key, _secret, _testnet)
            bal = state.bybit.get_balance("USDT")
            logger.info(f"✅ Bybit подключён | Баланс: {bal:.2f} USDT | Режим: {state.trading_mode}")
        else:
            # Use public-only mode: pass empty strings so BybitClient skips auth
            state.bybit = BybitClient("", "", _testnet)
            logger.info("📡 Bybit: публичные данные (ключи не заданы — только paper mode)")
            if not state.paper_mode:
                state.paper_mode = True
                logger.info("📄 Paper mode автоматически включён (нет API ключей)")
    except Exception as e:
        logger.error(f"❌ Bybit инициализация: {e}")

    # ── Paper mode ────────────────────────────────────────────
    if os.getenv("PAPER_TRADING", "false").lower() == "true":
        state.paper_mode = True
        # Восстанавливаем open positions в стратегии ПОСЛЕ того как paper_mode=True
        from main import _restore_paper_positions
        _restore_paper_positions()
        logger.info(
            f"📄 Paper Trading | Баланс: {state.paper.balance:.2f} USDT "
            f"| Позиций восстановлено: {len(state.paper.positions)}"
        )

    # ── Anomaly detector ──────────────────────────────────────
    if Path("data/models/anomaly_detector.pkl").exists():
        if state.anomaly_detector.load("data/models/anomaly_detector.pkl"):
            logger.info("🛡 Anomaly detector загружен")

    tasks = []

    # ── Новости ───────────────────────────────────────────────
    if state.news_enabled:
        tasks.append(asyncio.create_task(
            background_news_loop(state.news_manager, interval_min=15)
        ))
        logger.info("📡 News loop запущен")

    # ── Auto-train ────────────────────────────────────────────
    if state.ml_enabled and ML_AVAILABLE:
        tasks.append(asyncio.create_task(auto_train_loop()))
        logger.info("🔄 Auto-train loop запущен")

    # ── AI Orchestrator ───────────────────────────────────────
    _anthropic = os.getenv("ANTHROPIC_API_KEY")
    _openai    = os.getenv("OPENAI_API_KEY")
    _ollama    = os.getenv("OLLAMA_BASE_URL")
    if _anthropic or _openai or _ollama:
        state.orchestrator = ClaudeOrchestrator(
            api_key=_anthropic,
            openai_key=_openai,
            ollama_url=_ollama,
            execute_fn=_execute_bot_command,
            notify_fn=lambda msg, level: asyncio.create_task(state.telegram.send(msg)),
        )
        tasks.append(asyncio.create_task(orchestrator_loop()))
        logger.info("🤖 AI Orchestrator запущен")

    # ── Telegram Commander ────────────────────────────────────
    _tg_token   = os.getenv("TELEGRAM_TOKEN", "")
    _tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if _tg_token and _tg_chat_id:
        state.commander = TelegramCommander(
            bot_token=_tg_token,
            allowed_chat_id=_tg_chat_id,
            state_getter=lambda: state,
            start_fn=_tg_start_bot,
            stop_fn=_tg_stop_bot,
        )
        tasks.append(asyncio.create_task(state.commander.run()))
        logger.info("📱 Telegram Commander запущен")
    else:
        logger.warning("⚠️ Telegram не настроен — задайте TELEGRAM_TOKEN + TELEGRAM_CHAT_ID")

    # ── Часовой отчёт ─────────────────────────────────────────
    if state.telegram.enabled:
        tasks.append(asyncio.create_task(_hourly_report_loop()))
        logger.info("📊 Часовой отчёт запущен")

    # ── Торговый цикл ─────────────────────────────────────────
    if os.getenv("AUTO_START", "false").lower() == "true":
        if state.bybit or state.paper_mode:
            state.bot_running = True
            state.trading_loop_task = asyncio.create_task(trading_loop())
            tasks.append(state.trading_loop_task)
            logger.info("🚀 Торговый цикл запущен (AUTO_START=true)")
            await state.telegram.send("🚀 <b>Baibit запущен</b>\nРежим: Paper Trading\nУправление: /help")
        else:
            logger.warning("AUTO_START=true но нет Bybit API и Paper Mode — напишите /go в Telegram")
    else:
        logger.info("ℹ️ Напишите /go в Telegram чтобы начать торговлю")

    logger.info("✅ Бот готов к работе")

    # ── Держим процесс живым ──────────────────────────────────
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("🛑 Остановка бота...")
        state.bot_running = False
        for t in tasks:
            t.cancel()
        await state.telegram.send("🛑 <b>Baibit остановлен</b>")


if __name__ == "__main__":
    asyncio.run(main())
