"""
Telegram Commander — двусторонний интерфейс управления ботом.

Polling-режим через aiohttp (без python-telegram-bot, без вебхуков).
Запускается как asyncio-таск внутри FastAPI event loop.

Команды:
  /start      — приветствие + меню
  /help       — список команд
  /status     — статус бота (запущен/остановлен, режим, позиции)
  /balance    — баланс USDT
  /report     — дневной отчёт (сделки, PnL, WR)
  /positions  — открытые позиции
  /go         — запустить торговый цикл
  /stop       — остановить торговый цикл
  /paper_on   — включить paper mode
  /paper_off  — выключить paper mode
  /strategies — статус всех стратегий
  /enable S1  — включить стратегию
  /disable S1 — отключить стратегию
  /risk       — статус риск-менеджера
  /ai         — AI рекомендации (последние)
  /news       — последние новости (сентимент)
  /pause      — приостановить (не открывать новые сделки)
  /resume     — возобновить
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Timeout для long-polling (getUpdates)
_POLL_TIMEOUT = 30


class TelegramCommander:
    """
    Long-polling commander.
    Работает в одном event loop с FastAPI — запускать через asyncio.create_task(commander.run()).
    """

    def __init__(
        self,
        bot_token: str,
        allowed_chat_id: str,
        state_getter: Callable[[], Any],        # возвращает BotState
        start_fn: Callable[[], Coroutine],      # запустить торговый цикл
        stop_fn: Callable[[], Coroutine],       # остановить торговый цикл
    ):
        self.token         = bot_token
        self.allowed_id    = str(allowed_chat_id).strip()
        self._get_state    = state_getter
        self._start_fn     = start_fn
        self._stop_fn      = stop_fn
        self._base_url     = f"https://api.telegram.org/bot{bot_token}"
        self._offset       = 0
        self._paused       = False
        self._session: Optional[aiohttp.ClientSession] = None
        self.enabled       = bool(bot_token and allowed_chat_id)

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_POLL_TIMEOUT + 5)
            )
        return self._session

    async def _api(self, method: str, **kwargs) -> Optional[Dict]:
        try:
            session = await self._get_session()
            async with session.post(f"{self._base_url}/{method}", json=kwargs) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.debug(f"Telegram API {method}: {data.get('description')}")
                    return None
                return data.get("result")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Telegram API error ({method}): {e}")
            return None

    async def send(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        result = await self._api(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )
        return result is not None

    async def reply(self, update: Dict, text: str):
        chat_id = str(update["message"]["chat"]["id"])
        await self.send(chat_id, text)

    # ── Polling loop ─────────────────────────────────────────────────────────

    async def run(self):
        if not self.enabled:
            logger.info("[TgCommander] Отключён — задайте TELEGRAM_TOKEN + TELEGRAM_CHAT_ID")
            return

        logger.info("[TgCommander] Запущен (long-polling)")
        while True:
            try:
                updates = await self._api(
                    "getUpdates",
                    offset=self._offset,
                    timeout=_POLL_TIMEOUT,
                    allowed_updates=["message"],
                )
                if updates:
                    for upd in updates:
                        self._offset = upd["update_id"] + 1
                        await self._handle(upd)
            except asyncio.CancelledError:
                logger.info("[TgCommander] Остановлен")
                break
            except Exception as e:
                logger.warning(f"[TgCommander] Ошибка polling: {e}")
                await asyncio.sleep(5)

    async def _handle(self, upd: Dict):
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            return

        chat_id = str(msg["chat"]["id"])
        text = (msg.get("text") or "").strip()

        # Авторизация — только разрешённый chat_id
        if chat_id != self.allowed_id:
            logger.warning(f"[TgCommander] Отклонён chat_id={chat_id}")
            return

        if not text.startswith("/"):
            return

        parts = text.split(maxsplit=1)
        cmd   = parts[0].lower().split("@")[0]   # убираем @botname
        arg   = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/start":      self._cmd_start,
            "/help":       self._cmd_help,
            "/status":     self._cmd_status,
            "/balance":    self._cmd_balance,
            "/report":     self._cmd_report,
            "/positions":  self._cmd_positions,
            "/go":         self._cmd_go,
            "/stop":       self._cmd_stop,
            "/paper_on":   self._cmd_paper_on,
            "/paper_off":  self._cmd_paper_off,
            "/strategies": self._cmd_strategies,
            "/enable":     self._cmd_enable,
            "/disable":    self._cmd_disable,
            "/risk":       self._cmd_risk,
            "/ai":         self._cmd_ai,
            "/news":       self._cmd_news,
            "/pause":      self._cmd_pause,
            "/resume":     self._cmd_resume,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                await handler(upd, arg)
            except Exception as e:
                logger.error(f"[TgCommander] Ошибка команды {cmd}: {e}")
                await self.reply(upd, f"❌ Ошибка: {e}")
        else:
            await self.reply(upd, f"❓ Неизвестная команда: <code>{cmd}</code>\nНапишите /help")

    # ── Команды ──────────────────────────────────────────────────────────────

    async def _cmd_start(self, upd, arg):
        await self.reply(upd,
            "👋 <b>Baibit Trading Bot</b>\n\n"
            "Управление ботом через Telegram.\n"
            "Напишите /help для списка команд."
        )

    async def _cmd_help(self, upd, arg):
        await self.reply(upd,
            "📋 <b>Команды бота</b>\n\n"
            "<b>Информация:</b>\n"
            "  /status — состояние бота\n"
            "  /balance — баланс USDT\n"
            "  /report — дневной отчёт\n"
            "  /positions — открытые позиции\n"
            "  /risk — риск-менеджер\n"
            "  /ai — AI рекомендации\n"
            "  /news — сентимент новостей\n"
            "  /strategies — статус стратегий\n\n"
            "<b>Управление:</b>\n"
            "  /go — запустить торговлю\n"
            "  /stop — остановить\n"
            "  /pause — пауза (не открывать новые)\n"
            "  /resume — возобновить\n"
            "  /paper_on / /paper_off — paper mode\n"
            "  /enable S1 — включить стратегию\n"
            "  /disable S1 — отключить стратегию"
        )

    async def _cmd_status(self, upd, arg):
        s = self._get_state()
        mode = "📄 Paper" if s.paper_mode else "💰 Real"
        paused = " ⏸ ПАУЗА" if self._paused else ""
        running = "🟢 Запущен" if s.bot_running else "🔴 Остановлен"
        open_pos = sum(1 for st in s.strategies.values() if st.current_position)
        enabled  = sum(1 for st in s.strategies.values() if st.enabled and not st.auto_disabled)

        bybit_ok = "✅" if s.bybit else "❌"
        ai_ok    = "✅" if s.ai.enabled else "❌"
        news_ok  = "✅" if s.news_enabled else "❌"
        ml_ok    = "✅" if s.ml_enabled else "❌"

        await self.reply(upd,
            f"<b>Статус бота</b>\n\n"
            f"{running} | {mode}{paused}\n\n"
            f"Открытых позиций: <b>{open_pos}</b>\n"
            f"Активных стратегий: <b>{enabled}</b> / {len(s.strategies)}\n\n"
            f"Bybit API: {bybit_ok}  |  AI: {ai_ok}\n"
            f"Новости: {news_ok}  |  ML: {ml_ok}\n"
            f"<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>"
        )

    async def _cmd_balance(self, upd, arg):
        s = self._get_state()
        if s.paper_mode:
            ps = s.paper.get_stats()
            bal = ps.get("balance", 0)
            pnl = ps.get("total_pnl", 0)
            await self.reply(upd,
                f"💵 <b>Paper Balance</b>\n\n"
                f"Баланс: <b>{bal:.2f} USDT</b>\n"
                f"PnL: <b>{pnl:+.2f} USDT</b>\n"
                f"Сделок: {ps.get('trades', 0)}"
            )
        elif s.bybit:
            try:
                bal = s.bybit.get_balance("USDT")
                rm  = s.risk_manager.get_status()
                await self.reply(upd,
                    f"💵 <b>Баланс</b>\n\n"
                    f"USDT: <b>{bal:.4f}</b>\n"
                    f"Дневной PnL: <b>{rm.get('daily_pnl', 0):+.2f} USDT</b>\n"
                    f"Использование риска: {rm.get('open_positions', 0)}/{rm.get('max_positions', 5)} поз."
                )
            except Exception as e:
                await self.reply(upd, f"❌ Ошибка получения баланса: {e}")
        else:
            await self.reply(upd, "❌ Bybit API не подключён")

    async def _cmd_report(self, upd, arg):
        s = self._get_state()
        try:
            today = datetime.utcnow().date().isoformat()
            trades = s.journal.get_trades(start_date=today, limit=500)
            total  = len(trades)
            wins   = sum(1 for t in trades if (t.get("pnl_usd") or 0) > 0)
            losses = total - wins
            pnl    = sum(t.get("pnl_usd") or 0 for t in trades)
            wr     = wins / total * 100 if total else 0
            best   = max((t.get("pnl_usd") or 0 for t in trades), default=0)
            worst  = min((t.get("pnl_usd") or 0 for t in trades), default=0)

            await self.reply(upd,
                f"📊 <b>Дневной отчёт</b>\n"
                f"<i>{today}</i>\n\n"
                f"Сделок: <b>{total}</b> ({wins}W / {losses}L)\n"
                f"Win Rate: <b>{wr:.1f}%</b>\n"
                f"PnL: <b>{pnl:+.2f} USDT</b>\n"
                f"Лучшая: <code>{best:+.2f}</code>\n"
                f"Худшая: <code>{worst:+.2f}</code>"
            )
        except Exception as e:
            await self.reply(upd, f"❌ Ошибка: {e}")

    async def _cmd_positions(self, upd, arg):
        s = self._get_state()
        open_pos = [
            (sid, st) for sid, st in s.strategies.items() if st.current_position
        ]
        if not open_pos:
            await self.reply(upd, "📭 Нет открытых позиций")
            return

        lines = ["<b>Открытые позиции</b>\n"]
        for sid, st in open_pos:
            pos  = st.current_position
            side = pos.get("side", "?")
            e    = pos.get("entry", 0)
            sl   = pos.get("sl", 0)
            tp   = pos.get("tp", 0)
            emoji = "🟢" if side == "Buy" else "🔴"
            lines.append(
                f"{emoji} <b>{sid}</b> {st.symbol}\n"
                f"  Вход: {e:.4f} | SL: {sl:.4f} | TP: {tp:.4f}"
            )
        await self.reply(upd, "\n".join(lines))

    async def _cmd_go(self, upd, arg):
        s = self._get_state()
        if s.bot_running:
            await self.reply(upd, "ℹ️ Бот уже запущен")
            return
        if not s.bybit and not s.paper_mode:
            await self.reply(upd, "❌ Нет Bybit API и не включён Paper Mode")
            return
        self._paused = False
        await self._start_fn()
        await self.reply(upd, "🚀 Торговый цикл запущен")

    async def _cmd_stop(self, upd, arg):
        s = self._get_state()
        if not s.bot_running:
            await self.reply(upd, "ℹ️ Бот уже остановлен")
            return
        await self._stop_fn()
        await self.reply(upd, "🛑 Торговый цикл остановлен")

    async def _cmd_paper_on(self, upd, arg):
        s = self._get_state()
        s.paper_mode = True
        await self.reply(upd, "📄 Paper Mode включён")

    async def _cmd_paper_off(self, upd, arg):
        s = self._get_state()
        s.paper_mode = False
        await self.reply(upd, "💰 Paper Mode выключен — торговля реальная")

    async def _cmd_strategies(self, upd, arg):
        s = self._get_state()
        lines = ["<b>Стратегии</b>\n"]
        for sid, st in s.strategies.items():
            if st.auto_disabled:
                icon = "🚫"
            elif st.enabled:
                icon = "✅"
            else:
                icon = "⏹"
            pos = "📌" if st.current_position else "  "
            stats = st.get_rolling_stats()
            wr = stats.get("win_rate", 0)
            lines.append(f"{icon}{pos} <code>{sid}</code> {st.symbol} WR={wr:.0f}%")
        await self.reply(upd, "\n".join(lines))

    async def _cmd_enable(self, upd, arg):
        s = self._get_state()
        sid = arg.upper().strip()
        if sid not in s.strategies:
            await self.reply(upd, f"❌ Стратегия <code>{sid}</code> не найдена")
            return
        s.strategies[sid].enabled = True
        s.strategies[sid].auto_disabled = False
        await self.reply(upd, f"✅ {sid} включена")

    async def _cmd_disable(self, upd, arg):
        s = self._get_state()
        sid = arg.upper().strip()
        if sid not in s.strategies:
            await self.reply(upd, f"❌ Стратегия <code>{sid}</code> не найдена")
            return
        s.strategies[sid].enabled = False
        await self.reply(upd, f"⏹ {sid} отключена")

    async def _cmd_risk(self, upd, arg):
        s = self._get_state()
        rm = s.risk_manager.get_status()
        kill = "🛑 KILL SWITCH" if rm.get("kill_switch") else "✅ Активен"
        await self.reply(upd,
            f"🛡 <b>Риск-менеджер</b>\n\n"
            f"Статус: {kill}\n"
            f"Позиции: {rm.get('open_positions', 0)} / {rm.get('max_positions', 5)}\n"
            f"Дневной PnL: <b>{rm.get('daily_pnl', 0):+.2f} USDT</b>\n"
            f"Дневной лимит убытка: {rm.get('daily_max_loss_pct', 0):.0f}%\n"
            f"Серия убытков: {rm.get('consecutive_losses', 0)}"
        )

    async def _cmd_ai(self, upd, arg):
        s = self._get_state()
        rec = getattr(s, "_cached_ai_rec", None)
        if not rec or not rec.get("available"):
            await self.reply(upd, "ℹ️ AI рекомендации ещё не получены (обновляются каждые 30 мин)")
            return
        action = rec.get("action", "?").upper()
        risk   = rec.get("risk_level", "?")
        reason = rec.get("reasoning", "")[:300]
        fg     = rec.get("fear_greed", {})
        fg_val = fg.get("value", "?") if fg else "?"
        fg_lbl = fg.get("label", "") if fg else ""
        await self.reply(upd,
            f"🧠 <b>AI Рекомендация</b>\n\n"
            f"Действие: <b>{action}</b>\n"
            f"Риск: <b>{risk}</b>\n"
            f"Fear &amp; Greed: <b>{fg_val}</b> {fg_lbl}\n\n"
            f"<i>{reason}</i>"
        )

    async def _cmd_news(self, upd, arg):
        s = self._get_state()
        if not s.news_enabled:
            await self.reply(upd, "ℹ️ Новостной модуль отключён")
            return
        try:
            feat = s.news_manager.get_sentiment_features()
            score  = feat.get("sentiment_score", 0)
            count  = feat.get("news_count_24h", 0)
            bull   = feat.get("bull_count", 0)
            bear   = feat.get("bear_count", 0)
            emoji  = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
            await self.reply(upd,
                f"📰 <b>Новостной сентимент</b>\n\n"
                f"{emoji} Score: <b>{score:+.3f}</b>\n"
                f"Новостей за 24ч: <b>{count}</b>\n"
                f"Бычьих: {bull} | Медвежьих: {bear}"
            )
        except Exception as e:
            await self.reply(upd, f"❌ Ошибка: {e}")

    async def _cmd_pause(self, upd, arg):
        self._paused = True
        s = self._get_state()
        s.bot_running = False
        task = getattr(s, "trading_loop_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=3)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        await self.reply(upd, "⏸ Бот приостановлен — новые сделки не открываются")

    async def _cmd_resume(self, upd, arg):
        if not self._paused:
            await self.reply(upd, "ℹ️ Бот не на паузе")
            return
        self._paused = False
        s = self._get_state()
        s.bot_running = True
        await self._start_fn()
        await self.reply(upd, "▶️ Бот возобновлён")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
