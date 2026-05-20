"""
Telegram Commander — двусторонний интерфейс с инлайн-кнопками.
Polling через aiohttp. Кнопки: Старт/Стоп, Баланс, Сделки, Лучшая модель, Позиции.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)
_POLL_TIMEOUT = 30


# ── Хелперы для кнопок ────────────────────────────────────────
def _btn(text: str, data: str) -> Dict:
    return {"text": text, "callback_data": data}

def _keyboard(rows: List[List[Dict]]) -> Dict:
    return {"inline_keyboard": rows}


class TelegramCommander:
    def __init__(
        self,
        bot_token: str,
        allowed_chat_id: str,
        state_getter: Callable[[], Any],
        start_fn: Callable[[], Coroutine],
        stop_fn: Callable[[], Coroutine],
    ):
        self.token      = bot_token
        self.allowed_id = str(allowed_chat_id).strip()
        self._get_state = state_getter
        self._start_fn  = start_fn
        self._stop_fn   = stop_fn
        self._base_url  = f"https://api.telegram.org/bot{bot_token}"
        self._offset    = 0
        self._paused    = False
        self._session: Optional[aiohttp.ClientSession] = None
        self.enabled    = bool(bot_token and allowed_chat_id)
        # Ожидание ввода от пользователя: chat_id -> action
        self._pending_input: Dict[str, str] = {}

    # ── HTTP ─────────────────────────────────────────────────────

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
                    logger.debug(f"Telegram {method}: {data.get('description')}")
                    return None
                return data.get("result")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Telegram error ({method}): {e}")
            return None

    async def send(self, chat_id: str, text: str,
                   parse_mode: str = "HTML",
                   reply_markup: Optional[Dict] = None) -> Optional[Dict]:
        kwargs = dict(chat_id=chat_id, text=text,
                      parse_mode=parse_mode, disable_web_page_preview=True)
        if reply_markup:
            kwargs["reply_markup"] = reply_markup
        return await self._api("sendMessage", **kwargs)

    async def edit(self, chat_id: str, message_id: int, text: str,
                   reply_markup: Optional[Dict] = None):
        kwargs = dict(chat_id=chat_id, message_id=message_id,
                      text=text, parse_mode="HTML", disable_web_page_preview=True)
        if reply_markup:
            kwargs["reply_markup"] = reply_markup
        await self._api("editMessageText", **kwargs)

    async def answer_callback(self, callback_id: str, text: str = ""):
        await self._api("answerCallbackQuery", callback_query_id=callback_id, text=text)

    async def reply(self, upd: Dict, text: str, reply_markup: Optional[Dict] = None):
        msg = upd.get("message") or upd.get("edited_message")
        if msg:
            chat_id = str(msg["chat"]["id"])
            await self.send(chat_id, text, reply_markup=reply_markup)

    # ── Главное меню ─────────────────────────────────────────────

    def _main_menu(self) -> Dict:
        s = self._get_state()
        go_btn    = _btn("🛑 Стоп", "stop") if s.bot_running else _btn("🚀 Старт", "go")
        pause_btn = _btn("▶️ Возобновить", "resume") if self._paused else _btn("⏸ Пауза", "pause")
        bal_str   = f"{s.paper.balance:.0f}" if s.paper_mode else "реал"
        max_pos   = s.risk_manager.max_open_positions
        scalp_btn = _btn("⚡ Скальп: ВКЛ", "scalp_off") if s.scalp_active else _btn("⚡ Скальп: ВЫКЛ", "scalp_on")
        train_btn = _btn("🎓 Обучение: ВКЛ", "training_off") if getattr(s, "training_mode", False) \
                    else _btn("🎓 Обучение: ВЫКЛ", "training_on")
        min_trade = s.risk_manager.min_trade_usdt
        return _keyboard([
            [go_btn, pause_btn],
            [_btn("💰 Баланс", "balance"), _btn("📌 Позиции", "positions")],
            [_btn("📊 Сделки", "trades"), _btn("🏆 Лучшие модели", "best_models")],
            [_btn("📈 Стратегии", "strategies"), _btn("🛡 Риск", "risk")],
            [_btn("📰 Новости", "news"), _btn("🤖 AI", "ai")],
            [_btn(f"💵 Баланс: {bal_str} USDT", "set_balance"),
             _btn(f"📦 Макс сделок: {max_pos}", "set_max_pos")],
            [_btn(f"💲 Мин. сделка: {min_trade:.0f} USDT", "set_min_trade"), scalp_btn],
            [train_btn, _btn("🔄 Обновить меню", "menu")],
        ])

    # ── Polling ───────────────────────────────────────────────────

    async def run(self):
        if not self.enabled:
            return
        logger.info("[TgCommander] Запущен (long-polling)")
        while True:
            try:
                updates = await self._api(
                    "getUpdates",
                    offset=self._offset,
                    timeout=_POLL_TIMEOUT,
                    allowed_updates=["message", "callback_query"],
                )
                if updates:
                    for upd in updates:
                        self._offset = upd["update_id"] + 1
                        await self._handle(upd)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[TgCommander] Polling error: {e}")
                await asyncio.sleep(5)

    async def _handle(self, upd: Dict):
        # Callback от кнопки
        if "callback_query" in upd:
            await self._handle_callback(upd["callback_query"])
            return

        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            return

        chat_id = str(msg["chat"]["id"])
        if chat_id != self.allowed_id:
            return

        text = (msg.get("text") or "").strip()

        # Если ожидаем ввод от пользователя — обрабатываем любой текст
        if chat_id in self._pending_input:
            action = self._pending_input.pop(chat_id)
            await self._handle_pending_input(upd, action, text)
            return

        if not text.startswith("/"):
            return

        parts = text.split(maxsplit=1)
        cmd   = parts[0].lower().split("@")[0]
        arg   = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/start":      self._cmd_start,
            "/help":       self._cmd_help,
            "/menu":       self._cmd_menu,
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
            "/trades":     self._cmd_trades,
            "/best":       self._cmd_best_models,
            "/improve":    self._cmd_improve,
            "/advisor":    self._cmd_advisor,
            "/exportdb":   self._cmd_exportdb,
            "/scalp":      self._cmd_scalp,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                await handler(upd, arg)
            except Exception as e:
                logger.error(f"[TgCommander] {cmd}: {e}")
                await self.reply(upd, f"❌ Ошибка: {e}")
        else:
            await self.reply(upd, f"❓ Неизвестная команда. Напишите /menu")

    async def _handle_callback(self, cb: Dict):
        chat_id = str(cb["from"]["id"])
        if chat_id != self.allowed_id:
            return

        data       = cb.get("data", "")
        msg_id     = cb["message"]["message_id"]
        cb_id      = cb["id"]

        await self.answer_callback(cb_id)

        # Формируем псевдо-upd для совместимости с reply()
        fake_upd = {"message": {"chat": {"id": chat_id}}}

        action_map = {
            "menu":        self._show_menu_edit,
            "go":          self._cb_go,
            "stop":        self._cb_stop,
            "pause":       self._cb_pause,
            "resume":      self._cb_resume,
            "balance":     self._cmd_balance,
            "positions":   self._cmd_positions,
            "trades":      self._cmd_trades,
            "best_models": self._cmd_best_models,
            "strategies":  self._cmd_strategies,
            "risk":        self._cmd_risk,
            "ai":          self._cmd_ai,
            "news":        self._cmd_news,
            "set_balance":   self._cb_set_balance,
            "set_max_pos":   self._cb_set_max_pos,
            "set_min_trade": self._cb_set_min_trade,
            "scalp_on":      self._cb_scalp_on,
            "scalp_off":     self._cb_scalp_off,
            "training_on":   self._cb_training_on,
            "training_off":  self._cb_training_off,
        }

        # Закрытие отдельной позиции
        if data.startswith("close_pos:"):
            sid = data[len("close_pos:"):]
            await self._cmd_close_one_position(fake_upd, sid)
            return

        # AI-советы по стратегии
        if data.startswith("improve:"):
            sid = data[len("improve:"):]
            await self._cmd_improve(fake_upd, sid)
            return

        # Advisor кнопки
        if data == "advisor_run":
            await self._run_advisor_full(fake_upd)
            return
        if data == "advisor_all":
            await self._cmd_advisor(fake_upd, "")
            return
        if data.startswith("advisor:"):
            sid = data[len("advisor:"):]
            await self._cmd_advisor(fake_upd, sid)
            return

        fn = action_map.get(data)
        if fn:
            if data in ("menu",):
                await fn(chat_id, msg_id)
            else:
                await fn(fake_upd, "")

    async def _show_menu_edit(self, chat_id: str, msg_id: int):
        s = self._get_state()
        mode    = "📄 Paper" if s.paper_mode else "💰 Real"
        status  = "🟢 Работает" if s.bot_running else "🔴 Остановлен"
        balance = s.paper.balance if s.paper_mode else 0
        pnl     = sum(st.pnl for st in s.strategies.values())
        trades  = sum(st.trades for st in s.strategies.values())

        text = (
            f"🤖 <b>Baibit Trading Bot</b>\n\n"
            f"{status} | {mode}\n"
            f"Баланс: <b>{balance:.2f} USDT</b>\n"
            f"PnL сессии: <b>{pnl:+.2f} USDT</b>\n"
            f"Сделок: <b>{trades}</b>\n"
            f"<i>{datetime.utcnow().strftime('%H:%M UTC')}</i>"
        )
        await self.edit(chat_id, msg_id, text, reply_markup=self._main_menu())

    # ── Callback-версии Старт/Стоп/Пауза ────────────────────────

    async def _cb_go(self, upd, arg):
        s = self._get_state()
        if s.bot_running:
            await self.reply(upd, "ℹ️ Бот уже запущен", reply_markup=self._main_menu())
            return
        self._paused = False
        await self._start_fn()
        await self.reply(upd, "🚀 Торговый цикл запущен", reply_markup=self._main_menu())

    async def _cb_stop(self, upd, arg):
        s = self._get_state()
        if not s.bot_running:
            await self.reply(upd, "ℹ️ Бот уже остановлен", reply_markup=self._main_menu())
            return
        await self._stop_fn()
        await self.reply(upd, "🛑 Торговый цикл остановлен", reply_markup=self._main_menu())

    async def _cb_pause(self, upd, arg):
        await self._cmd_pause(upd, arg)

    async def _cb_resume(self, upd, arg):
        await self._cmd_resume(upd, arg)

    async def _cb_scalp_on(self, upd, arg):
        from main import activate_scalp_mode
        added = activate_scalp_mode()
        s = self._get_state()
        count = len([sid for sid in s.strategies if sid.startswith("SC_")])
        await self.reply(upd,
            f"⚡ <b>ScalperPro включён</b>\n"
            f"Активных скальперов: <b>{count}</b> символов\n"
            f"Добавлено: {', '.join(added) if added else 'уже были активны'}",
            reply_markup=self._main_menu())

    async def _cb_scalp_off(self, upd, arg):
        from main import deactivate_scalp_mode
        deactivate_scalp_mode()
        await self.reply(upd,
            "⚡ <b>ScalperPro отключён</b>\n"
            "SC_* стратегии приостановлены",
            reply_markup=self._main_menu())

    async def _cb_training_on(self, upd, arg):
        s = self._get_state()
        s.training_mode = True
        s.risk_manager.kill_switch = False
        # Сбрасываем Guard-счётчики чтобы стартовать обучение чисто
        s.trade_guard.reset_all()
        await self.reply(upd,
            "🎓 <b>Режим обучения ВКЛЮЧЁН</b>\n\n"
            "Все лимиты убраны:\n"
            "• Лимиты убытков (дневной, недельный)\n"
            "• Лимит сделок в день\n"
            "• Кулдауны после серии убытков\n"
            "• GlobalTradeGuard блокировки\n"
            "• AI-отклонение сигналов\n"
            "• ML аномалии и строгий режим\n"
            "• Оркестратор (только оповещения)\n\n"
            "⚠️ Бот торгует без ограничений для сбора данных.\n"
            "Выключи режим после набора статистики!",
            reply_markup=self._main_menu())

    async def _cb_training_off(self, upd, arg):
        s = self._get_state()
        s.training_mode = False
        # Сбрасываем Guard-счётчики: кулдауны накопленные в обучении — не должны
        # блокировать нормальную торговлю после выхода из обучения
        s.trade_guard.reset_all()
        await self.reply(upd,
            "🛡 <b>Режим обучения ВЫКЛЮЧЕН</b>\n\n"
            "Все защиты восстановлены:\n"
            "• Лимиты убытков и кулдауны\n"
            "• GlobalTradeGuard (все блоки)\n"
            "• Фильтр входа (EntryFilter)\n"
            "• AI-анализ сигналов\n"
            "• Оркестратор (авто-действия)\n\n"
            "Guard-счётчики сброшены — кулдауны из обучения не применяются.",
            reply_markup=self._main_menu())

    async def _cb_set_balance(self, upd, arg):
        """Запрашивает новый paper-баланс."""
        s = self._get_state()
        if not s.paper_mode:
            await self.reply(upd, "⚠️ Изменение баланса доступно только в Paper-режиме",
                             reply_markup=self._main_menu())
            return
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat_id = str(msg.get("chat", {}).get("id", self.allowed_id))
        self._pending_input[chat_id] = "set_balance"
        await self.reply(upd,
            f"💵 <b>Изменение Paper-баланса</b>\n\n"
            f"Текущий баланс: <b>{s.paper.balance:.2f} USDT</b>\n\n"
            f"Введи новый баланс числом (например: <code>1000</code>):\n"
            f"<i>Открытые позиции при этом сохранятся</i>",
            reply_markup=_keyboard([[_btn("❌ Отмена", "menu")]])
        )

    async def _cb_set_max_pos(self, upd, arg):
        """Запрашивает новый лимит одновременных позиций."""
        s = self._get_state()
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat_id = str(msg.get("chat", {}).get("id", self.allowed_id))
        self._pending_input[chat_id] = "set_max_pos"
        await self.reply(upd,
            f"📦 <b>Макс. одновременных сделок</b>\n\n"
            f"Текущее значение: <b>{s.risk_manager.max_open_positions}</b>\n\n"
            f"Введи новое число от 1 до 50 (например: <code>20</code>):",
            reply_markup=_keyboard([[_btn("❌ Отмена", "menu")]])
        )

    async def _cb_set_min_trade(self, upd, arg):
        """Запрашивает минимальный размер сделки в USDT."""
        s = self._get_state()
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat_id = str(msg.get("chat", {}).get("id", self.allowed_id))
        self._pending_input[chat_id] = "set_min_trade"
        await self.reply(upd,
            f"💲 <b>Минимальный размер сделки</b>\n\n"
            f"Текущее значение: <b>{s.risk_manager.min_trade_usdt:.0f} USDT</b>\n\n"
            f"Если расчётный объём сделки меньше этого порога — сделка пропускается.\n\n"
            f"Введи число от 1 до 1000 (например: <code>10</code>):",
            reply_markup=_keyboard([[_btn("❌ Отмена", "menu")]])
        )

    async def _handle_pending_input(self, upd: Dict, action: str, text: str):
        """Обрабатывает текстовый ввод после нажатия кнопки."""
        s = self._get_state()
        text = text.strip()

        if action == "set_balance":
            try:
                new_bal = float(text.replace(",", "."))
                if new_bal <= 0:
                    raise ValueError("отрицательный")
                old_bal = s.paper.balance
                s.paper.balance = round(new_bal, 2)
                s.paper._save_state()
                await self.reply(upd,
                    f"✅ <b>Баланс обновлён</b>\n\n"
                    f"{old_bal:.2f} USDT → <b>{new_bal:.2f} USDT</b>",
                    reply_markup=self._main_menu()
                )
            except (ValueError, TypeError):
                await self.reply(upd,
                    f"❌ Неверное значение: <code>{text}</code>\n"
                    f"Введи число, например <code>1000</code>",
                    reply_markup=self._main_menu()
                )

        elif action == "set_max_pos":
            try:
                new_max = int(text)
                if not 1 <= new_max <= 50:
                    raise ValueError("вне диапазона")
                old_max = s.risk_manager.max_open_positions
                s.risk_manager.max_open_positions = new_max
                await self.reply(upd,
                    f"✅ <b>Лимит позиций обновлён</b>\n\n"
                    f"{old_max} → <b>{new_max}</b> одновременных сделок",
                    reply_markup=self._main_menu()
                )
            except (ValueError, TypeError):
                await self.reply(upd,
                    f"❌ Неверное значение: <code>{text}</code>\n"
                    f"Введи целое число от 1 до 50",
                    reply_markup=self._main_menu()
                )

        elif action == "set_min_trade":
            try:
                new_min = float(text.replace(",", "."))
                if not 1 <= new_min <= 1000:
                    raise ValueError("вне диапазона")
                old_min = s.risk_manager.min_trade_usdt
                s.risk_manager.min_trade_usdt = round(new_min, 1)
                await self.reply(upd,
                    f"✅ <b>Мин. размер сделки обновлён</b>\n\n"
                    f"{old_min:.0f} USDT → <b>{new_min:.0f} USDT</b>",
                    reply_markup=self._main_menu()
                )
            except (ValueError, TypeError):
                await self.reply(upd,
                    f"❌ Неверное значение: <code>{text}</code>\n"
                    f"Введи число от 1 до 1000",
                    reply_markup=self._main_menu()
                )

        else:
            await self.reply(upd, "❓ Неизвестное действие", reply_markup=self._main_menu())

    # ── Команды ──────────────────────────────────────────────────

    async def _cmd_start(self, upd, arg):
        s = self._get_state()
        mode    = "📄 Paper" if s.paper_mode else "💰 Real"
        status  = "🟢 Работает" if s.bot_running else "🔴 Остановлен"
        balance = s.paper.balance if s.paper_mode else 0
        pnl     = sum(st.pnl for st in s.strategies.values())
        trades  = sum(st.trades for st in s.strategies.values())

        text = (
            f"🤖 <b>Baibit Trading Bot</b>\n\n"
            f"{status} | {mode}\n"
            f"Баланс: <b>{balance:.2f} USDT</b>\n"
            f"PnL сессии: <b>{pnl:+.2f} USDT</b>\n"
            f"Сделок: <b>{trades}</b>\n"
            f"<i>{datetime.utcnow().strftime('%H:%M UTC')}</i>"
        )
        await self.reply(upd, text, reply_markup=self._main_menu())

    async def _cmd_menu(self, upd, arg):
        await self._cmd_start(upd, arg)

    async def _cmd_help(self, upd, arg):
        await self.reply(upd,
            "📋 <b>Команды</b>\n\n"
            "/menu — главное меню с кнопками\n"
            "/balance — баланс\n"
            "/positions — открытые позиции\n"
            "/trades — последние сделки по стратегиям\n"
            "/best — лучшие стратегии\n"
            "/strategies — все стратегии\n"
            "/risk — риск-менеджер\n"
            "/go — запустить  |  /stop — стоп\n"
            "/pause — пауза  |  /resume — продолжить\n"
            "/paper_on  |  /paper_off\n"
            "/enable S1  |  /disable S1\n"
            "/ai — AI рекомендации\n"
            "/news — новостной сентимент\n"
            "/advisor — AI-анализ всех стратегий + рекомендации\n"
            "/advisor S1 — детальный анализ стратегии\n"
            "/exportdb — скачать базу данных сделок"
        )

    async def _cmd_status(self, upd, arg):
        await self._cmd_start(upd, arg)

    async def _cmd_balance(self, upd, arg):
        s = self._get_state()
        if s.paper_mode:
            ps  = s.paper.get_stats()
            bal = ps.get("balance", 0)
            pnl = ps.get("total_pnl", 0)
            wins   = ps.get("wins", 0)
            trades = ps.get("trades", 0)
            wr = wins / trades * 100 if trades else 0
            await self.reply(upd,
                f"💰 <b>Paper Balance</b>\n\n"
                f"Баланс: <b>{bal:.2f} USDT</b>\n"
                f"PnL: <b>{pnl:+.2f} USDT</b>\n"
                f"Сделок: {trades} | WR: {wr:.1f}%",
                reply_markup=self._main_menu()
            )
        elif s.bybit:
            try:
                bal = s.bybit.get_balance("USDT")
                rm  = s.risk_manager.get_status()
                await self.reply(upd,
                    f"💰 <b>Баланс</b>\n\n"
                    f"USDT: <b>{bal:.2f}</b>\n"
                    f"Дневной PnL: <b>{rm.get('daily_pnl', 0):+.2f} USDT</b>\n"
                    f"Позиций: {rm.get('open_positions', 0)}/{rm.get('max_positions', 4)}",
                    reply_markup=self._main_menu()
                )
            except Exception as e:
                await self.reply(upd, f"❌ {e}")
        else:
            await self.reply(upd, "❌ Нет подключения к Bybit")

    async def _cmd_trades(self, upd, arg):
        """Последние сделки по каждой стратегии."""
        s = self._get_state()
        lines = ["📊 <b>Сделки по стратегиям</b>\n"]
        has_data = False
        for sid, st in sorted(s.strategies.items()):
            if st.trades == 0:
                continue
            has_data = True
            wr    = st.wins / st.trades * 100 if st.trades else 0
            emoji = "🟢" if st.pnl >= 0 else "🔴"
            lines.append(
                f"{emoji} <b>{sid}</b> {st.symbol}\n"
                f"   Сделок: {st.trades} | WR: {wr:.0f}% | PnL: {st.pnl:+.2f}$"
            )
        if not has_data:
            await self.reply(upd, "📭 Сделок пока нет — бот только запустился")
            return
        await self.reply(upd, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_best_models(self, upd, arg):
        """Рейтинг стратегий по win rate и PnL."""
        s = self._get_state()
        rated = []
        for sid, st in s.strategies.items():
            if st.trades < 3:
                continue
            wr = st.wins / st.trades * 100 if st.trades else 0
            rated.append((sid, st, wr))

        if not rated:
            await self.reply(upd, "📭 Недостаточно данных — нужно минимум 3 сделки на стратегию")
            return

        # Сортируем по PnL
        by_pnl = sorted(rated, key=lambda x: x[1].pnl, reverse=True)
        # Сортируем по WR
        by_wr  = sorted(rated, key=lambda x: x[2], reverse=True)

        lines = ["🏆 <b>Лучшие стратегии</b>\n"]
        lines.append("<b>По прибыли (PnL):</b>")
        for i, (sid, st, wr) in enumerate(by_pnl[:5], 1):
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i-1]
            lines.append(f"{medal} {sid} {st.symbol}: <b>{st.pnl:+.2f}$</b> ({st.trades} сделок)")

        lines.append("\n<b>По Win Rate:</b>")
        for i, (sid, st, wr) in enumerate(by_wr[:5], 1):
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i-1]
            lines.append(f"{medal} {sid} {st.symbol}: <b>{wr:.0f}%</b> WR ({st.trades} сделок)")

        await self.reply(upd, "\n".join(lines), reply_markup=self._main_menu())

    async def _cmd_report(self, upd, arg):
        s = self._get_state()
        try:
            today  = datetime.utcnow().date().isoformat()
            trades = s.journal.get_trades(start_date=today, limit=500)
            total  = len(trades)
            wins   = sum(1 for t in trades if float(t.get("pnl_usd") or 0) > 0)
            pnl    = sum(float(t.get("pnl_usd") or 0) for t in trades)
            wr     = wins / total * 100 if total else 0
            best   = max((float(t.get("pnl_usd") or 0) for t in trades), default=0)
            worst  = min((float(t.get("pnl_usd") or 0) for t in trades), default=0)
            await self.reply(upd,
                f"📊 <b>Дневной отчёт</b> <i>{today}</i>\n\n"
                f"Сделок: <b>{total}</b> ({wins}W/{total-wins}L)\n"
                f"Win Rate: <b>{wr:.1f}%</b>\n"
                f"PnL: <b>{pnl:+.2f} USDT</b>\n"
                f"Лучшая: <code>{best:+.2f}</code> | Худшая: <code>{worst:+.2f}</code>",
                reply_markup=self._main_menu()
            )
        except Exception as e:
            await self.reply(upd, f"❌ {e}")

    async def _cmd_positions(self, upd, arg):
        from datetime import datetime, timezone
        s = self._get_state()
        open_pos = [(sid, st) for sid, st in s.strategies.items() if st.current_position]
        if not open_pos:
            await self.reply(upd, "📭 Нет открытых позиций", reply_markup=self._main_menu())
            return

        now = datetime.now(timezone.utc)
        lines = [f"📌 <b>Открытые позиции ({len(open_pos)})</b>\n"]
        close_buttons = []

        for sid, st in open_pos:
            pos  = st.current_position
            side = pos.get("side", "?")
            e    = pos.get("entry", 0)
            sl   = pos.get("sl", 0)
            tp   = pos.get("tp", 0)
            qty  = pos.get("qty", 0)
            lev  = pos.get("leverage", 1)
            emoji = "🟢" if side == "Buy" else "🔴"

            # Текущая рыночная цена
            cur_price = None
            if s.bybit:
                try:
                    ticker = s.bybit.get_ticker(st.symbol)
                    cur_price = ticker.get("price") if ticker else None
                except Exception:
                    pass

            # Длительность
            opened_at = pos.get("opened_at")
            dur_str = ""
            open_str = ""
            if opened_at:
                try:
                    oa = opened_at if isinstance(opened_at, str) else str(opened_at)
                    dt_open = datetime.fromisoformat(oa.replace("Z", "+00:00"))
                    if dt_open.tzinfo is None:
                        dt_open = dt_open.replace(tzinfo=timezone.utc)
                    open_str = dt_open.strftime("%d.%m.%Y %H:%M:%S UTC")
                    mins = int((now - dt_open).total_seconds() // 60)
                    h, m = divmod(mins, 60)
                    dur_str = f"{h}ч {m}мин" if h else f"{m}мин"
                except Exception:
                    pass

            # Нереализованный PnL по текущей цене
            unreal_str = ""
            if cur_price and e and qty:
                if side == "Buy":
                    unreal_pnl = qty * (cur_price - e)
                else:
                    unreal_pnl = qty * (e - cur_price)
                unreal_emoji = "📈" if unreal_pnl >= 0 else "📉"
                unreal_str = f"   {unreal_emoji} Текущий PnL: <b>{unreal_pnl:+.4f} USDT</b>\n"

            # Дистанция до TP и SL
            ref = cur_price if cur_price else e
            if e > 0 and ref > 0:
                if side == "Buy":
                    dist_tp = (tp - ref) / ref * 100
                    dist_sl = (sl - ref) / ref * 100
                    pnl_tp  = qty * (tp - ref)
                    pnl_sl  = qty * (sl - ref)
                else:
                    dist_tp = (ref - tp) / ref * 100
                    dist_sl = (ref - sl) / ref * 100
                    pnl_tp  = qty * (ref - tp)
                    pnl_sl  = qty * (ref - sl)
                dist_tp_str = f"+{dist_tp:.2f}% (≈{pnl_tp:+.2f} USDT)"
                dist_sl_str = f"{dist_sl:.2f}% (≈{pnl_sl:+.2f} USDT)"
            else:
                dist_tp_str = dist_sl_str = "—"

            margin = round(qty * e / lev, 2) if e and lev else 0
            cur_str = f"   📡 Текущая цена: <b>{cur_price:.4f}</b>\n" if cur_price else ""

            block = (
                f"{emoji} <b>{sid}</b> {st.symbol} {side} ×{lev}\n"
                + (f"   ⏰ Открыта: {open_str} ({dur_str})\n" if open_str else "")
                + f"   💰 Вход: <b>{e:.4f}</b> | Маржа: {margin:.2f} USDT\n"
                + cur_str
                + unreal_str
                + f"   🎯 До TP ({tp:.4f}): {dist_tp_str}\n"
                f"   🛑 До SL ({sl:.4f}): {dist_sl_str}"
            )
            lines.append(block)
            close_buttons.append([_btn(f"❌ Закрыть {sid} {st.symbol}", f"close_pos:{sid}")])

        close_buttons.append([_btn("🔄 Обновить", "positions"), _btn("🏠 Меню", "menu")])
        kb = _keyboard(close_buttons)
        await self.reply(upd, "\n\n".join(lines), reply_markup=kb)

    async def _cmd_close_one_position(self, upd, sid: str):
        s = self._get_state()
        strat = s.strategies.get(sid)
        if not strat or not strat.current_position:
            await self.reply(upd, f"❌ Позиция <b>{sid}</b> не найдена или уже закрыта",
                             reply_markup=self._main_menu())
            return

        symbol = strat.symbol
        pos = strat.current_position
        entry = pos.get("entry", 0)

        if s.paper_mode:
            cur_price = entry
            if s.bybit:
                try:
                    ticker = s.bybit.get_ticker(symbol)
                    cur_price = ticker.get("price", entry) if ticker else entry
                except Exception:
                    pass

            result = s.paper._close_position(symbol, cur_price, reason="manual_tg_close")
            strat.current_position = None
            s.risk_manager.register_position_close(sid)

            pnl = result.get("pnl_usd", 0) if result else 0
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            await self.reply(upd,
                f"{pnl_emoji} <b>Позиция {sid} закрыта вручную</b>\n"
                f"Цена закрытия: <b>{cur_price:.4f}</b>\n"
                f"PnL: <b>{pnl:+.4f} USDT</b>\n"
                f"Баланс: <b>{s.paper.balance:.2f} USDT</b>",
                reply_markup=self._main_menu()
            )
        else:
            side_close = "Sell" if pos.get("side") == "Buy" else "Buy"
            qty = str(pos.get("qty", 0))
            try:
                result = s.bybit.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side_close,
                    orderType="Market",
                    qty=qty,
                    reduceOnly=True,
                )
                if result.get("retCode") == 0:
                    strat.current_position = None
                    s.risk_manager.register_position_close(sid)
                    await self.reply(upd,
                        f"✅ <b>Позиция {sid} {symbol} закрыта вручную</b>",
                        reply_markup=self._main_menu()
                    )
                else:
                    await self.reply(upd, f"❌ Ошибка закрытия: {result.get('retMsg')}")
            except Exception as ex:
                await self.reply(upd, f"❌ Ошибка: {ex}")

    async def _cmd_go(self, upd, arg):
        s = self._get_state()
        if s.bot_running:
            await self.reply(upd, "ℹ️ Уже запущен", reply_markup=self._main_menu())
            return
        if not s.bybit and not s.paper_mode:
            await self.reply(upd, "❌ Нет Bybit API и Paper Mode не включён")
            return
        self._paused = False
        await self._start_fn()
        await self.reply(upd, "🚀 Торговый цикл запущен", reply_markup=self._main_menu())

    async def _cmd_stop(self, upd, arg):
        s = self._get_state()
        if not s.bot_running:
            await self.reply(upd, "ℹ️ Уже остановлен", reply_markup=self._main_menu())
            return
        await self._stop_fn()
        await self.reply(upd, "🛑 Остановлен", reply_markup=self._main_menu())

    async def _cmd_paper_on(self, upd, arg):
        self._get_state().paper_mode = True
        await self.reply(upd, "📄 Paper Mode включён", reply_markup=self._main_menu())

    async def _cmd_paper_off(self, upd, arg):
        self._get_state().paper_mode = False
        await self.reply(upd, "💰 Real Mode — реальная торговля!", reply_markup=self._main_menu())

    async def _cmd_strategies(self, upd, arg):
        s = self._get_state()
        lines = ["<b>Стратегии</b>\n"]
        improve_buttons = []
        for sid, st in s.strategies.items():
            icon = "🚫" if st.auto_disabled else ("✅" if st.enabled else "⏹")
            pos  = "📌" if st.current_position else "  "
            wr   = st.wins / st.trades * 100 if st.trades else 0
            pnl_str = f"{st.pnl:+.1f}$" if st.trades else "—"
            lines.append(f"{icon}{pos} <code>{sid}</code> WR={wr:.0f}% PnL={pnl_str}")
            if st.trades >= 5:
                improve_buttons.append(_btn(f"🤖 {sid}", f"improve:{sid}"))
        # Кнопки AI-советов (по 3 в ряд)
        kb_rows = [
            improve_buttons[i:i+3] for i in range(0, len(improve_buttons), 3)
        ]
        if improve_buttons:
            lines.append("\n<i>Нажми кнопку — AI-советы по стратегии</i>")
        kb_rows.append([_btn("🏠 Меню", "menu")])
        await self.reply(upd, "\n".join(lines), reply_markup=_keyboard(kb_rows))

    async def _cmd_improve(self, upd, arg):
        """AI-советы по улучшению стратегии на основе закрытых сделок."""
        s = self._get_state()
        sid = arg.upper().strip() if arg else ""

        if not s.ai.enabled:
            await self.reply(upd,
                "❌ AI недоступен — задайте <code>ANTHROPIC_API_KEY</code> или <code>OPENAI_API_KEY</code> в .env",
                reply_markup=self._main_menu())
            return

        if not sid:
            # Показываем список стратегий с кнопками
            lines = ["<b>🤖 AI-советы по стратегии</b>\n",
                     "Выберите стратегию или напишите <code>/improve S3</code>:"]
            btns = []
            for k, st in s.strategies.items():
                wr = st.wins / st.trades * 100 if st.trades else 0
                lines.append(f"  <code>{k}</code> — {st.NAME} ({st.trades} сд., WR={wr:.0f}%)")
                if st.trades >= 5:
                    btns.append(_btn(f"🤖 {k}", f"improve:{k}"))
            btns_rows = [btns[i:i+3] for i in range(0, len(btns), 3)]
            btns_rows.append([_btn("🏠 Меню", "menu")])
            await self.reply(upd, "\n".join(lines), reply_markup=_keyboard(btns_rows))
            return

        if sid not in s.strategies:
            await self.reply(upd,
                f"❌ Стратегия <code>{sid}</code> не найдена\nДоступны: {', '.join(s.strategies.keys())}")
            return

        strat = s.strategies[sid]
        if strat.trades < 5:
            await self.reply(upd,
                f"⚠️ У <code>{sid}</code> только {strat.trades} сделок — нужно минимум 5 для анализа")
            return

        await self.reply(upd, f"⏳ AI анализирует <b>{sid}</b> ({strat.trades} сделок)...")
        try:
            trades_hist = s.journal.get_trades(strategy_id=sid, limit=50)
            stats_list  = s.journal.get_stats_by_strategy()
            stats       = next((x for x in stats_list if x["strategy_id"] == sid), {})
            result = s.ai.analyze_strategy_performance({
                "id":     sid,
                "name":   strat.NAME,
                "symbol": strat.symbol,
                "trades": trades_hist,
                "stats":  stats,
            })
            if not result.get("available"):
                await self.reply(upd,
                    f"❌ AI недоступен: {result.get('message', '?')}",
                    reply_markup=self._main_menu())
                return
            analysis = result.get("analysis", "Нет данных")[:2000]
            wr = strat.wins / strat.trades * 100 if strat.trades else 0
            await self.reply(upd,
                f"🤖 <b>AI-советы: {sid} ({strat.NAME})</b>\n"
                f"<i>Сделок: {strat.trades} | WR: {wr:.0f}% | PnL: {strat.pnl:+.2f}$</i>\n\n"
                f"{analysis}",
                reply_markup=_keyboard([
                    [_btn(f"🔄 Обновить анализ {sid}", f"improve:{sid}")],
                    [_btn("📈 Стратегии", "strategies"), _btn("🏠 Меню", "menu")],
                ])
            )
        except Exception as e:
            logger.error(f"[TgCommander] improve {sid}: {e}")
            await self.reply(upd, f"❌ Ошибка анализа: {e}", reply_markup=self._main_menu())

    async def _cmd_advisor(self, upd, arg):
        """AI-анализ всех стратегий: /advisor или /advisor S1."""
        s = self._get_state()
        sid = arg.upper().strip() if arg else ""

        if not s.advisor.enabled:
            await self.reply(upd,
                "❌ AI-советник недоступен\n"
                "Задайте <code>ANTHROPIC_API_KEY</code> или <code>OPENAI_API_KEY</code> в .env",
                reply_markup=self._main_menu())
            return

        # Если указана стратегия — показываем детали из последнего анализа
        if sid:
            last = s.advisor.get_last_analysis()
            if last:
                analysis = last.get("analysis", {})
                analysis["available"] = True
                text = s.advisor.format_strategy_detail(analysis, sid)
                await self.reply(upd, text, reply_markup=_keyboard([
                    [_btn("🔄 Запустить новый анализ", "advisor_run"),
                     _btn("📊 Все стратегии", "advisor_all")],
                    [_btn("🏠 Меню", "menu")],
                ]))
            else:
                await self.reply(upd,
                    f"⚠️ Нет сохранённого анализа для <code>{sid}</code>\n"
                    "Запустите /advisor чтобы провести анализ.")
            return

        # Показываем последний анализ или предлагаем запустить
        last = s.advisor.get_last_analysis()
        if last:
            ts = last.get("timestamp", "")[:16].replace("T", " ")
            total = last.get("total_trades", 0)
            age_min = int((datetime.utcnow() - datetime.fromisoformat(
                last["timestamp"][:19]
            )).total_seconds() / 60) if last.get("timestamp") else 999
            age_str = f"{age_min} мин назад" if age_min < 60 else f"{age_min//60}ч назад"

            analysis = last.get("analysis", {})
            analysis["available"] = True
            text = s.advisor.format_telegram(analysis, brief=True)
            text += f"\n\n<i>Анализ от {ts} UTC ({age_str})</i>"
            await self.reply(upd, text, reply_markup=_keyboard([
                [_btn("🔄 Обновить анализ", "advisor_run"),
                 _btn("📋 Подробно", "advisor_all")],
                [_btn("🏠 Меню", "menu")],
            ]))
        else:
            journal_stats = s.journal.get_stats_by_strategy()
            total = sum(x.get("trades", 0) for x in journal_stats)
            await self.reply(upd,
                f"🤖 <b>AI-советник по стратегиям</b>\n\n"
                f"Сделок в БД: <b>{total}</b>\n"
                f"Нажми кнопку для запуска анализа всех стратегий.",
                reply_markup=_keyboard([
                    [_btn("🚀 Запустить анализ", "advisor_run")],
                    [_btn("🏠 Меню", "menu")],
                ]))

    async def _run_advisor_full(self, upd):
        """Запуск полного AI-анализа всех стратегий."""
        s = self._get_state()
        journal_stats = s.journal.get_stats_by_strategy()
        total = sum(x.get("trades", 0) for x in journal_stats)
        await self.reply(upd, f"⏳ AI анализирует все стратегии ({total} сделок в БД)...")
        try:
            market_ctx = {
                "balance": s.paper.balance if s.paper_mode else 0,
                "open_positions": s.risk_manager.open_positions_count,
                "daily_pnl": s.risk_manager.daily_pnl,
                "regime": "unknown",
            }
            result = await s.advisor.analyze_all(
                journal_stats=journal_stats,
                strategies=s.strategies,
                market_context=market_ctx,
                triggered_by="manual",
            )
            if result.get("available"):
                text = s.advisor.format_telegram(result, brief=False)
                await self.reply(upd, text[:4000], reply_markup=_keyboard([
                    [_btn("🔄 Обновить", "advisor_run"),
                     _btn("📊 Стратегии", "strategies")],
                    [_btn("🏠 Меню", "menu")],
                ]))
            else:
                await self.reply(upd,
                    f"❌ Ошибка анализа: {result.get('reason', '?')}",
                    reply_markup=self._main_menu())
        except Exception as e:
            logger.error(f"[TgCommander] advisor run: {e}")
            await self.reply(upd, f"❌ Ошибка: {e}", reply_markup=self._main_menu())

    async def _cmd_scalp(self, upd, arg):
        """
        /scalp        — статус скальпинга
        /scalp on     — включить
        /scalp off    — выключить
        """
        from main import activate_scalp_mode, deactivate_scalp_mode
        s = self._get_state()
        a = arg.lower().strip()
        if a in ("on", "вкл", "1"):
            added = activate_scalp_mode()
            count = len([sid for sid in s.strategies if sid.startswith("SC_")])
            await self.reply(upd, f"⚡ Скальпинг включён\nСтратегий: <b>{count}</b>\nДобавлено: {added}")
        elif a in ("off", "выкл", "0"):
            deactivate_scalp_mode()
            await self.reply(upd, "⏹ Скальпинг выключен, SC_* стратегии приостановлены")
        else:
            sc_strats = [(sid, st) for sid, st in s.strategies.items() if sid.startswith("SC_")]
            active = s.scalp_active
            status = "🟢 ВКЛ" if active else "🔴 ВЫКЛ"
            lines = [f"⚡ <b>Скальпинг:</b> {status}",
                     f"Скальперов: <b>{len(sc_strats)}</b>", ""]
            for sid, st in sc_strats[:15]:
                pos = "📌" if st.current_position else "—"
                wr = f"{st.win_rate:.0f}%" if st.trades else "—"
                pnl = f"{st.pnl:+.2f}$" if st.trades else "—"
                lines.append(f"{pos} <code>{sid}</code>  WR:{wr}  PnL:{pnl}  ({st.symbol})")
            lines.append("\n<i>/scalp on — включить, /scalp off — выключить</i>")
            await self.reply(upd, "\n".join(lines))

    async def _cmd_enable(self, upd, arg):
        s   = self._get_state()
        sid = arg.upper().strip()
        if sid not in s.strategies:
            await self.reply(upd, f"❌ Стратегия <code>{sid}</code> не найдена")
            return
        s.strategies[sid].enabled      = True
        s.strategies[sid].auto_disabled = False
        await self.reply(upd, f"✅ {sid} включена")

    async def _cmd_disable(self, upd, arg):
        s   = self._get_state()
        sid = arg.upper().strip()
        if sid not in s.strategies:
            await self.reply(upd, f"❌ Стратегия <code>{sid}</code> не найдена")
            return
        s.strategies[sid].enabled = False
        await self.reply(upd, f"⏹ {sid} отключена")

    async def _cmd_risk(self, upd, arg):
        s  = self._get_state()
        rm = s.risk_manager.get_status()
        training = getattr(s, "training_mode", False)
        if training:
            kill = "🎓 Режим обучения (лимиты ВЫКЛ)"
        elif rm.get("kill_switch"):
            kill = "🛑 KILL SWITCH"
        else:
            kill = "✅ Активен"
        training_note = (
            "\n\n⚠️ <b>Режим обучения АКТИВЕН</b>\n"
            "Все лимиты и фильтры отключены.\n"
            "Бот торгует без ограничений."
        ) if training else ""
        await self.reply(upd,
            f"🛡 <b>Риск-менеджер</b>\n\n"
            f"Статус: {kill}\n"
            f"Позиций: {rm.get('open_positions',0)}/{rm.get('max_positions',4)}\n"
            f"Сделок сегодня: <b>{rm.get('daily_trades_count',0)}</b>\n"
            f"Убытков сегодня: <b>{rm.get('daily_losses_count',0)}/{rm.get('max_daily_losses',3)}</b> (стоп после {rm.get('max_daily_losses',3)})\n"
            f"Дневной PnL: <b>{rm.get('daily_pnl',0):+.2f} USDT</b>\n"
            f"Лимит просадки: {rm.get('daily_max_loss_pct',0):.0f}%{training_note}",
            reply_markup=self._main_menu()
        )

    async def _cmd_ai(self, upd, arg):
        from datetime import datetime, timezone
        s   = self._get_state()
        rec = getattr(s, "_cached_ai_rec", None)

        # Если нет кэша или устарел — запускаем получение прямо сейчас
        if not rec or not rec.get("available"):
            reason = (rec or {}).get("reason", "")
            # Объясняем конкретную причину
            if "Anthropic API" in reason:
                await self.reply(upd,
                    "❌ <b>AI недоступен</b>\n\n"
                    "Не задан <code>ANTHROPIC_API_KEY</code> в .env\n"
                    "Добавьте ключ и перезапустите бот",
                    reply_markup=self._main_menu()
                )
                return
            if not s.news_enabled:
                await self.reply(upd,
                    "ℹ️ Новостной модуль отключён (<code>NEWS_ENABLED=false</code>)\n"
                    "AI рекомендации требуют новостных данных",
                    reply_markup=self._main_menu()
                )
                return
            # Пробуем получить прямо сейчас
            await self.reply(upd, "⏳ Запрашиваю AI рекомендации...")
            try:
                balance = s.paper.balance if s.paper_mode else 0
                portfolio_ctx = {
                    "balance": balance,
                    "open_positions": s.risk_manager.open_positions_count,
                    "daily_pnl": s.risk_manager.daily_pnl,
                    "active_strategies": [sid for sid, st in s.strategies.items() if st.enabled],
                }
                rec = await s.news_manager.get_trading_recommendations(portfolio_context=portfolio_ctx)
                s._cached_ai_rec = rec
                s._last_ai_rec = datetime.utcnow()
            except Exception as e:
                await self.reply(upd, f"❌ Ошибка получения AI рекомендаций: {e}",
                                 reply_markup=self._main_menu())
                return
            if not rec or not rec.get("available"):
                reason2 = (rec or {}).get("reason", "нет свежих данных")
                await self.reply(upd,
                    f"ℹ️ AI рекомендации недоступны\n<i>Причина: {reason2}</i>\n\n"
                    "Попробуйте через несколько минут — новостной модуль собирает данные каждые 15 мин",
                    reply_markup=self._main_menu()
                )
                return

        action = rec.get("action", "?").upper()
        risk   = rec.get("risk_level", "?")
        reason = rec.get("reasoning", "")[:400]
        fg     = rec.get("fear_greed", {})
        fg_val = fg.get("value","?") if fg else "?"
        fg_lbl = fg.get("label","") if fg else ""
        recs   = rec.get("recommendations", [])
        last_upd = getattr(s, "_last_ai_rec", None)
        upd_str = last_upd.strftime("%H:%M UTC") if last_upd else "?"

        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "extreme": "🔴"}.get(risk, "⚪")
        rec_block = ("\n" + "\n".join(f"• {r}" for r in recs[:4])) if recs else ""

        await self.reply(upd,
            f"🧠 <b>AI Рекомендация</b> <i>({upd_str})</i>\n\n"
            f"Действие: <b>{action}</b>\n"
            f"{risk_emoji} Риск: <b>{risk}</b>\n"
            f"Fear &amp; Greed: <b>{fg_val}</b> {fg_lbl}"
            + rec_block + "\n\n"
            f"<i>{reason}</i>",
            reply_markup=self._main_menu()
        )

    async def _cmd_news(self, upd, arg):
        s = self._get_state()
        if not s.news_enabled:
            await self.reply(upd, "ℹ️ Новостной модуль отключён")
            return
        try:
            feat  = s.news_manager.get_sentiment_features()
            score = feat.get("sentiment_score", 0)
            count = feat.get("news_count_24h", 0)
            bull  = feat.get("bull_count", 0)
            bear  = feat.get("bear_count", 0)
            emoji = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
            await self.reply(upd,
                f"📰 <b>Новостной сентимент</b>\n\n"
                f"{emoji} Score: <b>{score:+.3f}</b>\n"
                f"Новостей 24ч: <b>{count}</b>\n"
                f"Бычьих: {bull} | Медвежьих: {bear}",
                reply_markup=self._main_menu()
            )
        except Exception as e:
            await self.reply(upd, f"❌ {e}")

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
        await self.reply(upd, "⏸ Пауза — новые сделки не открываются", reply_markup=self._main_menu())

    async def _cmd_resume(self, upd, arg):
        if not self._paused:
            await self.reply(upd, "ℹ️ Бот не на паузе")
            return
        self._paused = False
        s = self._get_state()
        s.bot_running = True
        await self._start_fn()
        await self.reply(upd, "▶️ Возобновлён", reply_markup=self._main_menu())

    async def send_document(self, chat_id: str, file_path: str,
                             filename: str, caption: str = "") -> bool:
        """Отправить файл через Telegram (multipart/form-data)."""
        from pathlib import Path
        try:
            file_bytes = Path(file_path).read_bytes()
            session = await self._get_session()
            url = f"{self._base_url}/sendDocument"
            data = aiohttp.FormData()
            data.add_field("chat_id", str(chat_id))
            data.add_field("caption", caption, content_type="text/plain")
            data.add_field("parse_mode", "HTML")
            data.add_field(
                "document",
                file_bytes,
                filename=filename,
                content_type="application/octet-stream",
            )
            async with session.post(url, data=data) as resp:
                result = await resp.json()
                return result.get("ok", False)
        except Exception as e:
            logger.error(f"send_document: {e}")
            return False

    async def _cmd_exportdb(self, upd, arg):
        """Выгрузить базу данных сделок в Telegram."""
        chat_id = upd.get("chat", {}).get("id") or upd.get("message", {}).get("chat", {}).get("id")
        if not chat_id:
            return
        s = self._get_state()
        await self.reply(upd, "⏳ Готовлю базу данных...")
        try:
            from pathlib import Path
            db_path = getattr(s.journal.pool, "db_path", None)
            if db_path and Path(db_path).exists():
                ts = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M")
                fname = f"baibit_trades_{ts}.db"
                ok = await self.send_document(str(chat_id), db_path, fname,
                                              caption="🗃 SQLite база сделок")
                if not ok:
                    raise RuntimeError("Ошибка отправки файла")
            else:
                # Fallback: CSV-дамп
                csv_path = s.journal.export_to_csv()
                ts = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M")
                fname = f"baibit_trades_{ts}.csv"
                ok = await self.send_document(str(chat_id), csv_path, fname,
                                              caption="📊 CSV-экспорт сделок")
                if not ok:
                    raise RuntimeError("Ошибка отправки файла")
        except Exception as e:
            await self.reply(upd, f"❌ Ошибка экспорта: {e}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
