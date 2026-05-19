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
        go_btn = _btn("🛑 Стоп", "stop") if s.bot_running else _btn("🚀 Старт", "go")
        pause_btn = _btn("▶️ Возобновить", "resume") if self._paused else _btn("⏸ Пауза", "pause")
        return _keyboard([
            [go_btn, pause_btn],
            [_btn("💰 Баланс", "balance"), _btn("📌 Позиции", "positions")],
            [_btn("📊 Сделки", "trades"), _btn("🏆 Лучшие модели", "best_models")],
            [_btn("📈 Стратегии", "strategies"), _btn("🛡 Риск", "risk")],
            [_btn("📰 Новости", "news"), _btn("🤖 AI", "ai")],
            [_btn("🔄 Обновить меню", "menu")],
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
        }

        # Закрытие отдельной позиции
        if data.startswith("close_pos:"):
            sid = data[len("close_pos:"):]
            await self._cmd_close_one_position(fake_upd, sid)
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
            "/news — новостной сентимент"
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
            wins   = sum(1 for t in trades if (t.get("pnl_usd") or 0) > 0)
            pnl    = sum(t.get("pnl_usd") or 0 for t in trades)
            wr     = wins / total * 100 if total else 0
            best   = max((t.get("pnl_usd") or 0 for t in trades), default=0)
            worst  = min((t.get("pnl_usd") or 0 for t in trades), default=0)
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
        for sid, st in s.strategies.items():
            icon = "🚫" if st.auto_disabled else ("✅" if st.enabled else "⏹")
            pos  = "📌" if st.current_position else "  "
            wr   = st.wins / st.trades * 100 if st.trades else 0
            pnl_str = f"{st.pnl:+.1f}$" if st.trades else "—"
            lines.append(f"{icon}{pos} <code>{sid}</code> WR={wr:.0f}% PnL={pnl_str}")
        await self.reply(upd, "\n".join(lines), reply_markup=self._main_menu())

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
        kill = "🛑 KILL SWITCH" if rm.get("kill_switch") else "✅ Активен"
        await self.reply(upd,
            f"🛡 <b>Риск-менеджер</b>\n\n"
            f"Статус: {kill}\n"
            f"Позиций: {rm.get('open_positions',0)}/{rm.get('max_positions',4)}\n"
            f"Сделок сегодня: <b>{rm.get('daily_trades_count',0)}</b>\n"
            f"Убытков сегодня: <b>{rm.get('daily_losses_count',0)}/{rm.get('max_daily_losses',3)}</b> (стоп после {rm.get('max_daily_losses',3)})\n"
            f"Дневной PnL: <b>{rm.get('daily_pnl',0):+.2f} USDT</b>\n"
            f"Лимит просадки: {rm.get('daily_max_loss_pct',0):.0f}%",
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

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
