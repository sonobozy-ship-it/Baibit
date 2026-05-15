"""
Telegram уведомления о событиях бота.
Используется python-telegram-bot или httpx (прямой запрос к Bot API).
"""
import os
import logging
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bot_token and chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    async def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Отправить сообщение."""
        if not self.enabled:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return True
                    logger.warning(f"Telegram error: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def notify_trade_open(self, strategy_id: str, symbol: str, side: str,
                                 entry: float, sl: float, tp: float, reason: str):
        emoji = "🟢" if side == "BUY" else "🔴"
        text = (
            f"{emoji} <b>{side} {symbol}</b>\n"
            f"Стратегия: <code>{strategy_id}</code>\n"
            f"Вход: <b>{entry:.4f}</b>\n"
            f"SL: <code>{sl:.4f}</code>\n"
            f"TP: <code>{tp:.4f}</code>\n"
            f"Причина: <i>{reason}</i>"
        )
        await self.send(text)

    async def notify_trade_close(self, strategy_id: str, symbol: str, pnl: float, reason: str):
        emoji = "✅" if pnl > 0 else "❌"
        text = (
            f"{emoji} <b>Закрыта {symbol}</b>\n"
            f"Стратегия: <code>{strategy_id}</code>\n"
            f"PnL: <b>{'+' if pnl > 0 else ''}{pnl:.2f} USDT</b>\n"
            f"Причина: <i>{reason}</i>"
        )
        await self.send(text)

    async def notify_kill_switch(self, reason: str, balance: float):
        text = (
            f"🛑 <b>KILL SWITCH СРАБОТАЛ</b>\n\n"
            f"Причина: <i>{reason}</i>\n"
            f"Баланс: <code>{balance:.2f} USDT</code>\n\n"
            f"⚠️ Все стратегии остановлены до завтра."
        )
        await self.send(text)

    async def notify_strategy_disabled(self, strategy_id: str, name: str, reason: str):
        text = (
            f"⚠️ <b>Стратегия отключена</b>\n"
            f"<code>{strategy_id}</code> {name}\n"
            f"Причина: <i>{reason}</i>"
        )
        await self.send(text)

    async def notify_daily_report(self, stats: dict):
        text = (
            f"📊 <b>Дневной отчёт</b>\n\n"
            f"Сделок: <b>{stats.get('trades', 0)}</b>\n"
            f"Win Rate: <b>{stats.get('win_rate', 0):.1f}%</b>\n"
            f"PnL: <b>{stats.get('pnl', 0):+.2f} USDT</b>\n"
            f"Лучшая сделка: <code>{stats.get('best', 0):+.2f}</code>\n"
            f"Худшая сделка: <code>{stats.get('worst', 0):+.2f}</code>"
        )
        await self.send(text)
