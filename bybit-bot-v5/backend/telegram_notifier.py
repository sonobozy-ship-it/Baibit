"""
Telegram уведомления о событиях бота.
Используется python-telegram-bot или httpx (прямой запрос к Bot API).
"""
import os
import logging
import aiohttp
from typing import Optional

try:
    import pandas as pd
    from chart_generator import generate_trade_chart
    _CHART_OK = True
except Exception:
    _CHART_OK = False

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

    async def send_photo(self, photo_bytes: bytes, caption: str = "", parse_mode: str = "HTML") -> bool:
        """Отправить PNG-график."""
        if not self.enabled or not photo_bytes:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field("chat_id", self.chat_id)
                form.add_field("caption", caption[:1024])
                form.add_field("parse_mode", parse_mode)
                form.add_field("photo", photo_bytes, filename="chart.png", content_type="image/png")
                async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.warning(f"Telegram photo error {resp.status}: {body[:200]}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send_photo failed: {e}")
            return False

    async def notify_trade_open(self, strategy_id: str, symbol: str, side: str,
                                entry: float, sl: float, tp: float, reason: str,
                                qty: float = 0.0, leverage: int = 1,
                                df=None):
        from datetime import datetime, timezone
        emoji = "🟢" if side in ("BUY", "Buy") else "🔴"
        notional = qty * entry if qty else 0
        margin = round(notional / leverage, 2) if leverage and leverage > 0 else notional
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%d.%m.%Y %H:%M:%S UTC")
        caption = (
            f"{emoji} <b>ВХОД: {side} {symbol}</b>\n"
            f"⏰ Открыт: <b>{time_str}</b>\n"
            f"📊 Стратегия: <code>{strategy_id}</code>\n"
            f"💰 Вход: <b>{entry:.4f}</b>"
            + (f"\n💵 Маржа: <b>{margin:.2f} USDT</b> (×{leverage} = {notional:.1f} USDT)" if notional else "") +
            f"\n⚡ Плечо: <b>×{leverage}</b>\n"
            f"🛑 SL: <code>{sl:.4f}</code>\n"
            f"🎯 TP: <code>{tp:.4f}</code>"
        )
        if _CHART_OK and df is not None:
            try:
                chart = generate_trade_chart(
                    df=df, symbol=symbol, side=side,
                    entry=entry, sl=sl, tp=tp,
                    strategy_id=strategy_id,
                )
                if chart:
                    await self.send_photo(chart, caption=caption)
                    return
            except Exception as e:
                logger.warning(f"[Chart] notify_trade_open: {e}")
        await self.send(caption)

    async def notify_trade_close(self, strategy_id: str, symbol: str, pnl: float, reason: str,
                                 leverage: int = 1,
                                 df=None, entry: Optional[float] = None,
                                 side: Optional[str] = None, sl: Optional[float] = None,
                                 tp: Optional[float] = None, exit_price: Optional[float] = None,
                                 opened_at: Optional[str] = None):
        from datetime import datetime, timezone, timedelta
        emoji = "✅" if pnl > 0 else "❌"
        now = datetime.now(timezone.utc)
        close_str = now.strftime("%d.%m.%Y %H:%M:%S UTC")
        pnl_str = f"+{pnl:.2f}" if pnl > 0 else f"{pnl:.2f}"

        duration_str = ""
        open_str = ""
        if opened_at:
            try:
                if hasattr(opened_at, "isoformat"):
                    opened_at = opened_at.isoformat()
                dt_open = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
                if dt_open.tzinfo is None:
                    dt_open = dt_open.replace(tzinfo=timezone.utc)
                open_str = dt_open.strftime("%d.%m.%Y %H:%M:%S UTC")
                delta = now - dt_open
                total_min = int(delta.total_seconds() // 60)
                h, m = divmod(total_min, 60)
                duration_str = f"{h}ч {m}мин" if h else f"{m}мин"
            except Exception:
                pass

        caption = (
            f"{emoji} <b>ВЫХОД: {symbol}</b>\n"
            + (f"⏰ Открыт: <b>{open_str}</b>\n" if open_str else "")
            + f"⏰ Закрыт: <b>{close_str}</b>\n"
            + (f"⌛ Длительность: <i>{duration_str}</i>\n" if duration_str else "")
            + f"📊 Стратегия: <code>{strategy_id}</code>\n"
            f"⚡ Плечо: <b>×{leverage}</b>\n"
            f"💵 PnL: <b>{pnl_str} USDT</b>\n"
            f"📌 Причина: <i>{reason}</i>"
        )
        if _CHART_OK and df is not None and entry and side and sl and tp:
            try:
                norm_side = side.upper() if side else side
                chart = generate_trade_chart(
                    df=df, symbol=symbol, side=norm_side,
                    entry=entry, sl=sl, tp=tp,
                    exit_price=exit_price,
                    strategy_id=strategy_id,
                )
                if chart:
                    await self.send_photo(chart, caption=caption)
                    return
            except Exception as e:
                logger.warning(f"[Chart] notify_trade_close: {e}")
        await self.send(caption)

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
