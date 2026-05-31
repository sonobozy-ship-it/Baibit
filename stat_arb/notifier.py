"""Telegram notifications for the stat-arb bot."""
from __future__ import annotations

import logging

import aiohttp

from .config import ArbConfig
from .models import ArbPosition, SpreadSnapshot

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, cfg: ArbConfig):
        self._token   = cfg.telegram_token
        self._chat_id = cfg.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)

    async def _send(self, text: str) -> None:
        if not self._enabled or not text.strip():
            return
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }, timeout=aiohttp.ClientTimeout(total=8))
        except Exception as exc:
            logger.debug(f"[Notifier] send error: {exc}")

    async def send_entry(self, pos: ArbPosition) -> None:
        text = (
            f"🟢 <b>ARBITRAGE ENTRY</b>\n\n"
            f"Монета: <b>{pos.symbol}</b>\n"
            f"LONG:   {pos.long_exchange}  @ {pos.long_entry}\n"
            f"SHORT:  {pos.short_exchange} @ {pos.short_entry}\n\n"
            f"Spread: <b>{pos.entry_spread_pct:.3f}%</b>\n"
            f"Объём: {pos.notional_usdt:.0f}$ × {pos.leverage}x\n"
            f"Ожидаемая прибыль: <b>≈{pos.expected_pnl_usdt:.4f}$</b>"
        )
        await self._send(text)

    async def send_exit(self, pos: ArbPosition) -> None:
        emoji = "💰" if pos.realized_pnl_usdt >= 0 else "🔴"
        text = (
            f"{emoji} <b>ARBITRAGE EXIT</b>\n\n"
            f"Монета: <b>{pos.symbol}</b>\n"
            f"Вход:  {pos.entry_spread_pct:.3f}%\n"
            f"Выход: {pos.exit_spread_pct:.3f}%\n\n"
            f"PnL: <b>{pos.realized_pnl_usdt:+.5f}$</b>\n"
            f"Причина: {pos.exit_reason}"
        )
        await self._send(text)

    async def send_error(self, msg: str) -> None:
        await self._send(f"⚠️ <b>ARB ERROR</b>\n{msg}")

    async def send_status(self, paper_balance: float, open_count: int, summary: dict) -> None:
        text = (
            f"📊 <b>Stat-Arb Status</b>\n\n"
            f"Баланс: <b>{paper_balance:.2f}$</b>\n"
            f"Открытых: <b>{open_count}</b>\n"
            f"Сделок: {summary.get('total', 0)} | "
            f"WR: {summary.get('winrate', 0):.1%} | "
            f"PnL: {summary.get('pnl_usdt', 0):+.4f}$"
        )
        await self._send(text)
