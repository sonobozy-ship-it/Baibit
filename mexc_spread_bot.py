"""
MEXC Spot — spread capture market-maker bot.
Ставит BUY limit на best_bid и SELL limit на best_ask.
Автоматически перевыставляет при движении цены.
Максимальная ставка — MAX_BET_USDT за каждую сторону.

Запуск:
    pip install ccxt python-dotenv
    python mexc_spread_bot.py

Переменные окружения (.env):
    MEXC_API_KEY=...
    MEXC_API_SECRET=...
    TELEGRAM_BOT_TOKEN=...   (опционально)
    TELEGRAM_CHAT_ID=...     (опционально)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import ccxt
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ — всё здесь
# ═══════════════════════════════════════════════════════════════════
SYMBOL        = "PLB/USDT"   # тикер MEXC Spot (ccxt-формат)
MAX_BET_USDT  = 2.0          # максимальный размер ордера в USDT
REPRICE_SEC   = 3.0          # перевыставить ордер если старше N секунд
MIN_SPREAD_PCT = 0.05        # не входить если спред < 0.05%
MAX_INVENTORY_USDT = 6.0     # максимальная удержанная позиция в USDT
PAPER_MODE    = True         # True = симуляция без реальных ордеров

MEXC_API_KEY    = os.getenv("MEXC_API_KEY",    "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")
TG_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID",   "")
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BotState:
    # Открытые ордера
    buy_order_id:    Optional[str]   = None
    buy_order_price: float           = 0.0
    buy_order_ts:    float           = 0.0

    sell_order_id:    Optional[str]  = None
    sell_order_price: float          = 0.0
    sell_order_ts:    float          = 0.0

    # Инвентарь (накопленная нетто-позиция в базовой монете)
    inventory:       float           = 0.0   # в PLB

    # Статистика
    trades:          int             = 0
    wins:            int             = 0
    total_pnl:       float           = 0.0
    total_fees:      float           = 0.0
    session_start:   float           = field(default_factory=time.time)

    # Paper-режим: цена последней покупки для расчёта PnL
    paper_buy_price: float           = 0.0


# ── Exchange ──────────────────────────────────────────────────────

def make_exchange() -> ccxt.Exchange:
    return ccxt.mexc({
        "apiKey":          MEXC_API_KEY,
        "secret":          MEXC_API_SECRET,
        "enableRateLimit": True,
        "options":         {"defaultType": "spot"},
    })


# ── Telegram ──────────────────────────────────────────────────────

async def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID or not text.strip():
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={
                "chat_id":    TG_CHAT_ID,
                "text":       text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=aiohttp.ClientTimeout(total=5))
    except Exception:
        pass


# ── Helpers ───────────────────────────────────────────────────────

def round_price(price: float, precision: int = 4) -> float:
    return round(price, precision)


async def fetch_orderbook(ex: ccxt.Exchange) -> Optional[dict]:
    """Fetch top of orderbook (async via run_in_executor)."""
    loop = asyncio.get_event_loop()
    try:
        ob = await loop.run_in_executor(
            None, lambda: ex.fetch_order_book(SYMBOL, limit=5)
        )
        if ob["bids"] and ob["asks"]:
            return {
                "bid": float(ob["bids"][0][0]),
                "ask": float(ob["asks"][0][0]),
                "bid_size": float(ob["bids"][0][1]),
                "ask_size": float(ob["asks"][0][1]),
            }
    except Exception as exc:
        logger.warning(f"fetch_orderbook: {exc}")
    return None


async def cancel_order(ex: ccxt.Exchange, order_id: str) -> None:
    if PAPER_MODE or not order_id:
        return
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, lambda: ex.cancel_order(order_id, SYMBOL)
        )
        logger.debug(f"cancelled {order_id}")
    except Exception as exc:
        logger.debug(f"cancel {order_id}: {exc}")


async def place_limit(
    ex: ccxt.Exchange,
    side: str,    # "buy" | "sell"
    price: float,
    qty: float,
) -> Optional[str]:
    """Place limit order, return order_id (or fake id in paper mode)."""
    if PAPER_MODE:
        fake_id = f"PAPER-{side[:1].upper()}-{time.time():.0f}"
        logger.info(f"[PAPER] {side.upper()} {qty:.4f} {SYMBOL} @ {price}")
        return fake_id

    loop = asyncio.get_event_loop()
    try:
        order = await loop.run_in_executor(
            None,
            lambda: ex.create_limit_order(SYMBOL, side, qty, price),
        )
        oid = order.get("id")
        logger.info(f"[LIVE] {side.upper()} {qty:.4f} {SYMBOL} @ {price}  id={oid}")
        return oid
    except Exception as exc:
        logger.error(f"place_limit {side}: {exc}")
        return None


async def check_filled(
    ex: ccxt.Exchange,
    order_id: Optional[str],
) -> bool:
    """Return True if order is no longer open (filled or cancelled)."""
    if PAPER_MODE:
        return False   # fills handled separately in paper logic

    if not order_id:
        return False
    loop = asyncio.get_event_loop()
    try:
        open_ids = await loop.run_in_executor(
            None,
            lambda: {o["id"] for o in ex.fetch_open_orders(SYMBOL)},
        )
        return order_id not in open_ids
    except Exception:
        return False


# ── Paper fill simulation ─────────────────────────────────────────

def paper_try_fill_buy(ob: dict, order_price: float) -> bool:
    """Maker BUY fills when best_ask drops to or below our bid price."""
    return ob["ask"] <= order_price


def paper_try_fill_sell(ob: dict, order_price: float) -> bool:
    """Maker SELL fills when best_bid rises to or above our ask price."""
    return ob["bid"] >= order_price


# ── Main loop ─────────────────────────────────────────────────────

async def run() -> None:
    mode = "PAPER" if PAPER_MODE else "LIVE"
    logger.info(f"=== MEXC Spread Bot [{mode}] {SYMBOL} max={MAX_BET_USDT}$ ===")
    await tg_send(
        f"🤖 <b>MEXC Spread Bot [{mode}]</b>\n"
        f"Пара: {SYMBOL}\n"
        f"Ставка: {MAX_BET_USDT}$"
    )

    ex    = make_exchange()
    state = BotState(session_start=time.time())
    last_status_ts = time.time()

    while True:
        try:
            ob = await fetch_orderbook(ex)
            if not ob:
                await asyncio.sleep(1)
                continue

            bid        = ob["bid"]
            ask        = ob["ask"]
            spread_pct = (ask - bid) / bid * 100

            # ── Spread too thin ───────────────────────────────────
            if spread_pct < MIN_SPREAD_PCT:
                logger.debug(f"spread={spread_pct:.4f}% too thin, skip")
                await asyncio.sleep(0.5)
                continue

            now = time.time()
            qty = round(MAX_BET_USDT / bid, 4)

            # ═══════════════════════════════════════════════════════
            #  PAPER-MODE fill checks
            # ═══════════════════════════════════════════════════════
            if PAPER_MODE:
                # Check if existing BUY was filled
                if state.buy_order_id and paper_try_fill_buy(ob, state.buy_order_price):
                    fill_price      = state.buy_order_price
                    state.inventory += qty
                    state.paper_buy_price = fill_price
                    state.buy_order_id    = None
                    logger.info(
                        f"[PAPER FILL] BUY @ {fill_price}  inv={state.inventory:.4f} PLB"
                    )

                # Check if existing SELL was filled
                if state.sell_order_id and paper_try_fill_sell(ob, state.sell_order_price):
                    fill_price = state.sell_order_price
                    pnl = (fill_price - state.paper_buy_price) * qty - fill_price * qty * 0.002
                    state.inventory      -= qty
                    state.total_pnl      += pnl
                    state.total_fees     += fill_price * qty * 0.002
                    state.trades         += 1
                    if pnl > 0:
                        state.wins += 1
                    state.sell_order_id = None
                    logger.info(
                        f"[PAPER FILL] SELL @ {fill_price}  "
                        f"pnl={pnl:+.6f} USDT  total={state.total_pnl:+.5f}"
                    )
                    if abs(pnl) > 0.001:
                        await tg_send(
                            f"{'✅' if pnl>0 else '❌'} MEXC {SYMBOL}\n"
                            f"Спред: {state.paper_buy_price:.4f} → {fill_price:.4f}\n"
                            f"PnL: <b>{pnl:+.6f} USDT</b>"
                        )

            else:
                # ── LIVE fill checks ──────────────────────────────
                if state.buy_order_id and await check_filled(ex, state.buy_order_id):
                    state.inventory      += qty
                    state.paper_buy_price = state.buy_order_price
                    state.buy_order_id    = None
                    logger.info(f"[LIVE FILL] BUY @ {state.buy_order_price}")

                if state.sell_order_id and await check_filled(ex, state.sell_order_id):
                    pnl = (state.sell_order_price - state.paper_buy_price) * qty
                    state.inventory      -= qty
                    state.total_pnl      += pnl
                    state.trades         += 1
                    if pnl > 0:
                        state.wins += 1
                    state.sell_order_id = None
                    logger.info(f"[LIVE FILL] SELL @ {state.sell_order_price} pnl={pnl:+.6f}")

            # ═══════════════════════════════════════════════════════
            #  Place / reprice BUY order
            # ═══════════════════════════════════════════════════════
            inv_usdt = state.inventory * bid

            # Don't add more if inventory already at max
            if inv_usdt < MAX_INVENTORY_USDT:
                need_new_buy = (
                    state.buy_order_id is None
                    or abs(state.buy_order_price - bid) / bid > 0.001   # price drifted >0.1%
                    or (now - state.buy_order_ts) > REPRICE_SEC
                )
                if need_new_buy:
                    if state.buy_order_id:
                        await cancel_order(ex, state.buy_order_id)
                        state.buy_order_id = None

                    buy_price = round_price(bid)
                    oid = await place_limit(ex, "buy", buy_price, qty)
                    if oid:
                        state.buy_order_id    = oid
                        state.buy_order_price = buy_price
                        state.buy_order_ts    = now

            # ═══════════════════════════════════════════════════════
            #  Place / reprice SELL order
            # ═══════════════════════════════════════════════════════
            need_new_sell = (
                state.sell_order_id is None
                or abs(state.sell_order_price - ask) / ask > 0.001
                or (now - state.sell_order_ts) > REPRICE_SEC
            )
            if need_new_sell:
                if state.sell_order_id:
                    await cancel_order(ex, state.sell_order_id)
                    state.sell_order_id = None

                sell_price = round_price(ask)
                sell_qty   = min(qty, max(0.0001, state.inventory))
                if sell_qty > 0.0001:
                    oid = await place_limit(ex, "sell", sell_price, sell_qty)
                    if oid:
                        state.sell_order_id    = oid
                        state.sell_order_price = sell_price
                        state.sell_order_ts    = now

            # ── Console status ────────────────────────────────────
            wr = state.wins / state.trades * 100 if state.trades else 0
            logger.info(
                f"bid={bid:.4f} ask={ask:.4f} spr={spread_pct:.3f}%  "
                f"inv={state.inventory:.4f}PLB  "
                f"pnl={state.total_pnl:+.5f}$  "
                f"trades={state.trades} wr={wr:.0f}%"
            )

            # Hourly Telegram status
            if now - last_status_ts >= 3600:
                last_status_ts = now
                await tg_send(
                    f"📊 <b>MEXC Spread Bot</b>\n"
                    f"Пара: {SYMBOL}\n"
                    f"PnL: <b>{state.total_pnl:+.5f}$</b>\n"
                    f"Сделок: {state.trades} | WR: {wr:.0f}%\n"
                    f"Inventory: {state.inventory:.4f} PLB"
                )

        except asyncio.CancelledError:
            break
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error(f"main loop: {exc}")
            await asyncio.sleep(2)

        await asyncio.sleep(0.5)

    # Cleanup
    logger.info(f"Shutdown. PnL={state.total_pnl:+.5f}$ trades={state.trades}")
    if not PAPER_MODE:
        await cancel_order(ex, state.buy_order_id)
        await cancel_order(ex, state.sell_order_id)


if __name__ == "__main__":
    asyncio.run(run())
