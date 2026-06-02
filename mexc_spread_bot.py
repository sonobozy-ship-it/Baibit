"""
MEXC Spot — range trader (range-grid) для PLB/USDT.
Логика: собирает историю цен за скользящее окно, определяет
нижнюю и верхнюю границу диапазона и выставляет:
  BUY  ← вблизи нижней границы (накапливаем дёшево)
  SELL ← вблизи верхней границы (продаём дорого)

Максимальная ставка — MAX_BET_USDT за одну сторону.

Запуск:
    pip install ccxt python-dotenv aiohttp
    python mexc_spread_bot.py

.env:
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
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Optional, Tuple

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
#  НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════
SYMBOL             = "PLB/USDT"
MAX_BET_USDT       = 2.0        # макс. ордер в USDT
MAX_INVENTORY_USDT = 6.0        # макс. суммарный инвентарь в USDT

# Диапазон
RANGE_WINDOW       = 120        # секунд для построения диапазона
RANGE_PERCENTILE   = 10         # % от краёв диапазона для входа
MIN_RANGE_PCT      = 0.30       # не торгуем если диапазон < 0.30%
BUY_OFFSET_PCT     = 0.05       # BUY на RANGE_PERCENTILE + небольшой буфер
SELL_OFFSET_PCT    = 0.05       # SELL на RANGE_PERCENTILE - небольшой буфер

# Логика
REPRICE_SEC        = 5.0        # перевыставить если дрейф цели >0.15% или ордер старше
DRIFT_PCT          = 0.15       # % смещение цены ордера для перевыставки
PAPER_MODE         = True       # True = симуляция

MEXC_API_KEY    = os.getenv("MEXC_API_KEY",    "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")
TG_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID",   "")
# ═══════════════════════════════════════════════════════════════════


# ── Скользящая история цен ────────────────────────────────────────

@dataclass
class PriceTick:
    ts:  float
    mid: float  # (bid+ask)/2


class PriceRange:
    """Хранит историю mid-цен за последние RANGE_WINDOW секунд."""

    def __init__(self, window_sec: float = RANGE_WINDOW) -> None:
        self._buf: Deque[PriceTick] = deque()
        self._window = window_sec

    def add(self, mid: float) -> None:
        now = time.time()
        self._buf.append(PriceTick(now, mid))
        cutoff = now - self._window
        while self._buf and self._buf[0].ts < cutoff:
            self._buf.popleft()

    def bounds(self) -> Optional[Tuple[float, float]]:
        """Возвращает (нижняя_граница, верхняя_граница) или None если мало данных."""
        if len(self._buf) < 10:
            return None
        prices = [t.mid for t in self._buf]
        lo = min(prices)
        hi = max(prices)
        if (hi - lo) / lo * 100 < MIN_RANGE_PCT:
            return None   # диапазон слишком мал
        # Отступаем от краёв на RANGE_PERCENTILE%
        pct_range = hi - lo
        buy_level  = lo + pct_range * RANGE_PERCENTILE / 100 + lo * BUY_OFFSET_PCT / 100
        sell_level = hi - pct_range * RANGE_PERCENTILE / 100 - hi * SELL_OFFSET_PCT / 100
        if sell_level <= buy_level:
            return None
        return (buy_level, sell_level)

    def range_pct(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        prices = [t.mid for t in self._buf]
        lo, hi = min(prices), max(prices)
        return (hi - lo) / lo * 100 if lo else 0.0

    def __len__(self) -> int:
        return len(self._buf)


# ── Состояние бота ────────────────────────────────────────────────

@dataclass
class BotState:
    buy_order_id:    Optional[str] = None
    buy_order_price: float         = 0.0
    buy_order_ts:    float         = 0.0

    sell_order_id:    Optional[str] = None
    sell_order_price: float         = 0.0
    sell_order_ts:    float         = 0.0

    inventory:    float = 0.0   # PLB
    avg_buy_cost: float = 0.0   # средняя цена накопленного инвентаря (USDT/PLB)

    trades:    int   = 0
    wins:      int   = 0
    total_pnl: float = 0.0
    fees:      float = 0.0
    session_start: float = field(default_factory=time.time)


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


# ── Биржевые операции ─────────────────────────────────────────────

async def fetch_orderbook(ex: ccxt.Exchange) -> Optional[dict]:
    loop = asyncio.get_event_loop()
    try:
        ob = await loop.run_in_executor(
            None, lambda: ex.fetch_order_book(SYMBOL, limit=5)
        )
        if ob["bids"] and ob["asks"]:
            bid = float(ob["bids"][0][0])
            ask = float(ob["asks"][0][0])
            return {
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
            }
    except Exception as exc:
        logger.warning(f"orderbook: {exc}")
    return None


async def cancel_order(ex: ccxt.Exchange, order_id: Optional[str]) -> None:
    if PAPER_MODE or not order_id:
        return
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: ex.cancel_order(order_id, SYMBOL))
        logger.debug(f"cancelled {order_id}")
    except Exception as exc:
        logger.debug(f"cancel {order_id}: {exc}")


async def place_limit(
    ex: ccxt.Exchange, side: str, price: float, qty: float
) -> Optional[str]:
    if PAPER_MODE:
        fake = f"PAPER-{side[0].upper()}-{time.time():.0f}"
        logger.info(f"[PAPER] {side.upper():4} {qty:.4f} @ {price:.5f}")
        return fake

    loop = asyncio.get_event_loop()
    try:
        order = await loop.run_in_executor(
            None,
            lambda: ex.create_limit_order(SYMBOL, side, qty, price),
        )
        oid = order.get("id")
        logger.info(f"[LIVE] {side.upper():4} {qty:.4f} @ {price:.5f}  id={oid}")
        return oid
    except Exception as exc:
        logger.error(f"place {side}: {exc}")
        return None


async def is_filled(ex: ccxt.Exchange, order_id: Optional[str]) -> bool:
    if PAPER_MODE or not order_id:
        return False
    loop = asyncio.get_event_loop()
    try:
        open_ids = await loop.run_in_executor(
            None, lambda: {o["id"] for o in ex.fetch_open_orders(SYMBOL)}
        )
        return order_id not in open_ids
    except Exception:
        return False


# ── Paper: симуляция исполнения ────────────────────────────────────
# Maker BUY исполняется если ask опустился ≤ цены нашего ордера
# Maker SELL исполняется если bid поднялся ≥ цены нашего ордера

def paper_fill_buy(ob: dict, order_price: float) -> bool:
    return ob["ask"] <= order_price


def paper_fill_sell(ob: dict, order_price: float) -> bool:
    return ob["bid"] >= order_price


# ── Нужен ли reprice ──────────────────────────────────────────────

def needs_reprice(order_price: float, target_price: float, placed_at: float) -> bool:
    if order_price == 0:
        return True
    drift = abs(order_price - target_price) / target_price * 100
    age   = time.time() - placed_at
    return drift > DRIFT_PCT or age > REPRICE_SEC


# ── Обновление avg_buy_cost ───────────────────────────────────────

def update_avg_cost(state: BotState, fill_qty: float, fill_price: float) -> None:
    old_val = state.avg_buy_cost * state.inventory
    new_val = old_val + fill_price * fill_qty
    state.inventory += fill_qty
    state.avg_buy_cost = new_val / state.inventory if state.inventory else fill_price


# ── Главный цикл ──────────────────────────────────────────────────

async def run() -> None:
    mode = "PAPER" if PAPER_MODE else "LIVE"
    logger.info(f"=== MEXC Range Bot [{mode}] {SYMBOL} max={MAX_BET_USDT}$ ===")
    await tg_send(
        f"📈 <b>MEXC Range Bot [{mode}]</b>\n"
        f"Пара: {SYMBOL} | Ставка: {MAX_BET_USDT}$\n"
        f"Окно диапазона: {RANGE_WINDOW}s | Мин.диапазон: {MIN_RANGE_PCT}%"
    )

    ex            = make_exchange()
    state         = BotState(session_start=time.time())
    price_range   = PriceRange()
    last_status   = time.time()

    while True:
        try:
            ob = await fetch_orderbook(ex)
            if not ob:
                await asyncio.sleep(1)
                continue

            bid, ask, mid = ob["bid"], ob["ask"], ob["mid"]
            price_range.add(mid)

            bounds = price_range.bounds()
            now    = time.time()

            # ── PAPER: проверяем исполнение ────────────────────────
            if PAPER_MODE:
                if state.buy_order_id and paper_fill_buy(ob, state.buy_order_price):
                    fill_p = state.buy_order_price
                    fill_q = round(MAX_BET_USDT / fill_p, 4)
                    fee    = fill_p * fill_q * 0.001
                    state.fees += fee
                    update_avg_cost(state, fill_q, fill_p)
                    state.buy_order_id = None
                    logger.info(
                        f"[FILL] BUY  @ {fill_p:.5f}  "
                        f"inv={state.inventory:.4f} PLB  avg={state.avg_buy_cost:.5f}"
                    )

                if state.sell_order_id and paper_fill_sell(ob, state.sell_order_price):
                    fill_p = state.sell_order_price
                    fill_q = min(round(MAX_BET_USDT / fill_p, 4), state.inventory)
                    fee    = fill_p * fill_q * 0.001
                    gross  = (fill_p - state.avg_buy_cost) * fill_q
                    pnl    = gross - fee - state.avg_buy_cost * fill_q * 0.001  # обе комиссии
                    state.inventory    = max(0.0, state.inventory - fill_q)
                    state.total_pnl   += pnl
                    state.fees        += fee
                    state.trades      += 1
                    if pnl > 0:
                        state.wins += 1
                    state.sell_order_id = None
                    logger.info(
                        f"[FILL] SELL @ {fill_p:.5f}  pnl={pnl:+.6f}$  "
                        f"total={state.total_pnl:+.5f}$"
                    )
                    if abs(pnl) > 0.0005:
                        await tg_send(
                            f"{'✅' if pnl>0 else '❌'} {SYMBOL}\n"
                            f"Buy avg: {state.avg_buy_cost:.5f} → Sell: {fill_p:.5f}\n"
                            f"PnL: <b>{pnl:+.6f} USDT</b> | Всего: {state.total_pnl:+.5f}$"
                        )

            else:
                # LIVE: опрашиваем статус ордеров
                if state.buy_order_id and await is_filled(ex, state.buy_order_id):
                    fill_p = state.buy_order_price
                    fill_q = round(MAX_BET_USDT / fill_p, 4)
                    update_avg_cost(state, fill_q, fill_p)
                    state.buy_order_id = None
                    logger.info(f"[FILL] BUY  @ {fill_p:.5f}  inv={state.inventory:.4f}")

                if state.sell_order_id and await is_filled(ex, state.sell_order_id):
                    fill_p = state.sell_order_price
                    fill_q = min(round(MAX_BET_USDT / fill_p, 4), state.inventory)
                    pnl    = (fill_p - state.avg_buy_cost) * fill_q
                    state.inventory  = max(0.0, state.inventory - fill_q)
                    state.total_pnl += pnl
                    state.trades    += 1
                    if pnl > 0:
                        state.wins += 1
                    state.sell_order_id = None
                    logger.info(f"[FILL] SELL @ {fill_p:.5f}  pnl={pnl:+.6f}$")

            # ── Размещение ордеров (только если диапазон известен) ─
            if bounds is None:
                rng = price_range.range_pct()
                logger.debug(
                    f"range={rng:.3f}% ticks={len(price_range)} — нет сигнала"
                )
                await asyncio.sleep(0.5)
                continue

            buy_target, sell_target = bounds
            inv_usdt = state.inventory * bid
            qty_buy  = round(MAX_BET_USDT / buy_target,  4)
            qty_sell = round(MAX_BET_USDT / sell_target, 4)

            # BUY: только если инвентарь не переполнен
            if inv_usdt < MAX_INVENTORY_USDT:
                if needs_reprice(state.buy_order_price, buy_target, state.buy_order_ts):
                    await cancel_order(ex, state.buy_order_id)
                    state.buy_order_id = None
                    oid = await place_limit(ex, "buy", round(buy_target, 5), qty_buy)
                    if oid:
                        state.buy_order_id    = oid
                        state.buy_order_price = buy_target
                        state.buy_order_ts    = now
            else:
                # Инвентарь полон — отменяем покупку
                if state.buy_order_id:
                    await cancel_order(ex, state.buy_order_id)
                    state.buy_order_id = None

            # SELL: только если есть что продавать
            if state.inventory >= qty_sell * 0.5:
                sell_qty = min(qty_sell, state.inventory)
                if needs_reprice(state.sell_order_price, sell_target, state.sell_order_ts):
                    await cancel_order(ex, state.sell_order_id)
                    state.sell_order_id = None
                    oid = await place_limit(ex, "sell", round(sell_target, 5), sell_qty)
                    if oid:
                        state.sell_order_id    = oid
                        state.sell_order_price = sell_target
                        state.sell_order_ts    = now
            else:
                if state.sell_order_id:
                    await cancel_order(ex, state.sell_order_id)
                    state.sell_order_id = None

            # ── Лог в консоль ──────────────────────────────────────
            wr  = state.wins / state.trades * 100 if state.trades else 0
            rng = price_range.range_pct()
            logger.info(
                f"mid={mid:.5f}  range={rng:.3f}%  "
                f"B={buy_target:.5f} S={sell_target:.5f}  "
                f"inv={inv_usdt:.2f}$  pnl={state.total_pnl:+.5f}$  "
                f"tr={state.trades} wr={wr:.0f}%"
            )

            # Ежечасовой статус в Telegram
            if now - last_status >= 3600:
                last_status = now
                await tg_send(
                    f"📊 <b>MEXC Range Bot</b>\n"
                    f"Пара: {SYMBOL}\n"
                    f"Диапазон: {rng:.3f}% | BUY: {buy_target:.5f} | SELL: {sell_target:.5f}\n"
                    f"Инвентарь: {state.inventory:.4f} PLB ({inv_usdt:.2f}$)\n"
                    f"PnL: <b>{state.total_pnl:+.5f}$</b> | Fees: {state.fees:.5f}$\n"
                    f"Сделок: {state.trades} | WR: {wr:.0f}%"
                )

        except asyncio.CancelledError:
            break
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error(f"loop: {exc}", exc_info=True)
            await asyncio.sleep(2)

        await asyncio.sleep(0.5)

    logger.info(f"Shutdown. PnL={state.total_pnl:+.5f}$ trades={state.trades}")
    await cancel_order(ex, state.buy_order_id)
    await cancel_order(ex, state.sell_order_id)


if __name__ == "__main__":
    asyncio.run(run())
