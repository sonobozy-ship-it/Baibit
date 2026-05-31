"""
crypto_arbitrage_scanner.py — межбиржевой арбитражный сканер.
pip install ccxt

ВАЖНО: этот сканер показывает ценовые расхождения между биржами,
НЕ учитывая комиссии за вывод средств и время перевода.
Реальный арбитраж между биржами требует заранее размещённого
капитала на каждой из них.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import ccxt

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
]

MIN_NET_PROFIT_PERCENT = 0.25  # минимальная чистая прибыль после комиссий
TRADE_FEE_PERCENT      = 0.10  # % taker fee на вход и на выход (×2 суммарно)
CHECK_DELAY            = 3     # секунд между проходами

exchanges = {
    "binance": ccxt.binance({"enableRateLimit": True}),
    "bybit":   ccxt.bybit(  {"enableRateLimit": True}),
    "okx":     ccxt.okx(    {"enableRateLimit": True}),
    "mexc":    ccxt.mexc(   {"enableRateLimit": True}),
}


def _fetch_one(args: tuple):
    """Получить bid/ask с одной биржи (вызывается из ThreadPoolExecutor)."""
    name, exchange, symbol = args
    try:
        ticker = exchange.fetch_ticker(symbol)
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if bid is None or ask is None:
            return None
        return name, {"bid": float(bid), "ask": float(ask)}
    except Exception:
        return None


def scan():
    found_any = False
    for symbol in SYMBOLS:
        # Параллельный запрос со всех бирж — в 4× быстрее последовательного
        tasks = [(name, ex, symbol) for name, ex in exchanges.items()]
        prices: dict = {}
        with ThreadPoolExecutor(max_workers=len(exchanges)) as pool:
            for result in pool.map(_fetch_one, tasks):
                if result:
                    name, data = result
                    prices[name] = data

        if len(prices) < 2:
            continue

        best_buy  = min(prices.items(), key=lambda x: x[1]["ask"])
        best_sell = max(prices.items(), key=lambda x: x[1]["bid"])

        buy_exchange  = best_buy[0]
        sell_exchange = best_sell[0]

        # Нет смысла покупать и продавать на одной и той же бирже
        if buy_exchange == sell_exchange:
            continue

        buy_price  = best_buy[1]["ask"]
        sell_price = best_sell[1]["bid"]

        # sell должна быть выше buy — иначе арбитража нет
        if sell_price <= buy_price:
            continue

        gross_profit_pct = (sell_price - buy_price) / buy_price * 100
        total_fees_pct   = TRADE_FEE_PERCENT * 2   # вход + выход
        net_profit_pct   = gross_profit_pct - total_fees_pct

        if net_profit_pct >= MIN_NET_PROFIT_PERCENT:
            found_any = True
            print("=" * 60)
            print(f"Монета:         {symbol}")
            print(f"Купить:         {buy_exchange:<8}  @ {buy_price}")
            print(f"Продать:        {sell_exchange:<8}  @ {sell_price}")
            print(f"Грязный спред:  {gross_profit_pct:.3f}%")
            print(f"Комиссии:       {total_fees_pct:.3f}%")
            print(f"Чистая прибыль: {net_profit_pct:.3f}%")

    if not found_any:
        print("  нет возможностей выше порога")


def main():
    print("Запуск арбитражного сканера")
    print(f"Биржи:     {', '.join(exchanges)}")
    print(f"Монеты:    {', '.join(SYMBOLS)}")
    print(f"Мин. нет:  {MIN_NET_PROFIT_PERCENT}%  (после {TRADE_FEE_PERCENT * 2:.2f}% комиссий)")
    print("=" * 60)

    while True:
        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}]")
        try:
            scan()
        except KeyboardInterrupt:
            print("\nОстановлено.")
            break
        except Exception as exc:
            print(f"  Ошибка: {exc}")
        time.sleep(CHECK_DELAY)


if __name__ == "__main__":
    main()
