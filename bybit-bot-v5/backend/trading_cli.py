"""
Trading CLI — интерактивный торговый инструмент для Claude.

Использование:
  python trading_cli.py balance               — баланс счёта
  python trading_cli.py positions             — открытые позиции
  python trading_cli.py price BTCUSDT         — текущая цена
  python trading_cli.py analyze BTCUSDT [tf]  — анализ сигналов (все стратегии)
  python trading_cli.py open BTCUSDT Buy 0.001 [sl] [tp] [leverage]  — открыть сделку
  python trading_cli.py close BTCUSDT         — закрыть позицию
  python trading_cli.py pnl                   — P&L всех открытых позиций
  python trading_cli.py history [limit]       — история сделок

Переменные окружения: BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET
"""
import sys
import os
import json
from pathlib import Path

# Загрузка .env из родительской директории или текущей
for env_path in [Path(__file__).parent / ".env",
                 Path(__file__).parent.parent / ".env",
                 Path(".env")]:
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        break

sys.path.insert(0, str(Path(__file__).parent))

from bybit_client import BybitClient


def get_client() -> BybitClient:
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    testnet = os.getenv("BYBIT_TESTNET", "false").lower() in ("true", "1", "yes")
    if not api_key or not api_secret:
        print("ERROR: BYBIT_API_KEY / BYBIT_API_SECRET не заданы в .env")
        sys.exit(1)
    return BybitClient(api_key, api_secret, testnet=testnet)


def cmd_balance(client: BybitClient):
    """Показать баланс USDT + краткий обзор аккаунта."""
    usdt = client.get_balance("USDT")
    btc  = client.get_balance("BTC")
    print(f"\n{'='*45}")
    print(f"  БАЛАНС СЧЁТА BYBIT {'(TESTNET)' if client.testnet else '(MAINNET)'}")
    print(f"{'='*45}")
    print(f"  USDT: {usdt:>14.2f}")
    if btc > 0:
        print(f"  BTC:  {btc:>14.8f}")
    print(f"{'='*45}\n")


def cmd_positions(client: BybitClient):
    """Показать открытые позиции."""
    positions = client.get_positions()
    if not positions:
        print("\n  Нет открытых позиций.\n")
        return
    print(f"\n{'='*65}")
    print(f"  ОТКРЫТЫЕ ПОЗИЦИИ ({len(positions)})")
    print(f"{'='*65}")
    for p in positions:
        unrealised = float(p.get("unrealisedPnl", 0))
        pct = float(p.get("unrealisedPnl", 0)) / float(p.get("positionValue", 1)) * 100 if float(p.get("positionValue", 1)) else 0
        print(f"  {p['symbol']:<12} {p['side']:<5}  qty={p['size']:<10}"
              f"  entry={float(p['avgPrice']):<10.4f}"
              f"  PnL={unrealised:>+8.2f} USDT  ({pct:+.2f}%)")
    print(f"{'='*65}\n")


def cmd_price(client: BybitClient, symbol: str):
    """Показать текущую цену."""
    t = client.get_ticker(symbol)
    if not t:
        print(f"ERROR: не удалось получить цену для {symbol}")
        sys.exit(1)
    print(f"\n  {symbol}: {t['price']:.4f} USDT  "
          f"({t['change_24h']:+.2f}% 24h)  "
          f"Vol: {t['volume_24h']:,.0f}  "
          f"FR: {t['funding_rate']*100:.4f}%\n")


def cmd_analyze(client: BybitClient, symbol: str, timeframe: str = "15"):
    """Запустить все стратегии на символ и вывести сигналы."""
    import pandas_ta  # noqa — shim из backend/pandas_ta.py
    from strategies.all_strategies import (
        EMACrossoverStrategy, BollingerStrategy, RSIDivergenceStrategy,
        BreakoutStrategy, ScalperGridStrategy, TrendFollowerStrategy,
        MultiConfirmStrategy,
    )
    print(f"\n  Загружаем свечи {symbol} {timeframe}m...")
    df = client.get_klines(symbol, timeframe, limit=300)
    if df.empty:
        print("  ERROR: нет данных")
        sys.exit(1)
    current_price = df["close"].iloc[-1]
    print(f"  Свечей: {len(df)}  |  Последняя цена: {current_price:.4f}\n")

    strategies = [
        EMACrossoverStrategy(symbol=symbol),
        BollingerStrategy(symbol=symbol),
        RSIDivergenceStrategy(symbol=symbol),
        BreakoutStrategy(symbol=symbol),
        ScalperGridStrategy(symbol=symbol),
        TrendFollowerStrategy(symbol=symbol),
        MultiConfirmStrategy(symbol=symbol),
    ]

    signals = []
    for strat in strategies:
        try:
            sig = strat.analyze(df.copy())
            if sig:
                signals.append((strat.NAME, strat.ID, sig))
        except Exception as e:
            print(f"  [{strat.ID}] {strat.NAME}: ОШИБКА — {e}")

    print(f"{'='*65}")
    print(f"  АНАЛИЗ: {symbol} {timeframe}m  (цена: {current_price:.4f})")
    print(f"{'='*65}")
    if not signals:
        print("  Нет сигналов ни по одной стратегии.\n")
    else:
        for name, sid, sig in signals:
            action_icon = "🟢 LONG" if sig.action == "buy" else "🔴 SHORT"
            print(f"\n  [{sid}] {name}")
            print(f"    Сигнал:   {action_icon}")
            print(f"    Вход:     {sig.entry_price:.4f}")
            print(f"    StopLoss: {sig.stop_loss:.4f}  ({abs(sig.stop_loss/sig.entry_price-1)*100:.2f}%)")
            print(f"    TakeProf: {sig.take_profit:.4f}  ({abs(sig.take_profit/sig.entry_price-1)*100:.2f}%)")
            print(f"    Фильтры:  {json.dumps(sig.filters_passed, ensure_ascii=False)}")
            if hasattr(sig, "ml_confidence") and sig.ml_confidence:
                print(f"    ML conf:  {sig.ml_confidence:.2f}")
    print()


def cmd_open(client: BybitClient, symbol: str, side: str,
             qty: float, sl: float = 0.0, tp: float = 0.0, leverage: int = 1):
    """Открыть сделку."""
    side = side.capitalize()  # Buy / Sell
    if side not in ("Buy", "Sell"):
        print("ERROR: side должен быть Buy или Sell")
        sys.exit(1)

    print(f"\n  Открываем: {symbol} {side}  qty={qty}  lev={leverage}x"
          f"{'  SL='+str(sl) if sl else ''}{'  TP='+str(tp) if tp else ''}")

    confirm = input("  Подтвердить? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("  Отменено.\n")
        return

    result = client.place_order(
        symbol=symbol,
        side=side,
        qty=qty,
        stop_loss=sl if sl else None,
        take_profit=tp if tp else None,
        leverage=leverage,
    )
    if result.get("success"):
        print(f"\n  ✅ Ордер исполнен: orderId={result['data'].get('orderId')}\n")
    else:
        print(f"\n  ❌ Ошибка: {result.get('error')}\n")


def cmd_close(client: BybitClient, symbol: str):
    """Закрыть позицию по рынку."""
    positions = client.get_positions(symbol)
    if not positions:
        print(f"  Нет открытой позиции по {symbol}\n")
        return
    p = positions[0]
    print(f"\n  Закрыть {symbol} {p['side']} qty={p['size']}?")
    confirm = input("  Подтвердить? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("  Отменено.\n")
        return
    result = client.close_position(symbol)
    if result.get("success"):
        print(f"  ✅ Позиция закрыта\n")
    else:
        print(f"  ❌ Ошибка: {result.get('error')}\n")


def cmd_pnl(client: BybitClient):
    """Суммарный P&L открытых позиций."""
    positions = client.get_positions()
    if not positions:
        print("\n  Нет открытых позиций.\n")
        return
    total_pnl = sum(float(p.get("unrealisedPnl", 0)) for p in positions)
    total_value = sum(float(p.get("positionValue", 0)) for p in positions)
    print(f"\n  Открытых позиций:  {len(positions)}")
    print(f"  Суммарный P&L:     {total_pnl:+.2f} USDT")
    print(f"  Суммарная ценность: {total_value:.2f} USDT\n")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()

    if cmd in ("balance", "bal", "b"):
        cmd_balance(get_client())

    elif cmd in ("positions", "pos", "p"):
        cmd_positions(get_client())

    elif cmd == "price":
        symbol = args[1].upper() if len(args) > 1 else "BTCUSDT"
        cmd_price(get_client(), symbol)

    elif cmd in ("analyze", "analyse", "signal", "a"):
        symbol = args[1].upper() if len(args) > 1 else "BTCUSDT"
        tf = args[2] if len(args) > 2 else "15"
        cmd_analyze(get_client(), symbol, tf)

    elif cmd in ("open", "trade", "buy", "sell"):
        if cmd in ("buy", "sell"):
            symbol = args[1].upper() if len(args) > 1 else "BTCUSDT"
            side = cmd.capitalize()
            qty = float(args[2]) if len(args) > 2 else 0.001
            sl = float(args[3]) if len(args) > 3 else 0.0
            tp = float(args[4]) if len(args) > 4 else 0.0
            leverage = int(args[5]) if len(args) > 5 else 1
        else:
            symbol = args[1].upper() if len(args) > 1 else "BTCUSDT"
            side = args[2] if len(args) > 2 else "Buy"
            qty = float(args[3]) if len(args) > 3 else 0.001
            sl = float(args[4]) if len(args) > 4 else 0.0
            tp = float(args[5]) if len(args) > 5 else 0.0
            leverage = int(args[6]) if len(args) > 6 else 1
        cmd_open(get_client(), symbol, side, qty, sl, tp, leverage)

    elif cmd in ("close", "c"):
        symbol = args[1].upper() if len(args) > 1 else "BTCUSDT"
        cmd_close(get_client(), symbol)

    elif cmd in ("pnl",):
        cmd_pnl(get_client())

    elif cmd in ("history", "hist", "h"):
        limit = int(args[1]) if len(args) > 1 else 20
        print(f"  История: используйте GET /api/trades через браузер или FastAPI /docs")

    else:
        print(f"  Неизвестная команда: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
