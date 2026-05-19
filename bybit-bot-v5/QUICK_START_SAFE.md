# Быстрый старт — безопасный режим

## Шаг 1: Paper Mode (рекомендуется для начала)

```bash
cp .env.example .env
# .env уже содержит TRADING_MODE=PAPER, BYBIT_TESTNET=true
# API ключи не обязательны для paper mode
```

Запусти бота:
```bash
cd bybit-bot-v5/backend
python main.py
```

Открой Telegram и напиши `/start` — бот начнёт бумажную торговлю.

## Шаг 2: Testnet (после 2+ недель paper)

1. Зарегистрируйся на [testnet.bybit.com](https://testnet.bybit.com)
2. Получи API ключи (testnet)
3. В `.env`:
```
TRADING_MODE=TESTNET
BYBIT_TESTNET=true
BYBIT_API_KEY=<testnet_key>
BYBIT_API_SECRET=<testnet_secret>
```

## Шаг 3: Live (только после стабильной прибыли на testnet)

**Минимальные требования:**
- [ ] 2+ недели paper mode с положительным результатом
- [ ] 1+ неделя testnet с подтверждённой прибылью  
- [ ] Понимание рисков потери всего депозита
- [ ] Настроенный kill switch и лимиты

В `.env`:
```
TRADING_MODE=LIVE
BYBIT_TESTNET=false
LIVE_TRADING_CONFIRM=true
I_UNDERSTAND_REAL_MONEY_RISK=true
RISK_PER_TRADE_PCT=0.5
DAILY_MAX_LOSS_PCT=3.0
MAX_STRATEGY_DAILY_LOSSES=3
```

## Kill Switch

Если бот торгует слишком агрессивно:
- Telegram: нажми **Стоп** в главном меню
- Или установи `AUTO_START=false` и перезапусти бота

## Параметры риска по умолчанию

| Параметр | Paper | Testnet | Live (рекомендуется) |
|----------|-------|---------|---------------------|
| RISK_PER_TRADE_PCT | 1.0% | 0.5% | 0.5% |
| DAILY_MAX_LOSS_PCT | 5.0% | 3.0% | 3.0% |
| MAX_STRATEGY_DAILY_LOSSES | 5 | 3 | 3 |
| MAX_LEVERAGE | 5 | 3 | 3 |
