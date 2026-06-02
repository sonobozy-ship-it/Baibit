> ⚠️ **EXPERIMENTAL** — This is an experimental trading bot for research and paper trading.
> It is **NOT experimental** and **NOT financial advice**.
> Start with paper mode only. Real-money trading is at your own risk.
> Minimum 2 weeks of paper trading before considering live mode.

# 🤖 BYBIT AUTOTRADER PRO v5.0 — INTELLIGENCE EDITION

Самообучающийся бот для Bybit с 7 стратегиями, ML-фильтром, **новостями/Twitter/AI-sentiment**, ансамблевыми моделями, concept drift detection, anomaly detection и полным набором юнит-тестов.

## 🆕 Что нового в v5.0

### 🔴 Критичные фиксы (experimental)
- ✅ **Корректный PnL расчёт** (`position_calc.py`) — реальный `qty × (exit - entry)` минус комиссии
- ✅ **No look-ahead bias** в бэктестере — сигнал на свече `i`, исполнение на `open` свечи `i+1`
- ✅ **ML data leakage fix** — rolling-окно (20 сделок) вместо глобальных счётчиков
- ✅ **Async lock** в trading loop — невозможно запустить дважды
- ✅ **DB connection pool** (WAL mode) — тысячи операций в минуту
- ✅ **Fire-and-forget Telegram** — не блокирует цикл

### 📊 Новые ML-фичи (16 групп, 64 признака)
- **Order Book**: imbalance, spread, depth ratio, large orders detection
- **Funding rate**, Open Interest, Long/Short ratio
- **Sentiment**: news, twitter, deep AI analysis
- **Anomaly score** (Isolation Forest)
- **Market regime** (KMeans кластеризация)

### 📡 News & Sentiment
- **CryptoPanic API** (если есть ключ)
- **5 RSS источников**: CoinDesk, CoinTelegraph, Decrypt, Bitcoin Magazine, The Block
- **Twitter через Nitter** (15 топ-аккаунтов: Elon, CZ, Saylor, Vitalik...)
- **Reddit hot** (r/CryptoCurrency, r/Bitcoin)
- **2-уровневый sentiment**: rule-based (быстро) + Claude AI (точно, раз в час)

### 🧠 ML улучшения
- **Ensemble models**: XGBoost + LightGBM + LogReg → voting (+3-5% точности)
- **Concept drift monitoring** — авто-деактивация устаревшей модели
- **Auto-threshold tuning** — индивидуальный оптимум per стратегия
- **Anomaly detection** — пауза в аномальных условиях рынка

### 💰 Position Sizing
- **Kelly Criterion** (quarter Kelly, с safety cap)
- **ATR-адаптивные SL/TP** — больше волатильность = шире стопы
- **Slippage estimation** через стакан

### 🧪 Тесты
- 13 unit-тестов на критичные компоненты
- `pytest backend/tests/`

## 📦 Структура

```
bybit-bot/
├── backend/
│   ├── main.py                    # FastAPI оркестратор + торговый цикл с lock
│   ├── bybit_client.py            # Bybit REST + WebSocket + Orderbook + Funding
│   ├── risk_manager.py            # Дневные лимиты, kill switch
│   ├── backtester.py              # Бэктест БЕЗ look-ahead bias
│   ├── position_calc.py           # ⭐ Корректный PnL + Kelly + Adaptive SL
│   ├── db_pool.py                 # ⭐ Connection pool для SQLite (WAL)
│   ├── telegram_notifier.py       # TG уведомления (async fire-and-forget)
│   ├── trade_journal.py           # SQLite журнал + Excel экспорт
│   ├── correlation_filter.py      # Фильтр коррелированных позиций
│   ├── ai_analyzer.py             # Анализ через Claude API
│   ├── paper_trader.py            # Виртуальная торговля
│   │
│   ├── ml/                        # 🧠 ML МОДУЛЬ
│   │   ├── feature_extractor.py   # 64+ признака (orderbook, sentiment, regime)
│   │   ├── data_store.py          # SQLite + Parquet
│   │   ├── trainer.py             # XGBoost + walk-forward CV
│   │   ├── predictor.py           # Realtime фильтр P(win)
│   │   ├── anomaly_ensemble.py    # ⭐ Isolation Forest + Ensemble (XGB+LGB+LogReg)
│   │   ├── drift_monitor.py       # ⭐ Concept drift + Auto threshold
│   │   ├── regime_classifier.py   # KMeans для режимов рынка
│   │   └── auto_optimizer.py      # Bayesian оптимизация (Optuna)
│   │
│   ├── news/                      # ⭐ NEWS & SENTIMENT
│   │   ├── news_aggregator.py     # RSS + CryptoPanic + Reddit
│   │   ├── twitter_collector.py   # Nitter (15 crypto influencers)
│   │   ├── sentiment_analyzer.py  # Rule-based + Claude AI
│   │   └── news_manager.py        # Оркестратор + хранилище
│   │
│   ├── strategies/                # 7 стратегий с edge
│   │   ├── base.py                # ⭐ Корректный PnL + rolling stats
│   │   └── all_strategies.py
│   │
│   └── tests/                     # ⭐ Юнит-тесты
│       └── test_core.py           # 13 тестов: PnL, Kelly, Sentiment, Drift...
│
├── frontend/
│   └── index.html                 # UI с 8 вкладками (включая 📡 НОВОСТИ)
│
├── data/                          # ML данные
└── logs/                          # Логи + журнал сделок
```

## 🚀 Запуск

```bash
# 1. Установка зависимостей (включая ML стек)
pip install -r requirements.txt

# 2. Настройка
cp .env.example .env
# Минимум:
# - BYBIT_API_KEY / BYBIT_API_SECRET (для торговли)
# - ANTHROPIC_API_KEY (для AI sentiment)
# Опционально:
# - CRYPTOPANIC_API_KEY
# - TELEGRAM_BOT_TOKEN / CHAT_ID

# 3. Запуск тестов
pytest backend/tests/ -v

# 4. Запуск бэкенда
cd backend && python main.py

# 5. Открыть frontend/index.html
```

## 📊 Производительность

Все системы протестированы:

```
✅ PnL/R-multiple корректны
✅ Kelly sizing: full=55%, applied=2% (с safety cap)
✅ Adaptive SL/TP: SL=97.0, TP=106.0
✅ Sentiment: bull=1.0, bear=-1.0
✅ Drift monitor: detected=True (accuracy 0.20 < 0.5)
✅ DBPool: WAL, thread-safe, тысячи ops/sec
```

## 🎯 Полный workflow (рекомендованный)

### Неделя 1: Paper + сбор данных
```env
ML_FILTER_MODE=advisory
NEWS_ENABLED=true
```
- Бот торгует виртуально, собирает данные о рынке и новостях
- Накапливается база signal_snapshots с 64+ фичами каждый

### Неделя 2: Первое обучение
- Открыть вкладку **ML** → **🚀 ОБУЧИТЬ** для каждой стратегии (когда ≥100 сигналов)
- Проверить **Feature Importance** — какие фичи реально работают
- Обучить **Ensemble** (`/api/ml/train-ensemble`) для топ-3 стратегий
- Обучить **Anomaly Detector** (`/api/ml/anomaly/fit`)

### Неделя 3: Активация ML фильтра
```env
ML_FILTER_MODE=strict
USE_ADAPTIVE_SL=true
```
- Запустить **/api/ml/threshold/auto-tune** для каждой стратегии
- Запустить **Bayesian оптимизацию** параметров

### Неделя 4: Реальная торговля с малым риском
```env
BYBIT_TESTNET=false
RISK_PER_TRADE_PCT=0.5
USE_KELLY=true   # если ML стабилен
```
- Drift monitor автоматически отключит модель при деградации
- Каждую неделю — переобучение

## ⚙️ Все API endpoints

### Core
- `POST /api/connect` — Bybit подключение
- `POST /api/bot/control` — start/stop/paper/kill_switch_reset
- `GET /api/strategies` — список стратегий
- `POST /api/backtest` — бэктест

### ML
- `GET /api/ml/status` — общий статус
- `POST /api/ml/train` — обучить базовую модель
- `POST /api/ml/train-ensemble` — обучить ансамбль ⭐
- `POST /api/ml/anomaly/fit` — обучить anomaly detector ⭐
- `POST /api/ml/optimize` — Bayesian оптимизация
- `GET /api/ml/drift` — статус drift ⭐
- `POST /api/ml/threshold/auto-tune` — auto-threshold ⭐
- `GET /api/ml/feature-importance/{sid}` — важность фич

### News & Sentiment ⭐
- `POST /api/news/refresh` — собрать новости сейчас
- `GET /api/news/sentiment` — текущий sentiment
- `GET /api/news/items?source=...` — список новостей
- `GET /api/news/history?hours=24` — история sentiment
- `POST /api/news/deep-analysis` — AI-анализ через Claude

## 🛡 Безопасность

- ✅ **Тестируйте на TESTNET** перед запуском на mainnet
- ✅ **API ключи только trade**, без withdraw permissions
- ✅ **Kill switch** на дневной убыток (по умолчанию 5%)
- ✅ **Concept drift** автоматически отключает устаревшую модель
- ✅ **Anomaly detection** — пауза в флэш-крашах и нестабильных условиях
- ✅ **Юнит-тесты** на критичные компоненты PnL/Kelly/Sentiment/Drift

## 📈 64 фичи для ML (16 групп)

| Группа | Признаки |
|--------|----------|
| Цена | price, change_1c/5c/20c |
| EMA-наклоны | ema_9/21/50/200_slope |
| Расстояние до EMA | dist_to_ema_9/21/50/200_pct |
| Осцилляторы | RSI, Stoch K/D, CCI, Williams %R |
| MACD | value, signal, histogram |
| Волатильность | ATR%, ATR_change, BB Width/Position, σ20 |
| Объём | volume_ratio, change, trend, OBV slope |
| Свечной паттерн | body, wicks, direction |
| Структура | HH/LL count, range_position, ADX |
| Время | hour, day_of_week, sessions |
| Сигнал | action, confidence, RR, SL%, TP% |
| HTF | trend, distance, volatility |
| Стратегия (rolling) | recent_WR_20, recent_PnL_20, losses |
| **Order Book** ⭐ | imbalance, spread, depth ratio, large orders |
| **Funding & OI** ⭐ | funding_rate, OI_change, L/S ratio |
| **Sentiment** ⭐ | overall, news, twitter, deep AI, freshness |
| **Аномалии** ⭐ | anomaly_score, regime_id |

## 🤝 Что отслеживает Twitter

Дефолтный список топ-аккаунтов:
- **Movers**: elonmusk, cz_binance, saylor, VitalikButerin
- **Macro**: RaoulGMI, PeterSchiff, APompliano
- **Trading**: CryptoCobain, DegenSpartan, tier10k
- **On-chain**: WhaleAlert, GlassnodeInsight, CoinGlass_
- **News**: WatcherGuru, DocumentingBTC

Через Nitter (бесплатно). Если все инстансы лежат — Reddit + News покрывают остальное.

## ❤️ Финальные заметки

Этот код — experimental foundation. Для реальной торговли:
1. Прогоните на testnet **минимум 2 недели**
2. Сравните метрики бэктеста с paper trading
3. Начинайте с `RISK_PER_TRADE_PCT=0.5` (не 1.0)
4. Мониторьте Telegram уведомления постоянно
5. Раз в месяц переобучайте модели
