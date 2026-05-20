"""
AI-анализатор торгового бота.

Три функции:
  1. analyze_strategy_performance — почему стратегия теряет деньги
  2. analyze_market               — анализ рынка в реальном времени (новая)
  3. quick_signal_check           — оценка сигнала перед входом (0-10)

Подключён к мультипровайдеру (Anthropic / OpenAI / Ollama).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Импорт провайдеров из оркестратора ───────────────────────────────────────
try:
    from claude_orchestrator import (
        BaseProvider,
        create_provider,
        AnthropicProvider,
        ANTHROPIC_AVAILABLE,
        OPENAI_AVAILABLE,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    OPENAI_AVAILABLE = False

# ── Системный промпт (кешируется) ─────────────────────────────────────────────
_SYSTEM_PROMPT = """
Ты — опытный квант-трейдер и аналитик крипторынка, помогающий торговому боту Baibit.
Отвечай только на русском, конкретно и по делу. Без воды и лирических отступлений.
""".strip()


class AIAnalyzer:
    """
    AI-анализатор стратегий и рынка.

    Использование:
        ai = AIAnalyzer()                    # автодетект провайдера из .env
        ai = AIAnalyzer(anthropic_key="...") # явный ключ
    """

    # Модели для разных задач
    _MODEL_ANALYSIS = "claude-sonnet-4-6"   # анализ стратегий — качество важнее скорости
    _MODEL_MARKET   = "claude-sonnet-4-6"   # анализ рынка
    _MODEL_SIGNAL   = "claude-haiku-4-5"    # проверка сигнала — нужна скорость

    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        openai_key:    Optional[str] = None,
        ollama_url:    Optional[str] = None,
        provider:      Optional[Any] = None,
    ):
        # Поддержка старого API (api_key= positional)
        if anthropic_key is None:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if openai_key is None:
            openai_key = os.getenv("OPENAI_API_KEY")
        if ollama_url is None:
            ollama_url = os.getenv("OLLAMA_BASE_URL")

        if provider is not None:
            self._provider: Optional[BaseProvider] = provider
        elif ORCHESTRATOR_AVAILABLE:
            self._provider = create_provider(
                anthropic_key=anthropic_key,
                openai_key=openai_key,
                ollama_url=ollama_url,
            )
        else:
            self._provider = None

        self.enabled = self._provider is not None
        if self.enabled:
            logger.info(f"[AIAnalyzer] Провайдер: {self._provider.name} ({self._provider.model})")
        else:
            logger.warning("[AIAnalyzer] Недоступен — задайте ANTHROPIC_API_KEY / OPENAI_API_KEY")

    # ── Внутренний вызов LLM ─────────────────────────────────────────────────

    async def _ask(self, prompt: str, model_hint: str = "analysis") -> Optional[str]:
        """Асинхронный вызов провайдера."""
        if not self.enabled:
            return None
        if not prompt or not prompt.strip():
            logger.warning("[AIAnalyzer] Пустой промпт — пропускаем вызов LLM")
            return None
        try:
            # Для Anthropic меняем модель по задаче; для остальных — берём что есть
            if ORCHESTRATOR_AVAILABLE and ANTHROPIC_AVAILABLE and isinstance(self._provider, AnthropicProvider):
                model = {
                    "analysis": self._MODEL_ANALYSIS,
                    "market":   self._MODEL_MARKET,
                    "signal":   self._MODEL_SIGNAL,
                }.get(model_hint, self._MODEL_ANALYSIS)
                from claude_orchestrator import AnthropicProvider as AP
                tmp = AP(api_key=self._provider._client.api_key, model=model)
                return await tmp.complete(
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500 if model_hint == "analysis" else 600,
                )
            else:
                return await self._provider.complete(
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500 if model_hint == "analysis" else 600,
                )
        except Exception as e:
            logger.error(f"[AIAnalyzer] Ошибка LLM: {e}")
            return None

    def _ask_sync(self, prompt: str, model_hint: str = "analysis") -> Optional[str]:
        """Синхронная обёртка для вызова из не-async контекста.

        Работает корректно как из sync, так и из async (FastAPI-роут/trading loop) контекста:
        - Если event loop уже запущен (async контекст) — выполняет через ThreadPoolExecutor,
          чтобы не вызвать 'This event loop is already running'.
        - Если нет запущенного loop (чистый sync) — использует asyncio.run().
        """
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Вызов из async-контекста: запускаем _ask в отдельном потоке
            # с собственным event loop, чтобы не блокировать текущий.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._ask(prompt, model_hint))
                return future.result()
        else:
            return asyncio.run(self._ask(prompt, model_hint))

    # ── 1. Анализ производительности стратегии ────────────────────────────────

    def analyze_strategy_performance(self, strategy_data: Dict) -> Dict:
        """
        Анализ почему стратегия теряет деньги.

        strategy_data:
        {
            "id": "S3", "name": "RSI Divergence", "symbol": "BTCUSDT",
            "trades": [{"pnl_usd": -5.2, "side": "BUY", "exit_reason": "SL", ...}],
            "stats": {"win_rate": 35, "profit_factor": 0.7, "total_pnl": -45, ...}
        }
        """
        if not self.enabled:
            return {"available": False, "message": "AI недоступен — задайте API ключ"}

        try:
            prompt = self._build_strategy_prompt(strategy_data)
            text = self._ask_sync(prompt, "analysis")
            if not text:
                return {"available": False, "message": "Нет ответа от AI"}
            return {
                "available":   True,
                "analysis":    text,
                "strategy_id": strategy_data.get("id"),
            }
        except Exception as e:
            logger.error(f"[AIAnalyzer] analyze_strategy_performance: {e}")
            return {"available": False, "message": str(e)}

    def _build_strategy_prompt(self, data: Dict) -> str:
        stats  = data.get("stats", {})
        trades = data.get("trades", [])[-15:]

        trades_str = "\n".join(
            f"  {t.get('timestamp','')}: {t.get('side')} → "
            f"{t.get('pnl_usd', 0):+.2f} USDT ({t.get('exit_reason','?')})"
            for t in trades
        ) or "  Нет данных"

        n_trades = stats.get('trades', 0)
        wr       = stats.get('win_rate', 0)
        pnl      = stats.get('total_pnl', 0)
        pf       = stats.get('profit_factor', 0)

        # Предупреждение о малом числе сделок
        sample_warning = ""
        if n_trades < 10:
            sample_warning = f"\n⚠️ Малая выборка ({n_trades} сделок) — выводы предварительные.\n"

        return f"""Проанализируй производительность торговой стратегии крипто-бота.

ВАЖНО: Опирайся ТОЛЬКО на данные ниже. Не выдумывай факты которых нет в статистике.

**Стратегия:** {data.get('name')} ({data.get('id')}) | Символ: {data.get('symbol','?')}
{sample_warning}
**Факты из бота (реальные цифры):**
- Всего сделок: {n_trades}
- Win Rate: {wr:.1f}%
- Общий PnL: {pnl:+.2f} USDT
- Profit Factor: {pf:.2f}
- Средняя прибыль: {stats.get('avg_win', 0):+.2f} USDT
- Средний убыток: {stats.get('avg_loss', 0):+.2f} USDT
- Max Drawdown: {stats.get('max_drawdown_pct', 0):.1f}%
- Серия убытков подряд: {stats.get('longest_loss_streak', 0)}

**Последние {len(trades)} сделок:**
{trades_str}

Дай ответ в трёх частях:

1. **Диагноз** (2-3 предложения): что именно не так ИСХОДЯ ИЗ ДАННЫХ ВЫШЕ.
2. **Рекомендации** (3-5 конкретных пунктов): конкретные параметры SL/TP/фильтры.
3. **Вердикт** (одно из):
   - 🟢 EXCELLENT — оставить как есть (WR>60%, PF>1.5)
   - 🟡 GOOD — мелкие правки (WR>50%, PF>1.0)
   - 🟠 NEEDS_WORK — серьёзные правки (WR<50% или PF<1.0)
   - 🔴 DISABLE — отключить (PF<0.5 или серия 5+ убытков и PnL < -2$)

Не пиши "нет сделок" если сделок больше 0."""

    # ── 2. Анализ рынка в реальном времени ───────────────────────────────────

    def analyze_market(
        self,
        candles: Dict[str, pd.DataFrame],   # symbol -> DataFrame OHLCV
        current_prices: Dict[str, float],
        portfolio_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Анализ рынка в реальном времени перед торговым циклом.

        Возвращает:
        {
            "regime":      "trending_up" | "trending_down" | "flat" | "volatile",
            "risk_level":  "low" | "medium" | "high",
            "top_symbols": ["BTCUSDT", "ETHUSDT"],   # лучшие для торговли сейчас
            "avoid":       ["SOLUSDT"],               # избегать
            "summary":     "2-3 предложения о рынке",
            "trade_now":   True/False,
        }
        """
        if not self.enabled:
            return {"available": False, "regime": "unknown", "trade_now": True}

        try:
            prompt = self._build_market_prompt(candles, current_prices, portfolio_context)
            text = self._ask_sync(prompt, "market")
            if not text:
                return {"available": False, "regime": "unknown", "trade_now": True}

            # Парсим JSON из ответа
            parsed = self._extract_json(text)
            if parsed:
                parsed["available"] = True
                return parsed

            return {"available": True, "raw": text, "regime": "unknown", "trade_now": True}

        except Exception as e:
            logger.error(f"[AIAnalyzer] analyze_market: {e}")
            return {"available": False, "regime": "unknown", "trade_now": True}

    def _build_market_prompt(
        self,
        candles: Dict[str, pd.DataFrame],
        prices: Dict[str, float],
        ctx: Optional[Dict],
    ) -> str:
        import pandas_ta as ta

        symbols_info = []
        for sym, df in list(candles.items())[:8]:   # не больше 8 символов
            if len(df) < 20:
                continue
            try:
                close = df["close"]
                rsi   = ta.rsi(close, 14).iloc[-1]
                ema20 = ta.ema(close, 20).iloc[-1]
                atr   = ta.atr(df["high"], df["low"], close, 14).iloc[-1]
                vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
                chg_pct = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

                symbols_info.append(
                    f"  {sym}: price={prices.get(sym, close.iloc[-1]):.4f} "
                    f"chg5={chg_pct:+.2f}% RSI={rsi:.0f} "
                    f"ATR={atr/close.iloc[-1]*100:.2f}% vol×{vol_ratio:.1f} "
                    f"{'▲' if close.iloc[-1] > ema20 else '▼'}EMA20"
                )
            except Exception:
                continue

        ctx_str = ""
        if ctx:
            ctx_str = f"""
Контекст портфеля:
- Баланс: {ctx.get('balance', '?')} USDT
- Открытых позиций: {ctx.get('open_positions', 0)}
- Дневной PnL: {ctx.get('daily_pnl', 0):+.2f} USDT
- Режим бота: {'boost' if ctx.get('boost_active') else 'normal'}"""

        return f"""Проанализируй текущее состояние рынка для торгового бота.

**Данные по символам (последние свечи):**
{chr(10).join(symbols_info) or '  Нет данных'}
{ctx_str}

Ответь СТРОГО JSON (без markdown, без пояснений):
{{
  "regime": "trending_up|trending_down|flat|volatile",
  "risk_level": "low|medium|high",
  "top_symbols": ["SYMBOL1", "SYMBOL2"],
  "avoid": ["SYMBOL3"],
  "summary": "2-3 предложения о рынке",
  "trade_now": true/false,
  "reasoning": "одно предложение почему trade_now"
}}"""

    # ── 3. Быстрая проверка сигнала (синхронная, устаревшая) ─────────────────

    def quick_signal_check(
        self,
        signal_data: Dict,
        market_context: Optional[Dict] = None,
    ) -> Dict:
        """Синхронная обёртка — использует _ask_sync."""
        if not self.enabled:
            return {"score": 5, "comment": "AI недоступен", "approved": True}
        try:
            prompt = self._build_full_signal_prompt(signal_data, None, market_context)
            text = self._ask_sync(prompt, "signal")
            if not text:
                return {"score": 5, "comment": "Нет ответа AI", "approved": True}
            parsed = self._extract_json(text)
            if parsed and "score" in parsed:
                parsed.setdefault("approved", parsed["score"] >= 5)
                return parsed
            return {"score": 5, "comment": text[:200], "approved": True}
        except Exception:
            return {"score": 5, "comment": "AI ошибка", "approved": True}

    # ── 4. Полноценный async-анализ сигнала перед входом ──────────────────────

    async def analyze_signal_async(
        self,
        signal_data: Dict,
        df: Optional[pd.DataFrame] = None,
        sentiment: Optional[Dict] = None,
    ) -> Dict:
        """
        Async-анализ торгового сигнала с данными свечей.
        Возвращает {"score": 0-10, "comment": "...", "approved": bool, "reasoning": "..."}
        """
        if not self.enabled:
            return {"score": 5, "comment": "AI недоступен", "approved": True, "reasoning": ""}

        try:
            prompt = self._build_full_signal_prompt(signal_data, df, sentiment)
            text = await self._ask(prompt, "signal")
            if not text:
                return {"score": 5, "comment": "Нет ответа AI", "approved": True, "reasoning": ""}

            parsed = self._extract_json(text)
            if parsed and "score" in parsed:
                parsed.setdefault("approved", parsed["score"] >= 5)
                parsed.setdefault("reasoning", parsed.get("comment", ""))
                return parsed

            return {"score": 5, "comment": text[:200], "approved": True, "reasoning": text[:200]}

        except Exception as e:
            logger.error(f"[AIAnalyzer] analyze_signal_async: {e}")
            return {"score": 5, "comment": "AI ошибка", "approved": True, "reasoning": ""}

    def _build_full_signal_prompt(
        self,
        signal_data: Dict,
        df: Optional[pd.DataFrame],
        sentiment: Optional[Dict],
    ) -> str:
        entry = signal_data.get("entry_price", 0)
        sl    = signal_data.get("stop_loss",   0)
        tp    = signal_data.get("take_profit", 0)
        action = signal_data.get("action", "?")

        # R:R ratio
        try:
            risk   = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
        except Exception:
            rr = 0

        # Индикаторы из свечей
        indicators_str = ""
        if df is not None and len(df) >= 20:
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(__file__))
                import pandas_ta as ta
                close = df["close"].astype(float)
                high  = df["high"].astype(float)
                low   = df["low"].astype(float)
                vol   = df["volume"].astype(float)

                rsi   = ta.rsi(close, 14).iloc[-1]
                ema20 = ta.ema(close, 20).iloc[-1]
                ema50 = ta.ema(close, 50).iloc[-1] if len(df) >= 50 else ema20
                atr   = ta.atr(high, low, close, 14).iloc[-1]
                vol_avg = vol.rolling(20).mean().iloc[-1]
                vol_ratio = vol.iloc[-1] / vol_avg if vol_avg > 0 else 1.0

                # Направление тренда
                last_5 = close.iloc[-5:]
                trend = "восходящий" if last_5.iloc[-1] > last_5.iloc[0] else "нисходящий"
                price_vs_ema = "выше EMA20" if close.iloc[-1] > ema20 else "ниже EMA20"

                indicators_str = (
                    f"\n📊 Технические индикаторы:\n"
                    f"  RSI(14): {rsi:.1f} "
                    f"({'перекуплен' if rsi > 70 else 'перепродан' if rsi < 30 else 'нейтрал'})\n"
                    f"  Тренд (5 свечей): {trend}\n"
                    f"  Цена {price_vs_ema} | EMA50: {'выше' if close.iloc[-1] > ema50 else 'ниже'}\n"
                    f"  ATR: {atr/close.iloc[-1]*100:.2f}% от цены\n"
                    f"  Объём: {'высокий ×'+str(round(vol_ratio,1)) if vol_ratio > 1.5 else 'нормальный'}"
                )
            except Exception as _e:
                indicators_str = f"  (индикаторы недоступны: {_e})"

        # Сентимент
        sent_str = ""
        if sentiment:
            score = sentiment.get("sentiment_score", 0)
            news  = sentiment.get("news_count_24h", 0)
            sent_str = f"\n📰 Сентимент: {score:+.2f} | Новостей 24ч: {news}"

        return f"""Проанализируй торговый сигнал и реши — входить или нет.

🔔 Сигнал:
  Стратегия: {signal_data.get('strategy_name', '?')} ({signal_data.get('strategy_id', '?')})
  Символ: {signal_data.get('symbol', '?')} | Направление: {action}
  Вход: {entry} | SL: {sl} | TP: {tp}
  R:R = {rr:.2f} (риск {abs(entry-sl):.4f} → прибыль {abs(tp-entry):.4f})
  Уверенность стратегии: {signal_data.get('confidence', 0):.0%}
  Причина: {signal_data.get('reason', '?')}
{indicators_str}
{sent_str}

Критерии отказа (approved=false):
- R:R < 1.5
- RSI > 75 для BUY или RSI < 25 для SELL
- Объём падающий при пробое
- Цена против основного тренда без явного разворота

Ответь СТРОГО JSON (без markdown):
{{"score": число 0-10, "approved": true/false, "comment": "одно предложение итог", "reasoning": "2-3 причины решения"}}"""

    # ── Утилиты ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Извлекает JSON из ответа модели (игнорирует markdown-обёртку)."""
        text = text.strip()
        # Убираем ```json ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None
