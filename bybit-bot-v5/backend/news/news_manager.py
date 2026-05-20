"""
News Manager — главный оркестратор сбора, анализа и хранения новостей.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import os

from .news_aggregator import NewsAggregator
from .twitter_collector import TwitterCollector
from .sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class NewsManager:
    """Оркестратор сбора + анализ + хранение."""

    def __init__(
        self,
        db_pool,
        cryptopanic_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        twitter_accounts: Optional[List[str]] = None,
    ):
        self.db = db_pool
        self.news = NewsAggregator(cryptopanic_api_key=cryptopanic_key)
        self.twitter = TwitterCollector(custom_accounts=twitter_accounts)
        self.sentiment = SentimentAnalyzer(anthropic_api_key=anthropic_key, openai_api_key=openai_key)

        self.last_fetch: Optional[datetime] = None
        self.last_deep_analysis: Optional[datetime] = None
        self.cached_sentiment: Dict = {"score": 0, "magnitude": 0}
        self.cached_deep: Dict = {}
        self._init_db()

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, str):
                cleaned = value.replace("%", "").replace(",", ".").strip()
                return float(cleaned) if cleaned else default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value, default: int = 0) -> int:
        try:
            if value is None:
                return default
            if isinstance(value, str):
                cleaned = value.replace(",", ".").strip()
                return int(float(cleaned)) if cleaned else default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _init_db(self):
        """Таблицы для новостей."""
        ai = self.db.ai_pk()
        real = self.db.real_type()
        tables = [
            f"""CREATE TABLE IF NOT EXISTS news_items (
                id            {ai},
                source        VARCHAR(64)  NOT NULL,
                title         TEXT,
                text          TEXT,
                url           VARCHAR(255),
                published_at  VARCHAR(32),
                fetched_at    VARCHAR(32),
                sentiment_score    {real},
                sentiment_magnitude {real},
                bull_hits     INT,
                bear_hits     INT,
                item_type     VARCHAR(32),
                extra_json    TEXT,
                UNIQUE (source, url)
            )""",
            f"""CREATE TABLE IF NOT EXISTS sentiment_history (
                id                   {ai},
                timestamp            TEXT NOT NULL,
                source_type          TEXT,
                score                {real},
                magnitude            {real},
                count                INT,
                distribution_json    TEXT,
                deep_analysis_json   TEXT
            )""",
        ]
        indexes = [
            ("idx_news_fetched", "news_items",       "fetched_at"),
            ("idx_news_source",  "news_items",       "source"),
            ("idx_sent_ts",      "sentiment_history", "timestamp"),
        ]
        with self.db.cursor() as c:
            for ddl in tables:
                c.execute(ddl)
            for idx_name, tbl, col in indexes:
                if self.db.is_mysql:
                    try:
                        c.execute(f"CREATE INDEX {idx_name} ON {tbl}({col})")
                    except Exception:
                        pass
                else:
                    c.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl}({col})")

    async def collect_all(self, currencies: List[str] = None) -> Dict:
        """Сбор и анализ изо всех источников."""
        logger.info("📡 Сбор новостей и твитов...")
        currencies = currencies or ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]

        # Параллельный сбор
        news_task = asyncio.create_task(self.news.fetch_all(currencies))
        tweets_task = asyncio.create_task(self.twitter.fetch_all_tracked(max_per_account=5))

        news_items = await news_task
        tweets = await tweets_task

        # Sentiment анализ (rule-based, быстро)
        self.sentiment.analyze_batch(news_items, text_field="title")
        self.sentiment.analyze_batch(tweets, text_field="text")

        # Агрегация
        news_sentiment = self.sentiment.aggregate(news_items, weight_field="votes")
        twitter_sentiment = self.sentiment.aggregate(tweets)

        # Общий sentiment с весами (news 60%, twitter 40%)
        overall_score = news_sentiment["score"] * 0.6 + twitter_sentiment["score"] * 0.4
        overall_magnitude = (news_sentiment["magnitude"] + twitter_sentiment["magnitude"]) / 2

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "news_count": len(news_items),
            "tweet_count": len(tweets),
            "overall_score": round(overall_score, 3),
            "overall_magnitude": round(overall_magnitude, 3),
            "news_sentiment": news_sentiment,
            "twitter_sentiment": twitter_sentiment,
        }

        # Сохраняем в БД
        self._save_items(news_items + tweets)
        self._save_sentiment_snapshot(result)

        self.last_fetch = datetime.utcnow()
        self.cached_sentiment = result
        logger.info(
            f"📡 Sentiment: {overall_score:+.2f} "
            f"(news {news_sentiment['score']:+.2f}, twitter {twitter_sentiment['score']:+.2f})"
        )
        return result

    def _save_items(self, items: List[Dict]):
        """Сохранение в БД с дедупликацией."""
        if not items:
            return
        insert_sql = self.db.adapt(
            "INSERT IGNORE INTO news_items "
            "(source, title, text, url, published_at, fetched_at, "
            "sentiment_score, sentiment_magnitude, bull_hits, bear_hits, item_type, extra_json) "
            "VALUES (?,?,?,?,?,?, ?,?,?,?,?,?)"
        ) if self.db.is_mysql else self.db.adapt(
            "INSERT OR IGNORE INTO news_items "
            "(source, title, text, url, published_at, fetched_at, "
            "sentiment_score, sentiment_magnitude, bull_hits, bear_hits, item_type, extra_json) "
            "VALUES (?,?,?,?,?,?, ?,?,?,?,?,?)"
        )
        with self.db.cursor() as c:
            for item in items:
                try:
                    sent = item.get("sentiment", {})
                    text = item.get("text", item.get("summary", ""))
                    extra = {k: v for k, v in item.items()
                             if k not in ("source", "title", "text", "url",
                                          "published", "fetched_at", "sentiment", "summary")}
                    c.execute(insert_sql, (
                        item.get("source", ""),
                        item.get("title", "")[:500],
                        text[:1000] if text else "",
                        item.get("url", ""),
                        item.get("published", ""),
                        item.get("fetched_at", datetime.utcnow().isoformat()),
                        sent.get("score", 0),
                        sent.get("magnitude", 0),
                        sent.get("bull_hits", 0),
                        sent.get("bear_hits", 0),
                        "tweet" if item.get("source") == "twitter" else "news",
                        json.dumps(extra) if extra else None,
                    ))
                except Exception as e:
                    logger.debug(f"Item save skip: {e}")

    def _save_sentiment_snapshot(self, result: Dict):
        sql = self.db.adapt("""
            INSERT INTO sentiment_history (
                timestamp, source_type, score, magnitude, count, distribution_json
            ) VALUES (?,?,?,?,?,?)
        """)
        with self.db.cursor() as c:
            c.execute(sql, (
                result["timestamp"],
                "combined",
                result["overall_score"],
                result["overall_magnitude"],
                result["news_count"] + result["tweet_count"],
                json.dumps({
                    "news": result["news_sentiment"]["distribution"],
                    "twitter": result["twitter_sentiment"]["distribution"],
                }),
            ))

    async def deep_analysis(self, currencies: List[str] = None) -> Dict:
        """Глубокий AI-анализ (раз в час)."""
        cutoff = (datetime.utcnow() - timedelta(hours=6)).isoformat()
        query = self.db.adapt("""
            SELECT source, title, text, sentiment_score
            FROM news_items
            WHERE fetched_at >= ?
            ORDER BY fetched_at DESC LIMIT 30
        """)
        with self.db.connection() as conn:
            if self.db.is_mysql:
                with conn.cursor() as c:
                    c.execute(query, (cutoff,))
                    recent = [dict(row) for row in c.fetchall()]
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                recent = [dict(r) for r in conn.execute(query, (cutoff,)).fetchall()]
        if not recent:
            return {"available": False, "reason": "Нет свежих данных"}

        analysis = await self.sentiment.deep_analysis_with_claude(
            recent,
            context=f"Crypto market sentiment for: {', '.join(currencies or ['BTC', 'ETH'])}",
        )
        self.cached_deep = analysis
        self.last_deep_analysis = datetime.utcnow()
        if analysis.get("available"):
            sql = self.db.adapt("""
                INSERT INTO sentiment_history (
                    timestamp, source_type, score, magnitude, count, deep_analysis_json
                ) VALUES (?,?,?,?,?,?)
            """)
            with self.db.cursor() as c:
                c.execute(sql, (
                    datetime.utcnow().isoformat(),
                    "deep_claude",
                    analysis.get("overall_score", 0),
                    analysis.get("magnitude", 0),
                    analysis.get("analyzed_count", 0),
                    json.dumps(analysis),
                ))
        return analysis

    def get_current_sentiment(self) -> Dict:
        """Текущий sentiment (без блокирующего сбора). Использовать в торговом цикле."""
        return self.cached_sentiment

    def get_sentiment_features(self) -> Dict:
        """Sentiment-фичи для ML (текущее состояние)."""
        sent = self.cached_sentiment
        deep = self.cached_deep
        news_distribution = sent.get("news_sentiment", {}).get("distribution", {}) or {}
        twitter_distribution = sent.get("twitter_sentiment", {}).get("distribution", {}) or {}
        return {
            "sentiment_score": self._to_float(sent.get("overall_score", 0)),
            "sentiment_magnitude": self._to_float(sent.get("overall_magnitude", 0)),
            "news_sentiment": self._to_float(sent.get("news_sentiment", {}).get("score", 0)),
            "twitter_sentiment": self._to_float(sent.get("twitter_sentiment", {}).get("score", 0)),
            "news_count_24h": self._to_int(sent.get("news_count", 0)),
            "tweet_count_24h": self._to_int(sent.get("tweet_count", 0)),
            "bull_count": self._to_int(news_distribution.get("bullish", 0)) + self._to_int(twitter_distribution.get("bullish", 0)),
            "bear_count": self._to_int(news_distribution.get("bearish", 0)) + self._to_int(twitter_distribution.get("bearish", 0)),
            "neutral_count": self._to_int(news_distribution.get("neutral", 0)) + self._to_int(twitter_distribution.get("neutral", 0)),
            "deep_sentiment_score": self._to_float(deep.get("overall_score", 0)) if deep.get("available") else 0.0,
            "sentiment_data_freshness_min": (
                (datetime.utcnow() - self.last_fetch).total_seconds() / 60
                if self.last_fetch else 999
            ),
        }

    def get_recent_news(self, limit: int = 50, source_filter: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM news_items"
        params: list = []
        if source_filter:
            query += " WHERE source = ?"
            params.append(source_filter)
        query += " ORDER BY fetched_at DESC LIMIT ?"
        params.append(limit)
        with self.db.connection() as conn:
            if self.db.is_mysql:
                with conn.cursor() as c:
                    c.execute(self.db.adapt(query), params)
                    return list(c.fetchall())
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(self.db.adapt(query), params).fetchall()]

    def get_sentiment_history(self, hours: int = 24) -> List[Dict]:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        query = self.db.adapt("SELECT * FROM sentiment_history WHERE timestamp >= ? ORDER BY timestamp")
        with self.db.connection() as conn:
            if self.db.is_mysql:
                with conn.cursor() as c:
                    c.execute(query, (cutoff,))
                    return list(c.fetchall())
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(query, (cutoff,)).fetchall()]

    async def get_trading_recommendations(
        self,
        portfolio_context: Optional[Dict] = None,
        symbols: Optional[List[str]] = None,
    ) -> Dict:
        """
        AI-рекомендации на основе новостей, sentiment и контекста портфеля.

        portfolio_context:
            {"balance": 1000, "open_positions": 2, "daily_pnl": 15.5,
             "active_strategies": ["S1","S9"], "regime": "trending_up"}

        Возвращает:
            {"action": "trade|wait|reduce|stop",
             "risk_level": "low|medium|high|extreme",
             "focus_symbols": ["BTCUSDT"],
             "avoid_symbols": ["DOGEUSDT"],
             "recommendations": ["...", "..."],
             "fear_greed": {...},
             "reasoning": "...",
             "available": True}
        """
        if not self.sentiment.has_ai:
            return {"available": False, "reason": "AI API не настроен (нужен ANTHROPIC_API_KEY или OPENAI_API_KEY)", "action": "trade"}

        # Собираем последние данные
        cutoff = (datetime.utcnow() - timedelta(hours=4)).isoformat()
        query = self.db.adapt(
            "SELECT source, title, sentiment_score FROM news_items "
            "WHERE fetched_at >= ? ORDER BY fetched_at DESC LIMIT 40"
        )
        with self.db.connection() as conn:
            if self.db.is_mysql:
                with conn.cursor() as c:
                    c.execute(query, (cutoff,))
                    recent = [dict(row) for row in c.fetchall()]
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                recent = [dict(r) for r in conn.execute(query, (cutoff,)).fetchall()]

        # Fear & Greed Index
        fear_greed = await self.news.fetch_fear_greed()
        fg_value = self._to_int((fear_greed or {}).get("value"), 0)
        fg_change = self._to_float((fear_greed or {}).get("change"), 0.0)
        fg_label = (fear_greed or {}).get("label", "")

        # Символы по умолчанию
        syms = symbols or ["BTC", "ETH", "SOL", "BNB", "XRP"]

        # Строим промпт
        news_lines = []
        for it in recent[:25]:
            sentiment_score = self._to_float(it.get("sentiment_score"), 0.0)
            sentiment_prefix = "+" if sentiment_score > 0 else ""
            news_lines.append(
                f"  [{it.get('source', '?')}] {it.get('title', '')[:150]} "
                f"(sentiment: {sentiment_prefix}{sentiment_score:.2f})"
            )
        news_block = "\n".join(news_lines) or "  Нет свежих данных"

        fg_block = ""
        if fear_greed:
            fg_block = (
                f"\nFear & Greed Index: {fg_value}/100 "
                f"({fg_label}) "
                f"{'↑' if fg_change > 0 else '↓'}{abs(fg_change)} за день"
            )

        sentiment = self.cached_sentiment
        portfolio_block = ""
        if portfolio_context:
            balance = self._to_float(portfolio_context.get("balance"), 0.0)
            open_positions = self._to_int(portfolio_context.get("open_positions"), 0)
            daily_pnl = self._to_float(portfolio_context.get("daily_pnl"), 0.0)
            active_strategies = portfolio_context.get("active_strategies", [])
            if not isinstance(active_strategies, list):
                active_strategies = [str(active_strategies)]
            regime = str(portfolio_context.get("regime", "неизвестен"))
            portfolio_block = f"""
Портфель бота:
- Баланс: {balance:.2f} USDT
- Открытых позиций: {open_positions}
- Дневной PnL: {daily_pnl:+.2f} USDT
- Активные стратегии: {', '.join(active_strategies) if active_strategies else 'нет'}
- Режим рынка: {regime}"""

        prompt = f"""Ты — AI-аналитик торгового крипто-бота. Дай конкретные торговые рекомендации.

ТЕКУЩИЙ SENTIMENT РЫНКА:
- Общий: {self._to_float(sentiment.get('overall_score', 0)):+.2f} (−1 медведь .. +1 бык)
- Новости: {self._to_float(sentiment.get('news_sentiment', {}).get('score', 0)):+.2f}
- Twitter: {self._to_float(sentiment.get('twitter_sentiment', {}).get('score', 0)):+.2f}{fg_block}

СВЕЖИЕ НОВОСТИ И ТВИТЫ (последние 4 часа):
{news_block}
{portfolio_block}

ТОРГУЕМЫЕ СИМВОЛЫ: {', '.join(syms)}

Ответь СТРОГО JSON (только JSON, без пояснений):
{{
  "action": "trade" | "wait" | "reduce" | "stop",
  "risk_level": "low" | "medium" | "high" | "extreme",
  "focus_symbols": ["СИМВОЛ1USDT", ...],
  "avoid_symbols": ["СИМВОЛ2USDT", ...],
  "recommendations": [
    "конкретная рекомендация 1",
    "конкретная рекомендация 2",
    "конкретная рекомендация 3"
  ],
  "key_risks": ["риск1", "риск2"],
  "reasoning": "2-3 предложения почему такие рекомендации",
  "confidence": 0.0-1.0
}}

Правила:
- action=stop только при экстремальных рисках (FG < 15 или крупный hack/ban)
- action=reduce при высоком риске или негативных тенденциях
- action=wait при неопределённости
- action=trade при позитивном или нейтральном фоне
- Рекомендации на русском языке"""

        try:
            loop = asyncio.get_running_loop()
            if self.sentiment._anthropic_client:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.sentiment._anthropic_client.messages.create(
                        model="claude-haiku-4-5",
                        max_tokens=800,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                text = response.content[0].text.strip()
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.sentiment._openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=800,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                text = response.choices[0].message.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                result["available"] = True
                result["fear_greed"] = fear_greed
                result["sentiment_score"] = sentiment.get("overall_score", 0)
                result["timestamp"] = datetime.utcnow().isoformat()
                result["news_analyzed"] = len(recent)
                logger.info(
                    f"[AI Рекомендации] action={result.get('action')} "
                    f"risk={result.get('risk_level')} "
                    f"focus={result.get('focus_symbols', [])}"
                )
                return result
            return {"available": False, "reason": "JSON parse failed", "action": "trade"}
        except Exception as e:
            logger.error(f"AI рекомендации ошибка: {e}")
            return {"available": False, "reason": str(e), "action": "trade"}

    async def close(self):
        await self.news.close()
        await self.twitter.close()


# __init__ exports
async def background_news_loop(news_manager: NewsManager, interval_min: int = 15):
    """Фоновый таск: собирать новости каждые N минут."""
    while True:
        try:
            await news_manager.collect_all()
            # Раз в час — deep analysis через Claude
            if (
                news_manager.last_deep_analysis is None
                or (datetime.utcnow() - news_manager.last_deep_analysis).total_seconds() > 3600
            ):
                await news_manager.deep_analysis()
        except Exception as e:
            logger.error(f"News loop ошибка: {e}")
        await asyncio.sleep(interval_min * 60)
