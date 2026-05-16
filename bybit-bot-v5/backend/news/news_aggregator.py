"""
News Aggregator — сбор новостей из multiple источников.
Источники:
- CryptoPanic API (бесплатный tier, ~300 req/мес)
- CoinDesk RSS
- CoinTelegraph RSS
- Bitcoin Magazine RSS
- Reddit r/CryptoCurrency hot
"""
import aiohttp
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import re

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Сборщик новостей из разных источников."""

    RSS_FEEDS = {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
        "decrypt": "https://decrypt.co/feed",
        "bitcoinmagazine": "https://bitcoinmagazine.com/feed",
        "theblock": "https://www.theblock.co/rss.xml",
    }

    def __init__(self, cryptopanic_api_key: Optional[str] = None):
        self.cryptopanic_key = cryptopanic_api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Persistent session (быстрее)."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_rss(self, source: str, url: str, limit: int = 20) -> List[Dict]:
        """Парсинг RSS feed."""
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                content = await resp.text()
            return self._parse_rss(content, source, limit)
        except Exception as e:
            logger.warning(f"RSS {source} ошибка: {e}")
            return []

    def _parse_rss(self, content: str, source: str, limit: int) -> List[Dict]:
        """Парсинг XML RSS."""
        try:
            root = ET.fromstring(content)
            items = []
            channel = root.find("channel")
            if channel is None:
                return []
            for item in channel.findall("item")[:limit]:
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()
                description = self._strip_html(item.findtext("description", ""))[:500]
                items.append({
                    "source": source,
                    "title": title,
                    "url": link,
                    "published": pub_date,
                    "summary": description,
                    "fetched_at": datetime.utcnow().isoformat(),
                })
            return items
        except ET.ParseError as e:
            logger.warning(f"RSS parse {source}: {e}")
            return []

    @staticmethod
    def _strip_html(text: str) -> str:
        """Убрать HTML теги."""
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def fetch_cryptopanic(self, currencies: List[str] = None, limit: int = 50) -> List[Dict]:
        """CryptoPanic API - агрегатор криптоновостей."""
        if not self.cryptopanic_key:
            return []
        try:
            session = await self._get_session()
            params = {
                "auth_token": self.cryptopanic_key,
                "kind": "news",
                "filter": "hot",
                "public": "true",
            }
            if currencies:
                params["currencies"] = ",".join(currencies)

            url = "https://cryptopanic.com/api/v1/posts/"
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"CryptoPanic HTTP {resp.status}")
                    return []
                data = await resp.json()

            items = []
            for post in data.get("results", [])[:limit]:
                items.append({
                    "source": "cryptopanic",
                    "title": post.get("title", ""),
                    "url": post.get("url", ""),
                    "published": post.get("published_at", ""),
                    "summary": post.get("title", ""),
                    "currencies": [c.get("code") for c in post.get("currencies", [])],
                    "votes": post.get("votes", {}),
                    "kind": post.get("kind"),
                    "fetched_at": datetime.utcnow().isoformat(),
                })
            return items
        except Exception as e:
            logger.error(f"CryptoPanic error: {e}")
            return []

    async def fetch_reddit_hot(self, subreddit: str = "CryptoCurrency", limit: int = 25) -> List[Dict]:
        """Reddit hot posts (public JSON)."""
        try:
            session = await self._get_session()
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            async with session.get(url, headers={"User-Agent": "BybitBot/1.0"}) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            items = []
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                items.append({
                    "source": f"reddit/{subreddit}",
                    "title": post.get("title", ""),
                    "url": "https://reddit.com" + post.get("permalink", ""),
                    "summary": (post.get("selftext", "") or post.get("title", ""))[:500],
                    "score": post.get("score", 0),
                    "comments": post.get("num_comments", 0),
                    "published": datetime.fromtimestamp(post.get("created_utc", 0)).isoformat(),
                    "fetched_at": datetime.utcnow().isoformat(),
                })
            return items
        except Exception as e:
            logger.warning(f"Reddit error: {e}")
            return []

    async def fetch_fear_greed(self) -> Optional[Dict]:
        """Fear & Greed Index от Alternative.me (бесплатно, без ключа)."""
        try:
            session = await self._get_session()
            async with session.get("https://api.alternative.me/fng/?limit=2") as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            entries = data.get("data", [])
            if not entries:
                return None
            latest = entries[0]
            prev   = entries[1] if len(entries) > 1 else latest
            return {
                "value":      int(latest.get("value", 50)),
                "label":      latest.get("value_classification", "Neutral"),
                "prev_value": int(prev.get("value", 50)),
                "change":     int(latest.get("value", 50)) - int(prev.get("value", 50)),
                "timestamp":  latest.get("timestamp", ""),
            }
        except Exception as e:
            logger.warning(f"Fear&Greed ошибка: {e}")
            return None

    async def fetch_all(self, currencies: List[str] = None) -> List[Dict]:
        """Параллельная загрузка изо всех источников."""
        tasks = [self.fetch_rss(name, url, limit=15) for name, url in self.RSS_FEEDS.items()]
        tasks.append(self.fetch_cryptopanic(currencies, limit=30))
        tasks.append(self.fetch_reddit_hot("CryptoCurrency", limit=15))
        tasks.append(self.fetch_reddit_hot("Bitcoin", limit=10))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_items = []
        for r in results:
            if isinstance(r, list):
                all_items.extend(r)
            elif isinstance(r, Exception):
                logger.warning(f"Источник ошибка: {r}")

        # Дедупликация по title (нормализованному)
        seen = set()
        unique = []
        for item in all_items:
            title_norm = re.sub(r"[^\w]", "", item.get("title", "").lower())[:80]
            if title_norm and title_norm not in seen:
                seen.add(title_norm)
                unique.append(item)

        unique.sort(key=lambda x: x.get("published", ""), reverse=True)
        logger.info(f"📰 Собрано {len(unique)} уникальных новостей")
        return unique
