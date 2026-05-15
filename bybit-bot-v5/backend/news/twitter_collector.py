"""
Twitter/X сборщик.
Использует:
1. Nitter инстансы (бесплатно, не нужны ключи)
2. Опционально - X API v2 если есть ключ (платно)

Список аккаунтов для отслеживания: топ crypto influencers + on-chain аналитики.
"""
import aiohttp
import asyncio
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class TwitterCollector:
    """Сбор твитов через Nitter RSS (бесплатно, без авторизации)."""

    # Публичные Nitter инстансы (если один падает — пробуем следующий)
    NITTER_INSTANCES = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.net",
        "https://nitter.tiekoetter.com",
    ]

    # Ключевые crypto-аккаунты
    DEFAULT_ACCOUNTS = [
        "elonmusk",         # Markets-mover
        "cz_binance",       # CEO Binance
        "saylor",           # Microstrategy BTC
        "VitalikButerin",   # Ethereum
        "WatcherGuru",      # News aggregator
        "DocumentingBTC",   # BTC narrative
        "WhaleAlert",       # Large transfers
        "PeterSchiff",      # Bear thesis
        "RaoulGMI",         # Macro
        "APompliano",       # Bull thesis
        "CryptoCobain",     # Trader
        "GlassnodeInsight", # On-chain
        "CoinGlass_",       # Liquidations
        "tier10k",          # Macro flow
        "DegenSpartan",     # Trader
    ]

    def __init__(self, x_api_bearer_token: Optional[str] = None, custom_accounts: Optional[List[str]] = None):
        self.bearer_token = x_api_bearer_token
        self.accounts = custom_accounts or self.DEFAULT_ACCOUNTS
        self._session: Optional[aiohttp.ClientSession] = None
        self._working_instance: Optional[str] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _find_working_nitter(self) -> Optional[str]:
        """Найти живой Nitter инстанс."""
        if self._working_instance:
            return self._working_instance
        session = await self._get_session()
        for instance in self.NITTER_INSTANCES:
            try:
                async with session.get(f"{instance}/elonmusk/rss", timeout=5) as resp:
                    if resp.status == 200:
                        self._working_instance = instance
                        logger.info(f"🐦 Nitter: {instance}")
                        return instance
            except Exception:
                continue
        return None

    async def fetch_user_tweets(self, username: str, limit: int = 10) -> List[Dict]:
        """Получить твиты юзера через Nitter RSS."""
        instance = await self._find_working_nitter()
        if not instance:
            return []
        try:
            session = await self._get_session()
            async with session.get(f"{instance}/{username}/rss") as resp:
                if resp.status != 200:
                    return []
                content = await resp.text()

            tweets = self._parse_nitter_rss(content, username, limit)
            return tweets
        except Exception as e:
            logger.warning(f"Twitter {username}: {e}")
            # Сбрасываем кэш инстанса при ошибке
            self._working_instance = None
            return []

    def _parse_nitter_rss(self, content: str, username: str, limit: int) -> List[Dict]:
        """Парсинг Nitter RSS."""
        try:
            root = ET.fromstring(content)
            tweets = []
            channel = root.find("channel")
            if channel is None:
                return []
            for item in channel.findall("item")[:limit]:
                title = item.findtext("title", "")
                # Очищаем от "R: @user:" префиксов retweet'ов
                content_text = self._extract_text(item.findtext("description", ""))
                pub = item.findtext("pubDate", "")
                link = item.findtext("link", "")
                if not content_text and not title:
                    continue
                tweets.append({
                    "author": username,
                    "text": content_text or title,
                    "url": link,
                    "published": pub,
                    "fetched_at": datetime.utcnow().isoformat(),
                    "source": "twitter",
                })
            return tweets
        except ET.ParseError:
            return []

    @staticmethod
    def _extract_text(description: str) -> str:
        if not description:
            return ""
        text = re.sub(r"<[^>]+>", " ", description)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:500]

    async def fetch_all_tracked(self, max_per_account: int = 5) -> List[Dict]:
        """Параллельно собрать твиты со всех отслеживаемых аккаунтов."""
        # Batches по 5 чтобы не перегружать Nitter
        all_tweets = []
        for i in range(0, len(self.accounts), 5):
            batch = self.accounts[i:i + 5]
            results = await asyncio.gather(
                *[self.fetch_user_tweets(u, max_per_account) for u in batch],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, list):
                    all_tweets.extend(r)
            await asyncio.sleep(0.5)  # gentle to Nitter

        all_tweets.sort(key=lambda x: x.get("published", ""), reverse=True)
        logger.info(f"🐦 Собрано {len(all_tweets)} твитов с {len(self.accounts)} аккаунтов")
        return all_tweets

    async def search_keyword(self, query: str, limit: int = 20) -> List[Dict]:
        """Поиск твитов по ключевому слову через Nitter."""
        instance = await self._find_working_nitter()
        if not instance:
            return []
        try:
            session = await self._get_session()
            url = f"{instance}/search/rss"
            params = {"f": "tweets", "q": query}
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                content = await resp.text()
            return self._parse_nitter_rss(content, "search:" + query, limit)
        except Exception as e:
            logger.warning(f"Twitter search {query}: {e}")
            return []
