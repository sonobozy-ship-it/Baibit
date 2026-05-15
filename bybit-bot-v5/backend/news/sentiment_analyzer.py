"""
Sentiment Analyzer.
Двухуровневый подход:
1. VADER (быстро, бесплатно) — на каждый элемент
2. Claude AI (точно, медленно) — на агрегат

Возвращает sentiment score в [-1, 1]:
- -1.0: extreme bear
-  0.0: neutral
- +1.0: extreme bull
"""
import logging
import re
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Простой rule-based анализатор (не требует ML библиотек)
class SimpleSentiment:
    """Lightweight sentiment без зависимостей. VADER-style."""

    BULL_WORDS = {
        "bull", "bullish", "buy", "long", "moon", "rally", "surge", "pump",
        "breakout", "ath", "all-time-high", "soar", "gain", "profit", "win",
        "approve", "adoption", "partnership", "launch", "upgrade", "halving",
        "institutional", "etf approval", "rate cut", "cut", "easing",
        "to the moon", "wagmi", "lfg", "based", "alpha", "bullrun",
        "growth", "expansion", "recovery", "support", "buying",
        "лонг", "рост", "ралли", "пробой", "позитив", "купить",
    }
    BEAR_WORDS = {
        "bear", "bearish", "sell", "short", "dump", "crash", "drop", "fall",
        "decline", "loss", "liquidation", "hack", "exploit", "rug", "scam",
        "ban", "regulation", "lawsuit", "sec", "fraud", "fud", "panic",
        "capitulation", "rekt", "ngmi", "dead", "bubble", "fud",
        "rate hike", "hike", "tightening", "recession", "bankruptcy",
        "resistance", "rejection", "selling", "depressed",
        "шорт", "падение", "крах", "слив", "негатив", "продать",
    }

    @classmethod
    def analyze(cls, text: str) -> Dict:
        if not text:
            return {"score": 0.0, "magnitude": 0, "bull_hits": 0, "bear_hits": 0}
        text_lower = text.lower()

        bull_hits = sum(1 for w in cls.BULL_WORDS if w in text_lower)
        bear_hits = sum(1 for w in cls.BEAR_WORDS if w in text_lower)

        total = bull_hits + bear_hits
        if total == 0:
            score = 0.0
        else:
            score = (bull_hits - bear_hits) / total

        # Магнитуда: насколько уверенно (больше совпадений = выше)
        magnitude = min(1.0, total / 5)

        return {
            "score": round(score, 3),
            "magnitude": round(magnitude, 3),
            "bull_hits": bull_hits,
            "bear_hits": bear_hits,
        }


class SentimentAnalyzer:
    """Главный анализатор с агрегацией."""

    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self._anthropic_client = None
        try:
            from anthropic import Anthropic
            if self.anthropic_key:
                self._anthropic_client = Anthropic(api_key=self.anthropic_key)
        except ImportError:
            pass

    def analyze_item(self, text: str) -> Dict:
        """Быстрый анализ одного элемента (rule-based)."""
        return SimpleSentiment.analyze(text)

    def analyze_batch(self, items: List[Dict], text_field: str = "title") -> List[Dict]:
        """Анализ списка элементов. Добавляет 'sentiment' к каждому."""
        for item in items:
            text = item.get(text_field, "") or item.get("summary", "") or item.get("text", "")
            item["sentiment"] = self.analyze_item(text)
        return items

    def aggregate(self, items: List[Dict], weight_field: Optional[str] = None) -> Dict:
        """
        Агрегированный sentiment по списку.

        weight_field: 'score' для Reddit, 'votes' для CryptoPanic
        """
        if not items:
            return {"score": 0.0, "magnitude": 0, "count": 0, "distribution": {"bull": 0, "bear": 0, "neutral": 0}}

        scores = []
        weights = []
        bull_n = bear_n = neutral_n = 0

        for item in items:
            sent = item.get("sentiment") or self.analyze_item(item.get("title", item.get("text", "")))
            score = sent["score"]
            scores.append(score)

            # Веса
            if weight_field == "score" and "score" in item:
                weights.append(max(1, item.get("score", 1)))
            elif weight_field == "votes":
                votes = item.get("votes", {})
                w = max(1, votes.get("positive", 0) + votes.get("negative", 0) + 1)
                weights.append(w)
            else:
                weights.append(1)

            if score > 0.1:
                bull_n += 1
            elif score < -0.1:
                bear_n += 1
            else:
                neutral_n += 1

        # Weighted mean
        total_w = sum(weights)
        weighted = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w else 0

        magnitude = abs(weighted) * min(1.0, len(items) / 20)

        return {
            "score": round(weighted, 3),
            "magnitude": round(magnitude, 3),
            "count": len(items),
            "distribution": {"bull": bull_n, "bear": bear_n, "neutral": neutral_n},
            "weighted": True if weight_field else False,
        }

    async def deep_analysis_with_claude(self, items: List[Dict], context: str = "") -> Dict:
        """
        Глубокий анализ через Claude AI.
        Используется реже (раз в час), даёт content-aware sentiment.
        """
        if not self._anthropic_client or not items:
            return {"available": False, "reason": "No Anthropic client or items"}

        # Берём топ-20 свежих
        sample = items[:20]
        items_text = "\n".join([
            f"- [{it.get('source', '?')}] {(it.get('title') or it.get('text', ''))[:200]}"
            for it in sample
        ])

        prompt = f"""Проанализируй следующие криптовалютные новости/твиты и оцени общий sentiment.

Контекст: {context or 'Анализ крипторынка для торгового бота'}

ИСТОЧНИКИ:
{items_text}

Дай ответ строго в JSON:
{{
    "overall_score": число от -1.0 до 1.0,
    "magnitude": число от 0 до 1 (насколько сильный sentiment),
    "key_themes": ["тема1", "тема2", "тема3"],
    "risk_factors": ["риск1", "риск2"],
    "bullish_catalysts": ["катализатор1"],
    "summary": "1-2 предложения о настроении рынка на русском"
}}

ТОЛЬКО JSON, без других слов."""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            text = response.content[0].text.strip()

            # Извлекаем JSON
            import json
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                data["available"] = True
                data["analyzed_count"] = len(sample)
                data["timestamp"] = datetime.utcnow().isoformat()
                return data
            return {"available": False, "reason": "JSON parse failed", "raw": text[:200]}
        except Exception as e:
            logger.error(f"Claude sentiment ошибка: {e}")
            return {"available": False, "error": str(e)}
