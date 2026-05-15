"""
AI-анализатор стратегий через Claude API.
Помогает понять, почему стратегия теряет деньги, и даёт рекомендации.
"""
import os
import logging
from typing import Dict, List, Optional
import json

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.enabled = False
        if not ANTHROPIC_AVAILABLE:
            logger.warning("anthropic SDK не установлен. AI отключён.")
            return
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY не задан. AI отключён.")
            return
        self.client = Anthropic(api_key=api_key)
        self.enabled = True

    def analyze_strategy_performance(self, strategy_data: Dict) -> Dict:
        """
        Анализ производительности одной стратегии.

        strategy_data:
        {
            "id": "S3",
            "name": "RSI Divergence",
            "trades": [{"pnl": -5.2, "side": "BUY", "exit_reason": "SL", ...}, ...],
            "stats": {"win_rate": 35, "profit_factor": 0.7, ...}
        }
        """
        if not self.enabled:
            return {"available": False, "message": "AI отключён. Установите ANTHROPIC_API_KEY."}

        try:
            prompt = self._build_analysis_prompt(strategy_data)
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            return {
                "available": True,
                "analysis": response_text,
                "strategy_id": strategy_data.get("id"),
            }
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {"available": False, "message": str(e)}

    def _build_analysis_prompt(self, data: Dict) -> str:
        stats = data.get("stats", {})
        recent_trades = data.get("trades", [])[-30:]

        trades_summary = "\n".join([
            f"  - {t.get('timestamp', '')}: {t.get('side')} {t.get('symbol')} → "
            f"{t.get('pnl_usd', 0):+.2f} USDT ({t.get('exit_reason', '?')})"
            for t in recent_trades[-15:]
        ])

        return f"""Ты — опытный квант-трейдер, анализирующий производительность торговой стратегии.

**Стратегия:** {data.get('name')} ({data.get('id')})
**Символ:** {data.get('symbol', 'неизвестно')}

**Статистика:**
- Всего сделок: {stats.get('trades', 0)}
- Win Rate: {stats.get('win_rate', 0)}%
- Profit Factor: {stats.get('profit_factor', 0)}
- Общий PnL: {stats.get('total_pnl', 0)} USDT
- Средняя прибыль: {stats.get('avg_win', 0)} USDT
- Средний убыток: {stats.get('avg_loss', 0)} USDT
- Max Drawdown: {stats.get('max_drawdown_pct', 0)}%
- Серия убытков: {stats.get('longest_loss_streak', 0)}

**Последние 15 сделок:**
{trades_summary}

Проанализируй и дай ответ В ТРЁХ ЧАСТЯХ:

1. **Диагноз** (2-3 предложения): что не так / что хорошо.
2. **Рекомендации** (3-5 пунктов): конкретные действия — изменить SL/TP, фильтры, таймфрейм, отключить, итд.
3. **Вердикт**: одна из 4 категорий с эмодзи:
   - 🟢 EXCELLENT (оставить как есть)
   - 🟡 GOOD (мелкие правки)
   - 🟠 NEEDS_WORK (серьёзные правки)
   - 🔴 DISABLE (отключить)

Будь конкретным, без воды. На русском."""

    def quick_signal_check(self, signal_data: Dict) -> Dict:
        """
        Быстрая проверка сигнала перед открытием сделки.
        Возвращает score 0-10 и краткий комментарий.
        """
        if not self.enabled:
            return {"score": 5, "comment": "AI отключён, нейтральный score"}

        try:
            prompt = f"""Оцени торговый сигнал по шкале 0-10 (где 10 = идеальный сетап).

Стратегия: {signal_data.get('strategy_name')}
Символ: {signal_data.get('symbol')}
Направление: {signal_data.get('action')}
Вход: {signal_data.get('entry_price')}
SL: {signal_data.get('stop_loss')}
TP: {signal_data.get('take_profit')}
Фильтры прошли: {signal_data.get('filters_passed')}

Контекст рынка: {signal_data.get('market_context', 'неизвестен')}

Ответ строго JSON:
{{"score": число 0-10, "comment": "одно предложение"}}"""

            message = self.client.messages.create(
                model="claude-haiku-4-5-20251001",  # быстрая модель для realtime
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            # Парсим JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except Exception as e:
            logger.error(f"AI signal check failed: {e}")
            return {"score": 5, "comment": "AI недоступен"}
