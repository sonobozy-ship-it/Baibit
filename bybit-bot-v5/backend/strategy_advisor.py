"""
Strategy Advisor — AI-анализ производительности всех стратегий.

Что делает:
  1. Читает статистику из trade_journal (WR, PnL, profit_factor, тренд)
  2. Отправляет в Claude / GPT структурированный запрос
  3. Получает рекомендации по каждой стратегии (конкретные параметры)
  4. Сохраняет в БД (таблица strategy_advisor_log)
  5. Автозапуск: каждые AUTO_EVERY_TRADES сделок или раз в сутки

Команда Telegram: /advisor [SID]
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

AUTO_EVERY_TRADES = 30   # запускать анализ каждые N сделок
MIN_TRADES_TO_ANALYZE = 3  # минимум сделок у стратегии для включения в анализ


class StrategyAdvisor:
    """AI-советник по стратегиям."""

    def __init__(self, db_pool, anthropic_key: str = "", openai_key: str = ""):
        self.db = db_pool
        self._anthropic_client = None
        self._openai_client = None

        if anthropic_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            except Exception as e:
                logger.warning(f"StrategyAdvisor: Anthropic init: {e}")

        if openai_key and not self._anthropic_client:
            try:
                import openai as _openai
                self._openai_client = _openai.OpenAI(api_key=openai_key)
            except Exception as e:
                logger.warning(f"StrategyAdvisor: OpenAI init: {e}")

        self.enabled = bool(self._anthropic_client or self._openai_client)
        self._total_trades_at_last_run = 0
        self._last_run: Optional[datetime] = None
        self._init_db()

    # ── БД ───────────────────────────────────────────────────────────────────

    def _init_db(self):
        ai = self.db.ai_pk()
        sql = f"""
        CREATE TABLE IF NOT EXISTS strategy_advisor_log (
            id          {ai},
            timestamp   TEXT NOT NULL,
            triggered_by TEXT,
            total_trades INT,
            analysis_json TEXT,
            summary     TEXT
        )"""
        with self.db.cursor() as c:
            c.execute(sql)
            if not self.db.is_mysql:
                c.execute("CREATE INDEX IF NOT EXISTS idx_adv_ts ON strategy_advisor_log(timestamp)")

    def get_last_analysis(self) -> Optional[Dict]:
        """Последний сохранённый анализ из БД."""
        sql = self.db.adapt(
            "SELECT * FROM strategy_advisor_log ORDER BY timestamp DESC LIMIT 1"
        )
        with self.db.connection() as conn:
            if self.db.is_mysql:
                with conn.cursor() as c:
                    c.execute(sql)
                    row = c.fetchone()
                    row = dict(row) if row else None
            else:
                import sqlite3
                conn.row_factory = sqlite3.Row
                row = conn.execute(sql).fetchone()
                row = dict(row) if row else None
        if not row:
            return None
        try:
            row["analysis"] = json.loads(row["analysis_json"] or "{}")
        except Exception:
            row["analysis"] = {}
        return row

    def _save_analysis(self, analysis: Dict, triggered_by: str, total_trades: int):
        sql = self.db.adapt(
            "INSERT INTO strategy_advisor_log "
            "(timestamp, triggered_by, total_trades, analysis_json, summary) "
            "VALUES (?,?,?,?,?)"
        )
        summary = analysis.get("overall", "")[:500]
        with self.db.cursor() as c:
            c.execute(sql, (
                datetime.utcnow().isoformat(),
                triggered_by,
                total_trades,
                json.dumps(analysis, ensure_ascii=False),
                summary,
            ))

    # ── Подготовка данных ────────────────────────────────────────────────────

    def build_stats_block(self, journal_stats: List[Dict], strategies: Dict) -> str:
        """Формирует текстовый блок со статистикой для промпта."""
        lines = []
        for st in sorted(journal_stats, key=lambda x: x.get("trades", 0), reverse=True):
            sid = st["strategy_id"]
            strat_obj = strategies.get(sid)
            symbol = strat_obj.symbol if strat_obj else "?"
            tf = strat_obj.timeframe if strat_obj else "?"
            name = st.get("strategy_name", sid)
            trades = st.get("trades", 0)
            wr = st.get("win_rate", 0)
            pnl = st.get("total_pnl", 0)
            pf = st.get("profit_factor", 0)
            avg_w = st.get("avg_win", 0)
            avg_l = st.get("avg_loss", 0)
            best = st.get("best_trade", 0)
            worst = st.get("worst_trade", 0)
            lines.append(
                f"  {sid} ({name}) | {symbol} TF={tf}m | "
                f"Сделок={trades} WR={wr:.0f}% PnL={pnl:+.2f}$ PF={pf:.2f} | "
                f"Avg+={avg_w:.2f}$ Avg-={avg_l:.2f}$ | "
                f"Best={best:+.2f}$ Worst={worst:+.2f}$"
            )
        return "\n".join(lines) if lines else "  Нет данных"

    def build_no_data_block(self, strategies: Dict, journal_stats: List[Dict]) -> str:
        """Стратегии без сделок."""
        analyzed = {s["strategy_id"] for s in journal_stats}
        lines = []
        for sid, strat in strategies.items():
            if sid not in analyzed:
                lines.append(f"  {sid} ({strat.NAME}) | {strat.symbol} — 0 сделок")
        return "\n".join(lines) if lines else "  (все стратегии имеют сделки)"

    # ── AI-запрос ────────────────────────────────────────────────────────────

    async def analyze_all(
        self,
        journal_stats: List[Dict],
        strategies: Dict,
        market_context: Dict = None,
        triggered_by: str = "manual",
    ) -> Dict:
        """
        Анализирует все стратегии через AI.
        Возвращает dict с рекомендациями.
        """
        if not self.enabled:
            return {"available": False, "reason": "AI API не настроен"}

        stats_with_data = [s for s in journal_stats if s.get("trades", 0) >= MIN_TRADES_TO_ANALYZE]
        total_trades = sum(s.get("trades", 0) for s in journal_stats)

        stats_block = self.build_stats_block(stats_with_data, strategies)
        no_data_block = self.build_no_data_block(strategies, journal_stats)

        ctx = market_context or {}
        market_block = ""
        if ctx:
            market_block = f"""
РЫНОЧНЫЙ КОНТЕКСТ:
  Режим рынка: {ctx.get('regime', 'неизвестен')}
  BTC тренд: {ctx.get('btc_trend', '?')}
  Sentiment: {ctx.get('sentiment', '?')}
  Баланс: {ctx.get('balance', '?')} USDT
  Открытых позиций: {ctx.get('open_positions', '?')}"""

        prompt = f"""Ты — AI-советник торгового крипто-бота. Проанализируй производительность стратегий.

СТАТИСТИКА СТРАТЕГИЙ (закрытые сделки):
{stats_block}

СТРАТЕГИИ БЕЗ ДАННЫХ (нет сделок):
{no_data_block}
{market_block}

ЗАДАЧА: Дай конкретные рекомендации по каждой стратегии.

Ответь СТРОГО JSON (только JSON):
{{
  "overall": "2-3 предложения об общем состоянии портфеля стратегий",
  "top_strategies": ["S11", "S5"],
  "weak_strategies": ["S3"],
  "pause_suggestions": ["S6"],
  "strategies": {{
    "S1": {{
      "score": 7,
      "status": "good",
      "assessment": "краткая оценка 1 предложение",
      "issues": ["проблема 1", "проблема 2"],
      "recommendations": [
        "Конкретное изменение параметра 1 (например: RSI порог 35→42)",
        "Конкретное изменение параметра 2",
        "Конкретное изменение параметра 3"
      ],
      "priority": "low"
    }}
  }},
  "global_recommendations": [
    "Глобальная рекомендация 1",
    "Глобальная рекомендация 2"
  ],
  "next_review_after_trades": 50
}}

Правила:
- score: 1-10 (1=убыточная, 10=отличная)
- status: excellent/good/average/underperforming/broken/no_data
- priority: low/medium/high/critical
- Если < {MIN_TRADES_TO_ANALYZE} сделок — status=no_data, score=5, пустые issues/recommendations
- Рекомендации должны быть конкретными: какой параметр, было → стало
- Язык: русский
- Анализируй только стратегии: {list(strategies.keys())}"""

        try:
            loop = asyncio.get_running_loop()
            if self._anthropic_client:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._anthropic_client.messages.create(
                        model="claude-opus-4-7",
                        max_tokens=2500,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                text = response.content[0].text.strip()
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=2500,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
                text = response.choices[0].message.content.strip()

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                result["available"] = True
                result["timestamp"] = datetime.utcnow().isoformat()
                result["total_trades_analyzed"] = total_trades
                self._save_analysis(result, triggered_by, total_trades)
                self._last_run = datetime.utcnow()
                self._total_trades_at_last_run = total_trades
                logger.info(
                    f"[Advisor] Анализ завершён. "
                    f"Top: {result.get('top_strategies', [])} | "
                    f"Слабые: {result.get('weak_strategies', [])}"
                )
                return result
            return {"available": False, "reason": "JSON parse error", "raw": text[:300]}

        except Exception as e:
            logger.error(f"[Advisor] Ошибка анализа: {e}")
            return {"available": False, "reason": str(e)}

    # ── Автозапуск ───────────────────────────────────────────────────────────

    def should_auto_run(self, current_total_trades: int) -> bool:
        """Нужно ли запустить анализ автоматически?"""
        if not self.enabled:
            return False
        trades_since = current_total_trades - self._total_trades_at_last_run
        if trades_since >= AUTO_EVERY_TRADES:
            return True
        if self._last_run is None:
            return current_total_trades >= MIN_TRADES_TO_ANALYZE
        if (datetime.utcnow() - self._last_run).total_seconds() > 86400:  # раз в сутки
            return current_total_trades > self._total_trades_at_last_run
        return False

    # ── Форматирование для Telegram ──────────────────────────────────────────

    @staticmethod
    def format_telegram(analysis: Dict, brief: bool = False) -> str:
        """Форматирует результат анализа для Telegram."""
        if not analysis.get("available"):
            return f"❌ {analysis.get('reason', 'Нет данных')}"

        ts = analysis.get("timestamp", "")[:16].replace("T", " ")
        total = analysis.get("total_trades_analyzed", 0)
        lines = [
            f"🤖 <b>AI-советник по стратегиям</b>",
            f"<i>{ts} UTC | {total} сделок проанализировано</i>\n",
            f"📊 <b>Общая оценка:</b>",
            analysis.get("overall", "—"),
        ]

        top = analysis.get("top_strategies", [])
        weak = analysis.get("weak_strategies", [])
        pause = analysis.get("pause_suggestions", [])
        if top:
            lines.append(f"\n🏆 <b>Лучшие:</b> {' '.join(top)}")
        if weak:
            lines.append(f"⚠️ <b>Слабые:</b> {' '.join(weak)}")
        if pause:
            lines.append(f"⏸ <b>Рекомендую паузу:</b> {' '.join(pause)}")

        if brief:
            gr = analysis.get("global_recommendations", [])
            if gr:
                lines.append("\n💡 <b>Ключевые рекомендации:</b>")
                for r in gr[:3]:
                    lines.append(f"• {r}")
            lines.append("\nИспользуй /advisor S1 для деталей по стратегии")
            return "\n".join(lines)

        strats = analysis.get("strategies", {})
        if strats:
            lines.append("\n<b>По стратегиям:</b>")
            status_emoji = {
                "excellent": "🟢", "good": "✅", "average": "🟡",
                "underperforming": "🔴", "broken": "💀", "no_data": "⬜",
            }
            for sid in sorted(strats.keys()):
                st = strats[sid]
                em = status_emoji.get(st.get("status", ""), "❓")
                score = st.get("score", "?")
                assessment = st.get("assessment", "")
                lines.append(f"\n{em} <b>{sid}</b> [{score}/10] — {assessment}")
                recs = st.get("recommendations", [])
                for r in recs[:2]:
                    lines.append(f"  → {r}")
                if len(recs) > 2:
                    lines.append(f"  + ещё {len(recs)-2} рекомендации")

        gr = analysis.get("global_recommendations", [])
        if gr:
            lines.append("\n💡 <b>Глобальные рекомендации:</b>")
            for r in gr:
                lines.append(f"• {r}")

        next_rev = analysis.get("next_review_after_trades", AUTO_EVERY_TRADES)
        lines.append(f"\n<i>Следующий анализ через ~{next_rev} сделок</i>")
        return "\n".join(lines)

    @staticmethod
    def format_strategy_detail(analysis: Dict, sid: str) -> str:
        """Детальный отчёт по одной стратегии."""
        if not analysis.get("available"):
            return f"❌ Нет данных анализа. Запусти /advisor сначала."

        strats = analysis.get("strategies", {})
        if sid not in strats:
            available = ", ".join(strats.keys()) or "нет"
            return f"❌ Стратегия <code>{sid}</code> не найдена в последнем анализе.\nДоступны: {available}"

        st = strats[sid]
        status_emoji = {
            "excellent": "🟢", "good": "✅", "average": "🟡",
            "underperforming": "🔴", "broken": "💀", "no_data": "⬜",
        }
        em = status_emoji.get(st.get("status", ""), "❓")
        score = st.get("score", "?")
        status = st.get("status", "?")

        lines = [
            f"{em} <b>AI-анализ: {sid}</b> [{score}/10 — {status}]\n",
            f"<b>Оценка:</b> {st.get('assessment', '—')}\n",
        ]

        issues = st.get("issues", [])
        if issues:
            lines.append("<b>Проблемы:</b>")
            for issue in issues:
                lines.append(f"  ⚠️ {issue}")
            lines.append("")

        recs = st.get("recommendations", [])
        if recs:
            lines.append("<b>Рекомендации (конкретные изменения):</b>")
            for i, r in enumerate(recs, 1):
                lines.append(f"  {i}. {r}")
        else:
            lines.append("<i>Нет рекомендаций — стратегия работает хорошо</i>")

        ts = analysis.get("timestamp", "")[:16].replace("T", " ")
        lines.append(f"\n<i>Анализ от {ts} UTC</i>")
        return "\n".join(lines)
