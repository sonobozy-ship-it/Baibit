"""
Claude Orchestrator — ИИ-оркестратор торгового бота.

Архитектура:
  Бот → собирает состояние → отправляет Claude API → получает JSON-команды → исполняет.

Цикл работы (каждые 15 минут):
  1. Собрать полный snapshot: баланс, позиции, дневной PnL, WR, boost, скальп, риски
  2. Отправить в Claude (claude-sonnet-4-6, prompt caching)
  3. Получить решения в виде JSON
  4. Исполнить команды через внутренние хуки
  5. Логировать + Telegram-отчёт

Оркестратор принимает решения по правилам:
  • PnL дня < -15% → стоп торговли + алерт
  • WR последних 20 сделок < 50% → снизить плечо, перейти в advisory ML
  • 3 убытка подряд на скальпе → пауза скальпа 30 мин
  • Boost-прогресс: переход фазы / достижение цели → уведомление
  • Аномальный рынок → остановить до следующего цикла
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic SDK не установлен — оркестратор недоступен")


# ── Системный промпт (кешируется Claude API) ─────────────────────────────────
_SYSTEM_PROMPT = """
Ты — оркестратор криптовалютного торгового бота Baibit.
Получаешь JSON-snapshot состояния бота каждые 15 минут и отвечаешь JSON-командами.

== ДОСТУПНЫЕ КОМАНДЫ ==
{"cmd": "start"}                              — запустить торговый цикл
{"cmd": "stop", "reason": "..."}             — остановить торговый цикл
{"cmd": "paper_on"}                           — включить бумажную торговлю
{"cmd": "paper_off"}                          — выключить бумажную торговлю
{"cmd": "scalp_on"}                           — активировать ScalperPro (10 символов, 3m)
{"cmd": "scalp_off"}                          — деактивировать скальпирование
{"cmd": "boost_start", "initial": X, "target": Y, "days": N, "mode": "moderate|scalp|safe|aggressive"}
{"cmd": "boost_stop", "reason": "..."}       — остановить boost-сессию
{"cmd": "set_leverage", "sid": "S10", "value": 5}  — изменить плечо стратегии
{"cmd": "disable_strategy", "sid": "S3"}     — отключить стратегию
{"cmd": "enable_strategy", "sid": "S3"}      — включить стратегию
{"cmd": "set_ml_mode", "mode": "advisory|strict|off"}
{"cmd": "alert", "level": "info|warning|critical", "message": "..."}  — только оповещение
{"cmd": "wait"}                               — ничего не делать в этом цикле

== ПРАВИЛА ПРИНЯТИЯ РЕШЕНИЙ ==
1. Дневной PnL < -15% от начального баланса → stop + alert critical
2. WR последних 20 сделок < 48% → set_ml_mode strict + alert warning
3. WR последних 20 сделок > 70% → scalp_on (если не активен) + alert info
4. Boost: просадка от пика > 20% → boost_stop + alert warning
5. Boost: цель достигнута → boost_stop + alert info "🎉 Цель достигнута!"
6. Нет сделок > 2 часов при запущенном боте → alert warning (возможна проблема)
7. 5+ убытков подряд по одной стратегии → disable_strategy + alert warning
8. Баланс < $2 → stop + alert critical (нельзя торговать)
9. Если всё хорошо → wait (не вмешивайся без причины)

== ФОРМАТ ОТВЕТА (строго JSON) ==
{
  "analysis": "2-3 предложения: что видишь, что важно",
  "risk_level": "low|medium|high|critical",
  "decisions": [
    {"cmd": "...", ...},
    ...
  ],
  "next_check_min": 15
}

Отвечай ТОЛЬКО JSON. Никакого текста вне JSON.
""".strip()


class OrchestratorDecision:
    """Одно решение оркестратора."""
    def __init__(self, cmd: str, **kwargs):
        self.cmd    = cmd
        self.params = kwargs
        self.ts     = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {"cmd": self.cmd, **self.params}


class OrchestratorResult:
    """Результат одного цикла оркестратора."""
    def __init__(self, analysis: str, risk_level: str,
                 decisions: List[OrchestratorDecision], next_check_min: int):
        self.analysis      = analysis
        self.risk_level    = risk_level
        self.decisions     = decisions
        self.next_check_min = next_check_min
        self.timestamp     = datetime.utcnow().isoformat()
        self.executed: List[Dict] = []
        self.errors:   List[str]  = []

    def to_dict(self) -> Dict:
        return {
            "timestamp":     self.timestamp,
            "analysis":      self.analysis,
            "risk_level":    self.risk_level,
            "decisions":     [d.to_dict() for d in self.decisions],
            "next_check_min": self.next_check_min,
            "executed":      self.executed,
            "errors":        self.errors,
        }


class ClaudeOrchestrator:
    """
    Оркестратор на базе Claude API.

    Использование:
        orch = ClaudeOrchestrator(api_key="...", execute_fn=my_command_handler)
        result = await orch.run_cycle(full_status_dict)
    """

    MAX_HISTORY = 4        # сколько прошлых циклов держать в контексте

    def __init__(
        self,
        api_key: str,
        execute_fn: Optional[Callable[[Dict], Any]] = None,
        model: str = "claude-sonnet-4-6",
        notify_fn: Optional[Callable[[str, str], Any]] = None,
    ):
        """
        api_key    — Anthropic API key
        execute_fn — функция исполнения команды: fn({"cmd":"...", ...}) → dict
        model      — модель Claude
        notify_fn  — fn(message, level) для Telegram/лога
        """
        self.enabled  = ANTHROPIC_AVAILABLE and bool(api_key)
        self.model    = model
        self.execute  = execute_fn
        self.notify   = notify_fn
        self._history: List[Dict] = []   # последние MAX_HISTORY результатов
        self._last_run: Optional[datetime] = None
        self._next_run: Optional[datetime] = None
        self._client  = None

        if self.enabled:
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"[Orchestrator] Инициализирован: model={model}")
        else:
            logger.warning("[Orchestrator] Недоступен (нет ключа или anthropic SDK)")

    # ── Публичный API ─────────────────────────────────────────────────────────

    @property
    def is_due(self) -> bool:
        """Пора ли запускать следующий цикл?"""
        if self._next_run is None:
            return True
        return datetime.utcnow() >= self._next_run

    async def run_cycle(self, full_status: Dict) -> Optional[OrchestratorResult]:
        """
        Основной цикл: анализ состояния → команды → исполнение.
        Возвращает None если оркестратор отключён или не пора.
        """
        if not self.enabled:
            return None

        try:
            report   = self._build_report(full_status)
            raw      = await self._call_claude(report)
            result   = self._parse_response(raw)

            if result:
                self._execute_decisions(result)
                self._update_history(result, full_status)
                self._schedule_next(result.next_check_min)
                self._last_run = datetime.utcnow()

                # Telegram/лог уведомление
                self._send_summary(result)

            return result

        except Exception as e:
            logger.error(f"[Orchestrator] Ошибка цикла: {e}")
            self._schedule_next(15)
            return None

    def get_status(self) -> Dict:
        return {
            "enabled":        self.enabled,
            "model":          self.model,
            "last_run":       self._last_run.isoformat() if self._last_run else None,
            "next_run":       self._next_run.isoformat() if self._next_run else None,
            "is_due":         self.is_due,
            "cycles_done":    len(self._history),
            "last_result":    self._history[-1] if self._history else None,
        }

    # ── Формирование отчёта ───────────────────────────────────────────────────

    def _build_report(self, s: Dict) -> str:
        """Компактный JSON-отчёт о состоянии бота для Claude."""
        today = s.get("today", {})
        boost = s.get("boost", {})
        risk  = s.get("risk",  {})

        # Последние N сделок для WR
        recent = s.get("recent_trades", [])
        if recent:
            wins_r = sum(1 for t in recent if (t.get("pnl_usd") or 0) > 0)
            wr_r   = round(wins_r / len(recent) * 100, 1)
        else:
            wr_r = None

        # Скальп-итог
        scalpers = s.get("scalpers", {})
        scalp_total = sum(v.get("trades", 0) for v in scalpers.values())
        scalp_wins  = sum(v.get("wins",   0) for v in scalpers.values())
        scalp_wr    = round(scalp_wins / scalp_total * 100, 1) if scalp_total else None

        # История оркестратора
        prev = self._history[-2:] if len(self._history) >= 2 else self._history

        report = {
            "time_utc":       datetime.utcnow().isoformat(),
            "bot_running":    s.get("bot_running"),
            "paper_mode":     s.get("paper_mode"),
            "balance_usdt":   s.get("balance", {}).get("usdt"),
            "open_positions": s.get("open_count", 0),
            "positions_detail": [
                {"strategy": p["strategy"], "symbol": p["symbol"], "side": p["side"]}
                for p in s.get("open_positions", [])
            ],
            "today": {
                "trades":  today.get("trades", 0),
                "wins":    today.get("wins", 0),
                "losses":  today.get("losses", 0),
                "pnl_usd": today.get("pnl_usd", 0),
                "wr_pct":  today.get("wr_pct", 0),
            },
            "recent_wr_pct": wr_r,
            "scalp_active":  s.get("scalp_active"),
            "scalp_trades":  scalp_total,
            "scalp_wr_pct":  scalp_wr,
            "boost": {
                "active":           boost.get("active"),
                "mode":             boost.get("mode"),
                "progress_pct":     boost.get("progress_pct"),
                "current_balance":  boost.get("current_balance"),
                "target_balance":   boost.get("target_balance"),
                "days_remaining":   boost.get("days_remaining"),
                "required_daily_pct": boost.get("required_daily_pct"),
                "drawdown_from_peak": boost.get("drawdown_from_peak"),
                "phase":            boost.get("phase"),
                "consec_losses":    boost.get("consecutive_losses"),
            } if boost.get("active") else {"active": False},
            "risk": {
                "daily_loss_pct":    risk.get("daily_loss_pct"),
                "open_positions":    risk.get("open_positions_count"),
                "kill_switch":       risk.get("kill_switch_active"),
                "consec_losses":     risk.get("consecutive_losses"),
            },
            "ml_mode":        s.get("ml", {}).get("filter_mode"),
            "previous_cycles": prev,
        }
        return json.dumps(report, ensure_ascii=False, default=str)

    # ── Claude API вызов ──────────────────────────────────────────────────────

    async def _call_claude(self, report: str) -> str:
        """Отправляем отчёт в Claude, получаем JSON-ответ."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _sync_call():
            messages = [{"role": "user", "content": report}]

            # Добавляем историю диалога для контекста
            history_msgs = []
            for h in self._history[-self.MAX_HISTORY:]:
                history_msgs.append({
                    "role": "user",
                    "content": h.get("report_sent", ""),
                })
                history_msgs.append({
                    "role": "assistant",
                    "content": json.dumps(h.get("result", {}), ensure_ascii=False),
                })

            all_messages = history_msgs + messages

            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # prompt caching
                    }
                ],
                messages=all_messages,
            )
            return response.content[0].text

        return await loop.run_in_executor(None, _sync_call)

    # ── Парсинг ответа ────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Optional[OrchestratorResult]:
        try:
            # Claude иногда оборачивает JSON в ```json ... ```
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            decisions = [
                OrchestratorDecision(**d)
                for d in data.get("decisions", [])
                if isinstance(d, dict) and "cmd" in d
            ]
            return OrchestratorResult(
                analysis       = data.get("analysis", ""),
                risk_level     = data.get("risk_level", "low"),
                decisions      = decisions,
                next_check_min = int(data.get("next_check_min", 15)),
            )
        except Exception as e:
            logger.error(f"[Orchestrator] Парсинг ответа: {e}\nRaw: {raw[:300]}")
            return None

    # ── Исполнение команд ─────────────────────────────────────────────────────

    def _execute_decisions(self, result: OrchestratorResult):
        if not self.execute:
            logger.info("[Orchestrator] execute_fn не задан — команды только логируются")
            return

        for decision in result.decisions:
            cmd_dict = decision.to_dict()
            if cmd_dict.get("cmd") == "wait":
                result.executed.append({"cmd": "wait", "status": "skipped"})
                continue
            if cmd_dict.get("cmd") == "alert":
                # alert — только уведомление, не команда боту
                if self.notify:
                    self.notify(cmd_dict.get("message", ""), cmd_dict.get("level", "info"))
                result.executed.append({"cmd": "alert", "status": "notified"})
                continue
            try:
                exec_result = self.execute(cmd_dict)
                result.executed.append({
                    "cmd":    cmd_dict["cmd"],
                    "status": "ok",
                    "result": exec_result,
                })
                logger.info(f"[Orchestrator] ✅ {cmd_dict['cmd']}: {exec_result}")
            except Exception as e:
                result.errors.append(f"{cmd_dict['cmd']}: {e}")
                logger.error(f"[Orchestrator] ❌ {cmd_dict['cmd']}: {e}")

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _schedule_next(self, minutes: int):
        self._next_run = datetime.utcnow() + timedelta(minutes=max(5, minutes))

    def _update_history(self, result: OrchestratorResult, full_status: Dict):
        entry = result.to_dict()
        entry["report_sent"] = self._build_report(full_status)[:800]  # сокращаем для истории
        self._history.append(entry)
        if len(self._history) > self.MAX_HISTORY * 2:
            self._history = self._history[-self.MAX_HISTORY:]

    def _send_summary(self, result: OrchestratorResult):
        emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(
            result.risk_level, "⚪"
        )
        cmds = [d.cmd for d in result.decisions if d.cmd != "wait"]
        summary = (
            f"{emoji} <b>Orchestrator</b> [{result.risk_level.upper()}]\n"
            f"{result.analysis}\n"
        )
        if cmds:
            summary += f"Команды: {', '.join(cmds)}"
        else:
            summary += "Команды: wait (всё в норме)"

        logger.info(f"[Orchestrator] {result.analysis} | {cmds}")
        if self.notify:
            self.notify(summary, result.risk_level)
