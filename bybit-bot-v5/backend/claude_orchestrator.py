"""
Multi-Provider Orchestrator — ИИ-оркестратор торгового бота.

Поддерживаемые провайдеры:
  • Anthropic Claude (с prompt caching)
  • OpenAI GPT-4o
  • Ollama (локальные модели, OpenAI-совместимый API)

Цикл работы (каждые 15 минут):
  1. Собрать snapshot: баланс, позиции, PnL, WR, boost, скальп, риски
  2. Отправить в LLM (JSON)
  3. Получить решения в виде JSON
  4. Исполнить команды через внутренние хуки
  5. Логировать + Telegram-отчёт
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Проверка доступных SDK ────────────────────────────────────────────────────

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# ── Системный промпт ──────────────────────────────────────────────────────────
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

== ВАЖНО: БАЛАНС В ФЬЮЧЕРСАХ ==
- `balance_usdt` — свободные средства (не заблокированы)
- `equity_usdt` — реальный капитал = balance + locked_margin (ЭТО НАСТОЯЩИЙ БАЛАНС)
- `locked_margin` — маржа, заблокированная в открытых позициях
- Если locked_margin > 0, balance будет маленьким — ЭТО НОРМАЛЬНО, не паникуй!
- Оценивай просадку только по equity_usdt, НЕ по balance_usdt

== ПРАВИЛА ПРИНЯТИЯ РЕШЕНИЙ ==
1. equity_usdt упал на > 15% от стартового баланса за день → stop + alert critical
2. WR последних 20 сделок < 48% → set_ml_mode strict + alert warning
3. WR последних 20 сделок > 70% → scalp_on (если не активен) + alert info
4. Boost: просадка от пика > 20% → boost_stop + alert warning
5. Boost: цель достигнута → boost_stop + alert info "🎉 Цель достигнута!"
6. Нет сделок > 8 часов при запущенном боте И нет открытых позиций → alert warning (возможна проблема)
   Если есть открытые позиции — это нормально, бот ждёт TP/SL → wait
7. 5+ убытков подряд по одной стратегии → disable_strategy + alert warning
8. equity_usdt < $2 → stop + alert critical (нельзя торговать)
9. Если balance_usdt мал но equity_usdt нормальный — открыта позиция, всё ОК → wait
10. Если всё хорошо → wait (не вмешивайся без причины)

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


# ── Провайдеры ────────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    """Абстрактный LLM-провайдер."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int = 1024,
    ) -> str:
        """Выполнить запрос к LLM и вернуть текст ответа."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Название провайдера."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Идентификатор модели."""


class AnthropicProvider(BaseProvider):
    """Провайдер Anthropic Claude с prompt caching."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic SDK не установлен: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync():
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
            )
            return response.content[0].text

        return await loop.run_in_executor(None, _sync)


class OpenAIProvider(BaseProvider):
    """Провайдер OpenAI (GPT-4o и совместимые)."""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai SDK не установлен: pip install openai")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync():
            # Конвертируем в формат OpenAI (добавляем system в начало)
            oai_messages = [{"role": "system", "content": system}] + messages
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=oai_messages,
            )
            return response.choices[0].message.content or ""

        return await loop.run_in_executor(None, _sync)


class OllamaProvider(BaseProvider):
    """Провайдер Ollama — локальные модели через OpenAI-совместимый API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync():
            # Ollama поддерживает OpenAI-совместимый /v1/chat/completions
            if OPENAI_AVAILABLE:
                client = openai.OpenAI(
                    api_key="ollama",
                    base_url=f"{self._base_url}/v1",
                )
                oai_messages = [{"role": "system", "content": system}] + messages
                response = client.chat.completions.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=oai_messages,
                )
                return response.choices[0].message.content or ""
            elif HTTPX_AVAILABLE:
                # Фолбэк: прямой HTTP-запрос
                payload = {
                    "model": self._model,
                    "messages": [{"role": "system", "content": system}] + messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                }
                resp = httpx.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            else:
                raise RuntimeError("Для Ollama нужен openai или httpx: pip install openai")

        return await loop.run_in_executor(None, _sync)


# ── Фабрика провайдеров ───────────────────────────────────────────────────────

def create_provider(
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    ollama_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[BaseProvider]:
    """
    Создать провайдера по приоритету: Anthropic → OpenAI → Ollama.
    Возвращает None если ни один провайдер недоступен.
    """
    if anthropic_key and ANTHROPIC_AVAILABLE:
        m = model or "claude-sonnet-4-6"
        logger.info(f"[Orchestrator] Провайдер: Anthropic ({m})")
        return AnthropicProvider(api_key=anthropic_key, model=m)

    if openai_key and OPENAI_AVAILABLE:
        m = model or "gpt-4o"
        logger.info(f"[Orchestrator] Провайдер: OpenAI ({m})")
        return OpenAIProvider(api_key=openai_key, model=m)

    if ollama_url:
        m = model or "llama3"
        logger.info(f"[Orchestrator] Провайдер: Ollama ({m} @ {ollama_url})")
        return OllamaProvider(base_url=ollama_url, model=m)

    return None


# ── Вспомогательные классы ────────────────────────────────────────────────────

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
        self.analysis       = analysis
        self.risk_level     = risk_level
        self.decisions      = decisions
        self.next_check_min = next_check_min
        self.timestamp      = datetime.utcnow().isoformat()
        self.executed: List[Dict] = []
        self.errors:   List[str]  = []

    def to_dict(self) -> Dict:
        return {
            "timestamp":      self.timestamp,
            "analysis":       self.analysis,
            "risk_level":     self.risk_level,
            "decisions":      [d.to_dict() for d in self.decisions],
            "next_check_min": self.next_check_min,
            "executed":       self.executed,
            "errors":         self.errors,
        }


# ── Главный оркестратор ───────────────────────────────────────────────────────

class ClaudeOrchestrator:
    """
    Оркестратор торгового бота с поддержкой нескольких LLM-провайдеров.

    Использование (Anthropic):
        orch = ClaudeOrchestrator(api_key="sk-ant-...", execute_fn=handler)

    Использование (OpenAI):
        orch = ClaudeOrchestrator(openai_key="sk-...", execute_fn=handler)

    Использование (Ollama):
        orch = ClaudeOrchestrator(ollama_url="http://localhost:11434", execute_fn=handler)

    Использование (готовый провайдер):
        provider = AnthropicProvider(api_key="...")
        orch = ClaudeOrchestrator(provider=provider, execute_fn=handler)
    """

    MAX_HISTORY = 4

    def __init__(
        self,
        # Backward-compatible: старый параметр api_key = Anthropic key
        api_key:     Optional[str]                          = None,
        execute_fn:  Optional[Callable[[Dict], Any]]        = None,
        # Мультипровайдер
        openai_key:  Optional[str]                          = None,
        ollama_url:  Optional[str]                          = None,
        model:       Optional[str]                          = None,
        provider:    Optional[BaseProvider]                 = None,
        notify_fn:   Optional[Callable[[str, str], Any]]   = None,
    ):
        self.execute  = execute_fn
        self.notify   = notify_fn
        self._history: List[Dict] = []
        self._last_run: Optional[datetime] = None
        self._next_run: Optional[datetime] = None

        # Если передан готовый провайдер — используем его
        if provider is not None:
            self._provider: Optional[BaseProvider] = provider
        else:
            self._provider = create_provider(
                anthropic_key=api_key,
                openai_key=openai_key,
                ollama_url=ollama_url,
                model=model,
            )

        self.enabled = self._provider is not None

        if self.enabled:
            logger.info(
                f"[Orchestrator] Инициализирован: "
                f"provider={self._provider.name}, model={self._provider.model}"
            )
        else:
            logger.warning(
                "[Orchestrator] Недоступен — не задан ни один провайдер "
                "(ANTHROPIC_API_KEY / OPENAI_API_KEY / OLLAMA_BASE_URL)"
            )

    # ── Публичные свойства ────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._provider.model if self._provider else "none"

    @property
    def is_due(self) -> bool:
        if self._next_run is None:
            return True
        return datetime.utcnow() >= self._next_run

    # ── Публичный API ─────────────────────────────────────────────────────────

    async def run_cycle(self, full_status: Dict) -> Optional[OrchestratorResult]:
        if not self.enabled:
            return None

        try:
            report  = self._build_report(full_status)
            raw     = await self._call_llm(report)
            result  = self._parse_response(raw)

            if result:
                self._execute_decisions(result)
                self._update_history(result, full_status)
                self._schedule_next(result.next_check_min)
                self._last_run = datetime.utcnow()
                self._send_summary(result)

            return result

        except Exception as e:
            logger.error(f"[Orchestrator] Ошибка цикла: {e}")
            self._schedule_next(15)
            return None

    def get_status(self) -> Dict:
        return {
            "enabled":      self.enabled,
            "provider":     self._provider.name if self._provider else None,
            "model":        self.model,
            "last_run":     self._last_run.isoformat() if self._last_run else None,
            "next_run":     self._next_run.isoformat() if self._next_run else None,
            "is_due":       self.is_due,
            "cycles_done":  len(self._history),
            "last_result":  self._history[-1] if self._history else None,
        }

    # ── Построение отчёта ─────────────────────────────────────────────────────

    def _build_report(self, s: Dict) -> str:
        today   = s.get("today", {})
        boost   = s.get("boost", {})
        risk    = s.get("risk",  {})
        recent  = s.get("recent_trades", [])

        if recent:
            wins_r = sum(1 for t in recent if (t.get("pnl_usd") or 0) > 0)
            wr_r   = round(wins_r / len(recent) * 100, 1)
        else:
            wr_r = None

        scalpers    = s.get("scalpers", {})
        scalp_total = sum(v.get("trades", 0) for v in scalpers.values())
        scalp_wins  = sum(v.get("wins",   0) for v in scalpers.values())
        scalp_wr    = round(scalp_wins / scalp_total * 100, 1) if scalp_total else None

        prev = self._history[-2:] if len(self._history) >= 2 else self._history

        report = {
            "time_utc":         datetime.utcnow().isoformat(),
            "bot_running":      s.get("bot_running"),
            "paper_mode":       s.get("paper_mode"),
            "balance_usdt":     (s.get("balance") or {}).get("usdt"),
            "equity_usdt":      (s.get("balance") or {}).get("equity_usdt"),
            "locked_margin":    (s.get("balance") or {}).get("locked_margin", 0),
            "balance_note":     (s.get("balance") or {}).get("note", ""),
            "open_positions":   s.get("open_count", 0),
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
                "active":             boost.get("active"),
                "mode":               boost.get("mode"),
                "progress_pct":       boost.get("progress_pct"),
                "current_balance":    boost.get("current_balance"),
                "target_balance":     boost.get("target_balance"),
                "days_remaining":     boost.get("days_remaining"),
                "required_daily_pct": boost.get("required_daily_pct"),
                "drawdown_from_peak": boost.get("drawdown_from_peak"),
                "phase":              boost.get("phase"),
                "consec_losses":      boost.get("consecutive_losses"),
            } if boost.get("active") else {"active": False},
            "risk": {
                "daily_loss_pct":  risk.get("daily_loss_pct"),
                "open_positions":  risk.get("open_positions_count"),
                "kill_switch":     risk.get("kill_switch_active"),
                "consec_losses":   risk.get("consecutive_losses"),
            },
            "ml_mode":         s.get("ml", {}).get("filter_mode"),
            "previous_cycles": prev,
        }
        return json.dumps(report, ensure_ascii=False, default=str)

    # ── LLM вызов ─────────────────────────────────────────────────────────────

    async def _call_llm(self, report: str) -> str:
        """Формируем историю диалога и вызываем активный провайдер."""
        history_msgs: List[Dict] = []
        for h in self._history[-self.MAX_HISTORY:]:
            history_msgs.append({"role": "user",      "content": h.get("report_sent", "")})
            history_msgs.append({"role": "assistant",  "content": json.dumps(h.get("result", {}), ensure_ascii=False)})

        messages = history_msgs + [{"role": "user", "content": report}]
        return await self._provider.complete(
            system=_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=1024,
        )

    # ── Парсинг ответа ────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Optional[OrchestratorResult]:
        try:
            text = raw.strip()
            # Модель иногда добавляет пояснение ПЕРЕД блоком ```json — ищем блок явно.
            if "```" in text:
                parts = text.split("```")
                # parts[1] — содержимое первого блока кода
                if len(parts) >= 2:
                    text = parts[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
            # Фолбэк: вытащить первый {...} из ответа
            if not text.startswith("{"):
                start = text.find("{")
                end   = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]

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
        entry["report_sent"] = self._build_report(full_status)[:800]
        self._history.append(entry)
        if len(self._history) > self.MAX_HISTORY * 2:
            self._history = self._history[-self.MAX_HISTORY:]

    def _send_summary(self, result: OrchestratorResult):
        emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(
            result.risk_level, "⚪"
        )
        provider_tag = f"[{self._provider.name}]" if self._provider else ""
        cmds    = [d.cmd for d in result.decisions if d.cmd != "wait"]
        summary = (
            f"{emoji} <b>Orchestrator{provider_tag}</b> [{result.risk_level.upper()}]\n"
            f"{result.analysis}\n"
        )
        if cmds:
            summary += f"Команды: {', '.join(cmds)}"
        else:
            summary += "Команды: wait (всё в норме)"

        logger.info(f"[Orchestrator] {result.analysis} | {cmds}")
        if self.notify:
            self.notify(summary, result.risk_level)
