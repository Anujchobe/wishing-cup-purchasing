"""The AI client. One job: complete_json(system, user, schema) returns a checked Pydantic object.

How a call goes:
  1. Send the prompt (temperature 0, a time limit on each call).
  2. "Too many requests": wait (the provider's Retry-After if given, else 1, 2, 4... seconds,
     capped) and send again, up to ai.rate_limit_retries times.
  3. Pull the JSON out of the reply and check it against the schema. The schema can also check
     against a context, e.g. "the sku must be one of these five".
  4. If the answer fails the check, send it back once with what was wrong (ai.retries_on_bad_answer).
  5. Still bad, a timeout, or the provider fails: raise LLMError. Callers then use text matching
     and mark the result for a person to check.

With no key, or with the ai_down test switch on, make_client returns a NoAIClient whose calls
always raise LLMUnavailable, so every caller takes its text-matching path.

Every send, good or bad, is saved as one llm_calls row. A failure to save a log row is reported
in the Python log but never breaks the request itself.

Settings come only from .env (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL) through app.config. The
OpenAI-style API is used, so most providers and local servers work.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Protocol, TypeVar

from app.config import AIConfig, AppConfig, LLMSettings, ai_status
from app.llm.parsing import describe_error, extract_json

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
NO_AI_MODEL = "(no AI)"

T = TypeVar("T")


# ------------------------------------------------------------------ errors

class LLMError(Exception):
    """The AI couldn't give a usable answer. Callers fall back to text matching."""


class LLMUnavailable(LLMError):
    """No AI configured, AI switched off, timeout, network or provider error, or rate limits."""


class LLMBadAnswer(LLMError):
    """The model answered, but the answer still failed the checks after the retry."""


# What a transport may raise. OpenAITransport turns the openai library's errors into these.
class RateLimited(Exception):
    def __init__(self, retry_after: float | None = None, message: str = "too many requests"):
        super().__init__(message)
        self.retry_after = retry_after


class CallTimeout(Exception):
    pass


class ProviderError(Exception):
    pass


# ------------------------------------------------------------------ logging

@dataclass(frozen=True)
class CallRecord:
    model: str
    task: str
    duration_ms: int
    ok: bool
    error: str | None
    attempt: int
    request_id: int | None = None


CallLogger = Callable[[CallRecord], None]


def db_logger(engine) -> CallLogger:
    """Save each call in llm_calls, in its own transaction, so failed calls are kept even if the
    caller's work is rolled back."""
    from app import models
    from app.db import session_scope

    def write(record: CallRecord) -> None:
        with session_scope(engine) as session:
            session.add(models.LlmCall(model=record.model, task=record.task, duration_ms=record.duration_ms,
                                       ok=record.ok, error=record.error, attempt=record.attempt,
                                       request_id=record.request_id))

    return write


# ------------------------------------------------------------------ prompts

def render_prompt(prompt_name: str, /, **values: Any) -> str:
    """Load app/llm/prompts/<prompt_name>.txt and fill in its $placeholders.
    (Positional-only, so a placeholder may itself be called $name.)"""
    text = (PROMPTS_DIR / f"{prompt_name}.txt").read_text(encoding="utf-8")
    return Template(text).substitute({k: str(v) for k, v in values.items()})


# ------------------------------------------------------------------ transports

class Transport(Protocol):
    model: str

    def send(self, messages: list[dict[str, str]], timeout: float) -> str: ...


class OpenAITransport:
    """Any OpenAI-compatible chat endpoint (OpenAI, OpenRouter, Ollama, LM Studio, vLLM...)."""

    def __init__(self, settings: LLMSettings, timeout: float):
        from openai import OpenAI

        self.model = settings.model or ""
        # Retries are ours (see LLMClient), so the library's own are switched off.
        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url or None,
                              max_retries=0, timeout=timeout)
        self._send_temperature = True

    def send(self, messages: list[dict[str, str]], timeout: float) -> str:
        import openai

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "timeout": timeout}
        if self._send_temperature:
            kwargs["temperature"] = 0
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            raise RateLimited(_retry_after(exc), f"too many requests: {exc.message}") from exc
        except openai.APITimeoutError as exc:  # a subclass of APIConnectionError, so it goes first
            raise CallTimeout(str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"couldn't reach the AI provider ({exc})") from exc
        except openai.BadRequestError as exc:
            # Some models only allow their default temperature. Send once more without it.
            if self._send_temperature and "temperature" in str(exc.message).lower():
                self._send_temperature = False
                log.warning("Model %s rejected temperature=0; sending without it from now on.", self.model)
                return self.send(messages, timeout)
            raise ProviderError(f"the AI provider refused the request (HTTP 400: {exc.message})") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"the AI provider returned HTTP {exc.status_code} ({exc.message})") from exc
        if not response.choices:
            raise ProviderError("the AI provider sent an empty reply")
        return response.choices[0].message.content or ""


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


# ------------------------------------------------------------------ clients

class LLMClient:
    enabled = True

    def __init__(self, transport: Transport, *, ai_cfg: AIConfig, log_call: CallLogger | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic):
        self.transport = transport
        self.cfg = ai_cfg
        self._log_call = log_call or (lambda record: None)
        self._sleep = sleep
        self._monotonic = monotonic

    @property
    def model(self) -> str:
        return self.transport.model

    def complete_json(self, system: str, user: str, schema: type[T], *, task: str,
                      context: dict[str, Any] | None = None, request_id: int | None = None) -> T:
        """Ask the model and return a checked instance of `schema` (a Pydantic model class)."""
        schema_text = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        messages = [
            {"role": "system", "content": system.rstrip() + "\n\n" + render_prompt("json_format", schema=schema_text)},
            {"role": "user", "content": user},
        ]
        attempt = 0
        problem = ""
        for _ in range(1 + self.cfg.retries_on_bad_answer):
            reply, duration_ms, attempt = self._send(messages, task, request_id, attempt)
            try:
                answer = schema.model_validate(extract_json(reply), context=context)
            except ValueError as exc:  # broken JSON, or Pydantic's ValidationError (a ValueError)
                problem = describe_error(exc)
                self._log(task, duration_ms, False, f"bad answer: {problem}", attempt, request_id)
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": render_prompt("fix_reply", error=problem)},
                ]
                continue
            self._log(task, duration_ms, True, None, attempt, request_id)
            return answer
        raise LLMBadAnswer(f"The AI's answer was still wrong after a retry: {problem}")

    def _send(self, messages: list[dict[str, str]], task: str, request_id: int | None,
              attempt: int) -> tuple[str, int, int]:
        for wait_round in range(self.cfg.rate_limit_retries + 1):
            attempt += 1
            started = self._monotonic()
            try:
                reply = self.transport.send(messages, self.cfg.timeout_seconds)
                return reply, self._ms(started), attempt
            except RateLimited as exc:
                if wait_round == self.cfg.rate_limit_retries:
                    self._log(task, self._ms(started), False, "too many requests; gave up", attempt, request_id)
                    raise LLMUnavailable("The AI provider kept saying 'too many requests'.") from exc
                wait = exc.retry_after if exc.retry_after is not None else float(2 ** wait_round)
                wait = min(wait, self.cfg.rate_limit_max_wait_seconds)
                self._log(task, self._ms(started), False, f"too many requests; waited {wait:g}s", attempt, request_id)
                self._sleep(wait)
            except CallTimeout as exc:
                self._log(task, self._ms(started), False, f"timed out after {self.cfg.timeout_seconds:g}s",
                          attempt, request_id)
                raise LLMUnavailable(f"The AI didn't answer within {self.cfg.timeout_seconds:g} seconds.") from exc
            except ProviderError as exc:
                self._log(task, self._ms(started), False, str(exc), attempt, request_id)
                raise LLMUnavailable(f"The AI call failed: {exc}") from exc
            except Exception as exc:  # anything unexpected from a transport must not crash a request
                self._log(task, self._ms(started), False, f"{type(exc).__name__}: {exc}", attempt, request_id)
                raise LLMUnavailable(f"The AI call failed unexpectedly: {exc}") from exc
        raise AssertionError("unreachable")

    def _ms(self, started: float) -> int:
        return max(0, int((self._monotonic() - started) * 1000))

    def _log(self, task: str, duration_ms: int, ok: bool, error: str | None, attempt: int,
             request_id: int | None) -> None:
        try:
            self._log_call(CallRecord(self.model, task, duration_ms, ok, error, attempt, request_id))
        except Exception:
            log.exception("Couldn't save an llm_calls row")


class NoAIClient:
    """Used with no key, or with the ai_down switch on. Every call falls back to text matching."""

    enabled = False
    model = NO_AI_MODEL

    def __init__(self, reason: str, log_call: CallLogger | None = None):
        self.reason = reason
        self._log_call = log_call or (lambda record: None)

    def complete_json(self, system: str, user: str, schema: type[T], *, task: str,
                      context: dict[str, Any] | None = None, request_id: int | None = None) -> T:
        try:
            self._log_call(CallRecord(NO_AI_MODEL, task, 0, False, self.reason, 1, request_id))
        except Exception:
            log.exception("Couldn't save an llm_calls row")
        raise LLMUnavailable(self.reason)


AnyClient = LLMClient | NoAIClient


def make_client(cfg: AppConfig, llm: LLMSettings, *, log_call: CallLogger | None = None,
                transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep) -> AnyClient:
    status = ai_status(cfg, llm)
    if not status.enabled:
        return NoAIClient(status.reason, log_call)
    if transport is None:
        try:
            transport = OpenAITransport(llm, cfg.ai.timeout_seconds)
        except Exception as exc:  # e.g. the openai package missing or a bad base URL
            return NoAIClient(f"Couldn't start the AI client ({exc}). Using text matching.", log_call)
    return LLMClient(transport, ai_cfg=cfg.ai, log_call=log_call, sleep=sleep)
