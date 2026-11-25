from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationInfo, field_validator
from sqlmodel import select

from app import models
from app.config import LLMSettings
from app.llm.client import (
    CallTimeout,
    LLMBadAnswer,
    LLMClient,
    LLMUnavailable,
    NoAIClient,
    ProviderError,
    RateLimited,
    db_logger,
    make_client,
    render_prompt,
)
from tests.conftest import config_with
from tests.fake_ai import as_json, fake_client


class Pick(BaseModel):
    sku: str
    confidence: Decimal  # JSON decimals arrive as Decimal, never float

    @field_validator("sku")
    @classmethod
    def _listed(cls, value, info: ValidationInfo):
        allowed = (info.context or {}).get("skus")
        if allowed is not None and value not in allowed:
            raise ValueError(f"sku '{value}' is not in the list")
        return value


CTX = {"skus": {"A1", "B2"}}
GOOD = as_json({"sku": "A1", "confidence": 0.9})


def ask(client):
    return client.complete_json("system", "user", Pick, task="test", context=CTX, request_id=None)


def test_good_answer(cfg):
    client, transport, records, _ = fake_client(cfg, GOOD)
    assert ask(client) == Pick(sku="A1", confidence=Decimal("0.9"))
    assert [(r.ok, r.attempt, r.task, r.model) for r in records] == [(True, 1, "test", "fake-model")]
    assert transport.timeouts == [cfg.ai.timeout_seconds]  # every call has the time limit
    system = transport.sent[0][0]["content"]
    assert "JSON Schema" in system and '"sku"' in system   # the schema is in the prompt


def test_broken_json_is_retried_with_the_reason(cfg):
    client, transport, records, _ = fake_client(cfg, '{"sku": "A1", "confidence": ', GOOD)
    assert ask(client).sku == "A1"
    retry = transport.sent[1]
    assert retry[-2] == {"role": "assistant", "content": '{"sku": "A1", "confidence": '}
    assert "could not be used" in retry[-1]["content"] and "broken" in retry[-1]["content"]
    assert [r.ok for r in records] == [False, True]
    assert "bad answer" in records[0].error


def test_made_up_sku_is_retried_then_refused(cfg):
    made_up = as_json({"sku": "ZZ9", "confidence": 0.99})
    client, transport, records, _ = fake_client(cfg, made_up, made_up)
    with pytest.raises(LLMBadAnswer, match="ZZ9"):
        ask(client)
    assert "not in the list" in transport.sent[1][-1]["content"]  # the model was told what was wrong
    assert len(transport.sent) == 2                                  # one retry only
    assert [r.ok for r in records] == [False, False]


def test_made_up_sku_fixed_on_retry(cfg):
    client, _, records, _ = fake_client(cfg, as_json({"sku": "ZZ9", "confidence": 0.99}), GOOD)
    assert ask(client).sku == "A1"
    assert [r.attempt for r in records] == [1, 2]


def test_timeout_gives_up_at_once(cfg):
    client, transport, records, _ = fake_client(cfg, CallTimeout("slow"))
    with pytest.raises(LLMUnavailable, match="didn't answer within 30 seconds"):
        ask(client)
    assert len(transport.sent) == 1
    assert records[0].ok is False and "timed out" in records[0].error


def test_too_many_requests_waits_and_retries(cfg):
    client, transport, records, sleeps = fake_client(cfg, RateLimited(retry_after=3), RateLimited(), GOOD)
    assert ask(client).sku == "A1"
    assert sleeps == [3, 2.0]  # the provider's Retry-After first, then our own backoff (1, 2, 4...)
    assert [r.ok for r in records] == [False, False, True]
    assert [r.attempt for r in records] == [1, 2, 3]


def test_waits_are_capped(cfg):
    client, _, _, sleeps = fake_client(cfg, RateLimited(retry_after=600), GOOD)
    ask(client)
    assert sleeps == [cfg.ai.rate_limit_max_wait_seconds]


def test_too_many_requests_forever_gives_up(cfg):
    n = cfg.ai.rate_limit_retries + 1
    client, transport, records, sleeps = fake_client(cfg, *[RateLimited()] * n)
    with pytest.raises(LLMUnavailable, match="too many requests"):
        ask(client)
    assert len(transport.sent) == n and len(sleeps) == n - 1
    assert all(not r.ok for r in records)


def test_provider_errors_and_surprises_become_unavailable(cfg):
    for problem in (ProviderError("HTTP 500"), KeyError("bug in a transport")):
        client, _, records, _ = fake_client(cfg, problem)
        with pytest.raises(LLMUnavailable):
            ask(client)
        assert records[-1].ok is False


def test_a_broken_logger_never_breaks_the_call(cfg):
    from tests.fake_ai import FakeTransport

    def broken(record):
        raise RuntimeError("database is locked")

    client = LLMClient(FakeTransport(GOOD), ai_cfg=cfg.ai, log_call=broken)
    assert ask(client).sku == "A1"


def test_no_key_means_no_ai(cfg):
    client = make_client(cfg, LLMSettings())
    assert isinstance(client, NoAIClient) and not client.enabled
    with pytest.raises(LLMUnavailable, match="text matching"):
        ask(client)


def test_ai_down_switch_means_no_ai_even_with_a_key():
    cfg = config_with(lambda c: c["faults"].update(ai_down=True))
    client = make_client(cfg, LLMSettings(api_key="k", model="m"))
    assert isinstance(client, NoAIClient) and "ai_down" in client.reason


def test_key_and_model_give_a_real_client(cfg):
    from tests.fake_ai import FakeTransport

    client = make_client(cfg, LLMSettings(api_key="k", model="m"), transport=FakeTransport(GOOD))
    assert isinstance(client, LLMClient) and client.enabled


def test_calls_are_saved_in_llm_calls(cfg, engine):
    from tests.fake_ai import FakeTransport

    client = LLMClient(FakeTransport("not json", GOOD), ai_cfg=cfg.ai, log_call=db_logger(engine), sleep=lambda s: None)
    ask(client)
    with pytest.raises(LLMUnavailable):
        NoAIClient("AI is off", db_logger(engine)).complete_json("s", "u", Pick, task="read_request")
    from app.db import session_scope

    with session_scope(engine) as s:
        rows = s.exec(select(models.LlmCall).order_by(models.LlmCall.id)).all()
    assert [(r.model, r.task, r.ok, r.attempt) for r in rows] == [
        ("fake-model", "test", False, 1), ("fake-model", "test", True, 2), ("(no AI)", "read_request", False, 1)]
    assert all(r.duration_ms >= 0 for r in rows)


def test_prompts_are_text_files_and_fill_in():
    for name, values in {"read_request": {"products": "x | X | lb", "today": "Tuesday 2026-11-24"},
                         "read_request_user": {"text": "50 lb flour"},
                         "match_product": {"units": "lb"},
                         "match_product_user": {"name": "n", "code": "c", "base_unit": "lb", "suppliers": "s"},
                         "fix_reply": {"error": "oops"},
                         "json_format": {"schema": "{}"}}.items():
        text = render_prompt(name, **values)
        assert "$" not in text, name
