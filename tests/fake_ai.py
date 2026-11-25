"""A fake AI for tests: replays set answers instead of calling a provider.

Each reply is a string (the model's text), an exception to raise (CallTimeout(), RateLimited(...)),
or a function of the messages that returns a string.
"""
from __future__ import annotations

import json

from app.config import AppConfig
from app.llm.client import CallRecord, LLMClient


class FakeTransport:
    model = "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent: list[list[dict]] = []
        self.timeouts: list[float] = []

    def send(self, messages, timeout):
        self.sent.append([dict(m) for m in messages])
        self.timeouts.append(timeout)
        if not self.replies:
            raise AssertionError("the fake AI ran out of answers")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply(messages) if callable(reply) else reply


def fake_client(cfg: AppConfig, *replies):
    """An LLMClient around a FakeTransport. Returns (client, transport, call records, sleeps)."""
    records: list[CallRecord] = []
    sleeps: list[float] = []
    transport = FakeTransport(*replies)
    client = LLMClient(transport, ai_cfg=cfg.ai, log_call=records.append, sleep=sleeps.append)
    return client, transport, records, sleeps


def as_json(obj) -> str:
    return json.dumps(obj)
