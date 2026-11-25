"""Getting JSON out of a model's reply, and describing what was wrong with it.

Models wrap JSON in prose, code fences or <think> blocks. We take the first
complete JSON object. Numbers with decimals are read as Decimal, never float.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
MAX_ERROR_CHARS = 800


def _first_object(text: str) -> str | None:
    """The first balanced {...} in text, respecting strings and escapes."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json(reply: str) -> Any:
    """Parse the JSON object in a model reply. Raises ValueError with a readable reason."""
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("the reply was empty")
    text = _THINK.sub("", reply).strip()
    last_error = "there is no JSON object in the reply"
    for chunk in [m.group(1) for m in _FENCE.finditer(text)] + [text]:
        chunk = chunk.strip()
        try:
            return json.loads(chunk, parse_float=Decimal)
        except ValueError:
            pass
        obj = _first_object(chunk)
        if obj is not None:
            try:
                return json.loads(obj, parse_float=Decimal)
            except ValueError as exc:
                last_error = f"the JSON is broken ({exc})"
        elif "{" in chunk:
            last_error = "the JSON is broken or cut off (a '{' is never closed)"
    raise ValueError(last_error)


def describe_error(exc: Exception) -> str:
    """One readable line per problem. Works for Pydantic errors and plain ValueErrors."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = []
            for err in errors():
                where = ".".join(str(p) for p in err.get("loc", ())) or "(whole answer)"
                parts.append(f"{where}: {err.get('msg', '')}")
            text = "; ".join(parts)
        except Exception:  # an unusual error object; fall back to its text
            text = str(exc)
    else:
        text = str(exc)
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS] + "..."
