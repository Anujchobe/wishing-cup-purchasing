"""Fuzzy text matching with rapidfuzz, shared by intake (product names) and matching (catalogs).

Scorer: token_set_ratio on normalized text (lowercase, punctuation removed, plurals made
singular). It ignores word order and extra words, so "Unsalted butter" scores 100 against Sysco's
"BUTTER, UNSALTED" and US Foods' "Unsalted butter (bulk)". The flip side: a short query like
"flour" also scores 100 against every flour. That is why callers require a margin over the
second-best match before trusting the first.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping

from rapidfuzz import fuzz, process

from app.textparse import normalize


def score(a: str, b: str) -> float:
    return float(fuzz.token_set_ratio(a, b, processor=normalize))


def top_matches(query: str, choices: Mapping[Hashable, str], limit: int) -> list[tuple[Hashable, str, float]]:
    """Best `limit` choices as (key, text, score 0-100), best first; ties keep the choices' order."""
    if not choices:
        return []
    order = {key: n for n, key in enumerate(choices)}
    found = process.extract(query, choices, scorer=fuzz.token_set_ratio, processor=normalize, limit=None)
    ranked = sorted(((key, text, float(s)) for text, s, key in found), key=lambda r: (-r[2], order[r[0]]))
    return ranked[:limit]
