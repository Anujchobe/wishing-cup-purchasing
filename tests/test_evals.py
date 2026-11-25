"""The AI test cases themselves, and the scorer, without a real model."""
from datetime import date
from decimal import Decimal

from app import textparse
from app.config import LLMSettings
from app.llm.client import NoAIClient
from app.suppliers import fake_world
from evals import run as evals
from tests.fake_ai import as_json, fake_client

CASES = evals.load_cases()
PRODUCT_CODES = {p.code for p in fake_world.PRODUCTS}


def test_fifteen_of_each_with_unique_ids():
    assert len(CASES["read_request"]) == 15 and len(CASES["match_product"]) == 15
    ids = [c["id"] for c in CASES["read_request"] + CASES["match_product"]]
    assert len(ids) == len(set(ids))


def test_reading_cases_cover_the_hard_parts():
    texts = " ".join(c["text"].lower() for c in CASES["read_request"])
    for phrase in ("2 dozen", "half a case", "saffron", "1 1/2", "32 oz vanilla", "of milk", "the usual"):
        assert phrase in texts, phrase
    assert any(not c["expect"]["lines"] for c in CASES["read_request"])       # requests with nothing we sell
    assert any(c["expect"]["needed_by"] is None for c in CASES["read_request"])


def test_reading_answers_are_valid():
    today = date.fromisoformat(CASES["today"])
    for case in CASES["read_request"]:
        lines = case["expect"]["lines"]
        assert set(lines) <= PRODUCT_CODES, case["id"]
        assert all(Decimal(q) > 0 for q in lines.values()), case["id"]
        if case["expect"]["needed_by"]:
            assert date.fromisoformat(case["expect"]["needed_by"]) > today
        for word in case["expect"].get("unclear_mentions", []):
            assert word.lower() in case["text"].lower(), (case["id"], word)


def test_matching_answers_are_the_fake_suppliers_truth():
    for case in CASES["match_product"]:
        for supplier, right in case["expect"].items():
            sold = [o for o in fake_world.offers_for(supplier) if o.product_code == case["product"]]
            if right["sku"] is None:
                assert not sold, (case["id"], supplier)
            else:
                [offer] = sold
                assert (right["sku"], Decimal(right["pack"]), right["pack_text"]) == \
                       (offer.sku, offer.pack_base_qty, offer.raw_pack_text), (case["id"], supplier)
    assert any(r["sku"] is None for c in CASES["match_product"] for r in c["expect"].values())


def test_no_ai_scores_zero_and_says_why(cfg):
    results = evals.run_all(NoAIClient("No AI key"), cfg)
    assert len(results) == 30 and not any(r.passed for r in results)
    summary = evals.summarize(results, LLMSettings(model="m"), cfg, 1.0)
    assert summary["passed"] is False
    assert {k: v for k, v in summary["tasks"]["read_request"].items() if k != "seconds"} == {
        "score": "0.000", "passed": 0, "total": 15}
    assert len(summary["failures"]) == 30 and "the AI wasn't used" in summary["failures"][0]["detail"]


def test_a_right_reading_passes_and_a_wrong_one_fails(cfg):
    case = next(c for c in CASES["read_request"] if c["id"] == "two_dozen_eggs")
    today = date.fromisoformat(CASES["today"])
    right = as_json({"lines": [
        {"product_code": "eggs", "product_text": "eggs", "quantity": "2", "unit": "dozen"},
        {"product_code": "cream_cheese", "product_text": "cream cheese", "quantity": "10", "unit": "lbs"}],
        "needed_by": "for Friday", "unclear": []})
    client, *_ = fake_client(cfg, right)
    assert evals.run_read_case(case, client, cfg, today).passed
    wrong = right.replace('"dozen"', "null")   # 2 eggs instead of 24
    client, *_ = fake_client(cfg, wrong)
    result = evals.run_read_case(case, client, cfg, today)
    assert not result.passed and "eggs: 2" in result.detail


def test_a_right_match_passes(cfg):
    case = next(c for c in CASES["match_product"] if c["id"] == "match_vanilla_extract")
    picks = []
    for supplier, right in case["expect"].items():
        offer = fake_world.offer(supplier, "vanilla_extract")
        unit = {"fl_oz": "fl oz"}.get(offer.unit, offer.unit)
        picks.append({"supplier": supplier, "sku": right["sku"], "pack_count": offer.count,
                      "pack_size": str(offer.size), "pack_unit": unit, "confidence": 0.9})
    client, *_ = fake_client(cfg, as_json({"picks": picks}))
    result = evals.run_match_case(case, client, cfg)
    assert result.passed, result.detail


def test_the_case_dates_read_as_expected():
    today = date.fromisoformat(CASES["today"])
    assert textparse.parse_needed_by("for Friday", today) == date(2026, 11, 27)
