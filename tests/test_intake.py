from datetime import date
from decimal import Decimal

import pytest

from app.intake import ProductRef, read_request
from app.llm.client import CallTimeout, NoAIClient
from app.suppliers import fake_world
from tests.fake_ai import as_json, fake_client

TUE = date(2026, 11, 24)
PRODUCTS = [ProductRef(p.code, p.name, p.base_unit) for p in fake_world.PRODUCTS]
MSG = "Need 50 lb all-purpose flour, 30 lbs unsalted butter and 4 gallons of heavy cream by Thursday"


def line(code, text, qty, unit, note=""):
    return {"product_code": code, "product_text": text, "quantity": qty, "unit": unit, "note": note}


GOOD = as_json({"lines": [line("ap_flour", "all-purpose flour", "50", "lb"),
                          line("butter_unsalted", "unsalted butter", "30", "lbs"),
                          line("heavy_cream", "heavy cream", "4", "gallons")],
                "needed_by": "by Thursday", "unclear": []})


def read(cfg, client, text=MSG, model_tested=True, **kw):
    return read_request(text, products=PRODUCTS, client=client, today=TUE, model_tested=model_tested,
                        cfg=cfg.intake, **kw)


def got(result):
    return {l.product_code: l.qty for l in result.lines}


def test_good_answer_and_code_does_the_conversion(cfg):
    client, _, records, _ = fake_client(cfg, GOOD)
    result = read(cfg, client)
    assert result.source == "ai" and result.ai_error is None
    assert got(result) == {"ap_flour": Decimal(50), "butter_unsalted": Decimal(30), "heavy_cream": Decimal(512)}
    assert [(l.quantity_text, l.unit_text) for l in result.lines] == [("50", "lb"), ("30", "lbs"), ("4", "gallons")]
    assert result.needed_by == date(2026, 11, 26) and result.unclear == []
    assert not any(l.needs_check for l in result.lines)
    assert [r.ok for r in records] == [True]


def test_model_not_tested_means_a_person_checks_every_line(cfg):
    client, *_ = fake_client(cfg, GOOD)
    result = read(cfg, client, model_tested=False)
    assert all(l.needs_check and "model not tested" in l.check_reasons for l in result.lines)


def test_made_up_quantity_drops_that_line(cfg):
    answer = as_json({"lines": [line("ap_flour", "flour", "50", "lb"), line("butter_unsalted", "butter", "40", "lbs")],
                      "needed_by": "by Thursday"})
    client, *_ = fake_client(cfg, answer)
    result = read(cfg, client)
    assert got(result) == {"ap_flour": Decimal(50)}
    assert any("'40' isn't in the message" in u for u in result.unclear)


def test_made_up_unit_drops_that_line(cfg):
    answer = as_json({"lines": [line("heavy_cream", "heavy cream", "4", "liters")], "needed_by": "by Thursday"})
    client, *_ = fake_client(cfg, answer)
    result = read(cfg, client)
    assert result.lines == [] and any("'liters' isn't in the message" in u for u in result.unclear)


def test_made_up_product_code_is_retried(cfg):
    bad = as_json({"lines": [line("saffron_threads", "saffron", "50", "lb")], "needed_by": None})
    client, transport, *_ = fake_client(cfg, bad, GOOD)
    result = read(cfg, client)
    assert result.source == "ai" and len(result.lines) == 3
    assert "saffron_threads" in transport.sent[1][-1]["content"]


def test_product_not_on_the_list_goes_to_unclear(cfg):
    answer = as_json({"lines": [line(None, "saffron", "2", "lb")], "needed_by": None, "unclear": []})
    client, *_ = fake_client(cfg, answer)
    result = read(cfg, client, text="2 lb saffron by Friday")
    assert result.lines == []
    assert any("'saffron' isn't on our product list" in u for u in result.unclear)


@pytest.mark.parametrize("replies", [
    ["{broken", "{still broken"],
    [as_json({"lines": [line("made_up", "x", "1", "lb")]})] * 2,
    [CallTimeout("slow")],
])
def test_ai_failure_falls_back_to_text_matching_and_flags_everything(cfg, replies):
    client, *_ = fake_client(cfg, *replies)
    result = read(cfg, client)
    assert result.source == "text_match" and result.ai_error
    assert got(result) == {"ap_flour": Decimal(50), "butter_unsalted": Decimal(30), "heavy_cream": Decimal(512)}
    assert all(l.needs_check and "AI failed" in l.check_reasons[0] for l in result.lines)
    assert result.needed_by == date(2026, 11, 26)


def test_ai_date_must_be_in_the_message(cfg):
    answer = as_json({"lines": [line("ap_flour", "flour", "50", "lb")], "needed_by": "by Monday"})
    client, *_ = fake_client(cfg, answer)
    result = read(cfg, client)
    assert result.needed_by is None and any("'by Monday' isn't in the message" in u for u in result.unclear)


def test_form_date_wins(cfg):
    client, *_ = fake_client(cfg, GOOD)
    assert read(cfg, client, needed_by=date(2026, 12, 4)).needed_by == date(2026, 12, 4)


# ------------------------------------------------------------------ no AI

NO_AI = NoAIClient("No AI key and model in .env. Using text matching.")


def test_no_ai_example_message(cfg):
    msg = ("Hi! For the factory we need 50 lb AP flour, 2 dozen eggs, 3 gallons of heavy cream, "
           "1.5 kg dark chocolate and some saffron. Also vanilla 32 oz. By Friday please.")
    result = read(cfg, NO_AI, text=msg, model_tested=False)
    assert result.source == "text_match" and result.ai_error is None   # no AI is normal, not a failure
    assert got(result) == {"eggs": Decimal(24), "heavy_cream": Decimal(384), "dark_chocolate": Decimal("3.306933933")}
    assert not any(l.needs_check for l in result.lines)
    unclear = " | ".join(result.unclear)
    assert "'AP flour': not sure which product" in unclear       # never guesses which flour
    assert "'some saffron'" in unclear                           # not on the list
    assert "could be weight or fluid ounces" in unclear          # "32 oz" of vanilla
    assert result.needed_by == date(2026, 11, 27)


@pytest.mark.parametrize("text, problem", [
    ("3 cases unsalted butter by Friday", "depends on whose case"),
    ("20 gal milk by Friday", "not sure which product: Whole milk or Oat milk"),
    ("50 unsalted butter by Friday", "50 what?"),
    ("lemons by Friday", "no quantity given"),
])
def test_no_ai_unclear_cases(cfg, text, problem):
    result = read(cfg, NO_AI, text=text)
    assert result.lines == [] and any(problem in u for u in result.unclear), result.unclear


def test_counts_need_no_unit(cfg):
    assert got(read(cfg, NO_AI, text="30 lemons by Friday")) == {"lemons": Decimal(30)}


def test_two_lines_for_one_product_are_added_and_flagged(cfg):
    result = read(cfg, NO_AI, text="20 lb unsalted butter, 10 lb unsalted butter by Friday")
    assert got(result) == {"butter_unsalted": Decimal(30)}
    assert result.lines[0].needs_check


def test_missing_and_double_dates(cfg):
    assert any("No needed-by date" in u for u in read(cfg, NO_AI, text="30 lemons").unclear)
    assert any("more than one date" in u for u in read(cfg, NO_AI, text="30 lemons by Friday or by Monday").unclear)
