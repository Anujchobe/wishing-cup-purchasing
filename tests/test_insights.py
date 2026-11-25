from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app import insights
from app.insights import MatchItem, UnitPrice, classify_change, email_problems, match_candidates, template_email

D = Decimal
T0 = datetime(2026, 10, 26, 11, tzinfo=timezone.utc)


def weekly(*prices):
    return [(T0 + timedelta(weeks=n), D(p)) for n, p in enumerate(prices)]


# ------------------------------------------------------------------ price watch

@pytest.mark.parametrize("changes, kind, up", [
    ({"A": D("0.15"), "B": D("0.12")}, "market", ("A", "B")),
    ({"A": D("0.18"), "B": D("0.01"), "C": D("0.0")}, "one_supplier", ("A",)),
    ({"A": D("0.18"), "B": D("0.11"), "C": D("0.0")}, "some_suppliers", ("A", "B")),
    ({"A": D("0.05"), "B": D("0.02")}, None, ()),
    ({"A": D("0.30")}, None, ("A",)),           # one supplier can't tell market from supplier
    ({"A": D("0.10"), "B": D("0.10")}, None, ()),  # "more than" 10%
])
def test_classify(changes, kind, up):
    assert classify_change(changes, D("0.10")) == (kind, up)


def test_change_over_compares_with_n_weeks_earlier():
    assert insights.change_over(weekly(100, 100, 100, 100, 115), 4) == D("0.15")
    assert insights.change_over(weekly(100, 115), 4) is None   # not enough history
    assert insights.change_over([], 4) is None


def test_price_watch_rows():
    series = {"butter": {"Sysco": weekly(100, 100, 115, 115, 116), "US Foods": weekly(90, 90, 104, 104, 104)},
              "eggs": {"Sysco": weekly(50, 50, 50, 50, 51), "US Foods": weekly(50, 50, 59, 59, 59)},
              "flour": {"Sysco": weekly(30, 30, 30, 30, 30), "US Foods": weekly(28, 28, 28, 28, 29)}}
    rows = insights.price_watch_rows(series, {"butter": "Butter", "eggs": "Eggs"}, 4, D("0.10"))
    assert [(r.product_name, r.kind) for r in rows] == [("Butter", "market"), ("Eggs", "one_supplier")]
    assert "Only US Foods raised it" in rows[1].summary and "+18%" in rows[1].summary


# ------------------------------------------------------------------ price match

def up(supplier, cents_per_unit):
    return UnitPrice(supplier, D(cents_per_unit), 0, D(1))


def test_match_candidates_uses_the_threshold_per_unit():
    latest = {"butter": {"sysco": up("Sysco", 381), "usfoods": up("US Foods", 347), "pfg": up("PFG", 360)},
              "flour": {"sysco": up("Sysco", 100), "pfg": up("PFG", 96)},      # 4%: not enough
              "salt": {"usfoods": up("US Foods", 10)}}                         # main supplier doesn't sell it
    items = match_candidates(latest, {"butter": "Unsalted butter"}, {"butter": "lb"}, "sysco", D("0.05"))
    assert [(i.product_code, i.other.supplier) for i in items] == [("butter", "US Foods")]
    assert (items[0].main_label, items[0].other_label, insights.pct(items[0].cheaper_by)) == ("$3.81/lb", "$3.47/lb", "9%")


ITEM = MatchItem("butter", "Unsalted butter", "lb", up("Sysco", 381), up("US Foods", 347), D(34) / D(381))


def test_template_email_has_every_number():
    subject, body = template_email("Sysco", "Sam", [ITEM])
    assert subject == "Price match request: 1 item"
    assert email_problems(body, [ITEM]) == []
    assert "$3.81/lb" in body and "$3.47/lb" in body and "9%" in body and "Sam" in body


@pytest.mark.parametrize("body, problem", [
    ("Could you match $3.47/lb on butter? Thanks", "the price $3.81/lb is missing"),
    ("We pay $3.81/lb and were offered $3.47/lb, 9% less. Could you do $3.40?", "the amount $3.40"),
    ("We pay $3.81/lb and were offered $3.47/lb, 12% less.", "the percentage 12%"),
])
def test_ai_email_numbers_must_come_from_the_facts(body, problem):
    assert any(problem in p for p in email_problems(body, [ITEM]))


def test_good_ai_wording_passes():
    body = "Hi Sam, we pay $3.81/lb for butter and have been offered $3.47/lb, about 9% lower. Could you match it?"
    assert email_problems(body, [ITEM]) == []


def test_unit_price_label():
    assert insights.unit_price_label(D("381.4"), "lb") == "$3.81/lb"
    assert insights.unit_price_label(D("12.5"), "fl_oz") == "$0.125/fl oz"
    assert insights.unit_price_label(D("123456"), "lb") == "$1,234.56/lb"
