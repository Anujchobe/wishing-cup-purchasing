"""Each safety check on a hand-built snapshot: a passing case, then each way it can fail."""
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from app import checks, clock
from app.checks import LineView, OrderView, RequestView
from app.optimizer import SupplierTerms

NOW = clock.at_boston(date(2026, 11, 24)) + timedelta(hours=10)   # Tuesday 10:00, Boston
SYSCO = SupplierTerms("sysco", "Sysco", min_order_cents=50000, delivery_fee_cents=2500,
                      delivery_weekdays=("Tue", "Thu", "Sat"), lead_time_days=1, serves=frozenset({"factory"}))
PFG = SupplierTerms("pfg", "PFG", min_order_cents=75000, delivery_fee_cents=0, delivery_weekdays=("Tue", "Fri"),
                    lead_time_days=3, serves=frozenset({"factory"}))


def line(**changes) -> LineView:
    base = LineView(line_id=1, product_code="butter_unsalted", product_name="Unsalted butter", base_unit="lb",
                    qty_needed=Decimal(65), fill_status="ASSIGNED", supplier="sysco", sku="123", pack_size=Decimal(36),
                    cases=2, case_price_cents=30000, line_cost_cents=60000, has_quote=True,
                    quote_price_cents=30000, quote_in_stock=True, quoted_at=NOW - timedelta(hours=1),
                    most_ordered_before=Decimal(70))
    return replace(base, **changes)


def view(*lines: LineView, **changes) -> RequestView:
    lines = lines or (line(),)
    base = RequestView(request_id=1, location_code="factory", location_name="Factory", needed_by=date(2026, 12, 1),
                       now=NOW, lines=tuple(lines), suppliers={"sysco": SYSCO, "pfg": PFG}, plan_total_cents=62500)
    return replace(base, **changes)


def result(check, v) -> checks.CheckResult:
    return check(v)


def test_a_good_plan_passes_everything():
    results = checks.run_checks(view(), "approval")
    assert [r.name for r in results if not r.passed] == []
    assert len(results) == 10
    assert checks.summary(results) == "All safety checks passed."


@pytest.mark.parametrize("v, text", [
    (view(line(fill_status="OPEN", supplier=None)), "isn't on any supplier's list"),
    (view(line(fill_status="CANT_FILL", supplier=None)), "Nothing can be ordered"),
    (view(lines=()), "no items"),
])
def test_one_list(v, text):
    r = checks.check_one_list(v)
    assert not r.passed and any(text in x for x in r.reasons)


def test_cant_fill_items_are_allowed_next_to_ordered_ones():
    assert checks.check_one_list(view(line(), line(line_id=2, product_code="x", fill_status="CANT_FILL", supplier=None))).passed


@pytest.mark.parametrize("changes, text", [
    ({"cases": 1, "line_cost_cents": 30000}, "36 lb, less than the 65 lb needed"),
    ({"pack_size": None}, "case size at Sysco is unknown"),
])
def test_cases_cover(changes, text):
    r = checks.check_cases_cover(view(line(**changes), plan_total_cents=None))
    assert not r.passed and text in r.reasons[0]


@pytest.mark.parametrize("v, text", [
    (view(line(line_cost_cents=59999)), "2 x $300.00 is $600.00, but the line says $599.99"),
    (view(line(quote_price_cents=31000)), "the supplier's price is $310.00"),
    (view(plan_total_cents=60000), "plan total shown ($600.00) doesn't match the lines ($625.00)"),
    (view(orders=(OrderView("sysco", 59000, 2500, 61500, "PENDING"),)), "order says $590.00"),
    (view(orders=(OrderView("sysco", 60000, 2500, 60000, "PENDING"),)), "isn't items plus delivery"),
])
def test_totals(v, text):
    r = checks.check_totals(v)
    assert not r.passed and any(text in x for x in r.reasons), r.reasons


def test_minimums():
    r = checks.check_minimums(view(line(cases=1, line_cost_cents=30000), plan_total_cents=None))
    assert not r.passed and "Sysco: the order is $300.00, below its $500.00 minimum. Add $200.00" in r.reasons[0]


@pytest.mark.parametrize("v, text", [
    (view(needed_by=date(2026, 11, 24)), "can deliver Thu Nov 26 at the earliest, after the needed-by date Tue Nov 24"),
    (view(needed_by=date(2026, 11, 20)), "has passed"),
    (view(needed_by=None), "no needed-by date"),
    (view(location_code="court_st", location_name="Court St Cafe"), "Sysco doesn't deliver to Court St Cafe"),
])
def test_delivery(v, text):
    r = checks.check_delivery(v)
    assert not r.passed and any(text in x for x in r.reasons), r.reasons


@pytest.mark.parametrize("changes, text", [
    ({"quote_in_stock": False}, "out of stock at Sysco"),
    ({"item_exists": False}, "no longer lists SKU 123"),
    ({"has_quote": False, "quoted_at": None, "quote_price_cents": None}, "no price from Sysco"),
    ({"product_active": False}, "no longer on our product list"),
])
def test_in_stock(changes, text):
    r = checks.check_in_stock(view(line(**changes)))
    assert not r.passed and text in " ".join(r.reasons)


def test_fresh_prices():
    assert checks.check_fresh_prices(view(line(quoted_at=NOW - timedelta(hours=23, minutes=59)))).passed
    r = checks.check_fresh_prices(view(line(quoted_at=NOW - timedelta(hours=25))))
    assert not r.passed and "25 hours old, older than the 24-hour limit" in r.reasons[0]


def test_price_sanity():
    aside = line(price_set_aside="the price went up 70% since last week ($10.00 to $17.00)")
    r = checks.check_price_sanity(view(aside))
    assert not r.passed and "went up 70%" in r.reasons[0]
    assert checks.check_price_sanity(view(replace(aside, price_accepted=True))).passed


@pytest.mark.parametrize("changes, text", [
    ({"qty_needed": Decimal(211), "cases": 6, "line_cost_cents": 180000}, "more than 3 times the most Factory ever ordered (70 lb)"),
    ({"most_ordered_before": None}, "never ordered it before"),
])
def test_quantities(changes, text):
    r = checks.check_quantities(view(line(**changes), plan_total_cents=None))
    assert not r.passed and text in r.reasons[0]


def test_quantity_exactly_three_times_is_fine_and_confirmed_is_fine():
    assert checks.check_quantities(view(line(qty_needed=Decimal(210), cases=6, line_cost_cents=180000),
                                        plan_total_cents=None)).passed
    assert checks.check_quantities(view(line(most_ordered_before=None, qty_confirmed=True))).passed


def test_flags_cleared():
    r = checks.check_flags_cleared(view(line(line_needs_check=True, line_reason="model not tested",
                                             match_needs_check=True, match_reason="text match score 80 is below 90")))
    assert not r.passed and len(r.reasons) == 2
    assert "model not tested" in r.reasons[0] and "product match needs a person" in r.reasons[1]


def test_summary_lists_every_reason():
    results = checks.run_checks(view(line(quote_in_stock=False, quoted_at=NOW - timedelta(days=2))), "ordering")
    text = checks.summary(results)
    assert text.startswith("Blocked:") and "out of stock" in text and "48 hours old" in text


def test_unknown_stage_is_refused():
    with pytest.raises(ValueError):
        checks.run_checks(view(), "whenever")


def test_view_helpers():
    v = view(line(), line(line_id=2, product_code="eggs", supplier="pfg"))
    assert list(v.lists()) == ["pfg", "sysco"] and v.name_of("pfg") == "PFG" and v.today == date(2026, 11, 24)
    assert isinstance(v.now, datetime)
