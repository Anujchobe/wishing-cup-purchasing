from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.pricing import cases_to_buy, format_cents, format_qty, item_cost
from app.units import pack_base_qty, to_base

D = Decimal


@pytest.mark.parametrize(
    "need, case, cases",
    [("10", "4", 3), ("12", "4", 3), ("12.01", "4", 4), ("0", "4", 0), ("0.01", "50", 1), ("50", "50", 1),
     ("100", "22.046226218", 5)],
)
def test_rounds_up_to_whole_cases(need, case, cases):
    assert cases_to_buy(D(need), D(case)) == cases


@given(need=st.decimals(min_value=0, max_value=10**6, places=3),
       case=st.decimals(min_value=D("0.001"), max_value=10**5, places=3))
def test_cases_are_the_fewest_that_cover_the_need(need, case):
    cases = cases_to_buy(need, case)
    assert cases * case >= need
    assert cases == 0 or (cases - 1) * case < need


def test_item_cost_reports_everything():
    cost = item_cost(D(22), D(50), 2874)  # need 22 lb, 50 lb bag at $28.74
    assert (cost.cases, cost.cost_cents, cost.bought, cost.extra) == (1, 2874, D(50), D(28))
    assert cost.price_per_base_cents == D("57.4800")


def test_cost_is_for_whole_cases_not_the_exact_need():
    cost = item_cost(D(37), D(36), 10000)  # 37 lb of butter, 36 lb cases
    assert (cost.cases, cost.cost_cents, cost.extra) == (2, 20000, D(35))


def test_metric_need_and_metric_case_match_exactly():
    """10 kg needed, 10 kg case: both go through the same rounding, so it's 1 case, nothing extra."""
    need = to_base(D(10), "kg", "lb")
    cost = item_cost(need, pack_base_qty("1/10 KG", "lb"), 5000)
    assert (cost.cases, cost.extra) == (1, D(0))


@given(need=st.decimals(min_value=0, max_value=10**4, places=2),
       case=st.decimals(min_value=D("0.5"), max_value=500, places=2),
       price=st.integers(1, 10**6))
def test_cost_is_cases_times_price_and_extra_is_never_negative(need, case, price):
    cost = item_cost(need, case, price)
    assert cost.cost_cents == cost.cases * price
    assert type(cost.cost_cents) is int
    assert D(0) <= cost.extra < case


@pytest.mark.parametrize("price", [0, -100])
def test_zero_or_negative_price_is_refused(price):
    with pytest.raises(ValueError, match="flagged"):
        item_cost(D(1), D(1), price)


def test_bad_inputs_are_refused():
    with pytest.raises(TypeError):
        item_cost(D(1), D(1), 9.99)       # money must be whole cents
    with pytest.raises(TypeError):
        cases_to_buy(1.5, D(1))           # no floats
    with pytest.raises(ValueError):
        cases_to_buy(D(1), D(0))          # empty case
    with pytest.raises(ValueError):
        cases_to_buy(D(-1), D(1))


def test_formatting():
    assert format_cents(12345) == "$123.45"
    assert format_cents(5) == "$0.05"
    assert format_cents(-150) == "-$1.50"
    assert format_cents(123456789) == "$1,234,567.89"
    assert format_qty(D("12.500"), "lb") == "12.5 lb"
    assert format_qty(D("384"), "fl_oz") == "384 fl oz"
