import itertools
import math
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app import clock, optimizer
from app.optimizer import Need, Offer, Problem, SolveResult, SupplierTerms, make_plans

D = Decimal
ORDER_DAY = date(2026, 11, 24)  # a Tuesday


def one_case(supplier, item, price, in_stock=True, case=D(1), sku=None):
    return Offer(supplier, item, sku or f"{supplier}-{item}", case, price, in_stock)


def spec_example() -> Problem:
    """From the brief. A: no minimum, $5 fee. B: $15 minimum, no fee."""
    a = SupplierTerms("A", "Supplier A", min_order_cents=0, delivery_fee_cents=500)
    b = SupplierTerms("B", "Supplier B", min_order_cents=1500, delivery_fee_cents=0)
    prices = {"x": (1000, 1100), "y": (1000, 500), "z": (100, 900)}
    offers = [one_case(s, item, p) for item, pair in prices.items() for s, p in zip("AB", pair)]
    return Problem(needs=tuple(Need(i, D(1)) for i in prices), offers=tuple(offers), suppliers=(a, b),
                   order_date=ORDER_DAY)


# ------------------------------------------------------------------ the example from the brief

def test_spec_example_best_plan_costs_22_dollars():
    plans = make_plans(spec_example())
    assert plans.best.total_cents == 2200
    assert plans.best.assignment == {"x": "B", "y": "B", "z": "A"}
    assert plans.best.solved_by == "solver" and plans.best.meets_minimums
    by_supplier = {sl.supplier: sl for sl in plans.best.lists}
    assert (by_supplier["A"].subtotal_cents, by_supplier["A"].total_cents) == (100, 600)
    assert (by_supplier["B"].subtotal_cents, by_supplier["B"].total_cents) == (1600, 1600)


def test_spec_example_other_plans_savings_and_reasons():
    plans = make_plans(spec_example())
    # (a) cheapest per item ignores B's minimum: B would only get y ($5 < $15)
    assert plans.simple.assignment == {"x": "A", "y": "B", "z": "A"}
    assert plans.simple.total_cents == 2100 and not plans.simple.meets_minimums
    assert any("$10.00 short" in w for w in plans.simple.warnings)
    # (b) all from B ($25) beats all from A ($21 + $5 fee = $26)
    assert plans.one_supplier.assignment == {"x": "B", "y": "B", "z": "B"}
    assert plans.one_supplier.total_cents == 2500
    assert plans.savings_vs_one_supplier_cents == 300
    # only x differs between (a) and (c), because of B's minimum
    [diff] = plans.differences
    assert (diff.item, diff.simple_supplier, diff.best_supplier) == ("x", "A", "B")
    assert "$1.00 cheaper" in diff.reason and "Supplier B at $5.00" in diff.reason
    assert "$15.00 minimum" in diff.reason


# ------------------------------------------------------------------ explanations

def test_reason_delivery_fee():
    a = SupplierTerms("A", "Supplier A", delivery_fee_cents=1000)
    b = SupplierTerms("B", "Supplier B")
    p = Problem((Need("x", D(1)), Need("y", D(1))),
                (one_case("A", "x", 900), one_case("B", "x", 1000), one_case("B", "y", 2000)), (a, b),
                order_date=ORDER_DAY)
    plans = make_plans(p)
    assert plans.best.assignment == {"x": "B", "y": "B"}
    [diff] = plans.differences
    assert "$10.00 delivery fee" in diff.reason and "go up by $9.00" in diff.reason


def test_reason_new_suppliers_minimum():
    a = SupplierTerms("A", "Supplier A", min_order_cents=5000)
    b = SupplierTerms("B", "Supplier B")
    p = Problem((Need("x", D(1)),), (one_case("A", "x", 900), one_case("B", "x", 1000)), (a, b),
                order_date=ORDER_DAY)
    [diff] = make_plans(p).differences
    assert "would be $9.00, below its $50.00 minimum" in diff.reason


# ------------------------------------------------------------------ details

def test_cheapest_sku_is_used_when_a_supplier_has_several():
    s = SupplierTerms("A", "Supplier A")
    p = Problem((Need("flour", D(60)),),
                (Offer("A", "flour", "small", D(25), 1500), Offer("A", "flour", "big", D(50), 2600)), (s,),
                order_date=ORDER_DAY)
    [line] = make_plans(p).best.lists[0].lines
    assert (line.sku, line.cost.cases, line.cost_cents) == ("small", 3, 4500)  # 3x25 beats 2x50


def test_cant_fill_lists_the_reasons():
    sups = (
        SupplierTerms("A", "Supplier A"),
        SupplierTerms("B", "Supplier B", working=False),
        SupplierTerms("C", "Supplier C", lead_time_days=5),
        SupplierTerms("D", "Supplier D", serves=frozenset({"factory"})),
        SupplierTerms("E", "Supplier E"),
    )
    offers = (one_case("A", "x", 100, in_stock=False), one_case("B", "x", 100), one_case("C", "x", 100),
              one_case("D", "x", 100), one_case("E", "x", 0))
    p = Problem((Need("x", D(1)), Need("y", D(1))), offers, sups, needed_by=ORDER_DAY + timedelta(days=2),
                order_date=ORDER_DAY, location="court_st")
    plans = make_plans(p)
    reasons = {c.item: c.reason for c in plans.best.cant_fill}
    assert set(reasons) == {"x", "y"}
    for text in ("out of stock at Supplier A", "Supplier B didn't answer", "Supplier C can't deliver until Sun Nov 29",
                 "Supplier D doesn't deliver to this location", "Supplier E has it at an invalid price"):
        assert text in reasons["x"]
    assert "Supplier A doesn't sell it" in reasons["y"]
    assert plans.best.lists == () and plans.best.total_cents == 0
    assert any("Supplier B didn't answer" in w for w in plans.warnings)


def test_delivery_date():
    s = SupplierTerms("A", "A", delivery_weekdays=("Mon", "Thu"), lead_time_days=1)
    assert s.earliest_delivery(ORDER_DAY) == date(2026, 11, 26)            # Tue + 1 = Wed -> Thu
    assert s.earliest_delivery(date(2026, 11, 26)) == date(2026, 11, 30)   # Thu + 1 = Fri -> Mon
    assert SupplierTerms("B", "B", delivery_weekdays=()).earliest_delivery(ORDER_DAY) is None


def test_terms_from_config(cfg):
    t = SupplierTerms.from_config(cfg.supplier("pfg"))
    assert (t.min_order_cents, t.delivery_fee_cents, t.lead_time_days) == (75000, 0, 3)
    assert t.serves == frozenset({"factory"}) and t.delivery_weekdays == ("Tue", "Fri")


def test_one_supplier_needs_its_minimum():
    s = SupplierTerms("A", "Supplier A", min_order_cents=5000)
    p = Problem((Need("x", D(1)),), (one_case("A", "x", 100),), (s,), order_date=ORDER_DAY)
    plans = make_plans(p)
    assert plans.one_supplier is None and "below its $50.00 minimum" in plans.one_supplier_note
    assert plans.savings_vs_one_supplier_cents is None
    assert plans.best.solved_by == "fallback" and not plans.best.meets_minimums
    assert "can't be ordered until the minimums are met" in plans.best.warnings[0]


@pytest.mark.parametrize("bad", [
    lambda: Problem((Need("x", D(1)), Need("x", D(2))), (), ()),
    lambda: Problem((Need("x", 1.5),), (), ()),
    lambda: Problem((Need("x", D(0)),), (), ()),
    lambda: Problem((Need("x", D(1)),), (one_case("ghost", "x", 100),), ()),
])
def test_bad_problems_are_refused(bad):
    with pytest.raises((TypeError, ValueError)):
        bad()


# ------------------------------------------------------------------ when the solver fails

def failing_solver(model, time_limit):
    raise RuntimeError("CBC not found")


def test_solver_crash_falls_back_with_a_warning():
    plans = make_plans(spec_example(), solve=failing_solver)
    assert plans.best.solved_by == "fallback"
    assert "CBC not found" in plans.best.warnings[0]
    # fallback = better of one-supplier (B, $25) and greedy (drops B -> all A, $26)
    assert plans.best.total_cents == 2500 and plans.best.meets_minimums


def test_solver_nonsense_is_not_trusted():
    def zeros(model, time_limit):
        return SolveResult({v: 0 for v in model.variables}, "Optimal", True)

    plans = make_plans(spec_example(), solve=zeros)
    assert plans.best.solved_by == "fallback" and "didn't pass the checks" in plans.best.warnings[0]


def test_time_limit_hit_keeps_answer_but_warns():
    def unproven(model, time_limit):
        result = optimizer.solve_with_pulp(model, time_limit)
        return SolveResult(result.values, result.status, proven_optimal=False)

    plans = make_plans(spec_example(), solve=unproven)
    assert plans.best.solved_by == "solver" and plans.best.total_cents == 2200
    assert "10-second limit" in plans.best.warnings[0]


def test_pulp_and_cbc_work_on_this_machine():
    ok, message = optimizer.solver_check()
    assert ok, message


def test_solver_gets_the_10_second_limit():
    seen = []

    def spy(model, time_limit):
        seen.append(time_limit)
        return optimizer.solve_with_pulp(model, time_limit)

    make_plans(spec_example(), solve=spy)
    assert seen == [10]


# ------------------------------------------------------------------ random small cases vs brute force

WEEKDAY_SETS = [clock.WEEKDAYS, clock.WEEKDAYS, ("Mon", "Wed", "Fri"), ("Tue", "Thu"), ("Sat",)]


@st.composite
def problems(draw):
    """Up to 6 items and 3 suppliers, with enough overlap that plans really differ,
    plus every way an item can become unfillable (not sold, out of stock, $0, late, wrong place, down)."""
    suppliers = tuple(
        SupplierTerms(
            code=f"s{j}", name=f"Supplier {j}",
            min_order_cents=draw(st.sampled_from([0, 0, 1000, 2500, 5000, 12000])),
            delivery_fee_cents=draw(st.sampled_from([0, 300, 500, 1500, 4000])),
            delivery_weekdays=draw(st.sampled_from(WEEKDAY_SETS)),
            lead_time_days=draw(st.integers(0, 2)),
            serves=draw(st.sampled_from([None, None, None, frozenset({"cafe"}), frozenset({"factory"})])),
            working=draw(st.sampled_from([True] * 5 + [False])),
        )
        for j in range(draw(st.integers(1, 3)))
    )
    needs = tuple(Need(f"i{k}", draw(st.decimals(min_value=D("0.25"), max_value=D(60), places=2)))
                  for k in range(draw(st.integers(1, 6))))
    offers = []
    for need in needs:
        for s in suppliers:
            for m in range(draw(st.sampled_from([0, 1, 1, 1, 1, 2]))):
                offers.append(Offer(
                    s.code, need.item, f"{s.code}-{need.item}-{m}",
                    case_size=draw(st.sampled_from([D(1), D("2.5"), D(6), D(12), D(50)])),
                    case_price_cents=draw(st.integers(0, 6000)),
                    in_stock=draw(st.sampled_from([True] * 5 + [False])),
                ))
    return Problem(needs, tuple(offers), suppliers,
                   needed_by=ORDER_DAY + timedelta(days=draw(st.integers(1, 6))), order_date=ORDER_DAY,
                   location=draw(st.sampled_from([None, "cafe"])))


def brute_force(p: Problem):
    """Independent of app code: work out who can supply what, then try every choice."""
    eligible = {}
    for need in p.needs:
        eligible[need.item] = {}
        for s in p.suppliers:
            if not s.working or (p.location and s.serves is not None and p.location not in s.serves):
                continue
            days = [p.order_date + timedelta(days=s.lead_time_days + k) for k in range(7)]
            arrive = next((d for d in days if clock.WEEKDAYS[d.weekday()] in s.delivery_weekdays), None)
            if arrive is None or arrive > p.needed_by:
                continue
            costs = [math.ceil(Fraction(need.qty) / Fraction(o.case_size)) * o.case_price_cents
                     for o in p.offers if (o.supplier, o.item) == (s.code, need.item)
                     and o.in_stock and o.case_price_cents > 0]
            if costs:
                eligible[need.item][s.code] = min(costs)
    terms = {s.code: s for s in p.suppliers}
    fillable = [i for i, e in eligible.items() if e]
    best = None
    for choice in itertools.product(*[sorted(eligible[i].items()) for i in fillable]):
        subtotal = {}
        for s, cost in choice:
            subtotal[s] = subtotal.get(s, 0) + cost
        if any(subtotal[s] < terms[s].min_order_cents for s in subtotal):
            continue
        total = sum(subtotal.values()) + sum(terms[s].delivery_fee_cents for s in subtotal)
        best = total if best is None else min(best, total)
    return best, eligible


def assert_complete(plan, problem, eligible):
    placed = [line.item for sl in plan.lists for line in sl.lines]
    cant = [c.item for c in plan.cant_fill]
    assert len(placed) == len(set(placed)), "an item is on two lists"
    assert set(placed).isdisjoint(cant)
    assert sorted(placed + cant) == sorted(n.item for n in problem.needs)
    assert set(cant) == {i for i, e in eligible.items() if not e}
    for sl in plan.lists:
        assert sl.subtotal_cents == sum(line.cost_cents for line in sl.lines)
        assert sl.total_cents == sl.subtotal_cents + sl.delivery_fee_cents
        for line in sl.lines:
            assert line.cost_cents == eligible[line.item][sl.supplier]
    assert plan.total_cents == sum(sl.total_cents for sl in plan.lists)


@settings(max_examples=150, deadline=None)
@given(problems())
def test_best_plan_equals_trying_every_choice(problem):
    plans = make_plans(problem)
    best_total, eligible = brute_force(problem)
    assert_complete(plans.best, problem, eligible)
    if best_total is None:  # no choice meets every minimum
        assert plans.best.solved_by == "fallback" and not plans.best.meets_minimums
    else:
        assert plans.best.solved_by in ("solver", "rule")
        assert plans.best.meets_minimums
        assert plans.best.total_cents == best_total


@settings(max_examples=75, deadline=None)
@given(problems())
def test_best_never_costs_more_than_one_supplier(problem):
    plans = make_plans(problem)
    if plans.one_supplier is not None:
        assert plans.best.total_cents <= plans.one_supplier.total_cents
        assert plans.savings_vs_one_supplier_cents == plans.one_supplier.total_cents - plans.best.total_cents


@settings(max_examples=75, deadline=None)
@given(problems())
def test_every_item_is_placed_once_or_cant_fill(problem):
    plans = make_plans(problem)
    _, eligible = brute_force(problem)
    for plan in (plans.simple, plans.one_supplier, plans.best):
        if plan is not None:
            assert_complete(plan, problem, eligible)
    for item, suppliers in eligible.items():
        if suppliers:  # (a) really is the cheapest per item
            assert plans.simple.assignment[item] in suppliers
            assert min(suppliers.values()) == eligible[item][plans.simple.assignment[item]]
    differing = {i for i in plans.simple.assignment if plans.simple.assignment[i] != plans.best.assignment[i]}
    assert {d.item for d in plans.differences} == differing
    assert all(d.reason for d in plans.differences)


@settings(max_examples=100, deadline=None)
@given(problems())
def test_fallback_is_always_complete_and_never_beats_the_true_best(problem):
    plans = make_plans(problem, solve=failing_solver)
    best_total, eligible = brute_force(problem)
    assert_complete(plans.best, problem, eligible)
    assert plans.best.solved_by in ("fallback", "rule")
    if plans.best.meets_minimums and best_total is not None:
        assert plans.best.total_cents >= best_total
