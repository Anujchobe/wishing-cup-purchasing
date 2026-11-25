import inspect
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app import clock
from app.config import SupplierFaults
from app.suppliers import fake_world
from app.suppliers.base import (
    OrderLine,
    Supplier,
    SupplierOrder,
    SupplierRejected,
    SupplierTimeout,
    SupplierUnavailable,
    make_order_id,
)
from app.suppliers.mock import MockSupplier, build_suppliers


def make(cfg, code="sysco", **kwargs) -> MockSupplier:
    kwargs.setdefault("sleep", lambda seconds: None)
    return MockSupplier(cfg.supplier(code), seed=cfg.app.random_seed, **kwargs)


def an_order(sup: MockSupplier, cases: int = 1) -> SupplierOrder:
    item = next(i for i in sup.get_catalog() if i.in_stock and i.case_price_cents > 0)
    return SupplierOrder(supplier_code=sup.code, location_code="factory", deliver_on=date(2026, 11, 26),
                         lines=[OrderLine(sku=item.sku, cases=cases, expected_case_price_cents=item.case_price_cents)])


# ------------------------------------------------------------------ the contract

def test_base_class_cannot_be_used_directly():
    with pytest.raises(TypeError):
        Supplier("x", "X")  # abstract


def test_base_class_has_exactly_the_three_functions():
    abstract = Supplier.__abstractmethods__
    assert abstract == {"get_catalog", "get_quotes", "place_order"}
    assert list(inspect.signature(Supplier.get_quotes).parameters) == ["self", "skus"]
    assert list(inspect.signature(Supplier.place_order).parameters) == ["self", "order", "order_id"]


def test_order_id_is_request_plus_supplier():
    assert make_order_id("R-1a2b3c", "sysco") == "R-1a2b3c-sysco"


def test_order_refuses_duplicate_skus():
    with pytest.raises(ValueError):
        SupplierOrder(supplier_code="sysco", location_code="factory", deliver_on=date(2026, 1, 1),
                      lines=[OrderLine(sku="1", cases=1, expected_case_price_cents=100)] * 2)


# ------------------------------------------------------------------ catalog and quotes

def test_build_suppliers_makes_one_per_config_entry(cfg, tmp_path):
    suppliers = build_suppliers(cfg, ledger_dir=tmp_path)
    assert list(suppliers) == ["sysco", "usfoods", "pfg", "supplier4"]
    assert all(s.get_catalog() for s in suppliers.values())


def test_catalog_is_the_same_every_time(cfg):
    assert make(cfg).get_catalog() == make(cfg).get_catalog()


def test_seed_changes_prices(cfg):
    a = MockSupplier(cfg.supplier("sysco"), seed=1).get_catalog()
    b = MockSupplier(cfg.supplier("sysco"), seed=2).get_catalog()
    assert [i.sku for i in a] == [i.sku for i in b]
    assert [i.case_price_cents for i in a] != [i.case_price_cents for i in b]


def test_catalog_money_is_whole_cents(cfg):
    for code in ("sysco", "usfoods", "pfg", "supplier4"):
        for item in make(cfg, code).get_catalog():
            assert type(item.case_price_cents) is int


def test_quotes_skip_unknown_and_duplicate_skus(cfg):
    sup = make(cfg)
    skus = [i.sku for i in sup.get_catalog()][:3]
    quotes = sup.get_quotes([skus[0], "NO-SUCH-SKU", skus[1], skus[0], skus[2]])
    assert [q.sku for q in quotes] == skus
    assert all(q.quoted_at.tzinfo == clock.BOSTON for q in quotes)


def test_planted_problems_show_up_in_live_quotes(cfg):
    cream = fake_world.offer("pfg", "heavy_cream").sku
    matcha = fake_world.offer("sysco", "matcha_powder").sku
    assert make(cfg, "pfg").get_quotes([cream])[0].case_price_cents == 0
    assert make(cfg, "sysco").get_quotes([matcha])[0].in_stock is False


# ------------------------------------------------------------------ ordering

@settings(max_examples=50)
@given(order_id=st.text(min_size=1, max_size=60))
def test_same_order_id_gives_same_confirmation_and_books_once(cfg, order_id):
    sup = make(cfg)
    order = an_order(sup)
    first = sup.place_order(order, order_id)
    second = sup.place_order(order, order_id)
    assert first == second and first.startswith("FAKE-")
    assert sup.placements == 1


def test_reused_order_id_with_different_order_is_rejected(cfg):
    sup = make(cfg)
    sup.place_order(an_order(sup, cases=1), "R-x-sysco")
    with pytest.raises(SupplierRejected, match="already used"):
        sup.place_order(an_order(sup, cases=2), "R-x-sysco")
    assert sup.placements == 1


def test_ledger_survives_a_restart(cfg, tmp_path):
    ledger = tmp_path / "sysco.json"
    before = make(cfg, ledger_path=ledger)
    confirmation = before.place_order(an_order(before), "R-restart-sysco")
    after = make(cfg, ledger_path=ledger)  # a new process, same supplier
    assert after.place_order(an_order(after), "R-restart-sysco") == confirmation
    assert after.placements == 1


@pytest.mark.parametrize("problem", ["wrong supplier", "unknown sku", "out of stock"])
def test_bad_orders_are_rejected_and_not_booked(cfg, problem):
    sup = make(cfg)
    if problem == "wrong supplier":
        order = an_order(make(cfg, "usfoods"))
    else:
        sku = "NO-SUCH-SKU" if problem == "unknown sku" else fake_world.offer("sysco", "matcha_powder").sku
        order = SupplierOrder(supplier_code="sysco", location_code="court_st", deliver_on=date(2026, 11, 26),
                              lines=[OrderLine(sku=sku, cases=1, expected_case_price_cents=100)])
    with pytest.raises(SupplierRejected):
        sup.place_order(order, "R-bad-sysco")
    assert sup.placements == 0


# ------------------------------------------------------------------ fake problems

def test_down_supplier_never_answers(cfg):
    sup = make(cfg, faults=SupplierFaults(down=True))
    for call in (sup.get_catalog, lambda: sup.get_quotes(["1"])):
        with pytest.raises(SupplierUnavailable) as err:
            call()
        assert err.value.retryable


def test_slow_supplier_answers_late(cfg):
    slept = []
    sup = make(cfg, faults=SupplierFaults(slow=True), slow_seconds=0.5, timeout_seconds=2.0, sleep=slept.append)
    assert sup.get_catalog()
    assert slept == [0.5]


def test_very_slow_supplier_times_out(cfg):
    slept = []
    sup = make(cfg, faults=SupplierFaults(slow=True), slow_seconds=5.0, timeout_seconds=2.0, sleep=slept.append)
    with pytest.raises(SupplierTimeout):
        sup.get_catalog()
    assert slept == [2.0]


def test_flaky_supplier_always_failing_books_nothing(cfg):
    sup = make(cfg)
    order = an_order(sup)
    sup.faults = SupplierFaults(flaky=True)
    sup.flaky_failure_rate = 1.0
    with pytest.raises(SupplierUnavailable):
        sup.get_quotes(["1"])
    with pytest.raises(SupplierUnavailable):
        sup.place_order(order, "R-flaky-sysco")
    assert sup.placements == 0


def test_flaky_supplier_can_book_then_lose_the_reply_and_retry_is_safe(cfg):
    sup = make(cfg, flaky_failure_rate=0.5)
    order = an_order(sup)
    sup.faults = SupplierFaults(flaky=True)
    for n in range(200):
        order_id = f"R-{n}-sysco"
        try:
            sup.place_order(order, order_id)
        except SupplierUnavailable:
            if sup.has_booked(order_id):
                break  # booked, but we were told it failed
    else:
        pytest.fail("never hit the booked-but-reply-lost case")
    booked = sup.placements
    sup.faults = SupplierFaults()  # supplier recovers; we retry with the same id
    assert sup.place_order(order, order_id) == fake_world.confirmation_for("sysco", order_id)
    assert sup.placements == booked  # nothing new was booked


def test_flaky_failures_repeat_exactly(cfg):
    def outcomes():
        sup = make(cfg, faults=SupplierFaults(flaky=True), flaky_failure_rate=0.3)
        result = []
        for _ in range(40):
            try:
                sup.get_quotes(["1"])
                result.append("ok")
            except SupplierUnavailable:
                result.append("fail")
        return result

    first = outcomes()
    assert first == outcomes() and "ok" in first and "fail" in first
