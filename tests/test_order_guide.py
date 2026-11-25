"""The example real connector: an exported order guide in, order files out. Wired in through config."""
import os
import time
from datetime import date

import pytest

from app.suppliers.base import OrderLine, SupplierOrder, SupplierRejected, SupplierUnavailable
from app.suppliers.mock import build_suppliers, load_connector
from app.suppliers.order_guide import OrderGuideSupplier, dollars_to_cents
from tests.conftest import config_with

GUIDE = 'sku,description,pack,case_price,in_stock\n1630766,"BUTTER, UNSALTED",36/1 LB,150.95,yes\nX9,EGGS,15 DZ,$41.20,no\n'


@pytest.fixture
def guide_cfg(tmp_path):
    (tmp_path / "guide.csv").write_text(GUIDE, encoding="utf-8")

    def mutate(c):
        c["suppliers"][3].update(connector="app.suppliers.order_guide:OrderGuideSupplier",
                                 connector_settings={"order_guide": str(tmp_path / "guide.csv"),
                                                     "outbox": str(tmp_path / "outbox")})
    return config_with(mutate)


def test_one_config_entry_wires_it_in(guide_cfg, tmp_path):
    suppliers = build_suppliers(guide_cfg, ledger_dir=tmp_path / "ledgers")
    assert isinstance(suppliers["supplier4"], OrderGuideSupplier)
    assert type(suppliers["sysco"]).__name__ == "MockSupplier"


def test_reads_the_order_guide(guide_cfg, tmp_path):
    sup = build_suppliers(guide_cfg, ledger_dir=tmp_path)["supplier4"]
    items = sup.get_catalog()
    assert [(i.sku, i.case_price_cents, i.in_stock, i.raw_pack_text) for i in items] == [
        ("1630766", 15095, True, "36/1 LB"), ("X9", 4120, False, "15 DZ")]
    [quote] = sup.get_quotes(["1630766", "NOPE"])
    assert quote.case_price_cents == 15095


def test_prices_are_dated_by_the_export(guide_cfg, tmp_path):
    old = time.time() - 3 * 24 * 3600
    os.utime(tmp_path / "guide.csv", (old, old))
    sup = build_suppliers(guide_cfg, ledger_dir=tmp_path)["supplier4"]
    [quote] = sup.get_quotes(["1630766"])
    assert abs(quote.quoted_at.timestamp() - old) < 2   # so a 3-day-old export fails the freshness check


def test_orders_are_written_once(guide_cfg, tmp_path):
    sup = build_suppliers(guide_cfg, ledger_dir=tmp_path)["supplier4"]
    order = SupplierOrder(supplier_code="supplier4", location_code="court_st", deliver_on=date(2026, 12, 3),
                          lines=[OrderLine(sku="1630766", cases=2, expected_case_price_cents=15095)])
    first = sup.place_order(order, "R-abc-supplier4")
    assert sup.place_order(order, "R-abc-supplier4") == first and first.startswith("SENT-")
    assert (tmp_path / "outbox" / "R-abc-supplier4.csv").read_text(encoding="utf-8").splitlines() == [
        "sku,cases,expected_case_price,deliver_on", "1630766,2,150.95,2026-12-03"]
    bigger = order.model_copy(update={"lines": [OrderLine(sku="1630766", cases=3, expected_case_price_cents=15095)]})
    with pytest.raises(SupplierRejected, match="already used"):
        sup.place_order(bigger, "R-abc-supplier4")


def test_a_broken_order_guide_counts_as_no_answer(guide_cfg, tmp_path):
    (tmp_path / "guide.csv").write_text("sku,description\n1,x\n", encoding="utf-8")
    sup = build_suppliers(guide_cfg, ledger_dir=tmp_path)["supplier4"]
    with pytest.raises(SupplierUnavailable, match="no column"):
        sup.get_catalog()
    (tmp_path / "guide.csv").unlink()
    with pytest.raises(SupplierUnavailable, match="can't read"):
        sup.get_quotes(["1"])


@pytest.mark.parametrize("text, cents", [("150.95", 15095), ("$1,234.5", 123450), ("7", 700)])
def test_dollars_to_cents(text, cents):
    assert dollars_to_cents(text) == cents


@pytest.mark.parametrize("text", ["12.345", "abc", ""])
def test_bad_prices_are_refused(text):
    with pytest.raises(ValueError):
        dollars_to_cents(text)


def test_connector_paths_are_checked():
    with pytest.raises(ValueError):
        load_connector("no_colon_here")
    with pytest.raises(TypeError):
        load_connector("app.config:AppConfig")   # exists, but isn't a Supplier
