import csv
import io
from datetime import date

from app.po import CSV_COLUMNS, POLine, PurchaseOrder, save

PO = PurchaseOrder(
    order_key="R-abc123def4-sysco", supplier_name="Sysco", rep_name="Sam Rivera (fake)",
    rep_email="rep.sysco@example.com", location_name="Factory", deliver_to="Factory",
    needed_by=date(2026, 12, 1), deliver_on=date(2026, 11, 26),
    lines=(POLine("8762019", "FLOUR, ALL-PURPOSE", 2, 2874), POLine("1630766", "BUTTER, UNSALTED", 4, 15095)),
    delivery_fee_cents=2500, confirmation_id="FAKE-SYS-ABC", fake_label="FAKE TEST DATA: invented.",
)


def test_csv_has_the_columns_and_exact_money():
    rows = list(csv.DictReader(io.StringIO(PO.csv_text())))
    assert tuple(rows[0]) == CSV_COLUMNS == ("sku", "description", "cases", "case_price", "line_total",
                                             "deliver_to", "needed_by")
    assert rows[0] == {"sku": "8762019", "description": "FLOUR, ALL-PURPOSE", "cases": "2", "case_price": "28.74",
                       "line_total": "57.48", "deliver_to": "Factory", "needed_by": "2026-12-01"}
    assert rows[1]["line_total"] == "603.80"


def test_email_text():
    text = PO.email_text()
    assert text.startswith("To: Sam Rivera (fake) <rep.sysco@example.com>\nSubject: Purchase order R-abc123def4-sysco")
    assert "deliver this order to Factory on Thu Nov 26 (needed by Tue Dec 1)" in text
    for part in ("$57.48", "$603.80", "Items:", "$661.28", "Delivery:", "$25.00", "Total:", "$686.28",
                 "FAKE-SYS-ABC", "FAKE TEST DATA"):
        assert part in text, part
    assert PO.total_cents == 68628


def test_save_writes_both_files(tmp_path):
    txt, csv_path = save(PO, tmp_path / "purchase_orders")
    assert txt.name == "R-abc123def4-sysco.txt" and csv_path.name == "R-abc123def4-sysco.csv"
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_COLUMNS)
    assert save(PO, tmp_path / "purchase_orders") == [txt, csv_path]  # writing again replaces them
