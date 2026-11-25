"""Purchase orders: one per supplier order, as email text and as a CSV.

CSV columns: sku, description, cases, case_price, line_total, deliver_to, needed_by.
Money in the CSV is plain dollars with two decimals ("28.74"), built from whole cents, so a
spreadsheet reads it as a number and nothing is ever rounded through a float.

The email is written for a person to send (or keep as a record): the order itself already went to
the supplier through its connection, under the same order ID.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.pricing import format_cents
from app.textparse import day_label

CSV_COLUMNS = ("sku", "description", "cases", "case_price", "line_total", "deliver_to", "needed_by")


@dataclass(frozen=True)
class POLine:
    sku: str
    description: str
    cases: int
    case_price_cents: int

    @property
    def line_total_cents(self) -> int:
        return self.cases * self.case_price_cents


@dataclass(frozen=True)
class PurchaseOrder:
    order_key: str
    supplier_name: str
    rep_name: str
    rep_email: str
    location_name: str
    deliver_to: str
    needed_by: date
    deliver_on: date | None
    lines: tuple[POLine, ...]
    delivery_fee_cents: int
    confirmation_id: str | None
    fake_label: str = ""

    @property
    def subtotal_cents(self) -> int:
        return sum(line.line_total_cents for line in self.lines)

    @property
    def total_cents(self) -> int:
        return self.subtotal_cents + self.delivery_fee_cents

    @property
    def subject(self) -> str:
        return f"Purchase order {self.order_key} from Wishing Cup ({self.location_name})"

    def csv_text(self) -> str:
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for line in self.lines:
            writer.writerow([line.sku, line.description, line.cases, _dollars(line.case_price_cents),
                             _dollars(line.line_total_cents), self.deliver_to, self.needed_by.isoformat()])
        return out.getvalue()

    def email_text(self) -> str:
        rows = [f"  {'SKU':<14}{'Description':<32}{'Cases':>6}{'Case price':>13}{'Line total':>13}"]
        rows += [f"  {line.sku:<14}{line.description[:31]:<32}{line.cases:>6}"
                 f"{format_cents(line.case_price_cents):>13}{format_cents(line.line_total_cents):>13}"
                 for line in self.lines]
        width = 2 + 14 + 32 + 6 + 13 + 13
        totals = [("Items", self.subtotal_cents), ("Delivery", self.delivery_fee_cents), ("Total", self.total_cents)]
        rows += [f"{label + ':':>{width - 13}}{format_cents(cents):>13}" for label, cents in totals]
        arrive = (f"on {day_label(self.deliver_on)} " if self.deliver_on else "") + \
                 f"(needed by {day_label(self.needed_by)})"
        parts = [
            f"To: {self.rep_name} <{self.rep_email}>",
            f"Subject: {self.subject}",
            "",
            f"Hi {self.rep_name},",
            "",
            f"Please deliver this order to {self.deliver_to} {arrive}.",
            "",
            *rows,
            "",
            f"Order ID: {self.order_key}. It's the same ID our system sent you, so this can't be booked twice.",
        ]
        if self.confirmation_id:
            parts.append(f"Your confirmation: {self.confirmation_id}")
        parts += ["", "Thank you,", "Wishing Cup purchasing"]
        if self.fake_label:
            parts += ["", self.fake_label]
        return "\n".join(parts) + "\n"


def _dollars(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def save(po: PurchaseOrder, folder: Path) -> list[Path]:
    """Write <order id>.txt and <order id>.csv; writing again replaces them."""
    folder.mkdir(parents=True, exist_ok=True)
    txt, csv_path = folder / f"{po.order_key}.txt", folder / f"{po.order_key}.csv"
    txt.write_text(po.email_text(), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(po.csv_text())
    return [txt, csv_path]


def load_purchase_order(session, order_id: int, cfg) -> PurchaseOrder:
    from sqlmodel import select

    from app import models

    order = session.get(models.Order, order_id)
    req = session.get(models.Request, order.request_id)
    sup = session.get(models.Supplier, order.supplier_id)
    location = session.get(models.Location, req.location_id)
    lines = session.exec(select(models.RequestLine).where(
        models.RequestLine.request_id == req.id, models.RequestLine.assigned_supplier_id == sup.id,
        models.RequestLine.fill_status == models.LineFillStatus.ASSIGNED).order_by(models.RequestLine.line_no)).all()
    po_lines = []
    for line in lines:
        item = session.get(models.SupplierItem, line.supplier_item_id)
        po_lines.append(POLine(item.sku, item.description, line.cases, line.case_price_cents))
    address = cfg.location(location.code).address if any(l.code == location.code for l in cfg.locations) else ""
    return PurchaseOrder(
        order_key=order.order_key, supplier_name=sup.name, rep_name=sup.rep_name or sup.name,
        rep_email=sup.rep_email, location_name=location.name,
        deliver_to=f"{location.name}, {address}" if address else location.name,
        needed_by=req.needed_by, deliver_on=order.deliver_on, lines=tuple(po_lines),
        delivery_fee_cents=order.delivery_fee_cents, confirmation_id=order.confirmation_id,
        fake_label=cfg.app.fake_data_label,
    )
