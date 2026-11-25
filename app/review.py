"""What the Review page shows, built from the database. No Streamlit here, so it can be tested.

cost_matrix   every item x every supplier: the real cost for this need after rounding up to whole
              cases, or why there is none (no match, no price, out of stock, set aside, didn't answer...)
shopping_lists the current list per supplier, with cases, extra bought, totals and minimums
attention     the items a person has to look at, each with its reasons and the suppliers it could move to
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from sqlmodel import Session, select

from app import checks, clock, models
from app.config import AppConfig
from app.optimizer import SupplierTerms
from app.pricing import format_cents, format_qty, item_cost

OK = "ok"


@dataclass(frozen=True)
class Cell:
    status: str                  # ok | out_of_stock | dont_use | no_price | no_pack | no_match | no_answer | not_serving
    cost_cents: int | None = None
    cases: int | None = None
    case_price_cents: int | None = None
    note: str = ""

    @property
    def label(self) -> str:
        money = f"{format_cents(self.cost_cents)} ({self.cases} cs)" if self.cost_cents is not None else ""
        return {
            OK: money,
            "out_of_stock": f"out of stock {money}".strip(),
            "dont_use": f"don't use: {self.note}",
            "no_price": "no price",
            "no_pack": "case size unknown",
            "no_match": "not matched",
            "no_answer": "didn't answer",
            "not_serving": "doesn't deliver here",
        }[self.status]


@dataclass
class Matrix:
    products: list[tuple[str, str]]                       # (code, label with the amount needed)
    suppliers: list[tuple[str, str]]                      # (code, name)
    cells: dict[tuple[str, str], Cell]
    cheapest: dict[str, str | None]                       # product -> supplier with the lowest usable cost

    @property
    def empty(self) -> bool:
        return not self.products or not self.suppliers


def cost_matrix(session: Session, request_id: int, cfg: AppConfig) -> Matrix:
    from app.workflow import _latest_quotes

    req = session.get(models.Request, request_id)
    location = session.get(models.Location, req.location_id)
    lines = session.exec(select(models.RequestLine).where(models.RequestLine.request_id == request_id)
                         .order_by(models.RequestLine.line_no)).all()
    products = {p.id: p for p in session.exec(select(models.Product)).all()}
    rows = {s.code: s for s in session.exec(select(models.Supplier).where(models.Supplier.active)).all()}
    suppliers = [(sc.code, sc.name) for sc in cfg.suppliers if sc.code in rows]
    quotes = _latest_quotes(session, request_id)
    mappings = list(session.exec(select(models.ItemMapping)).all())

    cells: dict[tuple[str, str], Cell] = {}
    cheapest: dict[str, str | None] = {}
    labels = []
    for line in lines:
        product = products.get(line.product_id)
        if product is None or line.qty_needed is None:
            continue
        labels.append((product.code, f"{product.name} ({format_qty(line.qty_needed, product.base_unit.value)})"))
        best: tuple[int, str] | None = None
        for code, _ in suppliers:
            sc = cfg.supplier(code)
            m = next((x for x in mappings if x.product_id == product.id and x.supplier_id == rows[code].id), None)
            quote = quotes.get(m.supplier_item_id) if m else None
            if location.code not in sc.serves_locations:
                cell = Cell("not_serving")
            elif m is None:
                cell = Cell("no_match")
            elif code in req.suppliers_no_answer:
                cell = Cell("no_answer")
            elif quote is None:
                cell = Cell("no_price")
            elif m.pack_size_base is None:
                cell = Cell("no_pack")
            elif quote.dont_use and not quote.price_accepted_by:
                cell = Cell("dont_use", note=quote.dont_use_reason or "set aside")
            else:
                cost = item_cost(line.qty_needed, m.pack_size_base, quote.case_price_cents)
                cell = Cell(OK if quote.in_stock else "out_of_stock", cost.cost_cents, cost.cases,
                            quote.case_price_cents)
                if cell.status == OK and (best is None or cost.cost_cents < best[0]):
                    best = (cost.cost_cents, code)
            cells[(product.code, code)] = cell
        cheapest[product.code] = best[1] if best else None
    return Matrix(labels, suppliers, cells, cheapest)


@dataclass(frozen=True)
class ListLine:
    product_name: str
    sku: str
    description: str
    need: str
    case_size: str
    cases: int
    bought: str
    extra: str
    case_price_cents: int
    line_total_cents: int


@dataclass(frozen=True)
class ShoppingList:
    supplier: str
    name: str
    lines: tuple[ListLine, ...]
    delivery_fee_cents: int
    min_order_cents: int
    deliver_on: date | None

    @property
    def subtotal_cents(self) -> int:
        return sum(line.line_total_cents for line in self.lines)

    @property
    def total_cents(self) -> int:
        return self.subtotal_cents + self.delivery_fee_cents

    @property
    def meets_minimum(self) -> bool:
        return self.subtotal_cents >= self.min_order_cents


def shopping_lists(session: Session, request_id: int, cfg: AppConfig, *, now: datetime | None = None
                   ) -> tuple[list[ShoppingList], list[tuple[str, str]]]:
    """The current lists (after any of the owner's changes), and the can't-fill items with reasons."""
    now = now or clock.now()
    lines = session.exec(select(models.RequestLine).where(models.RequestLine.request_id == request_id)
                         .order_by(models.RequestLine.line_no)).all()
    products = {p.id: p for p in session.exec(select(models.Product)).all()}
    suppliers = {s.id: s for s in session.exec(select(models.Supplier)).all()}
    mappings = {m.supplier_item_id: m for m in session.exec(select(models.ItemMapping)).all()}
    grouped: dict[int, list[ListLine]] = {}
    cant = []
    for line in lines:
        product = products.get(line.product_id)
        name = product.name if product else line.raw_text
        unit = product.base_unit.value if product else "each"
        if line.fill_status != models.LineFillStatus.ASSIGNED:
            cant.append((name, line.plan_note or "not on any list yet"))
            continue
        item = session.get(models.SupplierItem, line.supplier_item_id)
        pack = mappings[line.supplier_item_id].pack_size_base if line.supplier_item_id in mappings else None
        bought = line.cases * pack if pack is not None else None
        grouped.setdefault(line.assigned_supplier_id, []).append(ListLine(
            name, item.sku, item.description, format_qty(line.qty_needed, unit),
            format_qty(pack, unit) if pack is not None else "unknown", line.cases,
            format_qty(bought, unit) if bought is not None else "unknown",
            format_qty(bought - line.qty_needed, unit) if bought is not None else "unknown",
            line.case_price_cents, line.line_cost_cents))
    order = {sc.code: n for n, sc in enumerate(cfg.suppliers)}
    out = []
    for sid, items in grouped.items():
        sup = suppliers[sid]
        sc = next((x for x in cfg.suppliers if x.code == sup.code), None)
        arrives = SupplierTerms.from_config(sc).earliest_delivery(now.date()) if sc else None
        out.append(ShoppingList(sup.code, sup.name, tuple(items), sup.delivery_fee_cents, sup.min_order_cents,
                                arrives))
    return sorted(out, key=lambda x: order.get(x.supplier, 99)), cant


@dataclass
class Attention:
    product_code: str
    product_name: str
    reasons: list[str] = field(default_factory=list)
    current_supplier: str | None = None
    can_confirm: bool = True
    move_to: list[tuple[str, str]] = field(default_factory=list)       # (code, "Name: $cost")
    set_aside: list[tuple[str, str]] = field(default_factory=list)     # (code, "Name: reason")


def attention(view: checks.RequestView, matrix: Matrix) -> list[Attention]:
    """Items a person must look at. Same rules as the checks, grouped per item."""
    out = []
    for line in view.lines:
        reasons = []
        if line.fill_status == "CANT_FILL":
            reasons.append("can't be filled from any supplier right now")
        if line.line_needs_check:
            reasons.append(line.line_reason or "marked for a person to check")
        if line.fill_status == "ASSIGNED":
            if line.match_needs_check:
                reasons.append(f"the product match at {view.name_of(line.supplier)} needs checking "
                               f"({line.match_reason or 'flagged'})")
            if line.price_set_aside and not line.price_accepted:
                reasons.append(f"the price was set aside ({line.price_set_aside})")
            if line.has_quote and line.quote_in_stock is False:
                reasons.append(f"out of stock at {view.name_of(line.supplier)}")
            if not line.qty_confirmed and line.qty_needed is not None:
                if line.most_ordered_before is None:
                    reasons.append(f"{view.location_name} has never ordered it before")
                elif line.qty_needed > view.quantity_multiple * line.most_ordered_before:
                    reasons.append(f"more than {view.quantity_multiple} times the most ever ordered "
                                   f"({format_qty(line.most_ordered_before, line.base_unit)})")
        if not reasons:
            continue
        options, aside = [], []
        for code, name in matrix.suppliers:
            cell = matrix.cells.get((line.product_code, code))
            if cell is None or code == line.supplier:
                continue
            if cell.status == OK:
                options.append((code, f"{name}: {format_cents(cell.cost_cents)}"))
            elif cell.status == "dont_use":
                aside.append((code, f"{name}: {cell.note}"))
        out.append(Attention(line.product_code, line.product_name, reasons, line.supplier,
                             can_confirm=line.fill_status != "CANT_FILL", move_to=options, set_aside=aside))
    return out
