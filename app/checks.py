"""Safety checks, run before approval and again before ordering. Any failure blocks.

From SPEC.md:
  1. every item is on exactly one supplier list or marked "can't fill"
  2. the cases cover what is needed
  3. totals add up from the lines
  4. each supplier's minimum order is met
  5. each supplier can deliver by the needed date
  6. the products exist and are in stock
  7. prices are not older than the limit
and from its other rules:
  8. no line uses a price that was set aside (0 or less, or up more than 50% since last week)
     unless a person checked it
  9. a quantity over 3x the most this location ever ordered has been confirmed by a person
 10. everything marked for a person to check (AI readings, product matches) has been checked

The checks are pure functions over a RequestView, a snapshot of everything they need. load_view
builds it from the database, so the rules can be tested without one. Every reason is a full
sentence a manager can act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

from app.optimizer import SupplierTerms
from app.pricing import format_cents, format_qty
from app.textparse import day_label

STAGES = ("approval", "ordering")


@dataclass(frozen=True)
class LineView:
    line_id: int
    product_code: str
    product_name: str
    base_unit: str
    qty_needed: Decimal | None
    fill_status: str                      # OPEN | ASSIGNED | CANT_FILL
    product_active: bool = True
    supplier: str | None = None
    sku: str | None = None
    item_exists: bool = True
    pack_size: Decimal | None = None      # from the product match
    match_needs_check: bool = False
    match_reason: str | None = None
    cases: int | None = None
    case_price_cents: int | None = None
    line_cost_cents: int | None = None
    has_quote: bool = False
    quote_price_cents: int | None = None
    quote_in_stock: bool | None = None
    quoted_at: datetime | None = None
    price_set_aside: str | None = None    # why the price wasn't used, if it was set aside
    price_accepted: bool = False
    line_needs_check: bool = False
    line_reason: str | None = None
    qty_confirmed: bool = False
    most_ordered_before: Decimal | None = None  # at this location; None = never ordered


@dataclass(frozen=True)
class OrderView:
    supplier: str
    subtotal_cents: int
    delivery_fee_cents: int
    total_cents: int
    status: str


@dataclass(frozen=True)
class RequestView:
    request_id: int
    location_code: str
    location_name: str
    needed_by: date | None
    now: datetime
    lines: tuple[LineView, ...]
    suppliers: dict[str, SupplierTerms]
    plan_total_cents: int | None = None
    orders: tuple[OrderView, ...] = ()
    price_max_age: timedelta = timedelta(hours=24)
    quantity_multiple: Decimal = Decimal(3)

    @property
    def today(self) -> date:
        return self.now.date()

    @property
    def assigned(self) -> list[LineView]:
        return [line for line in self.lines if line.fill_status == "ASSIGNED"]

    def lists(self) -> dict[str, list[LineView]]:
        out: dict[str, list[LineView]] = {}
        for line in self.assigned:
            out.setdefault(line.supplier, []).append(line)
        return dict(sorted(out.items()))

    def name_of(self, supplier: str) -> str:
        terms = self.suppliers.get(supplier)
        return terms.name if terms else supplier


@dataclass(frozen=True)
class CheckResult:
    name: str
    title: str
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'} {self.title}" + "".join(f"\n  - {r}" for r in self.reasons)


def _result(name: str, title: str, reasons: list[str]) -> CheckResult:
    return CheckResult(name, title, not reasons, tuple(reasons))


def _qty(line: LineView, qty: Decimal | None) -> str:
    return format_qty(qty, line.base_unit) if qty is not None else "an unknown amount"


# ------------------------------------------------------------------ the checks

def check_one_list(v: RequestView) -> CheckResult:
    reasons = [f"{line.product_name} isn't on any supplier's list and isn't marked can't fill."
               for line in v.lines if line.fill_status not in ("ASSIGNED", "CANT_FILL")]
    reasons += [f"{line.product_name} is marked as assigned but has no supplier."
                for line in v.assigned if not line.supplier]
    if v.lines and not v.assigned:
        reasons.append("Nothing can be ordered: every item is on the can't-fill list.")
    if not v.lines:
        reasons.append("The request has no items.")
    return _result("one_list", "Every item is on exactly one list or marked can't fill", reasons)


def check_cases_cover(v: RequestView) -> CheckResult:
    reasons = []
    for line in v.assigned:
        name = v.name_of(line.supplier)
        if line.qty_needed is None:
            reasons.append(f"{line.product_name}: the quantity needed is missing.")
        elif line.pack_size is None:
            reasons.append(f"{line.product_name}: the case size at {name} is unknown, so we can't tell "
                           f"whether {line.cases} cases cover {_qty(line, line.qty_needed)}.")
        elif not line.cases or line.cases * line.pack_size < line.qty_needed:
            reasons.append(f"{line.product_name}: {line.cases or 0} cases of {_qty(line, line.pack_size)} at {name} "
                           f"is {_qty(line, (line.cases or 0) * line.pack_size)}, less than the "
                           f"{_qty(line, line.qty_needed)} needed.")
    return _result("cases_cover", "The cases cover what is needed", reasons)


def check_totals(v: RequestView) -> CheckResult:
    reasons = []
    for line in v.assigned:
        if line.cases is None or line.case_price_cents is None or line.line_cost_cents is None:
            reasons.append(f"{line.product_name}: the line is missing its cases, case price or total.")
            continue
        if line.cases * line.case_price_cents != line.line_cost_cents:
            reasons.append(f"{line.product_name}: {line.cases} x {format_cents(line.case_price_cents)} is "
                           f"{format_cents(line.cases * line.case_price_cents)}, but the line says "
                           f"{format_cents(line.line_cost_cents)}.")
        if line.has_quote and line.quote_price_cents != line.case_price_cents:
            reasons.append(f"{line.product_name}: the line uses {format_cents(line.case_price_cents)} a case, "
                           f"but the supplier's price is {format_cents(line.quote_price_cents)}.")
    lists = v.lists()
    subtotals = {s: sum(line.line_cost_cents or 0 for line in lines) for s, lines in lists.items()}
    fees = {s: v.suppliers[s].delivery_fee_cents if s in v.suppliers else 0 for s in lists}
    total = sum(subtotals.values()) + sum(fees.values())
    if v.plan_total_cents is not None and v.plan_total_cents != total:
        reasons.append(f"The plan total shown ({format_cents(v.plan_total_cents)}) doesn't match the lines "
                       f"({format_cents(total)}). Recalculate the plan.")
    for order in v.orders:
        name = v.name_of(order.supplier)
        if order.supplier not in subtotals:
            reasons.append(f"There is an order for {name} but no items on its list.")
            continue
        if order.subtotal_cents != subtotals[order.supplier]:
            reasons.append(f"{name}: the order says {format_cents(order.subtotal_cents)} but its lines add up to "
                           f"{format_cents(subtotals[order.supplier])}.")
        if order.total_cents != order.subtotal_cents + order.delivery_fee_cents:
            reasons.append(f"{name}: the order total {format_cents(order.total_cents)} isn't items plus delivery.")
    return _result("totals", "Totals add up from the lines", reasons)


def check_minimums(v: RequestView) -> CheckResult:
    reasons = []
    for supplier, lines in v.lists().items():
        terms = v.suppliers.get(supplier)
        subtotal = sum(line.line_cost_cents or 0 for line in lines)
        if terms is not None and subtotal < terms.min_order_cents:
            reasons.append(f"{terms.name}: the order is {format_cents(subtotal)}, below its "
                           f"{format_cents(terms.min_order_cents)} minimum. Add "
                           f"{format_cents(terms.min_order_cents - subtotal)} or move its items to another supplier.")
    return _result("minimums", "Each supplier's minimum order is met", reasons)


def check_delivery(v: RequestView) -> CheckResult:
    reasons = []
    if v.needed_by is None:
        reasons.append("The request has no needed-by date.")
    elif v.needed_by < v.today:
        reasons.append(f"The needed-by date {day_label(v.needed_by)} has passed.")
    for supplier in v.lists():
        terms = v.suppliers.get(supplier)
        if terms is None:
            reasons.append(f"{supplier} is no longer one of our suppliers.")
            continue
        if terms.serves is not None and v.location_code not in terms.serves:
            reasons.append(f"{terms.name} doesn't deliver to {v.location_name}.")
            continue
        arrives = terms.earliest_delivery(v.today)
        if arrives is None:
            reasons.append(f"{terms.name} has no delivery days.")
        elif v.needed_by is not None and arrives > v.needed_by:
            reasons.append(f"{terms.name} can deliver {day_label(arrives)} at the earliest, after the "
                           f"needed-by date {day_label(v.needed_by)}.")
    return _result("delivery", "Each supplier can deliver by the needed date", reasons)


def check_in_stock(v: RequestView) -> CheckResult:
    reasons = []
    for line in v.lines:
        if not line.product_active:
            reasons.append(f"{line.product_name} is no longer on our product list.")
    for line in v.assigned:
        name = v.name_of(line.supplier)
        if not line.item_exists:
            reasons.append(f"{name} no longer lists SKU {line.sku} ({line.product_name}).")
        elif not line.has_quote:
            reasons.append(f"There is no price from {name} for {line.product_name}.")
        elif not line.quote_in_stock:
            reasons.append(f"{line.product_name} is out of stock at {name}.")
    return _result("in_stock", "The products exist and are in stock", reasons)


def check_fresh_prices(v: RequestView) -> CheckResult:
    reasons = []
    hours = int(v.price_max_age.total_seconds() // 3600)
    for line in v.assigned:
        if line.has_quote and line.quoted_at is not None and v.now - line.quoted_at > v.price_max_age:
            age = v.now - line.quoted_at
            reasons.append(f"{v.name_of(line.supplier)}'s price for {line.product_name} is "
                           f"{int(age.total_seconds() // 3600)} hours old, older than the {hours}-hour limit. "
                           "Get new prices.")
    return _result("fresh_prices", f"Prices are not older than {hours} hours", reasons)


def check_price_sanity(v: RequestView) -> CheckResult:
    reasons = [f"{v.name_of(line.supplier)}'s price for {line.product_name} was set aside ({line.price_set_aside}). "
               "A person must check it before it can be used."
               for line in v.assigned if line.price_set_aside and not line.price_accepted]
    return _result("price_sanity", "No price that looks wrong is used", reasons)


def check_quantities(v: RequestView) -> CheckResult:
    reasons = []
    for line in v.assigned:
        if line.qty_confirmed or line.qty_needed is None:
            continue
        if line.most_ordered_before is None:
            reasons.append(f"{line.product_name}: {v.location_name} has never ordered it before. "
                           f"Confirm {_qty(line, line.qty_needed)}.")
        elif line.qty_needed > v.quantity_multiple * line.most_ordered_before:
            reasons.append(f"{line.product_name}: {_qty(line, line.qty_needed)} is more than "
                           f"{v.quantity_multiple} times the most {v.location_name} ever ordered "
                           f"({_qty(line, line.most_ordered_before)}). Confirm it.")
    return _result("quantities", "Unusually large quantities are confirmed", reasons)


def check_flags_cleared(v: RequestView) -> CheckResult:
    reasons = []
    for line in v.lines:
        if line.line_needs_check and line.fill_status != "CANT_FILL":
            reasons.append(f"{line.product_name}: {line.line_reason or 'marked for a person to check'}. "
                           "Confirm the line.")
    for line in v.assigned:
        if line.match_needs_check:
            reasons.append(f"{line.product_name} at {v.name_of(line.supplier)}: the product match needs a person "
                           f"to check it ({line.match_reason or 'flagged'}).")
    return _result("flags_cleared", "Everything marked for a person to check has been checked", reasons)


ALL_CHECKS = (check_one_list, check_cases_cover, check_totals, check_minimums, check_delivery, check_in_stock,
              check_fresh_prices, check_price_sanity, check_quantities, check_flags_cleared)


def run_checks(view: RequestView, stage: str) -> list[CheckResult]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    return [check(view) for check in ALL_CHECKS]


def failures(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if not r.passed]


def summary(results: list[CheckResult]) -> str:
    bad = failures(results)
    if not bad:
        return "All safety checks passed."
    return "Blocked:\n" + "\n".join(f"- {reason}" for r in bad for reason in r.reasons)


# ------------------------------------------------------------------ building a view from the database

def load_view(session, request_id: int, cfg, now: datetime) -> RequestView:
    from sqlmodel import select

    from app import models

    req = session.get(models.Request, request_id)
    if req is None:
        raise KeyError(f"no request {request_id}")
    location = session.get(models.Location, req.location_id)
    suppliers = {s.id: s for s in session.exec(select(models.Supplier)).all()}
    lines = session.exec(select(models.RequestLine).where(models.RequestLine.request_id == request_id)
                         .order_by(models.RequestLine.line_no)).all()
    product_ids = {line.product_id for line in lines if line.product_id is not None}
    products = {p.id: p for p in session.exec(select(models.Product)).all() if p.id in product_ids}
    item_ids = {line.supplier_item_id for line in lines if line.supplier_item_id is not None}
    items = {i.id: i for i in session.exec(select(models.SupplierItem)).all() if i.id in item_ids}
    mappings = {m.supplier_item_id: m for m in session.exec(select(models.ItemMapping)).all()
                if m.supplier_item_id in item_ids}
    quote_ids = {line.quote_id for line in lines if line.quote_id is not None}
    quotes = {q.id: q for q in session.exec(select(models.Quote).where(models.Quote.request_id == request_id)).all()
              if q.id in quote_ids}
    history = most_ordered_before(session, req.location_id, product_ids, exclude_request=request_id)
    configured = {sc.code: sc for sc in cfg.suppliers}

    views = []
    for line in lines:
        product = products.get(line.product_id)
        supplier = suppliers.get(line.assigned_supplier_id)
        item = items.get(line.supplier_item_id)
        mapping = mappings.get(line.supplier_item_id)
        quote = quotes.get(line.quote_id)
        views.append(LineView(
            line_id=line.id,
            product_code=product.code if product else "?",
            product_name=product.name if product else (line.raw_text or "unknown product"),
            base_unit=product.base_unit.value if product else "each",
            qty_needed=line.qty_needed,
            fill_status=line.fill_status.value,
            product_active=bool(product and product.active),
            supplier=supplier.code if supplier else None,
            sku=item.sku if item else None,
            item_exists=item is not None or line.supplier_item_id is None,
            pack_size=mapping.pack_size_base if mapping and mapping.product_id == line.product_id else None,
            match_needs_check=bool(mapping and mapping.needs_check),
            match_reason=mapping.check_reason if mapping else None,
            cases=line.cases,
            case_price_cents=line.case_price_cents,
            line_cost_cents=line.line_cost_cents,
            has_quote=quote is not None,
            quote_price_cents=quote.case_price_cents if quote else None,
            quote_in_stock=quote.in_stock if quote else None,
            quoted_at=quote.quoted_at if quote else None,
            price_set_aside=quote.dont_use_reason if quote and quote.dont_use else None,
            price_accepted=bool(quote and quote.price_accepted_by),
            line_needs_check=line.needs_check,
            line_reason=line.check_reason,
            qty_confirmed=line.qty_confirmed,
            most_ordered_before=history.get(line.product_id),
        ))
    orders = tuple(OrderView(suppliers[o.supplier_id].code, o.subtotal_cents, o.delivery_fee_cents, o.total_cents,
                             o.status.value)
                   for o in session.exec(select(models.Order).where(models.Order.request_id == request_id)).all())
    terms = {code: SupplierTerms.from_config(sc) for code, sc in configured.items()}
    return RequestView(
        request_id=request_id, location_code=location.code, location_name=location.name, needed_by=req.needed_by,
        now=now, lines=tuple(views), suppliers=terms, plan_total_cents=req.plan_total_cents, orders=orders,
        price_max_age=timedelta(hours=cfg.approval.price_max_age_hours),
        quantity_multiple=cfg.purchasing.quantity_confirm_multiple,
    )


def most_ordered_before(session, location_id: int, product_ids: set[int], *, exclude_request: int | None = None
                        ) -> dict[int, Decimal]:
    """The largest quantity this location ever had ordered, per product (quantities are TEXT, so in Python)."""
    from sqlmodel import select

    from app import models

    ordered = {models.RequestStatus.ORDERED, models.RequestStatus.PARTIALLY_ORDERED}
    req_ids = {r.id for r in session.exec(select(models.Request).where(models.Request.location_id == location_id)).all()
               if r.status in ordered and r.id != exclude_request}
    most: dict[int, Decimal] = {}
    for line in session.exec(select(models.RequestLine)).all():
        if line.request_id in req_ids and line.product_id in product_ids and line.qty_needed is not None \
                and line.fill_status == models.LineFillStatus.ASSIGNED:
            most[line.product_id] = max(most.get(line.product_id, Decimal(0)), line.qty_needed)
    return most
