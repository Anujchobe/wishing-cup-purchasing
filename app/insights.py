"""Reports: savings replay, price watch, price-match emails, spend, and prices over time.

The rules are pure functions (easy to test); the load_* functions read the database.
Prices are compared per base unit (per lb, per fl oz, each) because pack sizes differ between
suppliers. Savings replay uses whole cases, like real ordering.

All of these only use product matches that are confirmed (by a person, or by a clear text match
with a readable pack). A match still waiting for a person could be the wrong product, and a
price-match email built on it would ask a real sales rep about the wrong item. Matching happens
when a request needs it, so early on most suppliers aren't matched; `coverage` says how much is,
and the Reports page offers to match every product at once.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from app import clock
from app.units import display_unit

SENDER = "Wishing Cup purchasing"


def unit_price_label(cents_per_unit: Decimal, base_unit: str) -> str:
    """Decimal('381.2') cents per lb -> '$3.81/lb'. Under $1 a unit shows a tenth of a cent too
    ('$0.081/each'), or cups at 8.1 and 7.0 cents would both read as pennies apart."""
    dollars = cents_per_unit / 100
    places = Decimal("0.001") if dollars < 1 else Decimal("0.01")
    return f"${dollars.quantize(places, ROUND_HALF_UP):,}/{display_unit(base_unit)}"


def pct(value: Decimal) -> str:
    return f"{(value * 100).quantize(Decimal(1), ROUND_HALF_UP)}%"


# ------------------------------------------------------------------ price watch

@dataclass(frozen=True)
class WatchRow:
    product_code: str
    product_name: str
    kind: str                          # market | one_supplier | some_suppliers
    changes: dict[str, Decimal]        # supplier -> change as a fraction (0.15 = up 15%)
    up: tuple[str, ...]

    @property
    def summary(self) -> str:
        parts = ", ".join(f"{s} {'+' if c >= 0 else ''}{pct(c)}" for s, c in sorted(self.changes.items()))
        label = {"market": "Market change: up at every supplier",
                 "one_supplier": f"Only {self.up[0]} raised it",
                 "some_suppliers": f"Up at {', '.join(self.up)} only"}[self.kind]
        return f"{label} ({parts})"


def classify_change(changes: dict[str, Decimal], threshold: Decimal) -> tuple[str | None, tuple[str, ...]]:
    """Market change: up more than threshold at every supplier we track (at least two).
    One supplier: up at exactly one of two or more. Fewer than two suppliers can't tell the difference."""
    up = tuple(sorted(s for s, c in changes.items() if c > threshold))
    if len(changes) < 2 or not up:
        return None, up
    if len(up) == len(changes):
        return "market", up
    return ("one_supplier" if len(up) == 1 else "some_suppliers"), up


def change_over(points: list[tuple[datetime, Decimal]], weeks: int) -> Decimal | None:
    """Latest price vs the latest price at least `weeks` weeks before it, as a fraction."""
    if not points:
        return None
    points = sorted(points)
    latest_at, latest = points[-1]
    earlier = [p for at, p in points if at <= latest_at - timedelta(weeks=weeks)]
    if not earlier or earlier[-1] <= 0:
        return None
    return latest / earlier[-1] - 1


def price_watch_rows(series: dict[str, dict[str, list[tuple[datetime, Decimal]]]], names: dict[str, str],
                     weeks: int, threshold: Decimal) -> list[WatchRow]:
    rows = []
    for product, by_supplier in sorted(series.items()):
        changes = {s: c for s, pts in by_supplier.items() if (c := change_over(pts, weeks)) is not None}
        kind, up = classify_change(changes, threshold)
        if kind is not None:
            rows.append(WatchRow(product, names.get(product, product), kind, changes, up))
    return sorted(rows, key=lambda r: (r.kind != "market", r.product_name))


# ------------------------------------------------------------------ price match

@dataclass(frozen=True)
class UnitPrice:
    supplier: str
    cents_per_unit: Decimal
    case_price_cents: int
    pack: Decimal


@dataclass(frozen=True)
class MatchItem:
    product_code: str
    product_name: str
    base_unit: str
    main: UnitPrice
    other: UnitPrice
    cheaper_by: Decimal            # fraction of the main supplier's price

    @property
    def main_label(self) -> str:
        return unit_price_label(self.main.cents_per_unit, self.base_unit)

    @property
    def other_label(self) -> str:
        return unit_price_label(self.other.cents_per_unit, self.base_unit)

    @property
    def fact(self) -> str:
        return (f"{self.product_name}: we pay {self.main_label}; we have been offered {self.other_label} "
                f"({pct(self.cheaper_by)} lower).")


def match_candidates(latest: dict[str, dict[str, UnitPrice]], names: dict[str, str], units: dict[str, str],
                     main: str, threshold: Decimal) -> list[MatchItem]:
    """Items where another supplier beats the main supplier per unit by more than `threshold`."""
    items = []
    for product, by_supplier in sorted(latest.items()):
        ours = by_supplier.get(main)
        others = [p for s, p in by_supplier.items() if s != main]
        if ours is None or not others or ours.cents_per_unit <= 0:
            continue
        best = min(others, key=lambda p: (p.cents_per_unit, p.supplier))
        cheaper_by = (ours.cents_per_unit - best.cents_per_unit) / ours.cents_per_unit
        if cheaper_by > threshold:
            items.append(MatchItem(product, names.get(product, product), units.get(product, "each"), ours, best,
                                   cheaper_by))
    return sorted(items, key=lambda i: -i.cheaper_by)


_MONEY = re.compile(r"\$\d[\d,]*(?:\.\d+)?")
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")


def email_problems(body: str, items: list[MatchItem]) -> list[str]:
    """Code owns the numbers: every price (with its unit) must appear, and no other amount may."""
    required = [i.main_label for i in items] + [i.other_label for i in items]
    allowed_money = {_MONEY.match(x).group(0) for x in required}
    allowed_pct = {pct(i.cheaper_by) for i in items}
    problems = [f"the price {x} is missing" for x in required if x not in body]
    problems += [f"the amount {m} isn't one of the facts" for m in _MONEY.findall(body) if m not in allowed_money]
    problems += [f"the percentage {p} isn't one of the facts" for p in _PERCENT.findall(body)
                 if p.replace(" ", "") not in allowed_pct]
    return problems


class PriceMatchEmail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject: Annotated[str, Field(min_length=3, max_length=150)]
    body: Annotated[str, Field(min_length=40)]

    @field_validator("body")
    @classmethod
    def _numbers_from_code(cls, body: str, info: ValidationInfo) -> str:
        items = (info.context or {}).get("items")
        if items:
            problems = email_problems(body, items)
            if problems:
                raise ValueError("; ".join(problems))
        return body


def template_email(supplier_name: str, rep_name: str, items: list[MatchItem]) -> tuple[str, str]:
    subject = f"Price match request: {len(items)} item{'s' if len(items) != 1 else ''}"
    lines = [f"  - {i.product_name}: you charge {i.main_label}; we have been offered {i.other_label} "
             f"({pct(i.cheaper_by)} lower)." for i in items]
    body = "\n".join([
        f"Hi {rep_name},", "",
        f"We value working with {supplier_name}, and we'd like to keep our orders with you. We have recently been "
        "offered lower prices on a few items we buy from you regularly. Could you match them?", "",
        *lines, "",
        "Happy to talk it through. Thank you,", SENDER,
    ])
    return subject, body


def write_price_match_email(client, supplier_name: str, rep_name: str, items: list[MatchItem]) -> tuple[str, str, str]:
    """(subject, body, written_by). The AI only writes the wording; any answer whose numbers differ
    from the facts is refused (one retry), and then the template is used."""
    from app.llm.client import LLMError, render_prompt

    if not items:
        raise ValueError("no items to ask about")
    system = render_prompt("price_match", rep_name=rep_name, supplier=supplier_name,
                           facts="\n".join(f"- {i.fact}" for i in items), sender=SENDER)
    try:
        email = client.complete_json(system, "Write the email.", PriceMatchEmail, task="write_email",
                                     context={"items": items})
        return email.subject, email.body, "ai"
    except LLMError:
        subject, body = template_email(supplier_name, rep_name, items)
        return subject, body, "template"


# ------------------------------------------------------------------ savings replay

@dataclass(frozen=True)
class ReplayRow:
    request_public_id: str
    week_of: date
    location: str
    actual_cents: int
    replay_cents: int | None
    note: str = ""

    @property
    def saved_cents(self) -> int | None:
        return None if self.replay_cents is None else self.actual_cents - self.replay_cents


@dataclass
class ReplayReport:
    rows: list[ReplayRow] = field(default_factory=list)

    @property
    def actual_cents(self) -> int:
        return sum(r.actual_cents for r in self.rows if r.replay_cents is not None)

    @property
    def replay_cents(self) -> int:
        return sum(r.replay_cents for r in self.rows if r.replay_cents is not None)

    @property
    def saved_cents(self) -> int:
        return self.actual_cents - self.replay_cents


@dataclass(frozen=True)
class Coverage:
    matched: int                            # confirmed matches
    possible: int
    missing: tuple[tuple[str, str], ...]   # (product name, supplier name)
    waiting: int = 0                        # matches saved but waiting for a person

    @property
    def complete(self) -> bool:
        return self.matched >= self.possible


# ------------------------------------------------------------------ database loaders

def _snapshots_by_item(session) -> dict[int, list[Any]]:
    from sqlmodel import select

    from app import models

    out: dict[int, list[Any]] = defaultdict(list)
    for s in session.exec(select(models.PriceSnapshot)).all():
        if not s.dont_use and s.case_price_cents > 0:
            out[s.supplier_item_id].append(s)
    return out


def _mapped(session) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any], list[Any]]:
    from sqlmodel import select

    from app import models

    products = {p.id: p for p in session.exec(select(models.Product)).all()}
    suppliers = {s.id: s for s in session.exec(select(models.Supplier).where(models.Supplier.active)).all()}
    items = {i.id: i for i in session.exec(select(models.SupplierItem)).all()}
    mappings = [m for m in session.exec(select(models.ItemMapping)).all()
                if m.pack_size_base is not None and not m.needs_check
                and m.supplier_id in suppliers and m.product_id in products]
    return products, suppliers, items, mappings


def load_coverage(session, cfg) -> Coverage:
    from sqlmodel import select

    from app import models

    products, suppliers, items, mappings = _mapped(session)
    have = {(m.product_id, m.supplier_id) for m in mappings}
    waiting = sum(1 for m in session.exec(select(models.ItemMapping)).all() if m.needs_check)
    possible, missing = 0, []
    for p in products.values():
        if not p.active:
            continue
        for s in suppliers.values():
            possible += 1
            if (p.id, s.id) not in have:
                missing.append((p.name, s.name))
    return Coverage(possible - len(missing), possible, tuple(missing), waiting)


def load_price_series(session) -> tuple[dict[str, dict[str, list[tuple[datetime, Decimal]]]], dict[str, str], dict[str, str]]:
    """product code -> supplier name -> [(when, cents per base unit)], plus names and units."""
    products, suppliers, items, mappings = _mapped(session)
    snaps = _snapshots_by_item(session)
    series: dict[str, dict[str, list[tuple[datetime, Decimal]]]] = defaultdict(lambda: defaultdict(list))
    for m in mappings:
        product, supplier = products[m.product_id], suppliers[m.supplier_id]
        for snap in snaps.get(m.supplier_item_id, []):
            series[product.code][supplier.name].append((snap.captured_at,
                                                        Decimal(snap.case_price_cents) / m.pack_size_base))
    names = {p.code: p.name for p in products.values()}
    unit = {p.code: p.base_unit.value for p in products.values()}
    return {k: dict(v) for k, v in series.items()}, names, unit


def load_price_watch(session, cfg) -> list[WatchRow]:
    series, names, _ = load_price_series(session)
    return price_watch_rows(series, names, cfg.purchasing.price_watch_weeks, cfg.purchasing.market_change_threshold)


def load_match_candidates(session, cfg) -> list[MatchItem]:
    products, suppliers, items, mappings = _mapped(session)
    snaps = _snapshots_by_item(session)
    latest: dict[str, dict[str, UnitPrice]] = defaultdict(dict)
    for m in mappings:
        usable = [s for s in snaps.get(m.supplier_item_id, []) if s.in_stock]
        if not usable:
            continue
        last = max(usable, key=lambda s: s.captured_at)
        sup = suppliers[m.supplier_id]
        latest[products[m.product_id].code][sup.code] = UnitPrice(
            sup.name, Decimal(last.case_price_cents) / m.pack_size_base, last.case_price_cents, m.pack_size_base)
    names = {p.code: p.name for p in products.values()}
    units = {p.code: p.base_unit.value for p in products.values()}
    return match_candidates(dict(latest), names, units, cfg.purchasing.main_supplier,
                            cfg.purchasing.match_threshold)


def load_savings_replay(session, cfg, *, now: datetime | None = None) -> ReplayReport:
    """Past orders from the last `replay_weeks` weeks, re-planned with the prices of their own week."""
    from sqlmodel import select

    from app import models
    from app.optimizer import Need, Offer, Problem, SupplierTerms, make_plans

    now = now or clock.now()
    since = clock.week_start(now.date()) - timedelta(weeks=cfg.purchasing.replay_weeks)
    products, suppliers, items, mappings = _mapped(session)
    snaps = _snapshots_by_item(session)
    locations = {loc.id: loc for loc in session.exec(select(models.Location)).all()}
    terms = tuple(SupplierTerms.from_config(sc) for sc in cfg.suppliers
                  if any(s.code == sc.code for s in suppliers.values()))
    done = {models.RequestStatus.ORDERED, models.RequestStatus.PARTIALLY_ORDERED}
    report = ReplayReport()
    for req in session.exec(select(models.Request).order_by(models.Request.created_at)).all():
        if req.status not in done or req.created_at.date() < since:
            continue
        orders = session.exec(select(models.Order).where(models.Order.request_id == req.id,
                                                         models.Order.status == models.OrderStatus.CONFIRMED)).all()
        actual = sum(o.total_cents for o in orders)
        lines = [ln for ln in session.exec(select(models.RequestLine).where(models.RequestLine.request_id == req.id)).all()
                 if ln.fill_status == models.LineFillStatus.ASSIGNED and ln.qty_needed]
        if not orders or not lines:
            continue
        week = clock.week_start(req.created_at.date())
        start, end = clock.at_boston(week), clock.at_boston(week + timedelta(days=7))
        wanted = {ln.product_id for ln in lines}
        offers = []
        for m in mappings:
            if m.product_id not in wanted:
                continue
            in_week = [s for s in snaps.get(m.supplier_item_id, []) if start <= s.captured_at < end]
            if not in_week:
                continue
            snap = max(in_week, key=lambda s: s.captured_at)
            offers.append(Offer(suppliers[m.supplier_id].code, products[m.product_id].code,
                                items[m.supplier_item_id].sku, m.pack_size_base, snap.case_price_cents, snap.in_stock))
        needs = tuple(Need(products[ln.product_id].code, ln.qty_needed, products[ln.product_id].name) for ln in lines)
        best = make_plans(Problem(needs, tuple(offers), terms, needed_by=req.needed_by,
                                  order_date=req.created_at.date(), location=locations[req.location_id].code)).best
        if best.cant_fill or not best.meets_minimums:
            report.rows.append(ReplayRow(req.public_id, week, locations[req.location_id].name, actual, None,
                                         "no prices for every item that week" if best.cant_fill
                                         else "no plan met every minimum"))
        else:
            report.rows.append(ReplayRow(req.public_id, week, locations[req.location_id].name, actual,
                                         best.total_cents))
    return report


@dataclass(frozen=True)
class SpendRow:
    name: str
    orders: int
    total_cents: int


def load_spend(session) -> tuple[list[SpendRow], list[SpendRow]]:
    """Confirmed orders, by supplier and by location."""
    from sqlmodel import select

    from app import models

    suppliers = {s.id: s.name for s in session.exec(select(models.Supplier)).all()}
    locations = {loc.id: loc.name for loc in session.exec(select(models.Location)).all()}
    requests = {r.id: r for r in session.exec(select(models.Request)).all()}
    by_supplier: dict[str, list[int]] = defaultdict(list)
    by_location: dict[str, list[int]] = defaultdict(list)
    for o in session.exec(select(models.Order).where(models.Order.status == models.OrderStatus.CONFIRMED)).all():
        by_supplier[suppliers.get(o.supplier_id, "?")].append(o.total_cents)
        by_location[locations.get(requests[o.request_id].location_id, "?")].append(o.total_cents)

    def rows(groups):
        return sorted((SpendRow(k, len(v), sum(v)) for k, v in groups.items()), key=lambda r: -r.total_cents)

    return rows(by_supplier), rows(by_location)
