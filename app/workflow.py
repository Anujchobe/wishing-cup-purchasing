"""The order flow, from a manager's message to confirmed orders.

    submit_request   read the message, match products          -> DRAFT
    get_prices       ask every supplier (wait and retry)        -> QUOTED
    make_plan        pick the best plan, fill the lists         -> PENDING_APPROVAL
    approve          owner's changes, recalculate, all checks   -> APPROVED
    place_orders     checks again, PENDING, then each supplier  -> ORDERED or PARTIALLY_ORDERED
    reject                                                      -> REJECTED

Status changes: a status is only changed with a guarded UPDATE ("set APPROVED where it is still
PENDING_APPROVAL"), and its audit_log row is written in the same transaction. Both are saved, or
neither. If two people approve at once, the second finds the status already moved and does nothing.

Ordering never places an order twice:
  1. Every order is saved as PENDING (unique key: request id + supplier) and committed BEFORE any
     supplier is contacted.
  2. Each supplier gets its order with that key. A supplier that sees a key again returns the same
     confirmation instead of booking again.
  3. If the app stops anywhere in between, recover_pending_orders (run at every start) sends each
     PENDING order again with the same key. Either the supplier never got it and books it now, or
     it already had it and just returns the confirmation.
  4. One failing supplier doesn't stop the others. Unreachable: the order stays PENDING, to be
     retried. Refused (e.g. out of stock): FAILED, and the owner decides.

Database writes never wait on a supplier: each step reads, then calls suppliers with no write
transaction open, then writes.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app import checks, clock, models
from app.config import AppConfig, SupplierCallsConfig, data_dir
from app.db import audit, session_scope
from app.intake import IntakeResult, ProductRef, read_request
from app.llm.client import AnyClient
from app.matching import MatchReport, match_products
from app.optimizer import Need, Offer, PlanSet, Problem, SupplierTerms, make_plans
from app.pricing import cases_to_buy, format_cents, item_cost, price_flag
from app.suppliers.base import (
    OrderLine,
    Supplier,
    SupplierError,
    SupplierOrder,
    SupplierRejected,
    SupplierUnavailable,
    make_order_id,
)

S = models.RequestStatus
OS = models.OrderStatus
T = TypeVar("T")

TRANSITIONS: dict[S, set[S]] = {
    S.DRAFT: {S.QUOTED, S.REJECTED},
    S.QUOTED: {S.QUOTED, S.PENDING_APPROVAL, S.REJECTED},       # QUOTED -> QUOTED: prices fetched again
    S.PENDING_APPROVAL: {S.QUOTED, S.APPROVED, S.REJECTED},     # back to QUOTED: fresh prices, new plan
    S.APPROVED: {S.ORDERED, S.PARTIALLY_ORDERED},
    S.PARTIALLY_ORDERED: {S.ORDERED},
    S.ORDERED: set(),
    S.REJECTED: set(),
}


class WorkflowError(Exception):
    """Something the person needs to know about, in plain words."""


class InvalidTransition(WorkflowError):
    pass


# ------------------------------------------------------------------ status changes

def set_status(session: Session, request: models.Request, to: S, actor: str, action: str, **details: Any) -> None:
    """Move a request to a new status and write its audit row in the same transaction."""
    current = S(request.status)
    if to not in TRANSITIONS[current]:
        raise InvalidTransition(f"Request {request.public_id} is {current.value}; it can't become {to.value}.")
    result = session.exec(update(models.Request)
                          .where(models.Request.id == request.id, models.Request.status == current)
                          .values(status=to))
    if result.rowcount != 1:
        raise WorkflowError(f"Request {request.public_id} was changed by someone else. Reload it and try again.")
    request.status = to
    audit(session, actor, action, "request", request.public_id, from_status=current.value, to_status=to.value,
          **details)
    session.flush()


def _request(session: Session, request_id: int) -> models.Request:
    req = session.get(models.Request, request_id)
    if req is None:
        raise WorkflowError(f"There is no request {request_id}.")
    return req


def _lines(session: Session, request_id: int) -> list[models.RequestLine]:
    return list(session.exec(select(models.RequestLine).where(models.RequestLine.request_id == request_id)
                             .order_by(models.RequestLine.line_no)).all())


def new_public_id() -> str:
    return f"R-{uuid.uuid4().hex[:10]}"


# ------------------------------------------------------------------ 1. a new request

@dataclass
class Submission:
    request: models.Request | None
    intake: IntakeResult
    matches: MatchReport | None
    problems: list[str] = field(default_factory=list)  # why no request was created


def submit_request(session: Session, *, location_code: str, text: str, created_by: str, client: AnyClient,
                   cfg: AppConfig, model_tested: bool, needed_by: date | None = None,
                   today: date | None = None) -> Submission:
    """Read a message, match its products, and save it as a DRAFT request.

    The AI calls (reading, matching) happen before this session writes anything, so the call log,
    which writes on its own connection, never waits on our lock.
    """
    today = today or clock.today()
    location = session.exec(select(models.Location).where(models.Location.code == location_code)).first()
    if location is None or not location.active:
        raise WorkflowError(f"There is no location '{location_code}'.")
    products = [ProductRef(p.code, p.name, p.base_unit.value)
                for p in session.exec(select(models.Product).where(models.Product.active)).all()]
    result = read_request(text, products=products, client=client, today=today, model_tested=model_tested,
                          cfg=cfg.intake, needed_by=needed_by)
    problems = []
    if not result.lines:
        problems.append("No items could be read from the message.")
    if result.needed_by is None:
        problems.append("There is no needed-by date. Pick one in the form.")
    if problems:
        return Submission(None, result, None, problems + result.unclear)

    inputs = [LineInput(line.product_code, line.qty, line.note,
                        " ".join(filter(None, [line.quantity_text, line.unit_text, line.product_text])),
                        line.needs_check, "; ".join(line.check_reasons) or None) for line in result.lines]
    req, report = save_request(session, location_code=location_code, text=text, needed_by=result.needed_by,
                               lines=inputs, created_by=created_by, client=client, cfg=cfg,
                               model_tested=model_tested, read_by=result.source, ai_error=result.ai_error,
                               unclear=result.unclear)
    return Submission(req, result, report)


@dataclass(frozen=True)
class LineInput:
    """One line as a person left it in the request table."""
    product_code: str
    qty: Decimal               # in the product's base unit
    note: str = ""
    raw_text: str = ""
    needs_check: bool = False
    check_reason: str | None = None


def save_request(session: Session, *, location_code: str, text: str, needed_by: date | None,
                 lines: Sequence[LineInput], created_by: str, client: AnyClient, cfg: AppConfig,
                 model_tested: bool, read_by: str = "form", ai_error: str | None = None,
                 unclear: Sequence[str] = ()) -> tuple[models.Request, MatchReport]:
    """Save a request as DRAFT from checked lines, matching its products first (AI calls before any write)."""
    location = session.exec(select(models.Location).where(models.Location.code == location_code)).first()
    if location is None or not location.active:
        raise WorkflowError(f"There is no location '{location_code}'.")
    if needed_by is None:
        raise WorkflowError("Pick a needed-by date.")
    if not lines:
        raise WorkflowError("The request has no lines.")
    by_code = {p.code: p for p in session.exec(select(models.Product).where(models.Product.active)).all()}
    seen = set()
    for line in lines:
        if line.product_code not in by_code:
            raise WorkflowError(f"'{line.product_code}' isn't on our product list.")
        if line.product_code in seen:
            raise WorkflowError(f"{by_code[line.product_code].name} is on two lines. Put it on one line.")
        seen.add(line.product_code)
        if not isinstance(line.qty, Decimal) or not line.qty.is_finite() or line.qty <= 0:
            raise WorkflowError(f"{by_code[line.product_code].name}: the amount must be more than zero.")

    report = match_products(session, [line.product_code for line in lines], client, cfg, model_tested=model_tested)
    req = models.Request(public_id=new_public_id(), location_id=location.id, needed_by=needed_by,
                         original_text=text, status=S.DRAFT, source="manager", created_by=created_by)
    session.add(req)
    session.flush()
    for n, line in enumerate(lines, start=1):
        session.add(models.RequestLine(request_id=req.id, line_no=n, raw_text=line.raw_text or line.note,
                                       product_id=by_code[line.product_code].id, qty_needed=line.qty,
                                       needs_check=line.needs_check, check_reason=line.check_reason))
    audit(session, created_by, "request_created", "request", req.public_id, status=S.DRAFT.value,
          location=location.code, read_by=read_by, ai_error=ai_error, unclear=list(unclear), lines=len(lines),
          matches_saved=sum(d.saved for d in report.decisions))
    session.flush()
    return req, report


# ------------------------------------------------------------------ 2. prices

def call_supplier(fn: Callable[[], T], calls: SupplierCallsConfig, sleep: Callable[[float], None]) -> T:
    """Try, then wait a short time and try again, up to calls.retries more times.
    Only 'didn't answer' is retried; a refusal is final."""
    last: SupplierUnavailable | None = None
    for attempt in range(1 + calls.retries):
        try:
            return fn()
        except SupplierUnavailable as exc:
            last = exc
            if attempt < calls.retries:
                sleep(calls.retry_wait_seconds)
    raise last


@dataclass
class PriceReport:
    answered: list[str] = field(default_factory=list)
    no_answer: dict[str, str] = field(default_factory=dict)          # supplier -> what went wrong
    nothing_to_ask: list[str] = field(default_factory=list)          # no matched products for this request
    not_serving: list[str] = field(default_factory=list)
    set_aside: list[str] = field(default_factory=list)               # prices not used, and why
    missing: list[str] = field(default_factory=list)                 # SKUs a supplier no longer lists
    warnings: list[str] = field(default_factory=list)


def get_prices(session: Session, request_id: int, suppliers: dict[str, Supplier], cfg: AppConfig, *,
               now: datetime | None = None, sleep: Callable[[float], None] = time.sleep,
               actor: str = "system") -> PriceReport:
    """Ask each supplier for this request's prices and save them with the time we got them."""
    now = now or clock.now()
    req = _request(session, request_id)
    if S(req.status) not in (S.DRAFT, S.QUOTED, S.PENDING_APPROVAL):
        raise InvalidTransition(f"Request {req.public_id} is {req.status.value}; prices can't be fetched now.")
    location = session.get(models.Location, req.location_id)
    lines = _lines(session, request_id)
    product_ids = {line.product_id for line in lines}
    rows = {s.code: s for s in session.exec(select(models.Supplier).where(models.Supplier.active)).all()}
    mappings = [m for m in session.exec(select(models.ItemMapping)).all()
                if m.product_id in product_ids and m.pack_size_base is not None]
    items = {i.id: i for i in session.exec(select(models.SupplierItem)).all()}
    week_start = clock.at_boston(clock.week_start(now.date()))
    mapped_ids = {m.supplier_item_id for m in mappings}
    snapshots: dict[int, list[models.PriceSnapshot]] = {}
    for snap in session.exec(select(models.PriceSnapshot)).all():
        if snap.supplier_item_id in mapped_ids:
            snapshots.setdefault(snap.supplier_item_id, []).append(snap)

    # Phase 1: call the suppliers. Nothing is written yet, so no lock is held while we wait.
    report = PriceReport()
    answers: dict[str, list] = {}
    for sc in cfg.suppliers:
        if sc.code not in rows:
            continue
        if location.code not in sc.serves_locations:
            report.not_serving.append(sc.code)
            continue
        wanted = sorted({items[m.supplier_item_id].sku for m in mappings if m.supplier_id == rows[sc.code].id})
        if not wanted:
            report.nothing_to_ask.append(sc.code)
            continue
        connector = suppliers.get(sc.code)
        if connector is None:
            report.no_answer[sc.code] = "no connection is set up for this supplier"
            continue
        try:
            answers[sc.code] = call_supplier(lambda c=connector, w=wanted: c.get_quotes(w), cfg.supplier_calls, sleep)
            report.answered.append(sc.code)
        except SupplierError as exc:
            report.no_answer[sc.code] = str(exc)

    # Phase 2: save prices, flag suspicious ones, keep a snapshot for price history.
    for code, quotes in answers.items():
        sup = rows[code]
        by_sku = {i.sku: i for i in items.values() if i.supplier_id == sup.id}
        returned = {q.sku for q in quotes}
        for m in mappings:
            item = items[m.supplier_item_id]
            if m.supplier_id == sup.id and item.sku not in returned:
                report.missing.append(f"{sup.name} no longer lists SKU {item.sku} ({item.description}).")
        for q in quotes:
            item = by_sku.get(q.sku)
            if item is None:
                continue
            last_week = _last_price(snapshots.get(item.id, []), before=week_start)
            reason = price_flag(q.case_price_cents, last_week, cfg.purchasing.price_jump_limit)
            session.add(models.Quote(request_id=req.id, supplier_id=sup.id, supplier_item_id=item.id,
                                     case_price_cents=q.case_price_cents, in_stock=q.in_stock,
                                     quoted_at=q.quoted_at, fetched_at=now, dont_use=reason is not None,
                                     dont_use_reason=reason))
            session.add(models.PriceSnapshot(supplier_item_id=item.id, case_price_cents=q.case_price_cents,
                                             in_stock=q.in_stock, captured_at=q.quoted_at,
                                             dont_use=reason is not None, dont_use_reason=reason, source="quote"))
            item.case_price_cents, item.in_stock, item.updated_at = q.case_price_cents, q.in_stock, now
            session.add(item)
            if reason:
                report.set_aside.append(f"{sup.name}, {item.description}: {reason}.")

    report.warnings = [f"{rows[code].name} didn't answer ({error}), so the plan uses the other suppliers."
                       for code, error in report.no_answer.items()]
    req.suppliers_no_answer = sorted(report.no_answer)
    session.add(req)
    set_status(session, req, S.QUOTED, actor, "prices_fetched", answered=report.answered,
               no_answer=report.no_answer, set_aside=report.set_aside, missing=report.missing)
    return report


def _last_price(snaps: list[models.PriceSnapshot], *, before: datetime | None = None) -> int | None:
    usable = [s for s in snaps if not s.dont_use and (before is None or s.captured_at < before)]
    return max(usable, key=lambda s: s.captured_at).case_price_cents if usable else None


# ------------------------------------------------------------------ 3. the plan

@dataclass
class PlanReport:
    plans: PlanSet
    warnings: list[str]
    checks: list[checks.CheckResult]


def _latest_quotes(session: Session, request_id: int) -> dict[int, models.Quote]:
    """The newest quote per supplier item for this request."""
    latest: dict[int, models.Quote] = {}
    for q in session.exec(select(models.Quote).where(models.Quote.request_id == request_id)).all():
        if q.supplier_item_id not in latest or q.id > latest[q.supplier_item_id].id:
            latest[q.supplier_item_id] = q
    return latest


def make_plan(session: Session, request_id: int, cfg: AppConfig, *, now: datetime | None = None,
              actor: str = "system") -> PlanReport:
    now = now or clock.now()
    req = _request(session, request_id)
    if S(req.status) != S.QUOTED:
        raise InvalidTransition(f"Request {req.public_id} is {req.status.value}; get prices before planning.")
    ctx = _plan_inputs(session, req, cfg, now)
    lines, products, rows, items, quotes, mappings = (ctx[k] for k in
                                                      ("lines", "products", "rows", "items", "quotes", "mappings"))
    code_of = {sid: s.code for sid, s in rows.items()}
    plans = make_plans(ctx["problem"])
    best = plans.best

    sku_to_item = {(code_of[i.supplier_id], i.sku): i for i in items.values() if i.supplier_id in rows}
    supplier_id = {s.code: sid for sid, s in rows.items()}
    placed = {line.item: (sl.supplier, line) for sl in best.lists for line in sl.lines}
    cant = {c.item: c.reason for c in best.cant_fill}
    for line in lines:
        code = products[line.product_id].code
        if code in placed:
            supplier, plan_line = placed[code]
            item = sku_to_item[(supplier, plan_line.sku)]
            _assign(line, supplier_id[supplier], item.id, quotes[item.id], plan_line.cost.cases)
        else:
            _cant_fill(line, cant.get(code, "no supplier can supply it"))
        session.add(line)
    req.plan_total_cents = best.total_cents
    session.flush()
    missing = [_cheaper_last_time(session, code, lines, products, mappings) for code in req.suppliers_no_answer]
    req.status_note = " ".join(missing) or None
    session.add(req)
    # The optimizer also says "X didn't answer"; ours says it with the items, so keep only ours.
    warnings = missing + [w for w in plans.warnings if "didn't answer" not in w] + list(best.warnings)
    set_status(session, req, S.PENDING_APPROVAL, actor, "plan_made", total_cents=best.total_cents,
               solved_by=best.solved_by, savings_vs_one_supplier_cents=plans.savings_vs_one_supplier_cents,
               warnings=warnings)
    results = checks.run_checks(checks.load_view(session, request_id, cfg, now), "approval")
    return PlanReport(plans, warnings, results)


def _plan_inputs(session: Session, req: models.Request, cfg: AppConfig, now: datetime) -> dict[str, Any]:
    """Everything the optimizer needs for a request, from its latest quotes."""
    location = session.get(models.Location, req.location_id)
    lines = _lines(session, req.id)
    products = {p.id: p for p in session.exec(select(models.Product)).all()}
    rows = {s.id: s for s in session.exec(select(models.Supplier).where(models.Supplier.active)).all()}
    code_of = {sid: s.code for sid, s in rows.items()}
    items = {i.id: i for i in session.exec(select(models.SupplierItem)).all()}
    quotes = _latest_quotes(session, req.id)
    mappings = {m.supplier_item_id: m for m in session.exec(select(models.ItemMapping)).all()}
    wanted = {line.product_id for line in lines}
    offers = []
    for item_id, q in quotes.items():
        m = mappings.get(item_id)
        if m is None or m.product_id not in wanted or m.pack_size_base is None or q.supplier_id not in rows:
            continue
        if q.dont_use and not q.price_accepted_by:
            continue
        offers.append(Offer(code_of[q.supplier_id], products[m.product_id].code, items[item_id].sku,
                            m.pack_size_base, q.case_price_cents, q.in_stock))
    terms = tuple(SupplierTerms.from_config(sc, working=sc.code not in req.suppliers_no_answer)
                  for sc in cfg.suppliers if any(r.code == sc.code for r in rows.values()))
    needs = tuple(Need(products[line.product_id].code, line.qty_needed, products[line.product_id].name)
                  for line in lines)
    problem = Problem(needs, tuple(offers), terms, needed_by=req.needed_by, order_date=now.date(),
                      location=location.code)
    return {"problem": problem, "lines": lines, "products": products, "rows": rows, "items": items,
            "quotes": quotes, "mappings": mappings}


def plan_options(session: Session, request_id: int, cfg: AppConfig, *, now: datetime | None = None) -> PlanSet | None:
    """Plans a, b and c from the request's latest prices, for showing. Nothing is saved."""
    req = _request(session, request_id)
    if not session.exec(select(models.Quote).where(models.Quote.request_id == request_id)).first():
        return None
    return make_plans(_plan_inputs(session, req, cfg, now or clock.now())["problem"])


def _cheaper_last_time(session: Session, code: str, lines: Sequence[models.RequestLine],
                       products: dict[int, models.Product], mappings: dict[int, models.ItemMapping]) -> str:
    """'X didn't answer...', plus the items X was cheaper for last time than what this plan pays.

    Compared on the real cost for this need (whole cases), against the plan's own line, because
    that is what not hearing from X actually costs.
    """
    sup = session.exec(select(models.Supplier).where(models.Supplier.code == code)).one()
    text = f"{sup.name} didn't answer, so the plan uses the other suppliers."
    cheaper = []
    for line in lines:
        m = next((x for x in mappings.values() if x.product_id == line.product_id and x.supplier_id == sup.id
                  and x.pack_size_base is not None), None)
        if m is None:
            continue
        snaps = session.exec(select(models.PriceSnapshot)
                             .where(models.PriceSnapshot.supplier_item_id == m.supplier_item_id)).all()
        last = _last_price(list(snaps))
        if not last:
            continue
        then = item_cost(line.qty_needed, m.pack_size_base, last).cost_cents
        name = products[line.product_id].name
        if line.fill_status != models.LineFillStatus.ASSIGNED:
            cheaper.append(f"{name} ({format_cents(then)} last time; it can't be filled now)")
        elif then < line.line_cost_cents:
            cheaper.append(f"{name} ({format_cents(then)} last time vs {format_cents(line.line_cost_cents)} now)")
    if cheaper:
        text += f" Last time {sup.name} was cheaper for: " + "; ".join(cheaper) + "."
    return text


def _assign(line: models.RequestLine, supplier_id: int, item_id: int, quote: models.Quote, cases: int) -> None:
    line.fill_status = models.LineFillStatus.ASSIGNED
    line.assigned_supplier_id, line.supplier_item_id, line.quote_id = supplier_id, item_id, quote.id
    line.cases, line.case_price_cents = cases, quote.case_price_cents
    line.line_cost_cents = cases * quote.case_price_cents
    line.plan_note = None


def _cant_fill(line: models.RequestLine, reason: str) -> None:
    line.fill_status = models.LineFillStatus.CANT_FILL
    line.assigned_supplier_id = line.supplier_item_id = line.quote_id = None
    line.cases = line.case_price_cents = line.line_cost_cents = None
    line.plan_note = reason


def _plan_total(session: Session, lines: Sequence[models.RequestLine]) -> int:
    used = {line.assigned_supplier_id for line in lines if line.fill_status == models.LineFillStatus.ASSIGNED}
    fees = sum(session.get(models.Supplier, sid).delivery_fee_cents for sid in used)
    return sum(line.line_cost_cents or 0 for line in lines if line.fill_status == models.LineFillStatus.ASSIGNED) + fees


# ------------------------------------------------------------------ 4. approval

@dataclass(frozen=True)
class MoveItem:
    product: str     # product code
    supplier: str    # supplier code


@dataclass(frozen=True)
class MarkCantFill:
    product: str
    reason: str = "set by the owner"


@dataclass(frozen=True)
class ConfirmItem:
    """The owner checked this line: its quantity, its reading and its product match."""
    product: str


@dataclass(frozen=True)
class AcceptPrice:
    """The owner checked a price that was set aside and allows it."""
    product: str
    supplier: str


Change = MoveItem | MarkCantFill | ConfirmItem | AcceptPrice


@dataclass
class ApprovalResult:
    approved: bool
    already: bool = False
    change_errors: list[str] = field(default_factory=list)
    checks: list[checks.CheckResult] = field(default_factory=list)
    total_cents: int | None = None

    @property
    def message(self) -> str:
        if self.change_errors:
            return "Nothing was changed:\n" + "\n".join(f"- {e}" for e in self.change_errors)
        if self.already:
            return "Already approved; nothing was changed."
        return "Approved." if self.approved else checks.summary(self.checks)


def approve(session: Session, request_id: int, approver: str, changes: Sequence[Change] = (),
            cfg: AppConfig | None = None, *, now: datetime | None = None) -> ApprovalResult:
    """Apply the owner's changes, recalculate the totals, run every check, and approve if all pass.

    Changes are validated together first: if any is invalid, none is applied. They are applied in
    this order: accepted prices, then moves and can't-fills (in the order given), then confirmations.
    Approving an already approved request does nothing, so a double click can't order twice.
    """
    if cfg is None:
        raise ValueError("cfg is required")
    now = now or clock.now()
    req = _request(session, request_id)
    status = S(req.status)
    if status in (S.APPROVED, S.PARTIALLY_ORDERED, S.ORDERED):
        errors = ["The request was already approved, so these changes were not applied."] if changes else []
        return ApprovalResult(True, already=True, change_errors=errors, total_cents=req.plan_total_cents)
    if status != S.PENDING_APPROVAL:
        raise InvalidTransition(f"Request {req.public_id} is {status.value}; it can't be approved now.")

    changed = apply_changes(session, request_id, approver, changes, cfg, now=now)
    if changed.change_errors:
        return ApprovalResult(False, change_errors=changed.change_errors, total_cents=req.plan_total_cents)
    results = changed.checks
    described = [_describe(c) for c in changes]
    if checks.failures(results):
        audit(session, approver, "approval_blocked", "request", req.public_id, changes=described,
              total_cents=req.plan_total_cents,
              reasons=[r for c in checks.failures(results) for r in c.reasons])
        session.flush()
        return ApprovalResult(False, checks=results, total_cents=req.plan_total_cents)
    req.approved_by, req.approved_at = approver, now
    session.add(req)
    set_status(session, req, S.APPROVED, approver, "approved", changes=described, total_cents=req.plan_total_cents)
    return ApprovalResult(True, checks=results, total_cents=req.plan_total_cents)


@dataclass
class ChangeResult:
    change_errors: list[str] = field(default_factory=list)
    checks: list[checks.CheckResult] = field(default_factory=list)
    total_cents: int | None = None


def apply_changes(session: Session, request_id: int, person: str, changes: Sequence[Change], cfg: AppConfig, *,
                  now: datetime | None = None) -> ChangeResult:
    """The owner's edits to a plan waiting for approval: validate all, apply all, recalculate, re-check."""
    now = now or clock.now()
    req = _request(session, request_id)
    if S(req.status) != S.PENDING_APPROVAL:
        raise InvalidTransition(f"Request {req.public_id} is {req.status.value}; its lists can't be changed now.")
    lines = _lines(session, request_id)
    errors = _apply_changes(session, req, lines, list(changes), person, now, dry_run=True)
    if errors:
        return ChangeResult(errors, total_cents=req.plan_total_cents)
    _apply_changes(session, req, lines, list(changes), person, now, dry_run=False)
    req.plan_total_cents = _plan_total(session, lines)
    session.add(req)
    if changes:
        audit(session, person, "plan_changed", "request", req.public_id, changes=[_describe(c) for c in changes],
              total_cents=req.plan_total_cents)
    session.flush()
    results = checks.run_checks(checks.load_view(session, request_id, cfg, now), "approval")
    return ChangeResult([], results, req.plan_total_cents)


def _describe(change: Change) -> str:
    return f"{type(change).__name__}({', '.join(f'{k}={v!r}' for k, v in vars(change).items())})"


def _apply_changes(session: Session, req: models.Request, lines: list[models.RequestLine], changes: list[Change],
                   approver: str, now: datetime, *, dry_run: bool) -> list[str]:
    products = {p.code: p for p in session.exec(select(models.Product)).all()}
    suppliers = {s.code: s for s in session.exec(select(models.Supplier).where(models.Supplier.active)).all()}
    items = {i.id: i for i in session.exec(select(models.SupplierItem)).all()}
    mappings = list(session.exec(select(models.ItemMapping)).all())
    quotes = _latest_quotes(session, req.id)
    line_of = {line.product_id: line for line in lines}
    errors: list[str] = []
    accepted: set[int] = {q.id for q in quotes.values() if q.price_accepted_by}

    def find(change) -> tuple[models.RequestLine | None, models.Product | None]:
        product = products.get(change.product)
        line = line_of.get(product.id) if product else None
        if line is None:
            errors.append(f"'{change.product}' isn't in this request.")
        return line, product

    def mapping_at(product, supplier) -> models.ItemMapping | None:
        return next((m for m in mappings if m.product_id == product.id and m.supplier_id == supplier.id), None)

    order = ([c for c in changes if isinstance(c, AcceptPrice)]
             + [c for c in changes if isinstance(c, (MoveItem, MarkCantFill))]
             + [c for c in changes if isinstance(c, ConfirmItem)])
    for change in order:
        if not isinstance(change, (MoveItem, MarkCantFill, ConfirmItem, AcceptPrice)):
            errors.append(f"Unknown change: {change!r}.")
            continue
        line, product = find(change)
        if line is None:
            continue
        if isinstance(change, (MoveItem, AcceptPrice)):
            supplier = suppliers.get(change.supplier)
            if supplier is None:
                errors.append(f"There is no supplier '{change.supplier}'.")
                continue
            m = mapping_at(product, supplier)
            quote = quotes.get(m.supplier_item_id) if m else None
            if m is None:
                errors.append(f"We have no product match for {product.name} at {supplier.name}.")
                continue
            if quote is None:
                errors.append(f"There is no price from {supplier.name} for {product.name} in this request.")
                continue
            if isinstance(change, AcceptPrice):
                if quote.case_price_cents <= 0:
                    errors.append(f"{supplier.name}'s price for {product.name} is "
                                  f"{format_cents(quote.case_price_cents)}; that can't be accepted.")
                    continue
                if not dry_run:
                    quote.price_accepted_by = approver
                    session.add(quote)
                accepted.add(quote.id)
                continue
            if m.pack_size_base is None:
                errors.append(f"The case size of {product.name} at {supplier.name} is unknown.")
                continue
            if quote.dont_use and quote.id not in accepted:
                errors.append(f"{supplier.name}'s price for {product.name} was set aside ({quote.dont_use_reason}). "
                              "Accept it first if a person checked it.")
                continue
            if not dry_run:
                _assign(line, supplier.id, m.supplier_item_id, quote, cases_to_buy(line.qty_needed, m.pack_size_base))
                session.add(line)
        elif isinstance(change, MarkCantFill):
            if not dry_run:
                _cant_fill(line, change.reason)
                session.add(line)
        elif isinstance(change, ConfirmItem) and not dry_run:
            line.needs_check = False
            line.qty_confirmed = True
            if line.check_reason:
                line.check_reason = f"confirmed by {approver}: {line.check_reason}"
            session.add(line)
            m = next((x for x in mappings if x.supplier_item_id == line.supplier_item_id), None)
            if m is not None and m.needs_check:
                m.needs_check, m.confirmed_by, m.confirmed_at = False, approver, now
                session.add(m)
    if not dry_run:
        session.flush()
    return errors


def reject(session: Session, request_id: int, by: str, reason: str) -> None:
    req = _request(session, request_id)
    req.status_note = reason
    session.add(req)
    set_status(session, req, S.REJECTED, by, "rejected", reason=reason)


# ------------------------------------------------------------------ 5. orders

@dataclass
class OrderOutcome:
    supplier: str
    order_key: str
    status: str               # CONFIRMED | PENDING | FAILED
    confirmation_id: str | None = None
    error: str | None = None


@dataclass
class PlacementReport:
    request_status: str = ""
    blocked: list[checks.CheckResult] = field(default_factory=list)
    outcomes: list[OrderOutcome] = field(default_factory=list)
    purchase_orders: list[Path] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.blocked:
            return checks.summary(self.blocked)
        parts = [f"{o.supplier}: {o.status.lower()}" + (f" ({o.error})" if o.error else "") for o in self.outcomes]
        return f"Request is {self.request_status}. " + "; ".join(parts)


def _contact_supplier(connector: Supplier, order: SupplierOrder, order_key: str, calls: SupplierCallsConfig,
                      sleep: Callable[[float], None]) -> str:
    """The one place an order leaves the app. (Tests stop the app here to check recovery.)"""
    return call_supplier(lambda: connector.place_order(order, order_key), calls, sleep)


def place_orders(engine: Engine, request_id: int, suppliers: dict[str, Supplier], cfg: AppConfig, *,
                 now: datetime | None = None, sleep: Callable[[float], None] = time.sleep,
                 po_dir: Path | None = None) -> PlacementReport:
    """Check again, save every order as PENDING, then contact each supplier."""
    now = now or clock.now()
    report = PlacementReport()
    try:
        blocked = _save_pending_orders(engine, request_id, cfg, now, report)
    except IntegrityError:  # another place_orders saved them first; its PENDING orders get sent below
        blocked = False
    if blocked or report.request_status == S.ORDERED.value:
        return report
    # The PENDING orders are committed now, before any supplier hears about them.
    _send_pending(engine, request_id, suppliers, cfg, now, sleep, report)
    with session_scope(engine) as session:
        report.request_status = _settle(session, request_id)
    report.purchase_orders = _write_purchase_orders(engine, request_id, cfg, po_dir)
    return report


def _save_pending_orders(engine: Engine, request_id: int, cfg: AppConfig, now: datetime,
                         report: PlacementReport) -> bool:
    """Run the checks again and save one PENDING order per supplier list. True if blocked."""
    with session_scope(engine) as session:
        req = _request(session, request_id)
        status = S(req.status)
        if status == S.ORDERED:
            report.request_status = status.value
            return False
        if status not in (S.APPROVED, S.PARTIALLY_ORDERED):
            raise InvalidTransition(f"Request {req.public_id} is {status.value}; approve it before ordering.")
        lines = [line for line in _lines(session, request_id) if line.fill_status == models.LineFillStatus.ASSIGNED]
        existing = {o.supplier_id for o in session.exec(select(models.Order)
                                                        .where(models.Order.request_id == request_id)).all()}
        new_suppliers = sorted({line.assigned_supplier_id for line in lines} - existing)
        if new_suppliers:
            results = checks.run_checks(checks.load_view(session, request_id, cfg, now), "ordering")
            if checks.failures(results):
                report.blocked = checks.failures(results)
                report.request_status = status.value
                audit(session, "system", "ordering_blocked", "request", req.public_id,
                      reasons=[r for c in report.blocked for r in c.reasons])
                return True
            location = session.get(models.Location, req.location_id)
            for sid in new_suppliers:
                sup = session.get(models.Supplier, sid)
                subtotal = sum(line.line_cost_cents for line in lines if line.assigned_supplier_id == sid)
                arrives = SupplierTerms.from_config(cfg.supplier(sup.code)).earliest_delivery(now.date())
                order = models.Order(order_key=make_order_id(req.public_id, sup.code), request_id=req.id,
                                     supplier_id=sid, status=OS.PENDING, subtotal_cents=subtotal,
                                     delivery_fee_cents=sup.delivery_fee_cents,
                                     total_cents=subtotal + sup.delivery_fee_cents,
                                     deliver_on=arrives or req.needed_by)
                session.add(order)
                audit(session, "system", "order_saved_pending", "order", order.order_key, supplier=sup.code,
                      location=location.code, total_cents=order.total_cents)
    return False


def _send_pending(engine: Engine, request_id: int | None, suppliers: dict[str, Supplier], cfg: AppConfig,
                  now: datetime, sleep: Callable[[float], None], report: PlacementReport) -> None:
    with session_scope(engine) as session:
        query = select(models.Order).where(models.Order.status == OS.PENDING)
        if request_id is not None:
            query = query.where(models.Order.request_id == request_id)
        pending = [o.id for o in session.exec(query.order_by(models.Order.id)).all()]
    for order_id in pending:
        outcome = _send_one(engine, order_id, suppliers, cfg, now, sleep)
        if outcome is not None:
            report.outcomes.append(outcome)


def _send_one(engine: Engine, order_id: int, suppliers: dict[str, Supplier], cfg: AppConfig, now: datetime,
              sleep: Callable[[float], None]) -> OrderOutcome | None:
    with session_scope(engine) as session:
        order = session.get(models.Order, order_id)
        if order is None or order.status != OS.PENDING:
            return None
        sup = session.get(models.Supplier, order.supplier_id)
        req = session.get(models.Request, order.request_id)
        location = session.get(models.Location, req.location_id)
        lines = session.exec(select(models.RequestLine).where(
            models.RequestLine.request_id == req.id, models.RequestLine.assigned_supplier_id == sup.id,
            models.RequestLine.fill_status == models.LineFillStatus.ASSIGNED).order_by(models.RequestLine.line_no)).all()
        try:
            payload = SupplierOrder(
                supplier_code=sup.code, location_code=location.code, deliver_on=order.deliver_on,
                lines=[OrderLine(sku=session.get(models.SupplierItem, line.supplier_item_id).sku, cases=line.cases,
                                 expected_case_price_cents=line.case_price_cents) for line in lines])
        except ValueError as exc:  # e.g. the list became empty; never send a broken order
            order.status, order.last_error = OS.FAILED, f"the order can't be built: {exc}"
            session.add(order)
            audit(session, "system", "order_refused", "order", order.order_key, supplier=sup.code,
                  error=order.last_error)
            return OrderOutcome(sup.code, order.order_key, OS.FAILED.value, None, order.last_error)
        order.attempts += 1
        session.add(order)
        key, code = order.order_key, sup.code

    connector = suppliers.get(code)
    try:
        if connector is None:
            raise SupplierUnavailable("no connection is set up for this supplier")
        confirmation, status, error = _contact_supplier(connector, payload, key, cfg.supplier_calls, sleep), OS.CONFIRMED, None
    except SupplierRejected as exc:
        confirmation, status, error = None, OS.FAILED, str(exc)
    except SupplierUnavailable as exc:
        confirmation, status, error = None, OS.PENDING, str(exc)

    with session_scope(engine) as session:
        order = session.get(models.Order, order_id)
        if order.status != OS.PENDING:  # someone else finished it meanwhile
            return OrderOutcome(code, key, order.status.value, order.confirmation_id)
        order.status, order.confirmation_id, order.last_error = status, confirmation, error
        if status == OS.CONFIRMED:
            order.placed_at = now
        session.add(order)
        action = {OS.CONFIRMED: "order_confirmed", OS.FAILED: "order_refused", OS.PENDING: "order_not_sent_yet"}[status]
        audit(session, "system", action, "order", key, supplier=code, confirmation_id=confirmation, error=error,
              attempt=order.attempts)
    return OrderOutcome(code, key, status.value, confirmation, error)


def _settle(session: Session, request_id: int) -> str:
    """ORDERED when every list is confirmed, PARTIALLY_ORDERED when some are."""
    req = _request(session, request_id)
    lists = {line.assigned_supplier_id for line in _lines(session, request_id)
             if line.fill_status == models.LineFillStatus.ASSIGNED}
    orders = session.exec(select(models.Order).where(models.Order.request_id == request_id)).all()
    confirmed = {o.supplier_id for o in orders if o.status == OS.CONFIRMED}
    current = S(req.status)
    target = S.ORDERED if lists and lists <= confirmed else S.PARTIALLY_ORDERED if confirmed else current
    if target != current and target in TRANSITIONS[current]:
        set_status(session, req, target, "system", "orders_settled",
                   confirmed=len(confirmed), lists=len(lists),
                   not_confirmed=[o.order_key for o in orders if o.status != OS.CONFIRMED])
    return S(req.status).value


@dataclass
class RecoveryReport:
    outcomes: list[OrderOutcome] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.outcomes:
            return "no pending orders"
        return "; ".join(f"{o.order_key}: {o.status.lower()}" + (f" ({o.error})" if o.error else "")
                         for o in self.outcomes)


def recover_pending_orders(engine: Engine, suppliers: dict[str, Supplier], cfg: AppConfig, *,
                           now: datetime | None = None, sleep: Callable[[float], None] = time.sleep,
                           po_dir: Path | None = None) -> RecoveryReport:
    """Run at every start: send every PENDING order again with its same key.

    The safety checks are not run again here on purpose. They passed before the order was saved as
    PENDING, and the supplier may already have this order; re-sending is the only way to find out,
    and the same key guarantees it can't be booked twice.
    """
    now = now or clock.now()
    placement = PlacementReport()
    _send_pending(engine, None, suppliers, cfg, now, sleep, placement)
    with session_scope(engine) as session:
        open_requests = {r.id for r in session.exec(select(models.Request)).all()
                         if S(r.status) in (S.APPROVED, S.PARTIALLY_ORDERED)}
        request_ids = {o.request_id for o in session.exec(select(models.Order)).all()} & open_requests
        for rid in sorted(request_ids):
            _settle(session, rid)
    for rid in sorted(request_ids):
        _write_purchase_orders(engine, rid, cfg, po_dir)
    return RecoveryReport(placement.outcomes)


def _write_purchase_orders(engine: Engine, request_id: int, cfg: AppConfig, po_dir: Path | None) -> list[Path]:
    from app import po

    folder = po_dir or data_dir(cfg) / "purchase_orders"
    paths = []
    with session_scope(engine) as session:
        orders = session.exec(select(models.Order).where(models.Order.request_id == request_id,
                                                         models.Order.status == OS.CONFIRMED)).all()
        for order in orders:
            paths += po.save(po.load_purchase_order(session, order.id, cfg), folder)
    return paths


# ------------------------------------------------------------------ the whole way, for scripts and tests

def ready_for_approval(engine: Engine, *, location_code: str, text: str, needed_by: date | None,
                       suppliers: dict[str, Supplier], client: AnyClient, cfg: AppConfig, model_tested: bool,
                       created_by: str = "manager", now: datetime | None = None,
                       sleep: Callable[[float], None] = time.sleep) -> tuple[int, PriceReport, PlanReport]:
    """submit_request + get_prices + make_plan, each in its own transaction."""
    with session_scope(engine) as session:
        sub = submit_request(session, location_code=location_code, text=text, created_by=created_by,
                             client=client, cfg=cfg, model_tested=model_tested, needed_by=needed_by)
        if sub.request is None:
            raise WorkflowError("The request wasn't created: " + " ".join(sub.problems))
        request_id = sub.request.id
    with session_scope(engine) as session:
        prices = get_prices(session, request_id, suppliers, cfg, now=now, sleep=sleep)
    with session_scope(engine) as session:
        plan = make_plan(session, request_id, cfg, now=now)
    return request_id, prices, plan
