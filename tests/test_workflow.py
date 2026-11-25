"""The order flow on the fake suppliers: statuses, prices, approval, ordering and recovery."""
from datetime import timedelta

import pytest
from sqlmodel import select

from app import clock, models, workflow
from app.config import SupplierFaults
from app.db import session_scope
from app.llm.client import NoAIClient
from app.suppliers import fake_world
from app.suppliers.mock import build_suppliers
from app.workflow import AcceptPrice, ConfirmItem, MoveItem

NO_AI = NoAIClient("No AI key and model in .env. Using text matching.")
NO_WAIT = lambda seconds: None  # noqa: E731

# Big enough that the best plan splits between two suppliers (US Foods and PFG on the fake data).
BIG = ("200 lb all-purpose flour, 90 lb cake flour, 180 lb granulated sugar, 70 lb powdered sugar, "
       "130 lb unsalted butter, 2000 fl oz heavy cream, 1440 eggs, 66 lb cream cheese, 44 lb dark chocolate, "
       "18 lb cocoa powder, 9 lb baking powder")
BIG_PRODUCTS = ["ap_flour", "cake_flour", "granulated_sugar", "powdered_sugar", "butter_unsalted", "heavy_cream",
                "eggs", "cream_cheese", "dark_chocolate", "cocoa_powder", "baking_powder"]
SMALL = "100 lb all-purpose flour, 65 lb unsalted butter, 1000 fl oz heavy cream, 720 eggs, 66 lb cream cheese, 90 lb granulated sugar"
SMALL_PRODUCTS = ["ap_flour", "butter_unsalted", "heavy_cream", "eggs", "cream_cheese", "granulated_sugar"]


class AppStopped(BaseException):
    """Stands in for the app being killed (BaseException, so nothing in the app catches it)."""


def prepare(engine, cfg, sups, text=BIG):
    return workflow.ready_for_approval(engine, location_code="factory", text=text,
                                       needed_by=clock.today() + timedelta(days=8), suppliers=sups, client=NO_AI,
                                       cfg=cfg, model_tested=False, sleep=NO_WAIT)


def approve(engine, cfg, request_id, changes=(), now=None):
    with session_scope(engine) as s:
        return workflow.approve(s, request_id, "owner", changes, cfg, now=now)


def confirm(products=BIG_PRODUCTS):
    return [ConfirmItem(p) for p in products]


def place(engine, cfg, request_id, sups, tmp_path):
    return workflow.place_orders(engine, request_id, sups, cfg, sleep=NO_WAIT, po_dir=tmp_path / "po")


def request_status(engine, request_id):
    with session_scope(engine) as s:
        return s.get(models.Request, request_id).status


def orders(engine, request_id):
    with session_scope(engine) as s:
        codes = {x.id: x.code for x in s.exec(select(models.Supplier)).all()}
        return {codes[o.supplier_id]: o for o in s.exec(select(models.Order)
                                                        .where(models.Order.request_id == request_id)).all()}


def plan_suppliers(engine, request_id):
    with session_scope(engine) as s:
        codes = {x.id: x.code for x in s.exec(select(models.Supplier)).all()}
        return sorted({codes[line.assigned_supplier_id] for line in s.exec(select(models.RequestLine).where(
            models.RequestLine.request_id == request_id)).all() if line.assigned_supplier_id})


def ready_and_approved(engine, cfg, sups):
    request_id, _, _ = prepare(engine, cfg, sups)
    result = approve(engine, cfg, request_id, confirm())
    assert result.approved, result.message
    return request_id


# ------------------------------------------------------------------ the whole flow

def test_full_flow_status_history_and_purchase_orders(live_engine, cfg, suppliers, tmp_path):
    request_id, prices, plan = prepare(live_engine, cfg, suppliers)
    assert set(prices.answered) == {"sysco", "usfoods", "pfg"} and prices.not_serving == ["supplier4"]
    assert any("$0.00" in x for x in prices.set_aside)          # PFG's planted $0 heavy cream is set aside
    assert request_status(live_engine, request_id) == models.RequestStatus.PENDING_APPROVAL
    assert not plan.plans.best.warnings, plan.plans.best.warnings

    result = approve(live_engine, cfg, request_id, confirm())
    assert result.approved, result.message
    report = place(live_engine, cfg, request_id, suppliers, tmp_path)
    assert report.request_status == "ORDERED", report.summary
    placed = orders(live_engine, request_id)
    assert sorted(placed) == plan_suppliers(live_engine, request_id)
    assert all(o.status == models.OrderStatus.CONFIRMED and o.confirmation_id.startswith("FAKE-") for o in placed.values())

    with session_scope(live_engine) as s:
        public_id = s.get(models.Request, request_id).public_id
        rows = s.exec(select(models.AuditLog).where(models.AuditLog.entity_id == public_id)
                      .order_by(models.AuditLog.id)).all()
    moves = [(r.details["from_status"], r.details["to_status"]) for r in rows if "to_status" in r.details]
    assert moves == [("DRAFT", "QUOTED"), ("QUOTED", "PENDING_APPROVAL"), ("PENDING_APPROVAL", "APPROVED"),
                     ("APPROVED", "ORDERED")]

    files = sorted(p.name for p in report.purchase_orders)
    assert files == sorted(f"{o.order_key}.{ext}" for o in placed.values() for ext in ("csv", "txt"))
    csv_text = next(p for p in report.purchase_orders if p.suffix == ".csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "sku,description,cases,case_price,line_total,deliver_to,needed_by"


def test_approving_twice_places_each_order_only_once(live_engine, cfg, suppliers, tmp_path):
    request_id, _, _ = prepare(live_engine, cfg, suppliers)
    assert approve(live_engine, cfg, request_id, confirm()).approved
    again = approve(live_engine, cfg, request_id, confirm())
    assert again.approved and again.already                    # nothing happens the second time
    first = place(live_engine, cfg, request_id, suppliers, tmp_path)
    second = place(live_engine, cfg, request_id, suppliers, tmp_path)
    assert first.request_status == second.request_status == "ORDERED" and second.outcomes == []
    used = plan_suppliers(live_engine, request_id)
    assert len(used) >= 2
    assert {code: suppliers[code].placements for code in used} == {code: 1 for code in used}
    assert len(orders(live_engine, request_id)) == len(used)


# ------------------------------------------------------------------ the app stops part way

def test_stop_after_pending_before_contacting_then_restart_places_exactly_once(live_engine, cfg, tmp_path,
                                                                               monkeypatch):
    ledgers = tmp_path / "ledgers"
    sups = build_suppliers(cfg, ledger_dir=ledgers, sleep=NO_WAIT)
    request_id = ready_and_approved(live_engine, cfg, sups)

    def stop(*args, **kwargs):
        raise AppStopped()

    monkeypatch.setattr(workflow, "_contact_supplier", stop)
    with pytest.raises(AppStopped):
        place(live_engine, cfg, request_id, sups, tmp_path)
    saved = orders(live_engine, request_id)
    assert saved and all(o.status == models.OrderStatus.PENDING for o in saved.values())   # saved first...
    assert all(s.placements == 0 for s in sups.values())                                  # ...nothing sent

    monkeypatch.undo()
    restarted = build_suppliers(cfg, ledger_dir=ledgers, sleep=NO_WAIT)   # a new process: suppliers from disk
    recovery = workflow.recover_pending_orders(live_engine, restarted, cfg, sleep=NO_WAIT, po_dir=tmp_path / "po")
    assert {o.status for o in recovery.outcomes} == {"CONFIRMED"}
    assert request_status(live_engine, request_id) == models.RequestStatus.ORDERED
    used = plan_suppliers(live_engine, request_id)
    assert {c: restarted[c].placements for c in used} == {c: 1 for c in used}

    assert workflow.recover_pending_orders(live_engine, restarted, cfg, sleep=NO_WAIT).summary == "no pending orders"
    assert {c: build_suppliers(cfg, ledger_dir=ledgers)[c].placements for c in used} == {c: 1 for c in used}


def test_stop_after_the_supplier_booked_then_restart_gets_the_same_confirmation(live_engine, cfg, tmp_path,
                                                                                monkeypatch):
    ledgers = tmp_path / "ledgers"
    sups = build_suppliers(cfg, ledger_dir=ledgers, sleep=NO_WAIT)
    request_id = ready_and_approved(live_engine, cfg, sups)
    real_contact = workflow._contact_supplier

    def booked_then_stop(*args, **kwargs):
        real_contact(*args, **kwargs)   # the supplier books it...
        raise AppStopped()             # ...and the app dies before saving the confirmation

    monkeypatch.setattr(workflow, "_contact_supplier", booked_then_stop)
    with pytest.raises(AppStopped):
        place(live_engine, cfg, request_id, sups, tmp_path)
    monkeypatch.undo()

    restarted = build_suppliers(cfg, ledger_dir=ledgers, sleep=NO_WAIT)
    workflow.recover_pending_orders(live_engine, restarted, cfg, sleep=NO_WAIT, po_dir=tmp_path / "po")
    for code, order in orders(live_engine, request_id).items():
        assert order.status == models.OrderStatus.CONFIRMED
        assert order.confirmation_id == fake_world.confirmation_for(code, order.order_key)
        assert restarted[code].placements == 1


# ------------------------------------------------------------------ checks that block

def test_old_prices_block_approval(live_engine, cfg, suppliers):
    request_id, _, _ = prepare(live_engine, cfg, suppliers)
    later = clock.now() + timedelta(hours=25)
    result = approve(live_engine, cfg, request_id, confirm(), now=later)
    assert not result.approved
    assert [c.name for c in result.checks if not c.passed] == ["fresh_prices"]
    assert "older than the 24-hour limit" in result.message
    assert request_status(live_engine, request_id) == models.RequestStatus.PENDING_APPROVAL

    with session_scope(live_engine) as s:   # the way out: fresh prices, a new plan
        workflow.get_prices(s, request_id, suppliers, cfg, sleep=NO_WAIT)
    with session_scope(live_engine) as s:
        workflow.make_plan(s, request_id, cfg)
    assert approve(live_engine, cfg, request_id, confirm()).approved


def test_large_quantity_needs_a_person_to_confirm(live_engine, cfg, suppliers):
    request_id, _, plan = prepare(live_engine, cfg, suppliers, text=SMALL.replace("65 lb unsalted butter",
                                                                                 "650 lb unsalted butter"))
    blocked = approve(live_engine, cfg, request_id, [ConfirmItem(p) for p in SMALL_PRODUCTS if p != "butter_unsalted"])
    assert not blocked.approved
    assert any("Unsalted butter: 650 lb is more than 3 times" in r for c in blocked.checks for r in c.reasons)
    assert approve(live_engine, cfg, request_id, confirm(SMALL_PRODUCTS)).approved


def test_ordering_runs_the_checks_again(live_engine, cfg, suppliers, tmp_path):
    request_id = ready_and_approved(live_engine, cfg, suppliers)
    report = workflow.place_orders(live_engine, request_id, suppliers, cfg, sleep=NO_WAIT,
                                   now=clock.now() + timedelta(hours=25), po_dir=tmp_path / "po")
    assert report.blocked and "older than the 24-hour limit" in report.summary
    assert orders(live_engine, request_id) == {} and all(s.placements == 0 for s in suppliers.values())


# ------------------------------------------------------------------ suppliers failing

def test_one_supplier_down_still_gives_a_plan_from_the_others(live_engine, cfg, suppliers):
    suppliers["sysco"].faults = SupplierFaults(down=True)
    calls = []
    real = suppliers["sysco"].get_quotes
    suppliers["sysco"].get_quotes = lambda skus: (calls.append(1), real(skus))[1]

    request_id, prices, plan = prepare(live_engine, cfg, suppliers, text=SMALL)
    assert len(calls) == 1 + cfg.supplier_calls.retries                   # asked, then 2 more tries
    assert "sysco" in prices.no_answer and "sysco" not in prices.answered
    assert plan_suppliers(live_engine, request_id) and "sysco" not in plan_suppliers(live_engine, request_id)
    warning = next(w for w in plan.warnings if w.startswith("Sysco didn't answer"))
    assert "Last time Sysco was cheaper for:" in warning and "Unsalted butter" in warning and "Large eggs" in warning
    with session_scope(live_engine) as s:
        assert s.get(models.Request, request_id).suppliers_no_answer == ["sysco"]


def test_one_supplier_failing_at_ordering_doesnt_stop_the_others(live_engine, cfg, suppliers, tmp_path):
    request_id = ready_and_approved(live_engine, cfg, suppliers)
    used = plan_suppliers(live_engine, request_id)
    assert len(used) >= 2
    down = used[-1]
    suppliers[down].faults = SupplierFaults(down=True)
    report = place(live_engine, cfg, request_id, suppliers, tmp_path)
    assert report.request_status == "PARTIALLY_ORDERED"
    placed = orders(live_engine, request_id)
    assert placed[down].status == models.OrderStatus.PENDING and "did not answer" in placed[down].last_error
    assert all(placed[c].status == models.OrderStatus.CONFIRMED for c in used if c != down)

    suppliers[down].faults = SupplierFaults()   # it comes back; the next start finishes the order
    workflow.recover_pending_orders(live_engine, suppliers, cfg, sleep=NO_WAIT, po_dir=tmp_path / "po")
    assert request_status(live_engine, request_id) == models.RequestStatus.ORDERED
    assert {c: suppliers[c].placements for c in used} == {c: 1 for c in used}


# ------------------------------------------------------------------ the owner's changes

def test_owner_moves_an_item_and_totals_are_recalculated(live_engine, cfg, suppliers):
    request_id, _, _ = prepare(live_engine, cfg, suppliers)
    with session_scope(live_engine) as s:
        product = s.exec(select(models.Product).where(models.Product.code == "eggs")).one()
        line = s.exec(select(models.RequestLine).where(models.RequestLine.request_id == request_id,
                                                       models.RequestLine.product_id == product.id)).one()
        current = s.get(models.Supplier, line.assigned_supplier_id).code
    target = next(c for c in ("usfoods", "pfg", "sysco") if c != current)
    result = approve(live_engine, cfg, request_id, [MoveItem("eggs", target)] + confirm())
    with session_scope(live_engine) as s:
        req = s.get(models.Request, request_id)
        lines = s.exec(select(models.RequestLine).where(models.RequestLine.request_id == request_id)).all()
        moved = next(x for x in lines if x.product_id == product.id)
        assert s.get(models.Supplier, moved.assigned_supplier_id).code == target
        used = {x.assigned_supplier_id for x in lines if x.assigned_supplier_id}
        fees = sum(s.get(models.Supplier, sid).delivery_fee_cents for sid in used)
        assert req.plan_total_cents == sum(x.line_cost_cents or 0 for x in lines) + fees
    # The move is saved either way; approval depends on the checks (e.g. a minimum may now be missed).
    assert result.approved or any(c.name == "minimums" for c in result.checks if not c.passed), result.message


def test_bad_changes_change_nothing(live_engine, cfg, suppliers):
    request_id, _, _ = prepare(live_engine, cfg, suppliers)
    result = approve(live_engine, cfg, request_id, [MoveItem("eggs", "supplier4"), MoveItem("saffron", "pfg"),
                                                    AcceptPrice("heavy_cream", "pfg")] + confirm())
    assert not result.approved
    text = " ".join(result.change_errors)
    assert "no product match for Large eggs at Supplier 4" in text or "no price from Supplier 4" in text
    assert "'saffron' isn't in this request" in text
    assert "$0.00; that can't be accepted" in text
    with session_scope(live_engine) as s:
        lines = s.exec(select(models.RequestLine).where(models.RequestLine.request_id == request_id)).all()
        assert all(x.needs_check or not x.qty_confirmed for x in lines)  # the confirms weren't applied either


def test_status_and_audit_row_are_saved_together(live_engine, cfg, suppliers, monkeypatch):
    request_id, _, _ = prepare(live_engine, cfg, suppliers)

    def broken_audit(*args, **kwargs):
        raise RuntimeError("audit_log is unavailable")

    monkeypatch.setattr(workflow, "audit", broken_audit)
    with pytest.raises(RuntimeError):
        approve(live_engine, cfg, request_id, confirm())
    assert request_status(live_engine, request_id) == models.RequestStatus.PENDING_APPROVAL


def test_moves_that_arent_allowed(live_engine, cfg, suppliers, tmp_path):
    request_id, _, _ = prepare(live_engine, cfg, suppliers)
    with pytest.raises(workflow.InvalidTransition):
        place(live_engine, cfg, request_id, suppliers, tmp_path)          # not approved yet
    with session_scope(live_engine) as s, pytest.raises(workflow.InvalidTransition):
        workflow.make_plan(s, request_id, cfg)                             # already planned
    with session_scope(live_engine) as s:
        workflow.reject(s, request_id, "owner", "not this week")
    with session_scope(live_engine) as s, pytest.raises(workflow.InvalidTransition):
        workflow.approve(s, request_id, "owner", (), cfg)


def test_message_without_a_date_or_items_is_not_saved(live_engine, cfg):
    with session_scope(live_engine) as s:
        sub = workflow.submit_request(s, location_code="factory", text="some saffron please", created_by="m",
                                      client=NO_AI, cfg=cfg, model_tested=False)
    assert sub.request is None and "No items could be read" in sub.problems[0]
    assert any("needed-by" in p for p in sub.problems)
