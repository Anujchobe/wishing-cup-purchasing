from datetime import timedelta

import pytest
from sqlmodel import select

from app import clock, models
from app.db import init_db, make_engine, session_scope, table_counts
from app.suppliers import fake_world
from scripts.seed import seed, seed_if_empty
from tests.conftest import FIXED_TODAY


def test_counts(seeded_engine, cfg):
    with session_scope(seeded_engine) as s:
        counts = table_counts(s)
    items = len(fake_world.all_offers())
    assert counts["products"] == len(fake_world.PRODUCTS) == 24
    assert counts["supplier_items"] == items
    assert counts["price_snapshots"] == items * (fake_world.HISTORY_WEEKS + 1)
    assert counts["item_mappings"] == 24              # main supplier only
    assert counts["requests"] == counts["orders"] == 2 * fake_world.HISTORY_WEEKS
    assert counts["locations"] == 2 and counts["suppliers"] == 4
    assert counts["audit_log"] == 1


def _fingerprint(engine):
    with session_scope(engine) as s:
        skus = {i.id: i.sku for i in s.exec(select(models.SupplierItem)).all()}
        snaps = sorted((skus[p.supplier_item_id], p.captured_at.isoformat(), p.case_price_cents, p.in_stock)
                       for p in s.exec(select(models.PriceSnapshot)).all())
        lines = sorted((l.request_id, l.line_no, str(l.qty_needed), l.cases, l.line_cost_cents)
                       for l in s.exec(select(models.RequestLine)).all())
        orders = sorted((o.order_key, o.total_cents, o.confirmation_id) for o in s.exec(select(models.Order)).all())
    return snaps, lines, orders


def test_seeding_is_repeatable(cfg, tmp_path):
    results = []
    for name in ("a.db", "b.db"):
        engine = make_engine(tmp_path / name)
        init_db(engine)
        with session_scope(engine) as s:
            seed(s, cfg, today=FIXED_TODAY)
        results.append(_fingerprint(engine))
        engine.dispose()
    assert results[0] == results[1]


def test_seed_runs_only_once(seeded_engine, cfg):
    assert seed_if_empty(seeded_engine, cfg) is None
    with pytest.raises(RuntimeError, match="--reset"):
        with session_scope(seeded_engine) as s:
            seed(s, cfg)


def test_history_orders_are_internally_consistent(seeded_engine, cfg):
    main = cfg.main_supplier
    with session_scope(seeded_engine) as s:
        sysco = s.exec(select(models.Supplier).where(models.Supplier.code == main.code)).one()
        packs = {m.supplier_item_id: m.pack_size_base for m in s.exec(select(models.ItemMapping)).all()}
        requests = {r.id: r for r in s.exec(select(models.Request)).all()}
        lines = s.exec(select(models.RequestLine)).all()
        orders = s.exec(select(models.Order)).all()

    for line in lines:
        assert line.fill_status == models.LineFillStatus.ASSIGNED
        assert line.assigned_supplier_id == sysco.id
        pack = packs[line.supplier_item_id]
        assert line.cases * pack >= line.qty_needed > (line.cases - 1) * pack  # enough, and not one case too many
        assert line.line_cost_cents == line.cases * line.case_price_cents

    for order in orders:
        req = requests[order.request_id]
        subtotal = sum(l.line_cost_cents for l in lines if l.request_id == order.request_id)
        assert order.status == models.OrderStatus.CONFIRMED
        assert order.confirmation_id.startswith("FAKE-")
        assert order.order_key == f"{req.public_id}-{main.code}"
        assert order.subtotal_cents == subtotal
        assert order.total_cents == subtotal + main.delivery_fee_cents
        assert order.subtotal_cents >= main.min_order_cents
        assert clock.weekday_name(order.deliver_on) in main.delivery_weekdays
        assert order.deliver_on >= req.created_at.date() + timedelta(days=main.lead_time_days)


def test_history_covers_the_last_8_weeks(seeded_engine):
    this_monday = clock.week_start(FIXED_TODAY)
    with session_scope(seeded_engine) as s:
        weeks = sorted({clock.week_start(r.created_at.date()) for r in s.exec(select(models.Request)).all()})
    assert len(weeks) == 8
    assert weeks[-1] == this_monday - timedelta(weeks=1)
    assert weeks[0] == this_monday - timedelta(weeks=8)


def test_no_pending_orders_after_seeding(seeded_engine):
    with session_scope(seeded_engine) as s:
        pending = s.exec(select(models.Order).where(models.Order.status == models.OrderStatus.PENDING)).all()
    assert pending == []


def test_only_main_supplier_is_premapped(seeded_engine, cfg):
    with session_scope(seeded_engine) as s:
        supplier_codes = {sup.id: sup.code for sup in s.exec(select(models.Supplier)).all()}
        mappings = s.exec(select(models.ItemMapping)).all()
    assert {supplier_codes[m.supplier_id] for m in mappings} == {cfg.main_supplier.code}
    assert all(m.source == models.MappingSource.person and not m.needs_check for m in mappings)
