from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError, StatementError
from sqlmodel import Session, select

from app import clock, models
from app.db import SchemaMismatch, init_db, session_scope, sync_config_to_db
from app.models import DecimalText, UtcDateTime
from tests.conftest import config_with

SPEC_TABLES = {
    "locations", "suppliers", "products", "supplier_items", "price_snapshots", "item_mappings",
    "requests", "request_lines", "quotes", "orders", "llm_calls", "audit_log",
}


def test_every_spec_table_exists(engine):
    assert SPEC_TABLES <= set(inspect(engine).get_table_names())


def test_init_is_safe_to_repeat(engine):
    init_db(engine)
    init_db(engine)


def test_old_schema_is_refused_with_a_fix(engine):
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA user_version = 999")
    with pytest.raises(SchemaMismatch, match="--reset"):
        init_db(engine)


def _request(session: Session, cfg) -> models.Request:
    sync_config_to_db(session, cfg)
    loc = session.exec(select(models.Location)).first()
    req = models.Request(public_id="R-test", location_id=loc.id, needed_by=date(2026, 12, 1))
    session.add(req)
    session.flush()
    return req


def _supplier(session: Session) -> models.Supplier:
    return session.exec(select(models.Supplier).where(models.Supplier.code == "sysco")).one()


def test_foreign_keys_are_enforced(engine):
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            s.add(models.SupplierItem(supplier_id=9999, sku="X", description="x", raw_pack_text="x",
                                      case_price_cents=100))


def test_quantity_is_stored_exactly(engine, cfg):
    qty = Decimal("1234.567890123456789")
    with session_scope(engine) as s:
        req = _request(s, cfg)
        s.add(models.RequestLine(request_id=req.id, line_no=1, qty_needed=qty))
    with session_scope(engine) as s:
        line = s.exec(select(models.RequestLine)).one()
        assert isinstance(line.qty_needed, Decimal) and line.qty_needed == qty


def test_float_quantity_is_refused(engine, cfg):
    with pytest.raises((StatementError, TypeError), match="float"):
        with session_scope(engine) as s:
            req = _request(s, cfg)
            s.add(models.RequestLine(request_id=req.id, line_no=1, qty_needed=1.5))


def test_naive_datetime_is_refused(engine):
    with pytest.raises((StatementError, ValueError), match="naive"):
        with session_scope(engine) as s:
            s.add(models.AuditLog(actor="t", action="t", at=datetime(2026, 1, 1, 12, 0)))


def test_timestamps_come_back_in_boston_time(engine):
    moment = datetime(2026, 7, 1, 16, 30, tzinfo=timezone.utc)
    with session_scope(engine) as s:
        s.add(models.AuditLog(actor="t", action="t", at=moment))
    with session_scope(engine) as s:
        row = s.exec(select(models.AuditLog)).one()
        assert row.at == moment
        assert row.at.tzinfo == clock.BOSTON and row.at.hour == 12  # EDT is UTC-4


def test_order_key_and_one_order_per_request_supplier(engine, cfg):
    with session_scope(engine) as s:
        req = _request(s, cfg)
        sup = _supplier(s)
        req_id, sup_id = req.id, sup.id

    def order(key):
        return models.Order(order_key=key, request_id=req_id, supplier_id=sup_id, subtotal_cents=100,
                            delivery_fee_cents=0, total_cents=100, deliver_on=date(2026, 12, 1))

    with session_scope(engine) as s:
        s.add(order("R-test-sysco"))
    with pytest.raises(IntegrityError):  # same request + supplier, different key
        with session_scope(engine) as s:
            s.add(order("R-test-sysco-again"))


@pytest.mark.parametrize(
    "changes",
    [
        {"fill_status": models.LineFillStatus.ASSIGNED},                       # assigned to nobody
        {"fill_status": models.LineFillStatus.CANT_FILL, "assigned_supplier_id": "SUP"},  # both
        {"cases": 2, "case_price_cents": 500, "line_cost_cents": 999},         # cost doesn't add up
    ],
)
def test_bad_request_lines_are_refused(engine, cfg, changes):
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            req = _request(s, cfg)
            sup = _supplier(s)
            fields = {k: (sup.id if v == "SUP" else v) for k, v in changes.items()}
            s.add(models.RequestLine(request_id=req.id, line_no=1, **fields))


def test_order_totals_must_add_up(engine, cfg):
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            req = _request(s, cfg)
            s.add(models.Order(order_key="k", request_id=req.id, supplier_id=_supplier(s).id,
                               subtotal_cents=100, delivery_fee_cents=25, total_cents=100,
                               deliver_on=date(2026, 12, 1)))


def test_confirmed_order_needs_a_confirmation_id(engine, cfg):
    with pytest.raises(IntegrityError):
        with session_scope(engine) as s:
            req = _request(s, cfg)
            s.add(models.Order(order_key="k", request_id=req.id, supplier_id=_supplier(s).id,
                               status=models.OrderStatus.CONFIRMED, subtotal_cents=0, delivery_fee_cents=0,
                               total_cents=0, deliver_on=date(2026, 12, 1)))


def test_config_changes_reach_the_database(engine, cfg):
    with session_scope(engine) as s:
        sync_config_to_db(s, cfg)
    changed = config_with(lambda c: c["suppliers"][0].update(min_order_cents=12345))
    with session_scope(engine) as s:
        sync_config_to_db(s, changed)
        sysco = _supplier(s)
        assert sysco.min_order_cents == 12345 and sysco.is_main


@given(st.decimals(allow_nan=False, allow_infinity=False, min_value=-10**12, max_value=10**12, places=12))
def test_decimal_column_roundtrip(value):
    col = DecimalText()
    assert col.process_result_value(col.process_bind_param(value, None), None) == value


@given(st.datetimes(min_value=datetime(1971, 1, 1), max_value=datetime(2100, 1, 1),
                    timezones=st.sampled_from([timezone.utc, clock.BOSTON]), allow_imaginary=False))
def test_datetime_column_roundtrip(value):
    col = UtcDateTime()
    out = col.process_result_value(col.process_bind_param(value, None), None)
    assert out.astimezone(timezone.utc) == value.astimezone(timezone.utc)
    assert out.tzinfo == clock.BOSTON


@given(st.lists(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1),
                             timezones=st.just(clock.BOSTON), allow_imaginary=False), min_size=2, max_size=20))
def test_stored_timestamps_sort_like_real_time(values):
    col = UtcDateTime()
    stored = [col.process_bind_param(v, None) for v in values]
    assert sorted(stored) == [col.process_bind_param(v, None)
                              for v in sorted(values, key=lambda v: v.astimezone(timezone.utc))]
