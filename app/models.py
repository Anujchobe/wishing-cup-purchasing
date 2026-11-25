"""Database tables (SQLModel on SQLite).

Storage rules are enforced in code, not left to habit:

* Money is whole cents, in integer columns named ``*_cents``.
* Quantities are ``Decimal`` and are stored as TEXT. SQLite's NUMERIC type
  would quietly turn them into floats. Writing a float raises an error.
  Because they are text, never SUM or compare quantities in SQL; do that math
  in Python (the rules say code does the math anyway).
* Timestamps must have a timezone. They are stored as UTC text, which sorts
  correctly across daylight-saving changes, and come back in Boston time.

Invariants the safety checks rely on are also database constraints, so a bug
can't write a bad row: one order per request and supplier, a unique order key,
a request line that is either on one supplier's list or marked can't-fill, and
order totals that add up.
"""
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Optional

from sqlalchemy import JSON, CheckConstraint, Column, Index, String, UniqueConstraint
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel

from app import clock

SCHEMA_VERSION = 3  # 3 = step 4 request statuses and plan columns. Bump on any table change; old databases are reset, not half-read


# ---------------------------------------------------------------- column types

class DecimalText(TypeDecorator):
    """Exact Decimal stored as text. Refuses floats."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (bool, float)):
            raise TypeError(f"refusing to store {type(value).__name__} {value!r} as a quantity; use Decimal")
        if isinstance(value, int):
            value = Decimal(value)
        if not isinstance(value, Decimal):
            raise TypeError(f"quantities must be Decimal, got {type(value).__name__}")
        if not value.is_finite():
            raise ValueError(f"quantity must be a finite number, got {value}")
        return format(value, "f")

    def process_result_value(self, value: Any, dialect: Any) -> Optional[Decimal]:
        return None if value is None else Decimal(value)


class UtcDateTime(TypeDecorator):
    """Timezone-aware datetime stored as UTC ISO text, read back in Boston time."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"expected a datetime, got {type(value).__name__}")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("refusing to store a naive datetime; use app.clock.now() or attach a timezone")
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")

    def process_result_value(self, value: Any, dialect: Any) -> Optional[datetime]:
        return None if value is None else datetime.fromisoformat(value).astimezone(clock.BOSTON)


def _json_col() -> Any:
    return Column(JSON, nullable=False)


def _qty_col(nullable: bool = False) -> Any:
    return Column(DecimalText(), nullable=nullable)


def _ts_col(nullable: bool = False) -> Any:
    return Column(UtcDateTime(), nullable=nullable)


# ---------------------------------------------------------------- enums
# Stored by name. Names equal values so the database text is readable.

class BaseUnit(StrEnum):
    lb = "lb"
    fl_oz = "fl_oz"
    each = "each"


class LocationKind(StrEnum):
    cafe = "cafe"
    factory = "factory"


class RequestStatus(StrEnum):
    """DRAFT -> QUOTED -> PENDING_APPROVAL -> APPROVED -> ORDERED | PARTIALLY_ORDERED, or REJECTED.
    The allowed moves are in app.workflow.TRANSITIONS."""

    DRAFT = "DRAFT"                          # read and matched, no prices yet
    QUOTED = "QUOTED"                        # prices fetched
    PENDING_APPROVAL = "PENDING_APPROVAL"    # a plan is waiting for the owner
    APPROVED = "APPROVED"                    # checks passed, owner approved; orders not all confirmed
    ORDERED = "ORDERED"                      # every supplier confirmed
    PARTIALLY_ORDERED = "PARTIALLY_ORDERED"  # some suppliers confirmed, others failed or are pending
    REJECTED = "REJECTED"


class LineFillStatus(StrEnum):
    OPEN = "OPEN"            # not planned yet
    ASSIGNED = "ASSIGNED"    # on exactly one supplier's list
    CANT_FILL = "CANT_FILL"  # no supplier can fill it; shown to the owner


class OrderStatus(StrEnum):
    PENDING = "PENDING"      # saved before contacting the supplier
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MappingSource(StrEnum):
    ai = "ai"
    text_match = "text_match"
    person = "person"


# ---------------------------------------------------------------- tables

class Location(SQLModel, table=True):
    __tablename__ = "locations"
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True)
    name: str
    kind: LocationKind
    active: bool = True


class Supplier(SQLModel, table=True):
    """Mirrors config.yaml; refreshed on every start so config edits take effect."""

    __tablename__ = "suppliers"
    __table_args__ = (
        CheckConstraint("min_order_cents >= 0", name="ck_suppliers_min_order"),
        CheckConstraint("delivery_fee_cents >= 0", name="ck_suppliers_delivery_fee"),
        CheckConstraint("lead_time_days >= 0", name="ck_suppliers_lead_time"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True)
    name: str
    min_order_cents: int
    delivery_fee_cents: int
    delivery_weekdays: list[str] = Field(default_factory=list, sa_column=_json_col())
    lead_time_days: int
    serves_locations: list[str] = Field(default_factory=list, sa_column=_json_col())  # location codes
    rep_name: str = ""
    rep_email: str = ""
    is_main: bool = False
    is_placeholder: bool = False
    data_is_fake: bool = True
    active: bool = True


class Product(SQLModel, table=True):
    """Our own ingredient list. Quantities for a product are always in its base unit."""

    __tablename__ = "products"
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True)
    name: str
    base_unit: BaseUnit
    category: str = ""
    active: bool = True


class SupplierItem(SQLModel, table=True):
    """A supplier's product as the supplier describes it (raw, unparsed)."""

    __tablename__ = "supplier_items"
    __table_args__ = (UniqueConstraint("supplier_id", "sku", name="uq_supplier_items_sku"),)
    id: Optional[int] = Field(default=None, primary_key=True)
    supplier_id: int = Field(foreign_key="suppliers.id", index=True)
    sku: str
    description: str
    raw_pack_text: str
    case_price_cents: int  # latest price; may be bad data (0, huge jump) until checked
    in_stock: bool = True
    updated_at: datetime = Field(default_factory=clock.now, sa_column=_ts_col())


class PriceSnapshot(SQLModel, table=True):
    __tablename__ = "price_snapshots"
    __table_args__ = (Index("ix_price_snapshots_item_time", "supplier_item_id", "captured_at"),)
    id: Optional[int] = Field(default=None, primary_key=True)
    supplier_item_id: int = Field(foreign_key="supplier_items.id")
    case_price_cents: int
    in_stock: bool = True
    captured_at: datetime = Field(sa_column=_ts_col())
    dont_use: bool = False
    dont_use_reason: Optional[str] = None
    source: str = "quote"  # quote | seed_history


class ItemMapping(SQLModel, table=True):
    """Our product <-> a supplier SKU. Confirmed rows are reused; only new SKUs go to the AI."""

    __tablename__ = "item_mappings"
    id: Optional[int] = Field(default=None, primary_key=True)
    product_id: int = Field(foreign_key="products.id", index=True)
    supplier_id: int = Field(foreign_key="suppliers.id", index=True)
    supplier_item_id: int = Field(foreign_key="supplier_items.id", unique=True)  # a SKU is one product
    pack_size_base: Optional[Decimal] = Field(default=None, sa_column=_qty_col(nullable=True))
    confidence: float = 0.0  # a score, not money or a quantity
    source: MappingSource
    needs_check: bool = True
    check_reason: Optional[str] = None
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[datetime] = Field(default=None, sa_column=_ts_col(nullable=True))
    created_at: datetime = Field(default_factory=clock.now, sa_column=_ts_col())


class Request(SQLModel, table=True):
    __tablename__ = "requests"
    id: Optional[int] = Field(default=None, primary_key=True)
    public_id: str = Field(unique=True, index=True)  # globally unique; part of every order id
    location_id: int = Field(foreign_key="locations.id", index=True)
    needed_by: date
    original_text: str = ""
    status: RequestStatus = RequestStatus.DRAFT
    source: str = "manager"  # manager | seed_history
    created_by: str = ""
    created_at: datetime = Field(default_factory=clock.now, sa_column=_ts_col())
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = Field(default=None, sa_column=_ts_col(nullable=True))
    plan_total_cents: Optional[int] = None  # items + delivery fees of the current plan
    suppliers_no_answer: list[str] = Field(default_factory=list, sa_column=_json_col())
    status_note: Optional[str] = None


class RequestLine(SQLModel, table=True):
    """One thing a manager asked for. Assignment lives here, so a line can only
    ever be on one supplier's list (or marked can't-fill)."""

    __tablename__ = "request_lines"
    __table_args__ = (
        UniqueConstraint("request_id", "line_no", name="uq_request_lines_no"),
        CheckConstraint(
            "fill_status != 'ASSIGNED' OR (assigned_supplier_id IS NOT NULL "
            "AND supplier_item_id IS NOT NULL AND cases > 0 AND line_cost_cents IS NOT NULL)",
            name="ck_request_lines_assigned_complete",
        ),
        CheckConstraint(
            "fill_status = 'ASSIGNED' OR (assigned_supplier_id IS NULL AND supplier_item_id IS NULL)",
            name="ck_request_lines_unassigned_empty",
        ),
        CheckConstraint(
            "line_cost_cents IS NULL OR line_cost_cents = cases * case_price_cents",
            name="ck_request_lines_cost",
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: int = Field(foreign_key="requests.id", index=True)
    line_no: int
    raw_text: str = ""
    product_id: Optional[int] = Field(default=None, foreign_key="products.id")
    qty_needed: Optional[Decimal] = Field(default=None, sa_column=_qty_col(nullable=True))  # base units
    needs_check: bool = False
    check_reason: Optional[str] = None
    qty_confirmed: bool = False  # a person confirmed an unusually large quantity
    fill_status: LineFillStatus = LineFillStatus.OPEN
    assigned_supplier_id: Optional[int] = Field(default=None, foreign_key="suppliers.id")
    supplier_item_id: Optional[int] = Field(default=None, foreign_key="supplier_items.id")
    cases: Optional[int] = None
    case_price_cents: Optional[int] = None
    line_cost_cents: Optional[int] = None
    quote_id: Optional[int] = Field(default=None, foreign_key="quotes.id")  # the price this line uses
    plan_note: Optional[str] = None  # e.g. why it can't be filled


class Quote(SQLModel, table=True):
    __tablename__ = "quotes"
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: Optional[int] = Field(default=None, foreign_key="requests.id", index=True)
    supplier_id: int = Field(foreign_key="suppliers.id")
    supplier_item_id: int = Field(foreign_key="supplier_items.id")
    case_price_cents: int
    in_stock: bool
    quoted_at: datetime = Field(sa_column=_ts_col())
    fetched_at: datetime = Field(default_factory=clock.now, sa_column=_ts_col())
    dont_use: bool = False
    dont_use_reason: Optional[str] = None
    price_accepted_by: Optional[str] = None  # a person checked a set-aside price and allowed it


class Order(SQLModel, table=True):
    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("request_id", "supplier_id", name="uq_orders_request_supplier"),
        CheckConstraint("total_cents = subtotal_cents + delivery_fee_cents", name="ck_orders_total"),
        CheckConstraint("status != 'CONFIRMED' OR confirmation_id IS NOT NULL", name="ck_orders_confirmed_has_id"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    order_key: str = Field(unique=True, index=True)  # request public id + supplier code
    request_id: int = Field(foreign_key="requests.id", index=True)
    supplier_id: int = Field(foreign_key="suppliers.id")
    status: OrderStatus = OrderStatus.PENDING
    confirmation_id: Optional[str] = None
    subtotal_cents: int
    delivery_fee_cents: int
    total_cents: int
    deliver_on: date
    attempts: int = 0
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=clock.now, sa_column=_ts_col())
    placed_at: Optional[datetime] = Field(default=None, sa_column=_ts_col(nullable=True))


class LlmCall(SQLModel, table=True):
    __tablename__ = "llm_calls"
    id: Optional[int] = Field(default=None, primary_key=True)
    at: datetime = Field(default_factory=clock.now, sa_column=_ts_col())
    model: str
    task: str  # read_request | match_product | read_pack | write_email | eval
    duration_ms: int
    ok: bool
    error: Optional[str] = None
    attempt: int = 1
    request_id: Optional[int] = Field(default=None, foreign_key="requests.id")


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"
    id: Optional[int] = Field(default=None, primary_key=True)
    at: datetime = Field(default_factory=clock.now, sa_column=_ts_col())
    actor: str
    action: str
    entity_type: str = ""
    entity_id: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict, sa_column=_json_col())


ALL_TABLES: tuple[type[SQLModel], ...] = (
    Location, Supplier, Product, SupplierItem, PriceSnapshot, ItemMapping,
    Request, RequestLine, Quote, Order, LlmCall, AuditLog,
)
