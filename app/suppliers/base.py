"""The contract every supplier connector follows.

A connector does three things: list its catalog, quote prices for SKUs, and
place an order. The data shapes are Pydantic models, so a connector that sends
back the wrong types fails loudly at the boundary.

These models check shape only (types, whole-cent ints, timezone-aware times).
They deliberately accept business nonsense such as a $0.00 price: suppliers
send bad data, and spotting it is the job of the app's own checks, which can
then explain the problem to a person.

Errors are split by what the caller should do next:
  SupplierUnavailable (and SupplierTimeout): try again later, same order id.
  SupplierRejected: the request itself is wrong; retrying won't help.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, model_validator


class SupplierError(Exception):
    retryable = False


class SupplierUnavailable(SupplierError):
    """No answer, a dropped connection, or a lost reply. Safe to retry with the same order id."""

    retryable = True


class SupplierTimeout(SupplierUnavailable):
    """The supplier took longer than our timeout."""


class SupplierRejected(SupplierError):
    """The supplier refused the request (unknown SKU, out of stock, reused order id...)."""


class _Wire(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CatalogItem(_Wire):
    sku: str = Field(min_length=1)
    description: str
    raw_pack_text: str
    case_price_cents: StrictInt
    in_stock: bool


class Quote(_Wire):
    sku: str = Field(min_length=1)
    case_price_cents: StrictInt
    in_stock: bool
    quoted_at: AwareDatetime


class OrderLine(_Wire):
    sku: str = Field(min_length=1)
    cases: StrictInt = Field(gt=0)
    expected_case_price_cents: StrictInt = Field(gt=0)


class SupplierOrder(_Wire):
    supplier_code: str
    location_code: str
    deliver_on: date
    lines: list[OrderLine] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_line_per_sku(self) -> SupplierOrder:
        skus = [line.sku for line in self.lines]
        if len(set(skus)) != len(skus):
            raise ValueError("each SKU may appear only once per order")
        return self


def make_order_id(request_public_id: str, supplier_code: str) -> str:
    """The idempotency key for one request's order at one supplier."""
    return f"{request_public_id}-{supplier_code}"


class Supplier(ABC):
    """One supplier. Subclasses talk to a real API, or (for now) fake one."""

    def __init__(self, code: str, name: str) -> None:
        self.code = code
        self.name = name

    @abstractmethod
    def get_catalog(self) -> list[CatalogItem]:
        """Everything this supplier sells right now, with current case prices."""

    @abstractmethod
    def get_quotes(self, skus: Sequence[str]) -> list[Quote]:
        """Current prices for the given SKUs.

        SKUs the supplier doesn't know are left out of the result (that is how
        the app learns a product no longer exists). Each known SKU appears once.
        """

    @abstractmethod
    def place_order(self, order: SupplierOrder, order_id: str) -> str:
        """Place an order and return the supplier's confirmation id.

        Must be idempotent: sending the same order_id again returns the same
        confirmation and books nothing new. Reusing an order_id for a different
        order raises SupplierRejected. This is what lets the app retry safely
        after a crash or a lost reply without ever ordering twice.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.code!r})"
