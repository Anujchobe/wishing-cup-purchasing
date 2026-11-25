"""Pricing for one item at one supplier.

Suppliers sell whole cases, so the real cost is the cost of the cases we must
buy, not need x unit price:

    cases = need / case size, rounded up      (exact Decimal, no float division)
    cost  = cases x case price                (integer cents)
    extra = cases x case size - need          (what we get beyond the need)

Price per base unit is for display and comparison only. It is never used to
compute what we pay.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.units import display_unit

PER_UNIT_PLACES = Decimal("0.0001")  # cents per lb / fl oz / each, shown to 1/100 of a cent


@dataclass(frozen=True)
class ItemCost:
    need: Decimal
    case_size: Decimal
    case_price_cents: int
    cases: int
    cost_cents: int
    bought: Decimal
    extra: Decimal
    price_per_base_cents: Decimal


def _as_decimal(value: Decimal | int, name: str) -> Decimal:
    if isinstance(value, (bool, float)) or not isinstance(value, (Decimal, int)):
        raise TypeError(f"{name} must be Decimal or int, got {type(value).__name__}")
    value = Decimal(value)
    if not value.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return value


def cases_to_buy(need: Decimal | int, case_size: Decimal | int) -> int:
    """Fewest whole cases that cover the need."""
    need = _as_decimal(need, "need")
    case_size = _as_decimal(case_size, "case size")
    if case_size <= 0:
        raise ValueError("case size must be more than zero")
    if need < 0:
        raise ValueError("need can't be negative")
    cases = int(need // case_size)  # exact for Decimals: integer part of the true quotient
    return cases if cases * case_size >= need else cases + 1


def item_cost(need: Decimal | int, case_size: Decimal | int, case_price_cents: int) -> ItemCost:
    if isinstance(case_price_cents, bool) or not isinstance(case_price_cents, int):
        raise TypeError("case price must be whole cents (an int)")
    if case_price_cents <= 0:
        raise ValueError("case price must be more than zero; zero or negative prices are flagged, not used")
    need = _as_decimal(need, "need")
    case_size = _as_decimal(case_size, "case size")
    cases = cases_to_buy(need, case_size)
    bought = cases * case_size
    return ItemCost(
        need=need,
        case_size=case_size,
        case_price_cents=case_price_cents,
        cases=cases,
        cost_cents=cases * case_price_cents,
        bought=bought,
        extra=bought - need,
        price_per_base_cents=(Decimal(case_price_cents) / case_size).quantize(PER_UNIT_PLACES, ROUND_HALF_UP),
    )


def format_cents(cents: int) -> str:
    """12345 -> '$123.45', -50 -> '-$0.50'."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def format_qty(qty: Decimal, unit: str) -> str:
    """Up to 2 decimals, trailing zeros dropped: Decimal('12.500') lb -> '12.5 lb'."""
    shown = qty.quantize(Decimal("0.01"), ROUND_HALF_UP).normalize()
    return f"{format(shown, 'f')} {display_unit(unit)}"


def price_flag(price_cents: int, last_week_cents: int | None, jump_limit: Decimal) -> str | None:
    """Why a price shouldn't be used without a person checking it, or None if it looks fine.

    Rules: 0 or less is never right; a change of more than `jump_limit` (0.5 = 50%) since last week
    is set aside. Both directions count: a price that halves is as likely a data error as one that doubles.
    """
    if price_cents <= 0:
        return f"the price is {format_cents(price_cents)}, which can't be right"
    if last_week_cents is None or last_week_cents <= 0:
        return None
    change = Decimal(price_cents) / Decimal(last_week_cents) - 1
    if abs(change) > jump_limit:
        pct = (abs(change) * 100).quantize(Decimal(1), ROUND_HALF_UP)
        way = "up" if change > 0 else "down"
        return (f"the price went {way} {pct}% since last week "
                f"({format_cents(last_week_cents)} to {format_cents(price_cents)})")
    return None
