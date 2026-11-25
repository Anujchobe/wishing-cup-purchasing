"""Reading a manager's request: text in, request lines out.

Who does what:
  AI (if available)  splits the message into items and picks product codes from OUR list,
                     copying quantity and unit exactly as written.
  Code               checks every number and unit is really in the message, converts to base
                     units (lb / fl oz / each), reads the needed-by date, and decides what is unclear.

Rules:
  * A line whose quantity (or unit) isn't in the original text is dropped and listed as unclear.
  * A product that isn't on our list goes to unclear. Codes are never made up: the answer schema
    only accepts codes from the list, and a wrong code costs the AI its one retry.
  * Unit words that depend on the supplier ("3 cases") go to unclear; whose case is meant?
  * If the AI fails twice, times out or is off, a plain text reader takes over. When that happens
    because the AI failed, every line is marked for a person to check.
  * If the model is "not tested" (see app.llm.model_check), every AI-read line is marked too.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationInfo, field_validator

from app import textparse, units
from app.config import IntakeConfig
from app.llm.client import AnyClient, LLMError, render_prompt
from app.pricing import format_qty
from app.textmatch import top_matches

TASK = "read_request"


@dataclass(frozen=True)
class ProductRef:
    code: str
    name: str
    base_unit: str  # lb | fl_oz | each


# ------------------------------------------------------------------ what the AI must send back

def _number_to_text(value: Any) -> Any:
    # JSON 30 or 1.5 instead of "30" / "1.5": keep exactly what was written.
    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        return str(value)
    return value


class AnswerLine(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_code: str | None = None
    product_text: str = Field(min_length=1, description="the manager's words for the item")
    quantity: Annotated[str, BeforeValidator(_number_to_text)] = Field(min_length=1)
    unit: Annotated[str | None, BeforeValidator(_number_to_text)] = None
    note: str = ""

    @field_validator("product_code")
    @classmethod
    def _code_from_our_list(cls, value: str | None, info: ValidationInfo) -> str | None:
        allowed = (info.context or {}).get("product_codes")
        if value is not None and allowed is not None and value not in allowed:
            raise ValueError(f"'{value}' is not one of the product codes given; use null for items not on the list")
        return value


class IntakeAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lines: list[AnswerLine] = Field(default_factory=list)
    needed_by: str | None = None
    unclear: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ what intake returns

@dataclass
class IntakeLine:
    product_code: str
    product_name: str
    base_unit: str
    qty: Decimal               # in base units, worked out by code
    product_text: str          # as written
    quantity_text: str         # as written
    unit_text: str | None      # as written
    note: str = ""
    needs_check: bool = False
    check_reasons: list[str] = field(default_factory=list)

    @property
    def qty_label(self) -> str:
        return format_qty(self.qty, self.base_unit)


@dataclass
class IntakeResult:
    lines: list[IntakeLine]
    needed_by: date | None
    needed_by_text: str | None
    unclear: list[str]
    source: str                   # "ai" or "text_match"
    ai_error: str | None = None   # why the AI wasn't used, if it was supposed to be


# ------------------------------------------------------------------ the reader

def read_request(text: str, *, products: Sequence[ProductRef], client: AnyClient, today: date,
                 model_tested: bool, cfg: IntakeConfig, needed_by: date | None = None,
                 request_id: int | None = None) -> IntakeResult:
    """Read a manager's message. `needed_by` from the form, if given, wins over the text."""
    by_code = {p.code: p for p in products}
    answer, ai_error = None, None
    try:  # the no-AI client fails at once (and logs the call), sending us to text matching
        answer = _ask_ai(text, products, client, today, request_id)
    except LLMError as exc:
        if client.enabled:
            ai_error = str(exc)

    unclear: list[str] = []
    lines: list[IntakeLine] = []
    if answer is not None:
        source = "ai"
        date_text = answer.needed_by
        unclear += [u for u in answer.unclear if u.strip()]
        for raw in answer.lines:
            product = by_code.get(raw.product_code) if raw.product_code else None
            if product is None:
                unclear.append(f"'{raw.product_text}' isn't on our product list.")
                continue
            line = _to_line(text, product, raw.product_text, raw.quantity, raw.unit, raw.note, unclear)
            if line is not None:
                if not model_tested:
                    _flag(line, "model not tested")
                lines.append(line)
    else:
        source = "text_match"
        date_text = None
        for raw in textparse.split_lines(text, today):
            product, problem = _product_by_text(raw.product_text, products, cfg)
            if product is None:  # never dropped silently: anything unread is listed
                if raw.quantity is None and not problem.startswith("not sure"):
                    unclear.append(f"Couldn't read '{raw.product_text}': no quantity, and not on our product list.")
                else:
                    unclear.append(f"'{raw.product_text}': {problem}.")
                continue
            if raw.quantity is None:
                unclear.append(f"{product.name}: no quantity given.")
                continue
            line = _to_line(text, product, raw.product_text, raw.quantity, raw.unit, "", unclear)
            if line is not None:
                if ai_error:
                    _flag(line, f"read by text matching because the AI failed ({ai_error})")
                lines.append(line)

    lines = _merge_duplicates(lines)
    found_date, found_text = _needed_by(text, today, needed_by, date_text, source, unclear)
    return IntakeResult(lines, found_date, found_text, unclear, source, ai_error)


def _ask_ai(text: str, products: Sequence[ProductRef], client: AnyClient, today: date,
            request_id: int | None) -> IntakeAnswer:
    product_list = "\n".join(f"{p.code} | {p.name} | {units.display_unit(p.base_unit)}" for p in products)
    system = render_prompt("read_request", products=product_list, today=f"{today:%A} {today.isoformat()}")
    user = render_prompt("read_request_user", text=text)
    return client.complete_json(system, user, IntakeAnswer, task=TASK, request_id=request_id,
                                context={"product_codes": {p.code for p in products}})


def _flag(line: IntakeLine, reason: str) -> None:
    line.needs_check = True
    if reason not in line.check_reasons:
        line.check_reasons.append(reason)


def _to_line(text: str, product: ProductRef, product_text: str, quantity: str, unit: str | None, note: str,
             unclear: list[str]) -> IntakeLine | None:
    """Check a line against the original text and convert it. Problems go to `unclear`."""
    name = product.name
    if not textparse.quantity_in_text(text, quantity):
        unclear.append(f"{name}: the quantity '{quantity}' isn't in the message, so this line was dropped.")
        return None
    if unit is not None and not textparse.unit_in_text(text, unit):
        unclear.append(f"{name}: the unit '{unit}' isn't in the message, so this line was dropped.")
        return None
    qty = textparse.parse_quantity(quantity)
    if qty is None or qty <= 0:
        unclear.append(f"{name}: couldn't read the quantity '{quantity}'.")
        return None
    base_label = units.display_unit(product.base_unit)
    if unit is None:
        if product.base_unit != "each":
            unclear.append(f"{name}: {quantity} what? Add a unit, such as {base_label}.")
            return None
        base_qty = qty
    elif unit.strip().lower() in textparse.CASE_WORDS:
        unclear.append(f"{name}: '{quantity} {unit}' depends on whose {unit.rstrip('s')} it is. "
                       f"Give the amount in {base_label}.")
        return None
    else:
        base_qty = units.to_base(qty, unit, product.base_unit)
        if base_qty is None:
            code = units.normalize_unit(unit)
            if code == "oz" and product.base_unit == "fl_oz":
                unclear.append(f"{name}: '{quantity} {unit}' could be weight or fluid ounces. Write 'fl oz'.")
            elif code is None:
                unclear.append(f"{name}: don't know the unit '{unit}'.")
            else:
                unclear.append(f"{name}: '{unit}' doesn't measure {base_label}.")
            return None
    return IntakeLine(product.code, name, product.base_unit, base_qty, product_text, quantity, unit, note)


def _product_by_text(words: str, products: Sequence[ProductRef], cfg: IntakeConfig) -> tuple[ProductRef | None, str]:
    """No-AI product lookup. Accept only a clear winner; otherwise say what was close."""
    ranked = top_matches(words, {p.code: p.name for p in products}, limit=3)
    if not ranked or ranked[0][2] < cfg.product_min_score:
        close = [name for _, name, s in ranked if s >= 60]
        return None, ("isn't on our product list" if not close else f"not sure which product (closest: {', '.join(close)})")
    if len(ranked) > 1 and ranked[0][2] - ranked[1][2] < cfg.product_min_margin:
        close = [name for _, name, s in ranked if ranked[0][2] - s < cfg.product_min_margin]
        return None, f"not sure which product: {' or '.join(close)}?"
    best = ranked[0][0]
    return next(p for p in products if p.code == best), ""


def _merge_duplicates(lines: list[IntakeLine]) -> list[IntakeLine]:
    merged: dict[str, IntakeLine] = {}
    for line in lines:
        if line.product_code not in merged:
            merged[line.product_code] = line
            continue
        first = merged[line.product_code]
        first.qty += line.qty
        first.quantity_text = f"{first.quantity_text} + {line.quantity_text}"
        first.needs_check = first.needs_check or line.needs_check
        first.check_reasons += [r for r in line.check_reasons if r not in first.check_reasons]
        _flag(first, "two lines for the same product were added together")
    return list(merged.values())


def _needed_by(text: str, today: date, form_date: date | None, ai_text: str | None, source: str,
               unclear: list[str]) -> tuple[date | None, str | None]:
    if form_date is not None:
        found, written = form_date, None
    elif source == "ai":
        if ai_text is None:
            found, written = None, None
        elif not textparse.mentions(text, ai_text):
            unclear.append(f"The needed-by date '{ai_text}' isn't in the message.")
            return None, None
        else:
            found, written = textparse.parse_needed_by(ai_text, today), ai_text
            if found is None:
                unclear.append(f"Couldn't read the date '{ai_text}'. Pick the needed-by date in the form.")
                return None, ai_text
    else:
        mentions = textparse.find_needed_by(text, today)
        values = {m.value for m in mentions}
        if len(values) > 1:
            unclear.append("The message mentions more than one date: "
                           + ", ".join(m.text for m in mentions) + ". Pick the needed-by date in the form.")
            return None, None
        found, written = (mentions[0].value, mentions[0].text) if mentions else (None, None)
    if found is None:
        unclear.append("No needed-by date. Pick one in the form.")
    elif found < today:
        unclear.append(f"The needed-by date {textparse.day_label(found)} is in the past.")
        return None, written
    return found, written
