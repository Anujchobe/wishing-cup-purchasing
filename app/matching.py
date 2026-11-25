"""Matching our products to supplier SKUs, only for the products in the current request.

For each product:
  1. Suppliers that already have a mapping for it are skipped: confirmed matches are reused and
     the AI is only asked about products it hasn't seen at that supplier.
  2. rapidfuzz picks each remaining supplier's 5 closest catalog items. Items with a confirmed
     mapping to another product are left out, so a SKU can't belong to two products. An item that is
     only *suggested* for another product stays available: a clear match takes it over (the
     suggestion is dropped), while another unsure match is not saved (the first suggestion stays).
  3. With AI, one call per product. For each supplier it returns one SKU from that supplier's list
     (or none), the pack count, size and unit, and how sure it is. The answer schema refuses any SKU
     that wasn't on the list, which costs the AI its one retry. Code converts the pack to base units.
  4. A person must check the match if: the AI is less than 70% sure, the AI and our pack reader read
     the pack differently, the pack can't be converted, the model is not tested, or the AI failed and
     text matching was used instead.
  5. With no AI: take the best text match if it scores at least 90, is 10 ahead of the next one, and
     our pack reader can read its pack text. Otherwise save it as a suggestion for a person to check
     (if it scores at least 60), or save nothing.
  6. Results are saved in item_mappings. All AI calls happen before any row is written, so the call
     log (which writes on its own connection) never waits on this transaction's lock.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from sqlmodel import Session, select

from app import clock, models, units
from app.config import AppConfig
from app.intake import ProductRef
from app.llm.client import AnyClient, LLMError, render_prompt
from app.pricing import format_qty
from app.textmatch import top_matches

TASK = "match_product"
PACK_UNITS = "lb, oz, kg, g, fl oz, cup, pt, qt, gal, ml, l, each, dz"


@dataclass(frozen=True)
class CatalogEntry:
    supplier: str
    sku: str
    description: str
    raw_pack_text: str
    item_id: int | None = None


@dataclass(frozen=True)
class Candidate:
    entry: CatalogEntry
    score: float  # 0-100


# ------------------------------------------------------------------ what the AI must send back

def _as_text(value: Any) -> Any:
    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        return str(value)  # numeric SKUs like 8762019 sometimes come back as numbers
    return value


def _as_int(value: Any) -> Any:
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    return value


class SupplierPick(BaseModel):
    model_config = ConfigDict(extra="ignore")

    supplier: str
    sku: Annotated[str | None, BeforeValidator(_as_text)] = None
    pack_count: Annotated[Annotated[int, Field(gt=0)] | None, BeforeValidator(_as_int)] = None
    pack_size: Annotated[Decimal, Field(gt=0)] | None = None
    pack_unit: str | None = None
    confidence: Annotated[Decimal, Field(ge=0, le=1)]
    reason: str = ""

    @field_validator("supplier")
    @classmethod
    def _asked_about(cls, value: str, info: ValidationInfo) -> str:
        asked = (info.context or {}).get("candidates")
        if asked is not None and value not in asked:
            raise ValueError(f"'{value}' is not one of the suppliers asked about ({', '.join(asked)})")
        return value

    @field_validator("sku")
    @classmethod
    def _sku_from_the_list(cls, value: str | None, info: ValidationInfo) -> str | None:
        asked = (info.context or {}).get("candidates")
        supplier = info.data.get("supplier")
        if value is None or asked is None or supplier not in asked:
            return value
        if value not in asked[supplier]:
            raise ValueError(f"sku '{value}' is not in {supplier}'s list; use one of "
                             f"{', '.join(asked[supplier]) or '(none)'} or null")
        return value

    @field_validator("pack_unit")
    @classmethod
    def _known_unit(cls, value: str | None) -> str | None:
        if value is not None and units.normalize_unit(value) is None:
            raise ValueError(f"unknown pack_unit '{value}'; use one of: {PACK_UNITS}")
        return value

    @model_validator(mode="after")
    def _pack_given_with_sku(self) -> SupplierPick:
        if self.sku is not None and None in (self.pack_count, self.pack_size, self.pack_unit):
            raise ValueError("pack_count, pack_size and pack_unit are all needed when sku is set")
        return self


class MatchAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    picks: list[SupplierPick]

    @field_validator("picks")
    @classmethod
    def _each_supplier_once(cls, picks: list[SupplierPick], info: ValidationInfo) -> list[SupplierPick]:
        asked = (info.context or {}).get("candidates")
        if asked is None:
            return picks
        got = [p.supplier for p in picks]
        missing = [s for s in asked if s not in got]
        twice = sorted({s for s in got if got.count(s) > 1})
        if missing or twice:
            raise ValueError("answer once for every supplier: "
                             + "; ".join(filter(None, [missing and f"missing {', '.join(missing)}",
                                                       twice and f"repeated {', '.join(twice)}"])))
        return picks


# ------------------------------------------------------------------ results

@dataclass
class MatchDecision:
    product_code: str
    product_name: str
    base_unit: str
    supplier: str
    supplier_name: str
    item: CatalogEntry | None        # None = no match saved
    pack_size_base: Decimal | None
    confidence: float
    source: str                      # ai | text_match
    needs_check: bool
    reasons: list[str] = field(default_factory=list)
    ai_pack: Decimal | None = None
    reader_pack: Decimal | None = None
    saved: bool = False
    blocked: bool = False            # not saved: its SKU belongs to another product

    @property
    def pack_label(self) -> str:
        return format_qty(self.pack_size_base, self.base_unit) if self.pack_size_base is not None else "unknown"


@dataclass
class MatchReport:
    decisions: list[MatchDecision] = field(default_factory=list)
    already_known: list[tuple[str, str]] = field(default_factory=list)  # (product, supplier) skipped
    replaced: list[tuple[str, str, str]] = field(default_factory=list)   # (old product, new product, sku)
    unknown_products: list[str] = field(default_factory=list)
    ai_errors: dict[str, str] = field(default_factory=dict)              # product -> why the AI wasn't used


# ------------------------------------------------------------------ pure steps (no database)

def candidates_for(product_name: str, entries: Sequence[CatalogEntry], limit: int) -> list[Candidate]:
    ranked = top_matches(product_name, {e.sku: e.description for e in entries}, limit)
    by_sku = {e.sku: e for e in entries}
    return [Candidate(by_sku[sku], score) for sku, _, score in ranked]


def ask_ai(client: AnyClient, product: ProductRef, candidates: dict[str, list[Candidate]],
           supplier_names: dict[str, str], request_id: int | None = None) -> MatchAnswer:
    blocks = []
    for code, cands in candidates.items():
        rows = "\n".join(f"  sku {c.entry.sku} | {c.entry.description} | pack: {c.entry.raw_pack_text}" for c in cands)
        blocks.append(f"Supplier {code} ({supplier_names.get(code, code)}):\n{rows}")
    system = render_prompt("match_product", units=PACK_UNITS)
    user = render_prompt("match_product_user", name=product.name, code=product.code,
                         base_unit=units.display_unit(product.base_unit), suppliers="\n\n".join(blocks))
    context = {"candidates": {code: [c.entry.sku for c in cands] for code, cands in candidates.items()}}
    return client.complete_json(system, user, MatchAnswer, task=TASK, context=context, request_id=request_id)


def decide_with_ai(product: ProductRef, supplier: str, supplier_name: str, cands: list[Candidate],
                   pick: SupplierPick, cfg: AppConfig, *, model_tested: bool) -> MatchDecision:
    if pick.sku is None:
        return MatchDecision(product.code, product.name, product.base_unit, supplier, supplier_name, None, None,
                             float(pick.confidence), "ai", False, ["the AI found no match among the closest products"])
    entry = next(c.entry for c in cands if c.entry.sku == pick.sku)
    base_label = units.display_unit(product.base_unit)
    ai_pack = units.to_base(Decimal(pick.pack_count) * pick.pack_size, pick.pack_unit, product.base_unit)
    reader = units.pack_base_qty(entry.raw_pack_text, product.base_unit)
    reasons = []
    if ai_pack is None:
        reasons.append(f"the AI read the pack as {pick.pack_count} x {pick.pack_size} {pick.pack_unit}, "
                       f"which doesn't convert to {base_label}")
    elif reader is not None and reader != ai_pack:
        reasons.append(f"the AI read the pack as {format_qty(ai_pack, product.base_unit)}, "
                       f"our pack reader as {format_qty(reader, product.base_unit)}")
    if pick.confidence < cfg.ai.match_confidence_min:
        reasons.append(f"the AI is only {pick.confidence} sure (needs {cfg.ai.match_confidence_min})")
    if not model_tested:
        reasons.append("model not tested")
    pack = ai_pack if ai_pack is not None else reader
    return MatchDecision(product.code, product.name, product.base_unit, supplier, supplier_name, entry, pack,
                         float(pick.confidence), "ai", bool(reasons), reasons, ai_pack, reader)


def decide_without_ai(product: ProductRef, supplier: str, supplier_name: str, cands: list[Candidate],
                      cfg: AppConfig, *, ai_error: str | None = None) -> MatchDecision:
    m = cfg.matching
    if not cands or cands[0].score < m.suggestion_min_score:
        return MatchDecision(product.code, product.name, product.base_unit, supplier, supplier_name, None, None,
                             0.0, "text_match", False, ["no close product in this catalog"])
    top = cands[0]
    second = cands[1] if len(cands) > 1 else None
    reader = units.pack_base_qty(top.entry.raw_pack_text, product.base_unit)
    reasons = []
    if ai_error:
        reasons.append(f"text matching was used because the AI failed ({ai_error})")
    if top.score < m.text_match_min_score:
        reasons.append(f"text match score {top.score:.0f} is below {m.text_match_min_score:.0f}")
    elif second is not None and top.score - second.score < m.text_match_min_margin:
        reasons.append(f"'{second.entry.description}' scores almost the same ({second.score:.0f})")
    if reader is None:
        reasons.append(f"the pack text '{top.entry.raw_pack_text}' can't be read safely")
    return MatchDecision(product.code, product.name, product.base_unit, supplier, supplier_name, top.entry, reader,
                         round(top.score / 100, 4), "text_match", bool(reasons), reasons, None, reader)


def match_one(product: ProductRef, candidates: dict[str, list[Candidate]], supplier_names: dict[str, str],
              client: AnyClient, cfg: AppConfig, *, model_tested: bool,
              request_id: int | None = None) -> tuple[list[MatchDecision], str | None]:
    """Decide one product at the given suppliers. Returns the decisions and the AI error, if any."""
    askable = {code: cands for code, cands in candidates.items() if cands}
    answer, ai_error = None, None
    if askable:
        try:
            answer = ask_ai(client, product, askable, supplier_names, request_id)
        except LLMError as exc:
            ai_error = str(exc) if client.enabled else None
    decisions = []
    for code, cands in candidates.items():
        name = supplier_names.get(code, code)
        if answer is not None and code in askable:
            pick = next(p for p in answer.picks if p.supplier == code)
            decisions.append(decide_with_ai(product, code, name, cands, pick, cfg, model_tested=model_tested))
        else:
            decisions.append(decide_without_ai(product, code, name, cands, cfg, ai_error=ai_error))
    return decisions, ai_error


# ------------------------------------------------------------------ who gets a SKU

@dataclass
class _Holder:
    product_name: str
    row: Any = None                       # an existing needs-check ItemMapping row
    decision: MatchDecision | None = None  # or a needs-check decision from this run


def claim_sku(d: MatchDecision, holders: dict[Any, _Holder], confirmed: set, report: MatchReport,
              to_delete: list) -> None:
    """One SKU, one product. A clear match beats a suggestion; two suggestions: the first stays."""
    key = d.item.item_id if d.item.item_id is not None else (d.supplier, d.item.sku)
    holder = holders.get(key)
    if holder is not None:
        if d.needs_check:
            d.blocked = True
            d.reasons.append(f"not saved: this SKU is already suggested for {holder.product_name}")
            return
        if holder.row is not None:
            to_delete.append(holder.row)
        if holder.decision is not None:
            holder.decision.blocked = True
            holder.decision.reasons.append(f"not saved: its SKU is a clear match for {d.product_name}")
        report.replaced.append((holder.product_name, d.product_name, d.item.sku))
    if d.needs_check:
        holders[key] = _Holder(d.product_name, decision=d)
    else:
        holders.pop(key, None)
        confirmed.add(key)


# ------------------------------------------------------------------ with the database

def match_products(session: Session, product_codes: Sequence[str], client: AnyClient, cfg: AppConfig, *,
                   model_tested: bool, request_id: int | None = None) -> MatchReport:
    """Match the products of one request and save the results in item_mappings (caller commits)."""
    report = MatchReport()
    codes = list(dict.fromkeys(product_codes))
    all_products = session.exec(select(models.Product)).all()
    products = {p.code: p for p in all_products if p.code in codes}
    name_by_id = {p.id: p.name for p in all_products}
    suppliers = [s for s in session.exec(select(models.Supplier).where(models.Supplier.active)).all()]
    order = {sc.code: n for n, sc in enumerate(cfg.suppliers)}
    suppliers.sort(key=lambda s: order.get(s.code, len(order)))
    names = {s.code: s.name for s in suppliers}
    mappings = session.exec(select(models.ItemMapping)).all()
    known_pairs = {(m.product_id, m.supplier_id) for m in mappings}
    confirmed = {m.supplier_item_id for m in mappings if not m.needs_check}
    holders = {m.supplier_item_id: _Holder(name_by_id.get(m.product_id, "?"), row=m)
               for m in mappings if m.needs_check}
    entries: dict[str, list[CatalogEntry]] = {s.code: [] for s in suppliers}
    supplier_by_id = {s.id: s for s in suppliers}
    for item in session.exec(select(models.SupplierItem)).all():
        sup = supplier_by_id.get(item.supplier_id)
        if sup is not None:
            entries[sup.code].append(CatalogEntry(sup.code, item.sku, item.description, item.raw_pack_text, item.id))

    to_delete: list = []
    for code in codes:
        row = products.get(code)
        if row is None:
            report.unknown_products.append(code)
            continue
        product = ProductRef(row.code, row.name, row.base_unit.value)
        todo = [s for s in suppliers if (row.id, s.id) not in known_pairs]
        report.already_known += [(code, s.code) for s in suppliers if (row.id, s.id) in known_pairs]
        if not todo:
            continue
        candidates = {s.code: candidates_for(product.name, [e for e in entries[s.code] if e.item_id not in confirmed],
                                             cfg.matching.candidates_per_supplier) for s in todo}
        decisions, ai_error = match_one(product, candidates, names, client, cfg, model_tested=model_tested,
                                        request_id=request_id)
        if ai_error:
            report.ai_errors[code] = ai_error
        for d in decisions:
            if d.item is not None:
                claim_sku(d, holders, confirmed, report, to_delete)
        report.decisions += decisions

    for old in to_delete:  # delete replaced suggestions first: a SKU may be in item_mappings only once
        session.delete(old)
    session.flush()
    now = clock.now()
    ids = {s.code: s.id for s in suppliers}
    for d in report.decisions:
        if d.item is None or d.blocked:
            continue
        session.add(models.ItemMapping(
            product_id=products[d.product_code].id,
            supplier_id=ids[d.supplier],
            supplier_item_id=d.item.item_id,
            pack_size_base=d.pack_size_base,
            confidence=d.confidence,
            source=models.MappingSource.ai if d.source == "ai" else models.MappingSource.text_match,
            needs_check=d.needs_check,
            check_reason="; ".join(d.reasons) or None,
            confirmed_by=None if d.needs_check else f"auto ({d.source.replace('_', ' ')})",
            confirmed_at=None if d.needs_check else now,
            created_at=now,
        ))
        d.saved = True
    session.flush()
    return report
