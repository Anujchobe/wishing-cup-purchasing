from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlmodel import select

from app import models
from app.db import session_scope
from app.intake import ProductRef
from app.llm.client import CallTimeout, NoAIClient, db_logger
from app.matching import (
    CatalogEntry,
    MatchAnswer,
    SupplierPick,
    candidates_for,
    decide_without_ai,
    match_products,
)
from app.suppliers import fake_world
from tests.fake_ai import as_json, fake_client

VANILLA = "vanilla_extract"
NO_AI = NoAIClient("No AI key and model in .env. Using text matching.")


def truth(supplier, product=VANILLA):
    return fake_world.offer(supplier, product)


def pick(supplier, count, size, unit, confidence=0.95, sku=None, product=VANILLA):
    return {"supplier": supplier, "sku": sku or truth(supplier, product).sku, "pack_count": count,
            "pack_size": size, "pack_unit": unit, "confidence": confidence, "reason": ""}


# Sysco's vanilla is already mapped in the seed data, so only these three are asked about.
RIGHT = [pick("usfoods", 1, "1", "gal"), pick("pfg", 12, "16", "fl oz"), pick("supplier4", 6, "16", "fl oz")]


def run(engine, cfg, client, model_tested=True, products=(VANILLA,)):
    with session_scope(engine) as s:
        report = match_products(s, list(products), client, cfg, model_tested=model_tested)
    with session_scope(engine) as s:
        rows = s.exec(select(models.ItemMapping).where(models.ItemMapping.source != models.MappingSource.person)).all()
        items = {i.id: i.sku for i in s.exec(select(models.SupplierItem)).all()}
    return report, {items[r.supplier_item_id]: r for r in rows}


def by_supplier(report):
    return {d.supplier: d for d in report.decisions}


def test_good_ai_answer_is_checked_and_saved(seeded_engine, cfg):
    client, transport, records, _ = fake_client(cfg, as_json({"picks": RIGHT}))
    report, saved = run(seeded_engine, cfg, client)
    assert report.already_known == [(VANILLA, "sysco")]
    asked = transport.sent[0][1]["content"]
    assert "Supplier sysco" not in asked and "Supplier pfg" in asked      # seen before: not asked
    assert asked.count("sku ") <= 3 * cfg.matching.candidates_per_supplier  # at most 5 per supplier
    d = by_supplier(report)
    assert {s: d[s].pack_size_base for s in d} == {"usfoods": Decimal(128), "pfg": Decimal(192), "supplier4": Decimal(96)}
    assert not any(x.needs_check for x in d.values())
    row = saved[truth("pfg").sku]
    assert (row.source, row.needs_check, row.pack_size_base, row.confidence) == \
           (models.MappingSource.ai, False, Decimal(192), 0.95)
    assert row.confirmed_by == "auto (ai)"
    assert len(records) == 1


def test_ai_and_pack_reader_disagree(seeded_engine, cfg):
    answer = [pick("usfoods", 1, "0.5", "gal")] + RIGHT[1:]   # true pack text says 1x1 gal
    client, *_ = fake_client(cfg, as_json({"picks": answer}))
    report, saved = run(seeded_engine, cfg, client)
    d = by_supplier(report)["usfoods"]
    assert d.needs_check and "AI read the pack as 64 fl oz, our pack reader as 128 fl oz" in d.reasons[0]
    assert saved[truth("usfoods").sku].needs_check


def test_low_confidence_goes_to_a_person(seeded_engine, cfg):
    answer = [pick("usfoods", 1, "1", "gal", confidence=0.6)] + RIGHT[1:]
    client, *_ = fake_client(cfg, as_json({"picks": answer}))
    d = by_supplier(run(seeded_engine, cfg, client)[0])
    assert d["usfoods"].needs_check and "only 0.6 sure" in d["usfoods"].reasons[0]
    assert not d["pfg"].needs_check


def test_model_not_tested_flags_every_ai_match(seeded_engine, cfg):
    client, *_ = fake_client(cfg, as_json({"picks": RIGHT}))
    report, _ = run(seeded_engine, cfg, client, model_tested=False)
    assert all(d.needs_check and "model not tested" in d.reasons for d in report.decisions)


def test_ai_reading_when_our_reader_cant(seeded_engine, cfg):
    """Sysco writes fluid ounces as 'OZ'. Our reader says 'unsure'; the AI reading is used as is."""
    with session_scope(seeded_engine) as s:
        for m in s.exec(select(models.ItemMapping)).all():
            s.delete(m)
    answer = [pick("sysco", 4, "32", "fl oz")] + RIGHT
    client, *_ = fake_client(cfg, as_json({"picks": answer}))
    d = by_supplier(run(seeded_engine, cfg, client)[0])["sysco"]
    assert (d.reader_pack, d.ai_pack, d.needs_check) == (None, Decimal(128), False)


@pytest.mark.parametrize("replies", [
    [as_json({"picks": [pick("usfoods", 1, "1", "gal", sku="NOT-A-SKU")] + RIGHT[1:]})] * 2,   # made-up sku
    ["not json at all", "{still not"],
    [CallTimeout("slow")],
])
def test_ai_failure_falls_back_to_text_matching(seeded_engine, cfg, replies):
    client, *_ = fake_client(cfg, *replies)
    report, saved = run(seeded_engine, cfg, client)
    assert VANILLA in report.ai_errors
    for d in report.decisions:
        assert d.source == "text_match" and d.needs_check
        assert "AI failed" in d.reasons[0]
    assert all(r.needs_check for r in saved.values())


def test_seen_products_are_not_asked_again(seeded_engine, cfg):
    client, transport, *_ = fake_client(cfg, as_json({"picks": RIGHT}))
    run(seeded_engine, cfg, client)
    second, _ = run(seeded_engine, cfg, client)
    assert len(transport.sent) == 1 and second.decisions == []
    assert {s for _, s in second.already_known} == {"sysco", "usfoods", "pfg", "supplier4"}


def test_only_products_in_the_request_are_matched(seeded_engine, cfg):
    report, _ = run(seeded_engine, cfg, NO_AI, products=(VANILLA,))
    assert {d.product_code for d in report.decisions} == {VANILLA}


def test_no_ai_run_saves_matches_and_logs_the_call(seeded_engine, cfg):
    report, saved = run(seeded_engine, cfg, NoAIClient("off", db_logger(seeded_engine)))
    d = by_supplier(report)
    assert not d["usfoods"].needs_check and d["usfoods"].item.sku == truth("usfoods").sku   # clear name, readable pack
    assert d["pfg"].needs_check                     # "PURE VAN EXT" is too far from the name to trust
    with session_scope(seeded_engine) as s:
        calls = s.exec(select(models.LlmCall)).all()
    assert [(c.model, c.ok) for c in calls] == [("(no AI)", False)]


def test_a_clear_match_takes_a_sku_from_a_wrong_suggestion(seeded_engine, cfg):
    """Supplier 4 has no cake flour, so its all-purpose flour is only a (wrong) suggestion for it.
    All-purpose flour's own clear match then takes the SKU, and the wrong suggestion is dropped."""
    s4_flour = truth("supplier4", "ap_flour").sku
    first, saved = run(seeded_engine, cfg, NO_AI, products=("cake_flour",))
    assert saved[s4_flour].needs_check
    second, saved = run(seeded_engine, cfg, NO_AI, products=("ap_flour",))
    assert ("Cake flour", "All-purpose flour", s4_flour) in second.replaced
    assert not saved[s4_flour].needs_check
    with session_scope(seeded_engine) as s:
        ap = s.exec(select(models.Product).where(models.Product.code == "ap_flour")).one()
        assert s.exec(select(models.ItemMapping).where(models.ItemMapping.supplier_item_id ==
                      s.exec(select(models.SupplierItem).where(models.SupplierItem.sku == s4_flour)).one().id)
                      ).one().product_id == ap.id


# ------------------------------------------------------------------ the answer schema

def ctx():
    return {"candidates": {"usfoods": ["US1", "US2"], "pfg": ["P-1"]}}


def test_schema_refuses_a_sku_not_on_the_list():
    with pytest.raises(ValidationError, match="not in usfoods's list"):
        MatchAnswer.model_validate({"picks": [
            {"supplier": "usfoods", "sku": "US9", "pack_count": 1, "pack_size": "1", "pack_unit": "gal", "confidence": 0.9},
            {"supplier": "pfg", "sku": None, "confidence": 0.9}]}, context=ctx())


def test_schema_needs_every_supplier_once():
    with pytest.raises(ValidationError, match="missing pfg"):
        MatchAnswer.model_validate({"picks": [{"supplier": "usfoods", "sku": None, "confidence": 0.5}]}, context=ctx())


def test_schema_needs_the_pack_with_a_sku_and_a_known_unit():
    with pytest.raises(ValidationError, match="all needed"):
        SupplierPick.model_validate({"supplier": "pfg", "sku": "P-1", "confidence": 0.9}, context=ctx())
    with pytest.raises(ValidationError, match="unknown pack_unit"):
        SupplierPick.model_validate({"supplier": "pfg", "sku": "P-1", "pack_count": 1, "pack_size": "2",
                                     "pack_unit": "barrels", "confidence": 0.9}, context=ctx())


def test_numeric_sku_and_decimal_pack_are_accepted():
    p = SupplierPick.model_validate({"supplier": "s", "sku": 8762019, "pack_count": Decimal("4"),
                                     "pack_size": Decimal("0.5"), "pack_unit": "GAL", "confidence": Decimal("0.8")})
    assert (p.sku, p.pack_count, p.pack_size) == ("8762019", 4, Decimal("0.5"))


# ------------------------------------------------------------------ no-AI safety over the whole fake catalog

def test_no_ai_never_accepts_a_wrong_match(cfg):
    """Every product at every supplier: an accepted text match is always the right SKU and pack."""
    accepted = set()
    for product in fake_world.PRODUCTS:
        ref = ProductRef(product.code, product.name, product.base_unit)
        for code in fake_world.SUPPLIER_CODES:
            entries = [CatalogEntry(code, o.sku, o.description, o.raw_pack_text) for o in fake_world.offers_for(code)]
            d = decide_without_ai(ref, code, code, candidates_for(product.name, entries, 5), cfg)
            if d.item is not None and not d.needs_check:
                right = fake_world.offer(code, product.code)
                assert (d.item.sku, d.pack_size_base) == (right.sku, right.pack_base_qty), (product.code, code)
                accepted.add((code, product.code))
    assert {("usfoods", "butter_unsalted"), ("supplier4", "heavy_cream"), ("sysco", "ap_flour")} <= accepted
    assert ("sysco", "vanilla_extract") not in accepted   # "4/32 OZ" on a liquid is never trusted without a person
