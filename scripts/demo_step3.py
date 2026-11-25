"""Step 3 examples: read one request and match one product, first with no AI, then with the
model in .env if a key is set.

    python scripts/demo_step3.py

Each run uses a throwaway copy of the fake data, so your database isn't changed. The only
thing read from your real database is the model's test score (for "model not tested").
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import select  # noqa: E402

from app import clock, models  # noqa: E402
from app.config import load_config, load_llm_settings  # noqa: E402
from app.db import init_db, make_engine, session_scope  # noqa: E402
from app.intake import IntakeResult, ProductRef, read_request  # noqa: E402
from app.llm.client import NoAIClient, db_logger, make_client  # noqa: E402
from app.llm.model_check import model_status  # noqa: E402
from app.matching import MatchReport, match_products  # noqa: E402
from app.textparse import day_label  # noqa: E402
from scripts.seed import seed  # noqa: E402

REQUEST = ("Hi! For the factory we need 50 lb AP flour, 2 dozen eggs, 3 gallons of heavy cream, "
           "1.5 kg dark chocolate and some saffron. Also vanilla 32 oz. By Friday please.")
MATCH_PRODUCT = "vanilla_extract"  # Sysco "4/32 OZ", PFG "PURE VAN EXT": the hard cases


def show_request(result: IntakeResult) -> None:
    print(f'  Message: "{REQUEST}"')
    print(f"  Read by: {'AI' if result.source == 'ai' else 'text matching'}"
          + (f" (AI failed: {result.ai_error})" if result.ai_error else ""))
    for line in result.lines:
        written = f"{line.quantity_text} {line.unit_text or ''}".strip()
        check = f"   CHECK: {'; '.join(line.check_reasons)}" if line.needs_check else ""
        print(f"    {line.product_name:<26} {written:<12} -> {line.qty_label}{check}")
    when = day_label(result.needed_by) if result.needed_by else "none"
    print(f"  Needed by: {when}" + (f'  (from "{result.needed_by_text}")' if result.needed_by_text else ""))
    for note in result.unclear:
        print(f"  Unclear: {note}")


def show_matching(report: MatchReport) -> None:
    for code, supplier in report.already_known:
        print(f"    {supplier:<10} already mapped; not asked again")
    for d in report.decisions:
        if d.item is None:
            print(f"    {d.supplier:<10} no match ({'; '.join(d.reasons)})")
            continue
        status = "needs a person: " + "; ".join(d.reasons) if d.needs_check else "accepted"
        print(f"    {d.supplier:<10} {d.item.sku:<16} {d.item.description:<30} pack '{d.item.raw_pack_text}' "
              f"= {d.pack_label}, {d.source}, confidence {d.confidence:.2f}\n{'':15}{status}")
    for product, error in report.ai_errors.items():
        print(f"  AI failed for {product}: {error}")


def run_once(title: str, folder: Path, client_factory, model_tested: bool) -> None:
    cfg = load_config()
    engine = make_engine(folder / "demo.db")
    try:
        init_db(engine)
        with session_scope(engine) as s:
            seed(s, cfg)
            products = [ProductRef(p.code, p.name, p.base_unit.value) for p in s.exec(select(models.Product)).all()]
            # Forget the seeded Sysco mapping so this product is matched at all four suppliers.
            product = s.exec(select(models.Product).where(models.Product.code == MATCH_PRODUCT)).one()
            for m in s.exec(select(models.ItemMapping).where(models.ItemMapping.product_id == product.id)).all():
                s.delete(m)
        client = client_factory(engine)
        print(f"\n=== {title} ===\n\nRequest example")
        show_request(read_request(REQUEST, products=products, client=client, today=clock.today(),
                                  model_tested=model_tested, cfg=cfg.intake))
        print(f"\nMatching example: {product.name}")
        with session_scope(engine) as s:
            show_matching(match_products(s, [MATCH_PRODUCT], client, cfg, model_tested=model_tested))
        with session_scope(engine) as s:
            calls = s.exec(select(models.LlmCall)).all()
        print(f"\n  llm_calls rows written: {len(calls)} "
              f"({sum(c.ok for c in calls)} worked, {sum(not c.ok for c in calls)} failed or fell back)")
    finally:
        engine.dispose()


def main() -> int:
    cfg = load_config()
    llm = load_llm_settings()
    with tempfile.TemporaryDirectory() as tmp:
        no_ai = Path(tmp) / "no_ai"
        no_ai.mkdir()
        run_once("No AI (text matching)", no_ai,
                 lambda engine: NoAIClient("demo: no AI", db_logger(engine)), model_tested=False)
        if not llm.configured:
            print("\nNo LLM_API_KEY / LLM_MODEL in .env, so there is no real-model example.")
            return 0
        status = model_status(cfg, llm)
        print(f"\nModel check: {status.label}")
        with_ai = Path(tmp) / "with_ai"
        with_ai.mkdir()
        run_once(f"With AI ({llm.model})", with_ai,
                 lambda engine: make_client(cfg, llm, log_call=db_logger(engine)), model_tested=status.tested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
