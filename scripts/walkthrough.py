"""Walk through the whole flow without the browser, the same way the screens do it.

    python scripts/walkthrough.py

Two runs on a throwaway copy of the fake data (your database isn't touched):
  1. every fake problem off
  2. Sysco down and the AI down
Each run: read the factory example, save it, get prices, plan, confirm what needs a person,
approve, place orders, write purchase orders, then run every report.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import select  # noqa: E402

from app import checks, clock, insights, models, review, workflow  # noqa: E402
from app.config import SupplierFaults, load_config, load_llm_settings  # noqa: E402
from app.db import init_db, make_engine, session_scope  # noqa: E402
from app.examples import example_text  # noqa: E402
from app.intake import ProductRef, read_request  # noqa: E402
from app.llm.client import db_logger, make_client  # noqa: E402
from app.llm.model_check import model_status  # noqa: E402
from app.matching import match_products  # noqa: E402
from app.pricing import format_cents  # noqa: E402
from app.suppliers.mock import build_suppliers  # noqa: E402
from scripts.seed import seed  # noqa: E402


def run(title: str, folder: Path, *, down: tuple[str, ...] = (), ai_down: bool = False) -> None:
    base = load_config()
    faults = base.faults.model_copy(update={
        "ai_down": ai_down,
        "suppliers": {sc.code: SupplierFaults(down=sc.code in down) for sc in base.suppliers}})
    cfg = base.model_copy(update={"faults": faults})
    engine = make_engine(folder / "walkthrough.db")
    try:
        init_db(engine)
        with session_scope(engine) as s:
            seed(s, cfg)
            products = [ProductRef(p.code, p.name, p.base_unit.value) for p in s.exec(select(models.Product)).all()]
        llm = load_llm_settings()
        status = model_status(cfg, llm)
        client = make_client(cfg, llm, log_call=db_logger(engine))
        print(f"\n=== {title} ===\nAI: {status.label}")

        text = example_text("factory", clock.today())
        result = read_request(text, products=products, client=client, today=clock.today(),
                              model_tested=status.tested, cfg=cfg.intake)
        print(f"Read by {result.source}: {len(result.lines)} lines; unclear: {result.unclear or 'nothing'}")
        lines = [workflow.LineInput(x.product_code, x.qty, x.note, x.product_text, x.needs_check,
                                    "; ".join(x.check_reasons) or None) for x in result.lines]
        with session_scope(engine) as s:
            req, _ = workflow.save_request(s, location_code="factory", text=text, needed_by=result.needed_by,
                                           lines=lines, created_by="manager", client=client, cfg=cfg,
                                           model_tested=status.tested, read_by=result.source)
            rid = req.id
        suppliers = build_suppliers(cfg, ledger_dir=folder / "ledgers", sleep=lambda s: None)
        with session_scope(engine) as s:
            prices = workflow.get_prices(s, rid, suppliers, cfg, sleep=lambda s: None)
        print(f"Prices from {prices.answered}; no answer: {list(prices.no_answer) or 'none'}; "
              f"set aside: {len(prices.set_aside)}")
        with session_scope(engine) as s:
            plan = workflow.make_plan(s, rid, cfg)
        for w in plan.warnings:
            print(f"  warning: {w}")
        p = plan.plans
        print(f"Plans: a {format_cents(p.simple.total_cents)}, "
              f"b {format_cents(p.one_supplier.total_cents) if p.one_supplier else '-'}, "
              f"c {format_cents(p.best.total_cents)} via {p.best.solved_by}, using {[sl.name for sl in p.best.lists]}")
        with session_scope(engine) as s:
            items = review.attention(checks.load_view(s, rid, cfg, clock.now()), review.cost_matrix(s, rid, cfg))
        print(f"Needs a person: {len(items)} items; the owner confirms them")
        with session_scope(engine) as s:
            changed = workflow.apply_changes(s, rid, "owner", [workflow.ConfirmItem(i.product_code)
                                                               for i in items if i.can_confirm], cfg)
        print("Checks:", "all pass" if not checks.failures(changed.checks) else checks.summary(changed.checks))
        with session_scope(engine) as s:
            approval = workflow.approve(s, rid, "owner", (), cfg)
        print("Approval:", approval.message)
        if approval.approved:
            report = workflow.place_orders(engine, rid, suppliers, cfg, sleep=lambda s: None,
                                           po_dir=folder / "purchase_orders")
            print("Orders:", report.summary)
            print("Purchase orders:", [x.name for x in report.purchase_orders])
        with session_scope(engine) as s:
            codes = [x.code for x in s.exec(select(models.Product)).all()]
        with session_scope(engine) as s:
            match_products(s, codes, client, cfg, model_tested=status.tested)
        with session_scope(engine) as s:
            replay = insights.load_savings_replay(s, cfg)
            watch = insights.load_price_watch(s, cfg)
            match = insights.load_match_candidates(s, cfg)
            by_supplier, _ = insights.load_spend(s)
        print(f"Reports: replay would have saved {format_cents(replay.saved_cents)} over {len(replay.rows)} orders; "
              f"price watch {[(w.product_name, w.kind) for w in watch]}; price match {len(match)} items; "
              f"spend {[(r.name, format_cents(r.total_cents)) for r in by_supplier]}")
        if match:
            subject, _, by = insights.write_price_match_email(client, cfg.main_supplier.name,
                                                              cfg.main_supplier.sales_rep.name, match)
            print(f"Price-match email: '{subject}' (wording by {by})")
    finally:
        engine.dispose()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "one").mkdir()
        (Path(tmp) / "two").mkdir()
        run("Every fake problem off", Path(tmp) / "one")
        run("Sysco down, AI down", Path(tmp) / "two", down=("sysco",), ai_down=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
