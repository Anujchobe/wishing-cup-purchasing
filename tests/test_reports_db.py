"""Reports and the review data on the seeded fake data."""
from datetime import timedelta

from sqlmodel import select

from app import checks, clock, insights, models, review, workflow
from app.db import session_scope
from app.llm.client import NoAIClient
from app.matching import match_products
from tests.fake_ai import as_json, fake_client
from tests.test_workflow import BIG, prepare

NO_AI = NoAIClient("off")


def match_everything(engine, cfg):
    with session_scope(engine) as s:
        codes = [p.code for p in s.exec(select(models.Product)).all()]
    with session_scope(engine) as s:
        match_products(s, codes, NO_AI, cfg, model_tested=False)


def test_reports_work_on_an_empty_history(engine, cfg):
    with session_scope(engine) as s:
        assert insights.load_savings_replay(s, cfg).rows == []
        assert insights.load_price_watch(s, cfg) == []
        assert insights.load_match_candidates(s, cfg) == []
        assert insights.load_spend(s) == ([], [])
        assert insights.load_price_series(s)[0] == {}


def test_coverage_and_matching_everything(live_engine, cfg):
    with session_scope(live_engine) as s:
        before = insights.load_coverage(s, cfg)
    assert before.matched == 24 and not before.complete          # only Sysco is matched in the seed data
    match_everything(live_engine, cfg)
    with session_scope(live_engine) as s:
        after = insights.load_coverage(s, cfg)
    assert after.matched > 60 and after.waiting > 0             # clear matches count; unsure ones wait


def test_savings_replay(live_engine, cfg):
    match_everything(live_engine, cfg)
    with session_scope(live_engine) as s:
        replay = insights.load_savings_replay(s, cfg)
    assert len(replay.rows) == 16                                   # 8 weeks x 2 locations
    assert replay.saved_cents > 0                                   # it would have saved money
    assert all(r.replay_cents is None or r.replay_cents <= r.actual_cents for r in replay.rows)


def test_price_watch_finds_the_planted_changes(live_engine, cfg):
    match_everything(live_engine, cfg)
    with session_scope(live_engine) as s:
        rows = {r.product_code: r for r in insights.load_price_watch(s, cfg)}
    assert rows["butter_unsalted"].kind == "market"                # +15% everywhere, 3 weeks ago
    assert rows["eggs"].kind == "one_supplier" and rows["eggs"].up == ("US Foods",)


def test_price_match_email_with_and_without_ai(live_engine, cfg):
    match_everything(live_engine, cfg)
    with session_scope(live_engine) as s:
        items = insights.load_match_candidates(s, cfg)
    assert items and all(i.main.supplier == "Sysco" and i.cheaper_by > cfg.purchasing.match_threshold for i in items)
    subject, body, by = insights.write_price_match_email(NO_AI, "Sysco", "Sam", items)
    assert by == "template" and insights.email_problems(body, items) == []
    made_up = as_json({"subject": "Prices", "body": "Hi Sam, could you go to $1.00 on everything? Thanks a lot!"})
    client, *_ = fake_client(cfg, made_up, made_up)
    assert insights.write_price_match_email(client, "Sysco", "Sam", items)[2] == "template"


def test_spend_by_supplier_and_location(live_engine, cfg):
    with session_scope(live_engine) as s:
        by_supplier, by_location = insights.load_spend(s)
    assert [r.name for r in by_supplier] == ["Sysco"] and by_supplier[0].orders == 16
    assert {r.name for r in by_location} == {"Court St Cafe", "Factory"}
    assert sum(r.total_cents for r in by_supplier) == sum(r.total_cents for r in by_location)


def test_review_data_for_a_planned_request(live_engine, cfg, suppliers):
    request_id, _, _ = prepare(live_engine, cfg, suppliers, text=BIG)
    with session_scope(live_engine) as s:
        matrix = review.cost_matrix(s, request_id, cfg)
        lists, cant = review.shopping_lists(s, request_id, cfg)
        view = checks.load_view(s, request_id, cfg, clock.now())
        items = review.attention(view, matrix)
        total = s.get(models.Request, request_id).plan_total_cents
        options = workflow.plan_options(s, request_id, cfg)
    assert matrix.cells[("heavy_cream", "pfg")].status == "dont_use"          # the planted $0 price
    assert matrix.cells[("eggs", "supplier4")].status == "not_serving"
    for code, supplier in matrix.cheapest.items():
        if supplier:
            costs = [c.cost_cents for (p, _), c in matrix.cells.items() if p == code and c.status == review.OK]
            assert matrix.cells[(code, supplier)].cost_cents == min(costs)
    assert sum(sl.total_cents for sl in lists) == total and cant == []
    assert options.best.total_cents == total
    assert all(i.reasons for i in items)


def test_review_data_is_safe_before_prices(live_engine, cfg):
    with session_scope(live_engine) as s:
        sub = workflow.submit_request(s, location_code="factory", text="30 lb unsalted butter", created_by="m",
                                      client=NO_AI, cfg=cfg, model_tested=False,
                                      needed_by=clock.today() + timedelta(days=7))
        rid = sub.request.id
    with session_scope(live_engine) as s:
        assert workflow.plan_options(s, rid, cfg) is None
        assert review.cost_matrix(s, rid, cfg).cells[("butter_unsalted", "sysco")].status == "no_price"
        assert review.shopping_lists(s, rid, cfg)[0] == []
