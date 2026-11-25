"""Click through the app with Streamlit's own test runner (no browser needed).

Runs the whole flow twice: with every fake problem off, and with Sysco down and the AI down.
"""
from pathlib import Path

import pytest
from sqlmodel import select

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

from app import models  # noqa: E402
from app.config import load_config  # noqa: E402
from app.db import dispose_engines, get_engine, session_scope  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")
PAGES = ("New request", "Review and approve", "Orders", "Reports")


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WISHING_CUP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WISHING_CUP_ENV_FILE", str(tmp_path / "no.env"))   # never a real AI in tests
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "WISHING_CUP_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path
    dispose_engines()


def run(at):
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def start(app_env):
    at = run(AppTest.from_file(APP, default_timeout=180))
    at.button(key="seed_now").click()
    return run(at)


def go(at, page):
    at.radio(key="page").set_value(page)
    return run(at)


def no_errors(at):
    assert [e.value for e in at.error] == []


def test_an_empty_app_offers_the_fake_data_and_every_page_works(app_env):
    at = run(AppTest.from_file(APP, default_timeout=180))
    assert at.button(key="seed_now")                       # no data yet: no crash, a way forward
    at.button(key="seed_now").click()
    run(at)
    for page in PAGES:                                     # data, but no requests yet
        go(at, page)
        no_errors(at)


def full_flow(at, *, location="Factory"):
    go(at, "New request")
    at.selectbox(key="new_location").set_value(location)
    run(at)
    at.button(key="load_example").click()
    run(at)
    assert at.text_area(key="request_text").value
    at.button(key="read_request").click()
    run(at)
    no_errors(at)
    at.button(key="get_prices").click()
    run(at)
    assert at.radio(key="page").value == "Review and approve"
    no_errors(at)
    try:
        at.button(key="confirm_all").click()               # the owner checked the flagged items
        run(at)
    except KeyError:
        pass                                               # nothing needed a person
    no_errors(at)
    at.button(key="approve").click()
    run(at)
    assert at.radio(key="page").value == "Orders"
    no_errors(at)
    for page in PAGES:
        go(at, page)
        no_errors(at)
    return at


def requests_in_db():
    cfg = load_config()
    with session_scope(get_engine(cfg)) as s:
        return [r for r in s.exec(select(models.Request).where(models.Request.source != "seed_history")).all()]


def test_full_flow_with_every_fake_problem_off(app_env):
    at = full_flow(start(app_env))
    [req] = requests_in_db()
    assert req.status == models.RequestStatus.ORDERED
    go(at, "Reports")
    if _has(at, "match_all"):              # the reports with every product matched
        at.button(key="match_all").click()
        run(at)
    no_errors(at)


def test_full_flow_with_a_supplier_down_and_the_ai_down(app_env):
    at = start(app_env)
    at.toggle(key="fault_ai_down").set_value(True)
    at.toggle(key="fault_sysco_down").set_value(True)
    run(at)
    assert any("ai_down" in i.value for i in at.sidebar.info)
    full_flow(at)
    [req] = requests_in_db()
    assert req.status == models.RequestStatus.ORDERED
    assert req.suppliers_no_answer == ["sysco"]
    cfg = load_config()
    with session_scope(get_engine(cfg)) as s:
        lines = s.exec(select(models.RequestLine).where(models.RequestLine.request_id == req.id)).all()
        sysco = s.exec(select(models.Supplier).where(models.Supplier.code == "sysco")).one()
    assert all(line.assigned_supplier_id != sysco.id for line in lines)


def test_cafe_flow(app_env):
    full_flow(start(app_env), location="Court St Cafe")
    [req] = requests_in_db()
    assert req.status == models.RequestStatus.ORDERED


def _has(at, key):
    try:
        at.button(key=key)
        return True
    except KeyError:
        return False
