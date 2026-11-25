import copy
from datetime import date
from pathlib import Path

import pytest
import yaml
from hypothesis import settings

# CI machines (Windows and Mac runners especially) can be slow; a timing limit there only adds noise.
settings.register_profile("wishing_cup", deadline=None)
settings.load_profile("wishing_cup")

from app.config import AppConfig, load_config, parse_config
from app.db import init_db, make_engine, session_scope
from scripts.seed import seed

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.yaml"
FIXED_TODAY = date(2026, 11, 24)  # a Tuesday; seeding with a fixed day makes every row repeatable


def raw_config() -> dict:
    """config.yaml as a plain dict, for tests that need a changed copy."""
    return copy.deepcopy(yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")))


def config_with(mutate) -> AppConfig:
    raw = raw_config()
    mutate(raw)
    return parse_config(raw)


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return load_config(CONFIG_FILE)  # explicit path: ignore any WISHING_CUP_CONFIG override


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(tmp_path / "test.db")
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def live_engine(engine, cfg):
    """Seeded for today's real date, so "last week's" prices line up with suppliers answering now."""
    with session_scope(engine) as session:
        seed(session, cfg)
    return engine


@pytest.fixture
def suppliers(cfg, tmp_path):
    from app.suppliers.mock import build_suppliers

    return build_suppliers(cfg, ledger_dir=tmp_path / "ledgers", sleep=lambda seconds: None)


@pytest.fixture
def seeded_engine(engine, cfg):
    with session_scope(engine) as session:
        seed(session, cfg, today=FIXED_TODAY)
    return engine
