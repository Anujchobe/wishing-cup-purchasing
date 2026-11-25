from decimal import Decimal

import pytest
import yaml

from app.config import ConfigError, ai_status, load_config, load_llm_settings
from tests.conftest import config_with, raw_config


def test_step1_settings(cfg):
    assert [s.name for s in cfg.suppliers] == ["Sysco", "US Foods", "PFG", "Supplier 4 (placeholder)"]
    assert cfg.purchasing.main_supplier == "sysco"
    assert cfg.main_supplier.name == "Sysco"
    assert cfg.purchasing.match_threshold == Decimal("0.05")  # exact, not a float
    assert cfg.ai.eval_threshold == Decimal("0.9")
    assert cfg.approval.approver == "owner"
    assert cfg.approval.price_max_age_hours == 24
    assert [loc.name for loc in cfg.locations] == ["Court St Cafe", "Factory"]
    assert {loc.kind for loc in cfg.locations} == {"cafe", "factory"}
    assert "FAKE" in cfg.app.fake_data_label


def test_all_fake_problems_off(cfg):
    assert cfg.faults.ai_down is False
    assert cfg.faults.any_on is False
    assert cfg.faults.describe_on() == []
    for sup in cfg.suppliers:
        assert not cfg.faults.for_supplier(sup.code).any_on


def test_suppliers_differ(cfg):
    for field in ("min_order_cents", "delivery_fee_cents", "lead_time_days"):
        values = [getattr(s, field) for s in cfg.suppliers]
        assert len(set(values)) == len(values), field
    days = [tuple(s.delivery_weekdays) for s in cfg.suppliers]
    assert len(set(days)) == len(days)
    # With 2 locations there are only 3 possible non-empty sets of locations,
    # so 4 suppliers can't all differ here. All 3 sets are used.
    served = {frozenset(s.serves_locations) for s in cfg.suppliers}
    assert served == {frozenset({"court_st", "factory"}), frozenset({"court_st"}), frozenset({"factory"})}


def test_every_location_has_a_supplier(cfg):
    for loc in cfg.locations:
        assert any(loc.code in s.serves_locations for s in cfg.suppliers), loc.code


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["purchasing"].update(main_supplier="nobody"), "main_supplier 'nobody'"),
        (lambda c: c["suppliers"][0].update(serves_locations=["moon_base"]), "unknown location"),
        (lambda c: c["suppliers"][0].update(min_order_cents=500.5), "min_order_cents"),
        (lambda c: c["suppliers"][0].update(min_order_cents=-1), "min_order_cents"),
        (lambda c: c["suppliers"][1].update(code="sysco"), "duplicate supplier"),
        (lambda c: c["faults"]["suppliers"].update(ghost={"down": True}), "unknown supplier"),
        (lambda c: c["suppliers"][0].update(delivery_weekdays=["Funday"]), "delivery_weekdays"),
        (lambda c: c["ai"].update(eval_threshold=1.5), "eval_threshold"),
        (lambda c: c.update(surprise=True), "surprise"),
    ],
)
def test_bad_config_is_refused_in_plain_words(mutate, message):
    with pytest.raises(ConfigError, match=message):
        config_with(mutate)


def test_missing_and_broken_files(tmp_path):
    with pytest.raises(ConfigError, match="Can't find"):
        load_config(tmp_path / "nope.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("suppliers: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(broken)


def test_roundtrip_through_a_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw_config()), encoding="utf-8")
    assert load_config(path).purchasing.match_threshold == Decimal("0.05")


# ------------------------------------------------------------------ AI settings

@pytest.fixture
def no_llm_env(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_no_key_means_text_matching(cfg, tmp_path, no_llm_env):
    llm = load_llm_settings(tmp_path / "missing.env")
    assert not llm.configured
    status = ai_status(cfg, llm)
    assert not status.enabled and "text matching" in status.reason


def test_model_comes_only_from_env_file(cfg, tmp_path, no_llm_env):
    env = tmp_path / ".env"
    env.write_text("LLM_BASE_URL=http://localhost:11434/v1\nLLM_API_KEY=ollama\nLLM_MODEL=test-model\n",
                   encoding="utf-8")
    llm = load_llm_settings(env)
    assert (llm.base_url, llm.model) == ("http://localhost:11434/v1", "test-model")
    assert ai_status(cfg, llm).enabled


def test_real_env_var_beats_env_file(tmp_path, no_llm_env, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=x\nLLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    assert load_llm_settings(env).model == "from-env"


def test_ai_down_switch_wins(tmp_path, no_llm_env):
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=x\nLLM_MODEL=m\n", encoding="utf-8")
    cfg = config_with(lambda c: c["faults"].update(ai_down=True))
    status = ai_status(cfg, load_llm_settings(env))
    assert not status.enabled and "ai_down" in status.reason
