import json
from decimal import Decimal

from app.config import LLMSettings
from app.llm.model_check import NOT_TESTED, eval_fingerprint, model_status, results_path, save_results
from tests.conftest import config_with

LLM = LLMSettings(base_url="http://localhost:11434/v1", api_key="k", model="llama3.1:8b")


def result(read="1.0", match="1.0", **changes):
    base = {"model": LLM.model, "base_url": LLM.base_url, "fingerprint": eval_fingerprint(),
            "tasks": {"read_request": {"score": read}, "match_product": {"score": match}}}
    return {**base, **changes}


def status(cfg, tmp_path, llm=LLM):
    return model_status(cfg, llm, results_dir=tmp_path)


def test_never_tested(cfg, tmp_path):
    st = status(cfg, tmp_path)
    assert st.ai_enabled and not st.tested and NOT_TESTED in st.label and "no test results" in st.label


def test_passing_results_count(cfg, tmp_path):
    save_results(LLM, result(), results_dir=tmp_path)
    st = status(cfg, tmp_path)
    assert st.tested and st.score == Decimal("1.0") and "passed its tests" in st.label


def test_every_task_must_pass(cfg, tmp_path):
    save_results(LLM, result(read="1.0", match="0.867"), results_dir=tmp_path)
    st = status(cfg, tmp_path)
    assert not st.tested and NOT_TESTED in st.label and "match product 0.867" in st.label


def test_results_belong_to_one_model_endpoint_and_set_of_prompts(cfg, tmp_path):
    save_results(LLM, result(fingerprint="something-else"), results_dir=tmp_path)
    assert "changed since" in status(cfg, tmp_path).label
    save_results(LLM, result(base_url="https://elsewhere/v1"), results_dir=tmp_path)
    assert "another model or endpoint" in status(cfg, tmp_path).label
    save_results(LLM, result(), results_dir=tmp_path)
    other = LLMSettings(base_url=LLM.base_url, api_key="k", model="other-model")
    assert not status(cfg, tmp_path, other).tested


def test_broken_results_file_means_not_tested(cfg, tmp_path):
    results_path(LLM.model, tmp_path).write_text("{not json", encoding="utf-8")
    assert not status(cfg, tmp_path).tested


def test_file_name_is_safe_for_any_model_name(tmp_path):
    assert results_path("openai/gpt-4o:latest", tmp_path).name == "openai_gpt-4o_latest.json"
    assert results_path("", tmp_path).name == "model.json"


def test_ai_off_is_reported_as_off(cfg, tmp_path):
    assert not status(cfg, tmp_path, LLMSettings()).ai_enabled
    off = config_with(lambda c: c["faults"].update(ai_down=True))
    assert "ai_down" in model_status(off, LLM, results_dir=tmp_path).label


def test_fingerprint_changes_with_a_prompt_or_the_cases(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a.txt").write_text("one", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text("{}", encoding="utf-8")
    before = eval_fingerprint(prompts, cases)
    (prompts / "a.txt").write_text("two", encoding="utf-8")
    after_prompt = eval_fingerprint(prompts, cases)
    cases.write_text('{"x": 1}', encoding="utf-8")
    assert len({before, after_prompt, eval_fingerprint(prompts, cases)}) == 3
    (prompts / "a.txt").write_bytes(b"line\r\nline")      # Windows line endings don't count as a change
    windows = eval_fingerprint(prompts, cases)
    (prompts / "a.txt").write_bytes(b"line\nline")
    assert eval_fingerprint(prompts, cases) == windows
    json.loads((tmp_path / "cases.json").read_text())
