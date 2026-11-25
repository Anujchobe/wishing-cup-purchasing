"""Score the AI model in .env on evals/cases.json and save the result for the model check.

    python -m evals.run

Prints the score and time for each task (reading requests, matching products) and every failure,
then saves evals/results/<model>.json. The app treats the model as tested only if every task scores
at least ai.eval_threshold with these exact prompts and cases. Exit code 0 means it passed.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import clock  # noqa: E402
from app.config import AppConfig, LLMSettings, load_config, load_llm_settings  # noqa: E402
from app.intake import ProductRef, read_request  # noqa: E402
from app.llm.client import AnyClient, LLMError, make_client  # noqa: E402
from app.llm.model_check import CASES_FILE, eval_fingerprint, save_results  # noqa: E402
from app.matching import CatalogEntry, ask_ai, candidates_for, decide_with_ai  # noqa: E402
from app.suppliers import fake_world  # noqa: E402

TASKS = ("read_request", "match_product")


@dataclass(frozen=True)
class CaseResult:
    task: str
    case: str
    passed: bool
    detail: str = ""
    seconds: float = 0.0


def load_cases(path: Path = CASES_FILE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def products() -> list[ProductRef]:
    return [ProductRef(p.code, p.name, p.base_unit) for p in fake_world.PRODUCTS]


def run_read_case(case: dict, client: AnyClient, cfg: AppConfig, today: date) -> CaseResult:
    started = time.monotonic()
    result = read_request(case["text"], products=products(), client=client, today=today, model_tested=True,
                          cfg=cfg.intake)
    seconds = time.monotonic() - started
    if result.source != "ai":
        return CaseResult("read_request", case["id"], False, f"the AI wasn't used: {result.ai_error}", seconds)
    expect = case["expect"]
    problems = []
    got = {line.product_code: line.qty for line in result.lines}
    want = {code: Decimal(qty) for code, qty in expect.get("lines", {}).items()}
    if got != want:
        problems.append(f"lines {_show(got)}, expected {_show(want)}")
    if "needed_by" in expect:
        wanted = date.fromisoformat(expect["needed_by"]) if expect["needed_by"] else None
        if result.needed_by != wanted:
            problems.append(f"needed-by {result.needed_by}, expected {wanted}")
    unclear = " ".join(result.unclear).lower()
    if expect.get("unclear") and not unclear:
        problems.append("expected something in unclear")
    problems += [f"unclear should mention '{word}'" for word in expect.get("unclear_mentions", [])
                 if word.lower() not in unclear]
    return CaseResult("read_request", case["id"], not problems, "; ".join(problems), seconds)


def run_match_case(case: dict, client: AnyClient, cfg: AppConfig) -> CaseResult:
    product = next(p for p in products() if p.code == case["product"])
    names = {s.code: s.name for s in cfg.suppliers}
    expect = case["expect"]
    codes = [s for s in expect if fake_world.offers_for(s)]
    cands = {s: candidates_for(product.name, [CatalogEntry(s, o.sku, o.description, o.raw_pack_text)
                                              for o in fake_world.offers_for(s)],
                               cfg.matching.candidates_per_supplier) for s in codes}
    for s in codes:  # a case is only fair if the right answer was among what the model was shown
        sku = expect[s]["sku"]
        if sku is not None and all(c.entry.sku != sku for c in cands[s]):
            return CaseResult("match_product", case["id"], False,
                              f"case problem: the right {s} SKU {sku} isn't among the candidates (search, not model)")
    started = time.monotonic()
    try:
        answer = ask_ai(client, product, cands, names)
    except LLMError as exc:
        return CaseResult("match_product", case["id"], False, str(exc), time.monotonic() - started)
    seconds = time.monotonic() - started
    problems = []
    for s in codes:
        pick = next(p for p in answer.picks if p.supplier == s)
        right = expect[s]
        if pick.sku != right["sku"]:
            problems.append(f"{s}: picked {pick.sku}, expected {right['sku']}")
        elif right["sku"] is not None:
            got = decide_with_ai(product, s, names[s], cands[s], pick, cfg, model_tested=True).ai_pack
            if got != Decimal(right["pack"]):
                problems.append(f"{s}: read '{right.get('pack_text', '')}' as {got}, expected {right['pack']}")
    return CaseResult("match_product", case["id"], not problems, "; ".join(problems), seconds)


def run_all(client: AnyClient, cfg: AppConfig, cases: dict[str, Any] | None = None) -> list[CaseResult]:
    cases = cases or load_cases()
    today = date.fromisoformat(cases["today"])
    return ([run_read_case(c, client, cfg, today) for c in cases["read_request"]]
            + [run_match_case(c, client, cfg) for c in cases["match_product"]])


def summarize(results: list[CaseResult], llm: LLMSettings, cfg: AppConfig, seconds: float) -> dict[str, Any]:
    tasks = {}
    for task in TASKS:
        mine = [r for r in results if r.task == task]
        passed = sum(r.passed for r in mine)
        score = (Decimal(passed) / Decimal(len(mine))).quantize(Decimal("0.001")) if mine else Decimal(0)
        tasks[task] = {"score": str(score), "passed": passed, "total": len(mine),
                       "seconds": round(sum(r.seconds for r in mine), 1)}
    return {
        "model": llm.model, "base_url": llm.base_url, "fingerprint": eval_fingerprint(),
        "threshold": str(cfg.ai.eval_threshold),
        "passed": all(Decimal(t["score"]) >= cfg.ai.eval_threshold for t in tasks.values()),
        "tasks": tasks, "seconds": round(seconds, 1), "finished_at": clock.now().isoformat(timespec="seconds"),
        "failures": [{"task": r.task, "case": r.case, "detail": r.detail} for r in results if not r.passed],
    }


def _show(lines: dict) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(lines.items())) + "}"


def main() -> int:
    cfg = load_config()
    llm = load_llm_settings()
    client = make_client(cfg, llm)
    if not client.enabled:
        print(f"Nothing to test: {client.reason}")
        return 1
    print(f"Testing {llm.model} on {CASES_FILE.name}...")
    started = time.monotonic()
    results = run_all(client, cfg)
    summary = summarize(results, llm, cfg, time.monotonic() - started)
    for task, t in summary["tasks"].items():
        print(f"  {task:<14} {t['passed']}/{t['total']}  score {t['score']}  ({t['seconds']} s)")
    for f in summary["failures"]:
        print(f"  FAIL {f['task']}/{f['case']}: {f['detail']}")
    path = save_results(llm, summary)
    verdict = "passed: AI answers no longer need a person by default" if summary["passed"] else \
        f"not passed (every task needs {summary['threshold']}): the app keeps showing 'model not tested'"
    print(f"Took {summary['seconds']} s. Saved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}. "
          f"Result: {verdict}.")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
