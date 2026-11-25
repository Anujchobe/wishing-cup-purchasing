"""Model check, the rule in SPEC.md: if the current model hasn't passed the AI test cases (score below
ai.eval_threshold), every AI answer needs a person to confirm and the app shows "model not tested".

`python -m evals.run` scores the model in .env and saves evals/results/<model>.json. This module reads
that file. A result only counts if it is for:
  * the same model at the same endpoint (LLM_MODEL and LLM_BASE_URL),
  * the same prompts and test cases (a fingerprint of app/llm/prompts/*.txt and evals/cases.json:
    edit either and the model is "not tested" again until the tests are re-run),
and every task (reading requests, matching products) scored at least ai.eval_threshold. One weak
task is enough to need a person, because both kinds of answer feed orders.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.config import ROOT, AppConfig, LLMSettings, ai_status
from app.llm.client import PROMPTS_DIR

RESULTS_DIR = ROOT / "evals" / "results"
CASES_FILE = ROOT / "evals" / "cases.json"
NOT_TESTED = "model not tested"


def eval_fingerprint(prompts_dir: Path = PROMPTS_DIR, cases_file: Path = CASES_FILE) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in prompts_dir.glob("*.txt") if p.is_file()) + ([cases_file] if cases_file.exists() else [])
    for path in files:
        # Line endings normalized, so a Windows checkout gives the same fingerprint.
        h.update(path.name.encode("utf-8") + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def results_path(model: str | None, results_dir: Path = RESULTS_DIR) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model or "").strip("._") or "model"
    return results_dir / f"{safe}.json"


@dataclass(frozen=True)
class ModelStatus:
    ai_enabled: bool
    tested: bool
    score: Decimal | None
    label: str


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def model_status(cfg: AppConfig, llm: LLMSettings, *, results_dir: Path = RESULTS_DIR,
                 fingerprint: str | None = None) -> ModelStatus:
    status = ai_status(cfg, llm)
    if not status.enabled:
        return ModelStatus(False, False, None, status.reason)
    threshold = cfg.ai.eval_threshold
    advice = "A person must confirm every AI answer. Run `python -m evals.run`."
    path = results_path(llm.model, results_dir)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ModelStatus(True, False, None, f"{NOT_TESTED}: {llm.model} has no test results yet. {advice}")
    if result.get("model") != llm.model or (result.get("base_url") or None) != (llm.base_url or None):
        return ModelStatus(True, False, None,
                           f"{NOT_TESTED}: the saved results are for another model or endpoint. {advice}")
    if result.get("fingerprint") != (fingerprint or eval_fingerprint()):
        return ModelStatus(True, False, None, f"{NOT_TESTED}: the prompts or test cases changed since "
                                              f"{llm.model} was tested. {advice}")
    tasks = {name: _decimal(t.get("score")) for name, t in (result.get("tasks") or {}).items()}
    scores = ", ".join(f"{name.replace('_', ' ')} {score}" for name, score in sorted(tasks.items()))
    low = min((s for s in tasks.values() if s is not None), default=None)
    if not tasks or low is None or any(s is None or s < threshold for s in tasks.values()):
        return ModelStatus(True, False, low, f"{NOT_TESTED}: {llm.model} scored {scores or 'nothing'}; every "
                                             f"task needs {threshold}. {advice}")
    return ModelStatus(True, True, low, f"AI model {llm.model}: passed its tests ({scores}; needs {threshold}).")


def save_results(llm: LLMSettings, result: dict[str, Any], *, results_dir: Path = RESULTS_DIR) -> Path:
    path = results_path(llm.model, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
