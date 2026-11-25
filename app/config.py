"""Load config.yaml and .env into validated, read-only objects.

A bad config stops the app with a plain-English list of what is wrong, rather
than failing later in some unrelated screen. Money in config must be whole
cents (an int); 500.5 is refused. Ratios such as match_threshold become
Decimal so money comparisons never touch floats.

AI settings come only from the environment or .env, so switching provider or
model never needs a code or config change. A real environment variable wins
over the .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV = "WISHING_CUP_CONFIG"      # optional: path to another config file
DATA_DIR_ENV = "WISHING_CUP_DATA_DIR"  # optional: put the database somewhere else (tests use this)
ENV_FILE_ENV = "WISHING_CUP_ENV_FILE"  # optional: read AI settings from another file (tests point it nowhere)
DB_FILENAME = "wishing_cup.db"


class ConfigError(Exception):
    """config.yaml is missing or invalid. The message is meant for a person."""


def _to_decimal(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("expected a number, got true/false")
    if isinstance(value, float):
        return Decimal(repr(value))  # 0.05 -> Decimal("0.05"), not 0.05000000000000000277
    if isinstance(value, (int, str)):
        try:
            return Decimal(value)
        except InvalidOperation:
            raise ValueError(f"{value!r} is not a number") from None
    return value


Ratio = Annotated[Decimal, BeforeValidator(_to_decimal), Field(ge=0)]
Cents = Annotated[int, Field(strict=True, ge=0)]
Code = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Weekday = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppSection(_Section):
    name: str = "Wishing Cup purchasing"
    data_dir: str = "data"
    random_seed: int = Field(default=42, strict=True)
    fake_data_label: str = Field(min_length=10)


class LocationConfig(_Section):
    code: Code
    name: str = Field(min_length=1)
    kind: Literal["cafe", "factory"]
    address: str = ""  # printed under "deliver to" on purchase orders


class SalesRep(_Section):
    name: str
    email: str


class SupplierConfig(_Section):
    code: Code
    name: str = Field(min_length=1)
    placeholder: bool = False
    min_order_cents: Cents
    delivery_fee_cents: Cents
    delivery_weekdays: list[Weekday] = Field(min_length=1)
    lead_time_days: int = Field(strict=True, ge=0)
    serves_locations: list[str] = Field(min_length=1)
    sales_rep: SalesRep
    connector: str = "mock"                   # "mock", or "package.module:ClassName" for a real connection
    connector_settings: dict[str, str] = Field(default_factory=dict)


class PurchasingConfig(_Section):
    main_supplier: str
    match_threshold: Ratio
    price_jump_limit: Ratio
    quantity_confirm_multiple: Ratio
    market_change_threshold: Ratio
    replay_weeks: int = Field(strict=True, ge=1)
    price_watch_weeks: int = Field(default=4, strict=True, ge=1)


class ApprovalConfig(_Section):
    approver: Literal["owner"]
    price_max_age_hours: int = Field(strict=True, gt=0)


Share = Annotated[Decimal, BeforeValidator(_to_decimal), Field(ge=0, le=1)]


class AIConfig(_Section):
    eval_threshold: Share
    retries_on_bad_answer: int = Field(default=1, strict=True, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    rate_limit_retries: int = Field(default=3, strict=True, ge=0)
    rate_limit_max_wait_seconds: float = Field(default=30.0, ge=0)
    match_confidence_min: Share = Decimal("0.7")


class IntakeConfig(_Section):
    """No-AI reading of product names: accept the best name only if clearly the best."""

    product_min_score: float = Field(default=90, ge=0, le=100)
    product_min_margin: float = Field(default=10, ge=0, le=100)


class MatchingConfig(_Section):
    candidates_per_supplier: int = Field(default=5, strict=True, ge=1)
    text_match_min_score: float = Field(default=90, ge=0, le=100)   # no-AI: accept at or above this...
    text_match_min_margin: float = Field(default=10, ge=0, le=100)  # ...if clearly ahead of the next one
    suggestion_min_score: float = Field(default=60, ge=0, le=100)   # below this, don't even suggest it


class SupplierCallsConfig(_Section):
    timeout_seconds: float = Field(gt=0)
    retries: int = Field(strict=True, ge=0)
    retry_wait_seconds: float = Field(ge=0)


class SupplierFaults(BaseModel):
    """Fake-problem switches for one supplier. Not frozen, so tests can flip them."""

    model_config = ConfigDict(extra="forbid")
    down: bool = False
    slow: bool = False
    flaky: bool = False

    @property
    def any_on(self) -> bool:
        return self.down or self.slow or self.flaky


class FaultsConfig(_Section):
    ai_down: bool = False
    slow_seconds: float = Field(default=1.0, ge=0)
    flaky_failure_rate: float = Field(default=0.3, ge=0, le=1)
    suppliers: dict[str, SupplierFaults] = Field(default_factory=dict)

    def for_supplier(self, code: str) -> SupplierFaults:
        return self.suppliers.get(code, SupplierFaults()).model_copy()

    @property
    def any_on(self) -> bool:
        return self.ai_down or any(f.any_on for f in self.suppliers.values())

    def describe_on(self) -> list[str]:
        """Plain-English list of the switches that are on."""
        out = ["AI is down"] if self.ai_down else []
        for code, faults in self.suppliers.items():
            out += [f"{code} is {name}" for name in ("down", "slow", "flaky") if getattr(faults, name)]
        return out


class AppConfig(_Section):
    app: AppSection
    locations: list[LocationConfig] = Field(min_length=1)
    suppliers: list[SupplierConfig] = Field(min_length=1)
    purchasing: PurchasingConfig
    approval: ApprovalConfig
    ai: AIConfig
    supplier_calls: SupplierCallsConfig
    intake: IntakeConfig = Field(default_factory=IntakeConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    faults: FaultsConfig = Field(default_factory=FaultsConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> AppConfig:
        loc_codes = [loc.code for loc in self.locations]
        sup_codes = [sup.code for sup in self.suppliers]
        for kind, codes in (("location", loc_codes), ("supplier", sup_codes)):
            dupes = sorted({c for c in codes if codes.count(c) > 1})
            if dupes:
                raise ValueError(f"duplicate {kind} code(s): {', '.join(dupes)}")
        if self.purchasing.main_supplier not in sup_codes:
            raise ValueError(
                f"main_supplier '{self.purchasing.main_supplier}' is not one of the suppliers "
                f"({', '.join(sup_codes)})"
            )
        for sup in self.suppliers:
            unknown = sorted(set(sup.serves_locations) - set(loc_codes))
            if unknown:
                raise ValueError(f"supplier '{sup.code}' serves unknown location(s): {', '.join(unknown)}")
        unknown_faults = sorted(set(self.faults.suppliers) - set(sup_codes))
        if unknown_faults:
            raise ValueError(f"faults are set for unknown supplier(s): {', '.join(unknown_faults)}")
        return self

    def supplier(self, code: str) -> SupplierConfig:
        for sup in self.suppliers:
            if sup.code == code:
                return sup
        raise KeyError(code)

    def location(self, code: str) -> LocationConfig:
        for loc in self.locations:
            if loc.code == code:
                return loc
        raise KeyError(code)

    @property
    def main_supplier(self) -> SupplierConfig:
        return self.supplier(self.purchasing.main_supplier)


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV) or ROOT / "config.yaml")


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path is not None else config_path()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Can't find the config file at {path}.") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML:\n{exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} should be a set of settings (key: value), but it isn't.")
    return parse_config(raw, source=path.name)


def parse_config(raw: dict[str, Any], source: str = "config.yaml") -> AppConfig:
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_plain_errors(exc, source)) from None


def _plain_errors(exc: ValidationError, source: str) -> str:
    lines = [f"{source} has {exc.error_count()} problem(s):"]
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "(whole file)"
        lines.append(f"  - {where}: {err['msg']}")
    return "\n".join(lines)


def data_dir(cfg: AppConfig) -> Path:
    override = os.environ.get(DATA_DIR_ENV)
    path = Path(override) if override else Path(cfg.app.data_dir)
    return path if path.is_absolute() else ROOT / path


def db_path(cfg: AppConfig) -> Path:
    return data_dir(cfg) / DB_FILENAME


def ledger_dir(cfg: AppConfig) -> Path:
    """Where the fake suppliers remember the orders they received (see mock.py)."""
    return data_dir(cfg) / "fake_supplier_ledgers"


# ---------------------------------------------------------------- AI settings

LLM_ENV_VARS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


class LLMSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)


def load_llm_settings(env_file: Path | str | None = None) -> LLMSettings:
    """Read LLM_BASE_URL, LLM_API_KEY, LLM_MODEL. Never modifies os.environ."""
    file_values = dotenv_values(Path(env_file or os.environ.get(ENV_FILE_ENV) or ROOT / ".env"))

    def pick(name: str) -> str | None:
        value = (os.environ.get(name) or file_values.get(name) or "").strip()
        return value or None

    base_url, api_key, model = (pick(name) for name in LLM_ENV_VARS)
    return LLMSettings(base_url=base_url, api_key=api_key, model=model)


@dataclass(frozen=True)
class AIStatus:
    enabled: bool
    reason: str


def ai_status(cfg: AppConfig, llm: LLMSettings) -> AIStatus:
    """Whether AI is used at all. Step 3 adds the "model not tested" check."""
    if cfg.faults.ai_down:
        return AIStatus(False, "AI is switched off by the ai_down test switch. Using text matching.")
    if not llm.configured:
        return AIStatus(False, "No AI key and model in .env. Using text matching.")
    return AIStatus(True, f"AI model: {llm.model}")
