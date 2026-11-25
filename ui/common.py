"""Shared pieces for the screens: this session's settings, the AI client, safety nets, page switching."""
from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

import streamlit as st

from app.config import AppConfig, FaultsConfig, LLMSettings, SupplierFaults
from app.llm.client import AnyClient, db_logger, make_client
from app.llm.model_check import ModelStatus
from app.suppliers.mock import MockSupplier, build_suppliers
from app.workflow import WorkflowError

try:  # Streamlit's own control-flow signals (st.rerun, st.stop) must never be swallowed
    from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException
except ImportError:  # older layout
    try:
        from streamlit.runtime.scriptrunner.exceptions import RerunException, StopException
    except ImportError:
        RerunException = StopException = ()

PAGES = ("New request", "Review and approve", "Orders", "Reports")
FAULT_KINDS = ("down", "slow", "flaky")


def fault_key(code: str, kind: str) -> str:
    return f"fault_{code}_{kind}"


def init_fault_state(base: AppConfig) -> None:
    """The switches start from config.yaml and then belong to this browser session."""
    st.session_state.setdefault("fault_ai_down", base.faults.ai_down)
    for sc in base.suppliers:
        current = base.faults.for_supplier(sc.code)
        for kind in FAULT_KINDS:
            st.session_state.setdefault(fault_key(sc.code, kind), getattr(current, kind))


def effective_config(base: AppConfig) -> AppConfig:
    """config.yaml with this session's fake-problem switches."""
    faults = FaultsConfig(
        ai_down=bool(st.session_state.get("fault_ai_down")),
        slow_seconds=base.faults.slow_seconds,
        flaky_failure_rate=base.faults.flaky_failure_rate,
        suppliers={sc.code: SupplierFaults(**{k: bool(st.session_state.get(fault_key(sc.code, k)))
                                              for k in FAULT_KINDS}) for sc in base.suppliers},
    )
    return base.model_copy(update={"faults": faults})


@dataclass
class Ctx:
    cfg: AppConfig
    engine: object
    llm: LLMSettings
    status: ModelStatus

    def client(self) -> AnyClient:
        return make_client(self.cfg, self.llm, log_call=db_logger(self.engine))

    def suppliers(self) -> dict[str, MockSupplier]:
        return build_suppliers(self.cfg)


def safe(render: Callable[[Ctx], None]) -> Callable[[Ctx], None]:
    """A page that hits an unexpected error shows it instead of taking the app down."""
    @wraps(render)
    def wrapper(ctx: Ctx) -> None:
        try:
            render(ctx)
        except (RerunException, StopException):
            raise
        except WorkflowError as exc:
            st.error(str(exc))
        except Exception as exc:  # st.rerun/st.stop are BaseExceptions, so they pass through
            st.error(f"This page hit a problem and stopped: {exc}. The other pages still work.")
            with st.expander("Details for a developer"):
                st.code(traceback.format_exc())
    return wrapper


def goto(page: str) -> None:
    st.session_state["_goto"] = page
    st.rerun()


def flash(kind: str, text: str) -> None:
    """A message that survives the next rerun (shown by show_flash)."""
    st.session_state.setdefault("_flash", []).append((kind, text))


def show_flash() -> None:
    for kind, text in st.session_state.pop("_flash", []):
        getattr(st, kind)(text)
