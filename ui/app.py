"""Wishing Cup purchasing: the Streamlit app. Start it with `python run.py`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Streamlit puts ui/ first on the path, where ui/app.py would hide the app/ package. Put the root first.
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Wishing Cup purchasing", page_icon="☕", layout="wide")

from app.config import ConfigError, load_config, load_llm_settings  # noqa: E402
from app.db import SchemaMismatch, get_engine, init_db, is_seeded, session_scope  # noqa: E402
from app.llm.model_check import model_status  # noqa: E402
from scripts.seed import seed_if_empty  # noqa: E402
from ui import common, page_new, page_orders, page_reports, page_review  # noqa: E402

try:
    base = load_config()
    engine = get_engine(base)
    init_db(engine)
except (ConfigError, SchemaMismatch) as exc:
    st.error(str(exc))
    st.stop()

with session_scope(engine) as session:
    seeded = is_seeded(session)
if not seeded:
    st.title("Wishing Cup purchasing")
    st.info("There is no data yet. Load the fake test data to start.")
    if st.button("Load the fake test data", key="seed_now", type="primary"):
        seed_if_empty(engine, base)
        st.rerun()
    st.stop()

common.init_fault_state(base)
cfg = common.effective_config(base)
llm = load_llm_settings()
status = model_status(cfg, llm)
ctx = common.Ctx(cfg, engine, llm, status)

with st.sidebar:
    st.title("Wishing Cup purchasing")
    if "_goto" in st.session_state:
        st.session_state["page"] = st.session_state.pop("_goto")
    st.radio("Page", common.PAGES, key="page", label_visibility="collapsed")

    st.subheader("AI")
    st.caption(f"Model in .env: {llm.model or 'none'}")
    if not status.ai_enabled:
        st.info(status.label)
    elif status.tested:
        st.success(status.label)
    else:
        st.warning(status.label)

    st.warning(cfg.app.fake_data_label, icon="⚠️")

    with st.expander("Fake problems (test switches)", expanded=cfg.faults.any_on):
        st.caption("For this browser session only. config.yaml sets where they start.")
        st.toggle("AI is down", key="fault_ai_down")
        for sc in base.suppliers:
            st.markdown(f"**{sc.name}**")
            for column, kind in zip(st.columns(3), common.FAULT_KINDS):
                column.toggle(kind, key=common.fault_key(sc.code, kind))

pages = {"New request": page_new.render, "Review and approve": page_review.render,
         "Orders": page_orders.render, "Reports": page_reports.render}
common.safe(pages[st.session_state.get("page", common.PAGES[0])])(ctx)
