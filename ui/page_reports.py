"""Page 4: Reports. Savings replay, price watch, price-match emails, spend, prices over time."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlmodel import select

from app import insights, models
from app.db import session_scope
from app.matching import match_products
from app.pricing import format_cents
from app.units import display_unit
from ui import common


def _coverage(ctx: common.Ctx) -> None:
    with session_scope(ctx.engine) as session:
        cov = insights.load_coverage(session, ctx.cfg)
    if cov.complete:
        return
    st.info(f"{cov.matched} of {cov.possible} product and supplier pairs have a confirmed match"
            + (f", and {cov.waiting} more are waiting for a person" if cov.waiting else "")
            + ". Reports only use confirmed matches. Products are matched when a request needs them.")
    if st.button("Match every product now", key="match_all"):
        with st.spinner("Matching every product at every supplier..."):
            with session_scope(ctx.engine) as session:
                codes = [p.code for p in session.exec(select(models.Product).where(models.Product.active)).all()]
            with session_scope(ctx.engine) as session:
                report = match_products(session, codes, ctx.client(), ctx.cfg, model_tested=ctx.status.tested)
        flagged = sum(d.needs_check for d in report.decisions if d.saved)
        common.flash("success", f"Saved {sum(d.saved for d in report.decisions)} matches "
                                f"({flagged} marked for a person to check).")
        st.rerun()


def _savings(ctx: common.Ctx) -> None:
    with session_scope(ctx.engine) as session:
        replay = insights.load_savings_replay(session, ctx.cfg)
    weeks = ctx.cfg.purchasing.replay_weeks
    if not replay.rows:
        st.info(f"No orders in the last {weeks} weeks to replay.")
        return
    st.caption(f"Each order from the last {weeks} weeks, planned again with the prices of its own week, "
               "the same needed-by date and today's supplier terms.")
    a, b, c = st.columns(3)
    a.metric("What we paid", format_cents(replay.actual_cents))
    b.metric("With this app", format_cents(replay.replay_cents))
    c.metric("Would have saved", format_cents(replay.saved_cents))
    st.dataframe(pd.DataFrame([{
        "Week of": f"{r.week_of:%b} {r.week_of.day}", "Location": r.location, "Request": r.request_public_id,
        "Paid": format_cents(r.actual_cents),
        "With this app": format_cents(r.replay_cents) if r.replay_cents is not None else "-",
        "Saved": format_cents(r.saved_cents) if r.saved_cents is not None else "-", "Note": r.note,
    } for r in replay.rows]), hide_index=True, use_container_width=True)


def _watch(ctx: common.Ctx) -> None:
    with session_scope(ctx.engine) as session:
        rows = insights.load_price_watch(session, ctx.cfg)
    p = ctx.cfg.purchasing
    st.caption(f"Latest price against {p.price_watch_weeks} weeks earlier, per supplier. Up more than "
               f"{insights.pct(p.market_change_threshold)} at every supplier is a market change; at just one, "
               "that supplier raised it.")
    if not rows:
        st.info("No price moved that much at two or more suppliers we track.")
        return
    st.dataframe(pd.DataFrame([{"Item": r.product_name, "What happened": r.summary} for r in rows]),
                 hide_index=True, use_container_width=True)


def _price_match(ctx: common.Ctx) -> None:
    with session_scope(ctx.engine) as session:
        items = insights.load_match_candidates(session, ctx.cfg)
    main = ctx.cfg.main_supplier
    threshold = insights.pct(ctx.cfg.purchasing.match_threshold)
    if not items:
        st.info(f"No other supplier is more than {threshold} cheaper than {main.name} on any matched item.")
        return
    st.caption(f"Items where another supplier is more than {threshold} cheaper than {main.name}, per unit.")
    st.dataframe(pd.DataFrame([{"Item": i.product_name, f"{main.name}": i.main_label, "Best other": i.other_label,
                                "From": i.other.supplier, "Cheaper by": insights.pct(i.cheaper_by)} for i in items]),
                 hide_index=True, use_container_width=True)
    if st.button("Write the price-match email", key="write_email"):
        with st.spinner("Writing..."):
            subject, body, by = insights.write_price_match_email(ctx.client(), main.name, main.sales_rep.name, items)
        st.session_state["price_match_email"] = (subject, body, by)
    email = st.session_state.get("price_match_email")
    if email:
        subject, body, by = email
        st.caption(f"Wording by {'the AI (numbers checked against the facts)' if by == 'ai' else 'the template'}. "
                   f"Send it yourself to {main.sales_rep.name} <{main.sales_rep.email}>.")
        st.code(f"Subject: {subject}", language=None)
        st.code(body, language=None)


def _spend(ctx: common.Ctx) -> None:
    with session_scope(ctx.engine) as session:
        by_supplier, by_location = insights.load_spend(session)
    if not by_supplier:
        st.info("No confirmed orders yet.")
        return
    left, right = st.columns(2)
    for column, title, rows in ((left, "By supplier", by_supplier), (right, "By location", by_location)):
        with column:
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame([{"Name": r.name, "Orders": r.orders, "Spend": format_cents(r.total_cents)}
                                       for r in rows]), hide_index=True, use_container_width=True)
            # Charts need numbers; dollars as floats are for drawing only, never stored.
            st.bar_chart(pd.DataFrame({"Spend ($)": [r.total_cents / 100 for r in rows]},
                                      index=[r.name for r in rows]))


def _history(ctx: common.Ctx) -> None:
    with session_scope(ctx.engine) as session:
        series, names, unit = insights.load_price_series(session)
    if not series:
        st.info("No price history yet.")
        return
    codes = sorted(series, key=lambda c: names.get(c, c))
    code = st.selectbox("Ingredient", codes, format_func=lambda c: names.get(c, c), key="history_product")
    rows = [{"Date": at.date(), "Supplier": supplier, "Price": float(cents) / 100}
            for supplier, points in series[code].items() for at, cents in points]
    if not rows:
        st.info("No prices for this ingredient yet.")
        return
    frame = pd.DataFrame(rows).groupby(["Date", "Supplier"])["Price"].last().unstack("Supplier").sort_index()
    frame.index = pd.to_datetime(frame.index)
    st.caption(f"Price per {display_unit(unit.get(code, 'each'))}, in dollars, by supplier.")
    st.line_chart(frame)


def render(ctx: common.Ctx) -> None:
    st.header("Reports")
    common.show_flash()
    _coverage(ctx)
    tabs = st.tabs(["Savings replay", "Price watch", "Price match", "Spend", "Prices over time"])
    for tab, part in zip(tabs, (_savings, _watch, _price_match, _spend, _history)):
        with tab:
            common.safe(part)(ctx)
