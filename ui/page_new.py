"""Page 1: New request. Location, needed-by date and a message; read it into an editable table; get prices."""
from __future__ import annotations

import math
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import streamlit as st
from sqlmodel import select

from app import clock, models
from app.db import session_scope
from app.examples import example_text
from app.intake import IntakeResult, ProductRef, read_request
from app.textparse import day_label
from app.units import display_unit
from app.workflow import LineInput, WorkflowError, get_prices, make_plan, save_request
from ui import common

COLUMNS = ["Product", "Amount", "As written", "Note", "Needs a person", "Why"]


def _load_example(kind: str, today: date) -> None:
    st.session_state["request_text"] = example_text(kind, today)


def _label(p: ProductRef) -> str:
    return f"{p.name} ({display_unit(p.base_unit)})"


def _table(result: IntakeResult, products: list[ProductRef]) -> pd.DataFrame:
    by_code = {p.code: p for p in products}
    rows = [{"Product": _label(by_code[line.product_code]), "Amount": float(line.qty),
             "As written": " ".join(filter(None, [line.quantity_text, line.unit_text, line.product_text])),
             "Note": line.note, "Needs a person": line.needs_check, "Why": "; ".join(line.check_reasons)}
            for line in result.lines if line.product_code in by_code]
    if rows:
        return pd.DataFrame(rows, columns=COLUMNS)
    return pd.DataFrame({"Product": pd.Series(dtype="object"), "Amount": pd.Series(dtype="float"),
                         "As written": pd.Series(dtype="object"), "Note": pd.Series(dtype="object"),
                         "Needs a person": pd.Series(dtype="bool"), "Why": pd.Series(dtype="object")})


def _lines_from_table(edited: pd.DataFrame, products: list[ProductRef], result: IntakeResult
                      ) -> tuple[list[LineInput], list[str]]:
    """The table as the person left it. A line they changed no longer carries the reader's flags."""
    by_label = {_label(p): p for p in products}
    read = {line.product_code: line for line in result.lines}
    lines, problems, seen = [], [], set()
    for _, row in edited.iterrows():
        label = row.get("Product")
        if not isinstance(label, str) or not label:
            continue
        product = by_label.get(label)
        amount = row.get("Amount")
        if product is None:
            problems.append(f"'{label}' isn't on our product list.")
            continue
        if amount is None or (isinstance(amount, float) and math.isnan(amount)) or float(amount) <= 0:
            problems.append(f"{product.name}: enter an amount above zero.")
            continue
        if product.code in seen:
            problems.append(f"{product.name} is on two lines. Put it on one line.")
            continue
        seen.add(product.code)
        qty = Decimal(repr(float(amount)))  # the table hands back floats; repr keeps exactly what was typed
        original = read.get(product.code)
        unchanged = original is not None and original.qty == qty
        note = row.get("Note") if isinstance(row.get("Note"), str) else ""
        lines.append(LineInput(product.code, qty, note,
                               raw_text=row.get("As written") if isinstance(row.get("As written"), str) else "",
                               needs_check=bool(original.needs_check) if unchanged else False,
                               check_reason="; ".join(original.check_reasons) or None if unchanged else None))
    if not lines and not problems:
        problems.append("Add at least one line.")
    return lines, problems


def render(ctx: common.Ctx) -> None:
    st.header("New request")
    with session_scope(ctx.engine) as session:
        locations = [(loc.code, loc.name, loc.kind.value)
                     for loc in session.exec(select(models.Location).where(models.Location.active)).all()]
        products = [ProductRef(p.code, p.name, p.base_unit.value)
                    for p in session.exec(select(models.Product).where(models.Product.active)).all()]
    if not locations:
        st.info("No locations are set up yet. Add them in config.yaml and restart.")
        return
    if not products:
        st.info("There are no products yet. Run `python run.py --reset` to load the fake test data.")
        return

    today = clock.today()
    left, right = st.columns(2)
    name = left.selectbox("Location", [n for _, n, _ in locations], key="new_location")
    code, _, kind = next(x for x in locations if x[1] == name)
    fallback = right.date_input("Needed by, if the message doesn't say", value=today + timedelta(days=7),
                                min_value=today, key="new_needed_by")
    st.button("Load example", key="load_example", on_click=_load_example, args=(kind, today))
    text = st.text_area("What do you need?", key="request_text", height=140,
                        placeholder="For example: 50 lb all-purpose flour and 2 dozen eggs by Friday")

    if st.button("Read request", key="read_request", type="primary", disabled=not (text or "").strip()):
        with st.spinner("Reading the request..."):
            result = read_request(text, products=products, client=ctx.client(), today=today,
                                  model_tested=ctx.status.tested, cfg=ctx.cfg.intake)
        st.session_state["intake"] = {"result": result, "text": text, "location": code}
        st.session_state["intake_round"] = st.session_state.get("intake_round", 0) + 1

    intake = st.session_state.get("intake")
    if not intake:
        st.caption("Type what you need, or load an example, then press Read request.")
        return
    result: IntakeResult = intake["result"]
    if intake["text"] != text or intake["location"] != code:
        st.info("The message or location changed since it was read. Press Read request again to re-read it.")

    message, lines_col = st.columns([2, 3])
    with message:
        st.subheader("The message")
        st.text(intake["text"])
        st.caption("Read by " + ("the AI" if result.source == "ai" else "text matching (no AI)"))
        if result.ai_error:
            st.warning(f"The AI failed, so text matching was used and every line needs a person: {result.ai_error}")
    with lines_col:
        st.subheader("Lines")
        st.caption("Amounts are in each product's unit (lb, fl oz or each). Add, fix or delete lines here.")
        edited = st.data_editor(
            _table(result, products), key=f"lines_{st.session_state.get('intake_round', 0)}", num_rows="dynamic",
            hide_index=True, use_container_width=True,
            column_config={
                "Product": st.column_config.SelectboxColumn("Product", options=[_label(p) for p in products],
                                                            required=True),
                "Amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=0.01, format="%.2f"),
                "As written": st.column_config.TextColumn("As written", disabled=True),
                "Note": st.column_config.TextColumn("Note"),
                "Needs a person": st.column_config.CheckboxColumn("Needs a person", disabled=True),
                "Why": st.column_config.TextColumn("Why", disabled=True),
            })
    for note in result.unclear:
        st.warning(note)

    needed_by = result.needed_by or fallback
    st.write(f"**Needed by {day_label(needed_by)}** "
             + ("(from the message)" if result.needed_by else "(from the date above)"))

    if st.button("Get prices", key="get_prices", type="primary"):
        lines, problems = _lines_from_table(edited, products, result)
        if problems:
            for problem in problems:
                st.error(problem)
            return
        try:
            client = ctx.client()
            with st.spinner("Matching products and asking every supplier for prices..."):
                with session_scope(ctx.engine) as session:
                    req, _ = save_request(session, location_code=intake["location"], text=intake["text"],
                                          needed_by=needed_by, lines=lines, created_by="manager", client=client,
                                          cfg=ctx.cfg, model_tested=ctx.status.tested, read_by=result.source,
                                          ai_error=result.ai_error, unclear=result.unclear)
                    request_id = req.id
                with session_scope(ctx.engine) as session:
                    prices = get_prices(session, request_id, ctx.suppliers(), ctx.cfg)
                with session_scope(ctx.engine) as session:
                    make_plan(session, request_id, ctx.cfg)
        except WorkflowError as exc:
            st.error(str(exc))
            return
        for warning in prices.warnings + prices.set_aside:
            common.flash("warning", warning)
        st.session_state["current_request"] = request_id
        st.session_state.pop("intake", None)
        common.goto("Review and approve")
