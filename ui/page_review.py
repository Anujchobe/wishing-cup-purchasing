"""Page 2: Review and approve. Prices, the three plans, the lists, what needs a person, the checks."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlmodel import select

from app import checks, clock, models, review, workflow
from app.db import session_scope
from app.pricing import format_cents
from app.textparse import day_label
from app.workflow import AcceptPrice, ConfirmItem, MarkCantFill, MoveItem
from ui import common

S = models.RequestStatus
CHEAPEST = "background-color: #cdeed6; color: #0b3d1a; font-weight: 600"
PROBLEM = "background-color: #fbe0de; color: #5c1410"
MUTED = "color: #8a8a8a"


def _requests(engine) -> list[tuple[int, str]]:
    with session_scope(engine) as session:
        places = {loc.id: loc.name for loc in session.exec(select(models.Location)).all()}
        reqs = session.exec(select(models.Request).where(models.Request.source != "seed_history")
                            .order_by(models.Request.id.desc())).all()
        return [(r.id, f"{r.public_id} · {places.get(r.location_id, '?')} · {r.status.value.replace('_', ' ').lower()}"
                        f" · needed by {day_label(r.needed_by)}") for r in reqs]


def _apply(ctx: common.Ctx, request_id: int, changes: list) -> None:
    with session_scope(ctx.engine) as session:
        result = workflow.apply_changes(session, request_id, ctx.cfg.approval.approver, changes, ctx.cfg)
    if result.change_errors:
        for error in result.change_errors:
            common.flash("error", error)
    else:
        common.flash("success", "Saved. Totals recalculated and the checks run again.")
    st.rerun()


def _prices_and_plan(ctx: common.Ctx, request_id: int) -> None:
    with st.spinner("Asking every supplier for prices and making the plan..."):
        with session_scope(ctx.engine) as session:
            prices = workflow.get_prices(session, request_id, ctx.suppliers(), ctx.cfg)
        with session_scope(ctx.engine) as session:
            workflow.make_plan(session, request_id, ctx.cfg)
    for warning in prices.warnings + prices.set_aside:
        common.flash("warning", warning)
    st.rerun()


def _matrix_table(matrix: review.Matrix) -> None:
    if matrix.empty:
        st.info("No prices yet for this request.")
        return
    labels = [label for _, label in matrix.products]
    names = [name for _, name in matrix.suppliers]
    text = pd.DataFrame("", index=labels, columns=names)
    style = pd.DataFrame("", index=labels, columns=names)
    for i, (product, _) in enumerate(matrix.products):
        for j, (supplier, _) in enumerate(matrix.suppliers):
            cell = matrix.cells.get((product, supplier), review.Cell("no_match"))
            text.iat[i, j] = cell.label
            if matrix.cheapest.get(product) == supplier:
                style.iat[i, j] = CHEAPEST
            elif cell.status in ("out_of_stock", "dont_use"):
                style.iat[i, j] = PROBLEM
            elif cell.status != review.OK:
                style.iat[i, j] = MUTED
    st.dataframe(text.style.apply(lambda _: style, axis=None), use_container_width=True)
    st.caption("Each price is the real cost for the amount needed, after rounding up to whole cases. "
               "Green: cheapest. Red: out of stock, or a price set aside for a person to check.")


def _plans(plans) -> None:
    if plans is None:
        st.info("No plans yet: there are no prices.")
        return
    columns = st.columns(3)
    for column, (letter, plan, note) in zip(columns, [("a", plans.simple, "Ignores minimum orders."),
                                                      ("b", plans.one_supplier, plans.one_supplier_note),
                                                      ("c", plans.best, "Lowest total with fees and minimums.")]):
        with column:
            if plan is None:
                st.markdown(f"**{letter}) One supplier**")
                st.caption(note or "Not possible for this request.")
                continue
            st.markdown(f"**{letter}) {plan.title}**")
            st.metric("Total", format_cents(plan.total_cents))
            st.caption(note)
            for sl in plan.lists:
                flag = "" if sl.meets_minimum else f" (short {format_cents(sl.short_by_cents)} of the minimum)"
                st.write(f"{sl.name}: {format_cents(sl.total_cents)}, {len(sl.lines)} items{flag}")
            for w in plan.warnings:
                st.warning(w)
    if plans.savings_vs_one_supplier_cents is not None:
        st.success(f"Plan c saves {format_cents(plans.savings_vs_one_supplier_cents)} compared with plan b.")
    if plans.differences:
        with st.expander(f"Why plan c differs from plan a ({len(plans.differences)} items)"):
            for d in plans.differences:
                st.write(f"**{d.name}**: {d.reason}")


def _lists(lists, cant) -> None:
    if not lists:
        st.info("No supplier lists yet.")
    for sl in lists:
        minimum = "meets" if sl.meets_minimum else f"is {format_cents(sl.min_order_cents - sl.subtotal_cents)} short of"
        when = day_label(sl.deliver_on) if sl.deliver_on else "no delivery day"
        st.markdown(f"**{sl.name}**: {format_cents(sl.total_cents)} ({format_cents(sl.subtotal_cents)} items + "
                    f"{format_cents(sl.delivery_fee_cents)} delivery); {minimum} the "
                    f"{format_cents(sl.min_order_cents)} minimum; earliest delivery {when}")
        st.dataframe(pd.DataFrame([{
            "Item": x.product_name, "SKU": x.sku, "Supplier's name": x.description, "Need": x.need,
            "Case": x.case_size, "Cases": x.cases, "Buying": x.bought, "Extra": x.extra,
            "Case price": format_cents(x.case_price_cents), "Line total": format_cents(x.line_total_cents),
        } for x in sl.lines]), hide_index=True, use_container_width=True)
    if cant:
        st.markdown("**Can't fill**")
        for name, why in cant:
            st.write(f"- {name}: {why}")


def _attention(ctx: common.Ctx, request_id: int, items: list[review.Attention]) -> None:
    if not items:
        st.success("Nothing needs a person.")
        return
    confirmable = [i for i in items if i.can_confirm]
    if confirmable and st.button(f"Confirm all {len(confirmable)} (I checked them)", key="confirm_all"):
        _apply(ctx, request_id, [ConfirmItem(i.product_code) for i in confirmable])
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.product_name}**")
            for reason in item.reasons:
                st.write(f"- {reason}")
            confirm, move, cant = st.columns([1, 3, 1])
            if item.can_confirm and confirm.button("Confirm", key=f"confirm_{item.product_code}"):
                _apply(ctx, request_id, [ConfirmItem(item.product_code)])
            if item.move_to:
                options = dict(item.move_to)
                target = move.selectbox("Move to", list(options), format_func=options.get,
                                        key=f"move_to_{item.product_code}")
                if move.button("Move", key=f"move_{item.product_code}"):
                    _apply(ctx, request_id, [MoveItem(item.product_code, target)])
            for supplier, label in item.set_aside:
                if move.button(f"Accept the set-aside price and move ({label})",
                               key=f"accept_{item.product_code}_{supplier}"):
                    _apply(ctx, request_id, [AcceptPrice(item.product_code, supplier),
                                             MoveItem(item.product_code, supplier)])
            if item.current_supplier and cant.button("Can't fill", key=f"cant_{item.product_code}"):
                _apply(ctx, request_id, [MarkCantFill(item.product_code)])


def _checklist(results: list[checks.CheckResult]) -> None:
    for r in results:
        st.markdown(f"{'✅' if r.passed else '❌'} {r.title}")
        for reason in r.reasons:
            st.caption(f"    {reason}")


def _pending(ctx: common.Ctx, request_id: int) -> None:
    now = clock.now()
    with session_scope(ctx.engine) as session:
        note = session.get(models.Request, request_id).status_note
        matrix = review.cost_matrix(session, request_id, ctx.cfg)
        plans = workflow.plan_options(session, request_id, ctx.cfg, now=now)
        lists, cant = review.shopping_lists(session, request_id, ctx.cfg, now=now)
        view = checks.load_view(session, request_id, ctx.cfg, now)
        results = checks.run_checks(view, "approval")
        items = review.attention(view, matrix)
        total = session.get(models.Request, request_id).plan_total_cents
    if note:
        st.warning(note)
    if st.button("Get fresh prices (makes a new plan)", key="refresh_prices"):
        _prices_and_plan(ctx, request_id)

    st.subheader("Prices: items x suppliers")
    _matrix_table(matrix)
    st.subheader("Plans")
    _plans(plans)
    st.subheader(f"Shopping lists (current total {format_cents(total or 0)})")
    _lists(lists, cant)
    st.subheader("Needs a person")
    _attention(ctx, request_id, items)
    st.subheader("Safety checks")
    _checklist(results)

    st.subheader("Decision")
    passed = not checks.failures(results)
    approve_col, reject_col = st.columns(2)
    with approve_col:
        if st.button("Approve and place orders", key="approve", type="primary", disabled=not passed):
            with session_scope(ctx.engine) as session:
                approval = workflow.approve(session, request_id, ctx.cfg.approval.approver, (), ctx.cfg)
            if not approval.approved:
                common.flash("error", approval.message)
                st.rerun()
            with st.spinner("Placing the orders..."):
                report = workflow.place_orders(ctx.engine, request_id, ctx.suppliers(), ctx.cfg)
            common.flash("info", report.summary)
            common.goto("Orders")
        if not passed:
            st.caption("Approve is available once every safety check passes.")
    with reject_col:
        reason = st.text_input("Reason, to reject", key=f"reject_reason_{request_id}")
        if st.button("Reject", key="reject", disabled=not reason.strip()):
            with session_scope(ctx.engine) as session:
                workflow.reject(session, request_id, ctx.cfg.approval.approver, reason.strip())
            common.flash("info", "Rejected.")
            st.rerun()


def render(ctx: common.Ctx) -> None:
    st.header("Review and approve")
    common.show_flash()
    options = _requests(ctx.engine)
    if not options:
        st.info("No requests yet. Start one on the New request page.")
        return
    ids = [i for i, _ in options]
    labels = dict(options)
    current = st.session_state.get("current_request")
    request_id = st.selectbox("Request", ids, index=ids.index(current) if current in ids else 0,
                              format_func=labels.get)
    st.session_state["current_request"] = request_id

    with session_scope(ctx.engine) as session:
        req = session.get(models.Request, request_id)
        status, text, note = S(req.status), req.original_text, req.status_note
    with st.expander("The original message"):
        st.text(text or "(no message)")

    if status in (S.DRAFT, S.QUOTED):
        st.info("This request has no plan yet." if status == S.DRAFT else "Prices are in; the plan isn't made yet.")
        if st.button("Get prices and make the plan", key="make_plan", type="primary"):
            _prices_and_plan(ctx, request_id)
    elif status == S.PENDING_APPROVAL:
        _pending(ctx, request_id)
    elif status == S.REJECTED:
        st.info(f"Rejected: {note or 'no reason given'}")
    else:
        st.success(f"Approved; orders are {status.value.replace('_', ' ').lower()}.")
        with session_scope(ctx.engine) as session:
            lists, cant = review.shopping_lists(session, request_id, ctx.cfg)
        _lists(lists, cant)
        if st.button("See the orders", key="see_orders"):
            common.goto("Orders")
