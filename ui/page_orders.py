"""Page 3: Orders. Status and confirmation per supplier, purchase orders to download, retry pending."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlmodel import select

from app import models, po, workflow
from app.db import session_scope
from app.pricing import format_cents
from app.textparse import day_label
from ui import common

S = models.RequestStatus
OS = models.OrderStatus


def _load(ctx: common.Ctx, include_history: bool) -> list[dict]:
    out = []
    with session_scope(ctx.engine) as session:
        places = {loc.id: loc.name for loc in session.exec(select(models.Location)).all()}
        suppliers = {s.id: s.name for s in session.exec(select(models.Supplier)).all()}
        reqs = session.exec(select(models.Request).order_by(models.Request.id.desc())).all()
        for r in reqs:
            if r.status not in (S.APPROVED, S.PARTIALLY_ORDERED, S.ORDERED):
                continue
            if r.source == "seed_history" and not include_history:
                continue
            orders = session.exec(select(models.Order).where(models.Order.request_id == r.id)
                                  .order_by(models.Order.id)).all()
            out.append({
                "id": r.id, "public_id": r.public_id, "status": S(r.status), "location": places.get(r.location_id, "?"),
                "needed_by": r.needed_by, "history": r.source == "seed_history",
                "orders": [{"supplier": suppliers.get(o.supplier_id, "?"), "key": o.order_key, "status": o.status,
                            "confirmation": o.confirmation_id, "total": o.total_cents, "deliver_on": o.deliver_on,
                            "attempts": o.attempts, "error": o.last_error,
                            "po": po.load_purchase_order(session, o.id, ctx.cfg) if o.status == OS.CONFIRMED else None}
                           for o in orders],
            })
    return out


def render(ctx: common.Ctx) -> None:
    st.header("Orders")
    common.show_flash()
    include_history = st.toggle("Also show the fake past orders", key="orders_history")
    requests = _load(ctx, include_history)
    if not requests:
        st.info("No orders yet. Approve a request on the Review and approve page.")
        return
    for n, r in enumerate(requests):
        total = sum(o["total"] for o in r["orders"])
        title = (f"{r['public_id']} · {r['location']} · {r['status'].value.replace('_', ' ').lower()} · "
                 f"{format_cents(total)} · needed by {day_label(r['needed_by'])}")
        with st.expander(title, expanded=n == 0 and not r["history"]):
            if not r["orders"]:
                st.info("Approved, but no orders are saved yet.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Supplier": o["supplier"], "Status": o["status"].value.lower(),
                    "Confirmation": o["confirmation"] or "", "Total": format_cents(o["total"]),
                    "Delivers": day_label(o["deliver_on"]) if o["deliver_on"] else "", "Tries": o["attempts"],
                    "Problem": o["error"] or "", "Order ID": o["key"],
                } for o in r["orders"]]), hide_index=True, use_container_width=True)
            for o in r["orders"]:
                if o["po"] is None:
                    continue
                st.markdown(f"**Purchase order for {o['supplier']}**")
                email_col, csv_col = st.columns(2)
                email_col.download_button("Download email text", o["po"].email_text(), file_name=f"{o['key']}.txt",
                                          mime="text/plain", key=f"po_txt_{o['key']}")
                csv_col.download_button("Download CSV", o["po"].csv_text(), file_name=f"{o['key']}.csv",
                                        mime="text/csv", key=f"po_csv_{o['key']}")
            waiting = r["status"] in (S.APPROVED, S.PARTIALLY_ORDERED) and (
                not r["orders"] or any(o["status"] == OS.PENDING for o in r["orders"]))
            if waiting and st.button("Send the waiting orders now", key=f"send_{r['id']}"):
                with st.spinner("Contacting suppliers..."):
                    report = workflow.place_orders(ctx.engine, r["id"], ctx.suppliers(), ctx.cfg)
                common.flash("info", report.summary)
                st.rerun()
