"""A real connector that needs no API: the supplier's exported order guide in, order files out.

Many suppliers let you export an "order guide" (your usual items with today's prices) from their
ordering website. Save it as CSV with these columns:

    sku, description, pack, case_price, in_stock
    1630766, "BUTTER, UNSALTED", 36/1 LB, 150.95, yes

and point a supplier at it in config.yaml:

    connector: app.suppliers.order_guide:OrderGuideSupplier
    connector_settings: {order_guide: order_guides/acme.csv, outbox: outbox/acme}

Prices are dated by the file's own modified time, so an old export fails the "prices are not older
than 24 hours" check, exactly like an old quote. Placing an order writes <order id>.csv to the outbox
for a person to email or upload (the purchase-order email text is made too, as for every order).
The same order id always gives the same reference, so an order can't be written twice.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app import clock
from app.config import ROOT, SupplierConfig
from app.suppliers.base import (
    CatalogItem,
    Quote,
    Supplier,
    SupplierOrder,
    SupplierRejected,
    SupplierUnavailable,
)

COLUMNS = ("sku", "description", "pack", "case_price", "in_stock")
YES = {"yes", "y", "true", "1", "in stock"}


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def dollars_to_cents(text: str) -> int:
    """'150.95' or '$1,234.5' -> whole cents, exactly. Anything else raises ValueError."""
    cleaned = text.strip().replace("$", "").replace(",", "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        raise ValueError(f"'{text}' is not a price") from None
    cents = value * 100
    if cents != cents.to_integral_value():
        raise ValueError(f"'{text}' has fractions of a cent")
    return int(cents)


def _dollars(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"  # from whole cents, never through a float


class OrderGuideSupplier(Supplier):
    def __init__(self, cfg: SupplierConfig):
        super().__init__(cfg.code, cfg.name)
        try:
            self.order_guide = _path(cfg.connector_settings["order_guide"])
            self.outbox = _path(cfg.connector_settings["outbox"])
        except KeyError as exc:
            raise ValueError(f"{cfg.name}: connector_settings needs '{exc.args[0]}'") from None

    def _read(self) -> tuple[list[CatalogItem], datetime]:
        try:
            exported = datetime.fromtimestamp(self.order_guide.stat().st_mtime, tz=clock.BOSTON)
            with self.order_guide.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError as exc:
            raise SupplierUnavailable(f"{self.name}: can't read the order guide {self.order_guide} ({exc})") from exc
        items = []
        for n, row in enumerate(rows, start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            missing = [c for c in COLUMNS if c not in row]
            if missing:
                raise SupplierUnavailable(f"{self.name}: the order guide has no column {', '.join(missing)}")
            try:
                items.append(CatalogItem(sku=row["sku"], description=row["description"], raw_pack_text=row["pack"],
                                         case_price_cents=dollars_to_cents(row["case_price"]),
                                         in_stock=row["in_stock"].lower() in YES))
            except ValueError as exc:
                raise SupplierUnavailable(f"{self.name}: order guide line {n}: {exc}") from exc
        return items, exported

    def get_catalog(self) -> list[CatalogItem]:
        return self._read()[0]

    def get_quotes(self, skus: Sequence[str]) -> list[Quote]:
        items, exported = self._read()
        by_sku = {i.sku: i for i in items}
        seen, out = set(), []
        for sku in skus:
            if sku in by_sku and sku not in seen:
                seen.add(sku)
                out.append(Quote(sku=sku, case_price_cents=by_sku[sku].case_price_cents,
                                 in_stock=by_sku[sku].in_stock, quoted_at=exported))
        return out

    def place_order(self, order: SupplierOrder, order_id: str) -> str:
        if order.supplier_code != self.code:
            raise SupplierRejected(f"This order is for '{order.supplier_code}', not {self.name}.")
        body = json.dumps(order.model_dump(mode="json"), sort_keys=True)
        reference = f"SENT-{hashlib.sha256(f'{self.code}|{order_id}'.encode()).hexdigest()[:10].upper()}"
        target = self.outbox / f"{order_id}.csv"
        stamp = self.outbox / f"{order_id}.json"
        if stamp.exists():
            if stamp.read_text(encoding="utf-8") != body:
                raise SupplierRejected(f"Order id {order_id} was already used for a different order.")
            return reference
        self.outbox.mkdir(parents=True, exist_ok=True)
        lines = ["sku,cases,expected_case_price,deliver_on"] + [
            f"{line.sku},{line.cases},{_dollars(line.expected_case_price_cents)},{order.deliver_on.isoformat()}"
            for line in order.lines]
        tmp = target.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        stamp.write_text(body, encoding="utf-8")  # written last: it marks the order as done
        return reference
