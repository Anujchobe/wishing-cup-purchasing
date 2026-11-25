"""Fake supplier connectors. FAKE TEST DATA: no real supplier is contacted.

Each MockSupplier serves its catalog from fake_world.py and behaves like a
remote system you don't control:

* Fault switches (config.yaml -> faults): "down" never answers, "slow" answers
  late (and times out if slow_seconds > timeout_seconds), "flaky" fails some
  calls. Flaky failures come from a seeded random generator, so a test run
  fails at exactly the same calls every time.
* A flaky place_order can fail AFTER the supplier has recorded the order. That
  is the worst real case (the supplier booked it, the reply was lost) and the
  reason order ids must be idempotent.
* The supplier keeps a ledger of order ids it has booked. With a ledger_path it
  survives app restarts, like a real supplier would, so "finish PENDING orders
  at startup" can be tested honestly. Same id + same order -> same confirmation;
  same id + different order -> rejected.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from app import clock
from app.config import AppConfig, SupplierConfig, SupplierFaults
from app.config import ledger_dir as default_ledger_dir
from app.suppliers import fake_world
from app.suppliers.base import (
    CatalogItem,
    Quote,
    Supplier,
    SupplierOrder,
    SupplierRejected,
    SupplierTimeout,
    SupplierUnavailable,
)

MAX_ORDER_ID_LENGTH = 200


class MockSupplier(Supplier):
    def __init__(
        self,
        cfg: SupplierConfig,
        *,
        seed: int,
        faults: SupplierFaults | None = None,
        slow_seconds: float = 1.0,
        flaky_failure_rate: float = 0.3,
        timeout_seconds: float = 2.0,
        ledger_path: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = clock.now,
    ) -> None:
        super().__init__(cfg.code, cfg.name)
        self.cfg = cfg
        self.seed = seed
        self.faults = faults if faults is not None else SupplierFaults()
        self.slow_seconds = slow_seconds
        self.flaky_failure_rate = flaky_failure_rate
        self.timeout_seconds = timeout_seconds
        self._ledger_path = ledger_path
        self._sleep = sleep
        self._now = now
        self._offers = {o.sku: o for o in fake_world.offers_for(cfg.code)}
        self._rng = random.Random(fake_world.stable_seed(seed, "flaky", cfg.code))
        self._ledger: dict[str, dict[str, Any]] = self._load_ledger()

    # ------------------------------------------------------------ the contract

    def get_catalog(self) -> list[CatalogItem]:
        self._network("catalog")
        return [self._catalog_item(o) for o in sorted(self._offers.values(), key=lambda o: o.sku)]

    def get_quotes(self, skus: Sequence[str]) -> list[Quote]:
        self._network("quotes")
        quoted_at = self._now()
        quotes, seen = [], set()
        for sku in skus:
            if sku in seen or sku not in self._offers:
                continue
            seen.add(sku)
            item = self._catalog_item(self._offers[sku])
            quotes.append(Quote(sku=sku, case_price_cents=item.case_price_cents,
                                in_stock=item.in_stock, quoted_at=quoted_at))
        return quotes

    def place_order(self, order: SupplierOrder, order_id: str) -> str:
        if not order_id or len(order_id) > MAX_ORDER_ID_LENGTH:
            raise SupplierRejected(f"Order id must be 1-{MAX_ORDER_ID_LENGTH} characters.")
        if order.supplier_code != self.code:
            raise SupplierRejected(f"This order is addressed to '{order.supplier_code}', not {self.name}.")
        self._network("place order")

        fingerprint = _fingerprint(order)
        booked = self._ledger.get(order_id)
        if booked is not None:
            if booked["fingerprint"] != fingerprint:
                raise SupplierRejected(f"Order id {order_id} was already used for a different order.")
            confirmation = booked["confirmation_id"]
        else:
            self._check_lines(order)
            confirmation = fake_world.confirmation_for(self.code, order_id)
            self._ledger[order_id] = {
                "confirmation_id": confirmation,
                "fingerprint": fingerprint,
                "booked_at": self._now().isoformat(),
            }
            self._save_ledger()

        if self._flaky_hit():  # booked, but the reply never reaches us
            raise SupplierUnavailable(f"{self.name}: connection dropped before the reply arrived (flaky switch).")
        return confirmation

    # ------------------------------------------------------------ test helpers

    @property
    def placements(self) -> int:
        """How many distinct orders this supplier has booked."""
        return len(self._ledger)

    def has_booked(self, order_id: str) -> bool:
        return order_id in self._ledger

    # ------------------------------------------------------------ internals

    def _catalog_item(self, o: fake_world.FakeOffer) -> CatalogItem:
        return CatalogItem(
            sku=o.sku,
            description=o.description,
            raw_pack_text=o.raw_pack_text,
            case_price_cents=fake_world.case_price_cents(o, 0, self.seed),
            in_stock=fake_world.in_stock(o, 0),
        )

    def _check_lines(self, order: SupplierOrder) -> None:
        for line in order.lines:
            o = self._offers.get(line.sku)
            if o is None:
                raise SupplierRejected(f"{self.name} has no product with SKU {line.sku}.")
            if not fake_world.in_stock(o, 0):
                raise SupplierRejected(f"{self.name}: SKU {line.sku} ({o.description}) is out of stock.")

    def _network(self, action: str) -> None:
        if self.faults.down:
            raise SupplierUnavailable(f"{self.name} did not answer ({action}). The 'down' test switch is on.")
        if self.faults.slow:
            if self.slow_seconds > self.timeout_seconds:
                self._sleep(self.timeout_seconds)
                raise SupplierTimeout(
                    f"{self.name} took longer than {self.timeout_seconds:g}s ({action}). "
                    "The 'slow' test switch is on."
                )
            self._sleep(self.slow_seconds)
        if self._flaky_hit():
            raise SupplierUnavailable(f"{self.name} failed ({action}). The 'flaky' test switch is on.")

    def _flaky_hit(self) -> bool:
        # The generator only advances while flaky is on, so runs stay repeatable.
        return self.faults.flaky and self._rng.random() < self.flaky_failure_rate

    def _load_ledger(self) -> dict[str, dict[str, Any]]:
        if self._ledger_path is None or not self._ledger_path.exists():
            return {}
        return json.loads(self._ledger_path.read_text(encoding="utf-8"))

    def _save_ledger(self) -> None:
        if self._ledger_path is None:
            return
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._ledger, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._ledger_path)  # atomic on Windows, Mac and Linux


def _fingerprint(order: SupplierOrder) -> str:
    data = order.model_dump(mode="json")
    data["lines"] = sorted(data["lines"], key=lambda line: line["sku"])
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def build_suppliers(
    cfg: AppConfig,
    *,
    ledger_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Supplier]:
    """One connector per supplier in config.

    connector: mock (the default) builds a fake supplier with that supplier's fault switches.
    connector: "package.module:ClassName" builds a real one: ClassName(supplier_config).
    """
    ledgers = ledger_dir if ledger_dir is not None else default_ledger_dir(cfg)
    out: dict[str, Supplier] = {}
    for sup in cfg.suppliers:
        if sup.connector == "mock":
            out[sup.code] = MockSupplier(
                sup,
                seed=cfg.app.random_seed,
                faults=cfg.faults.for_supplier(sup.code),
                slow_seconds=cfg.faults.slow_seconds,
                flaky_failure_rate=cfg.faults.flaky_failure_rate,
                timeout_seconds=cfg.supplier_calls.timeout_seconds,
                ledger_path=ledgers / f"{sup.code}.json",
                sleep=sleep,
            )
        else:
            out[sup.code] = load_connector(sup.connector)(sup)
    return out


def load_connector(path: str) -> type[Supplier]:
    """'app.suppliers.order_guide:OrderGuideSupplier' -> the class. It must be a Supplier."""
    import importlib

    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"connector '{path}' should look like 'package.module:ClassName'")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Supplier)):
        raise TypeError(f"{path} is not a Supplier class")
    return cls
