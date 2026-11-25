"""Load the fake test data into the database.

    python scripts/seed.py           seed an empty database
    python scripts/seed.py --reset   delete the database and fake supplier ledgers, then seed

What it creates (all FAKE, all from the fixed seed in config.yaml):
  - our ingredient list (products)
  - each supplier's catalog with raw, messy pack text (supplier_items)
  - 9 weekly price snapshots per item: this week plus 8 past weeks
  - mappings from our products to the main supplier's SKUs, marked as confirmed
    by a person, because the business already orders from that supplier.
    Other suppliers are left unmapped so step 3's matching has real work to do.
  - 8 weeks of past orders, all from the main supplier, the way the business
    bought before this app. Savings replay (step 5) compares against these.

Seeding reads fake_world.py directly rather than calling the fake supplier
API, so the fault switches can't break it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import date, time, timedelta
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/seed.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from app import clock, models
from app.config import AppConfig, db_path, ledger_dir, load_config
from app.db import audit, dispose_engines, get_engine, init_db, is_seeded, session_scope, sync_config_to_db
from app.suppliers import fake_world
from app.pricing import cases_to_buy, format_qty
from app.suppliers.base import make_order_id


@dataclass
class SeedSummary:
    products: int = 0
    supplier_items: int = 0
    price_snapshots: int = 0
    item_mappings: int = 0
    requests: int = 0
    request_lines: int = 0
    orders: int = 0


def next_delivery_day(earliest: date, weekdays: list[str]) -> date:
    for offset in range(7):
        day = earliest + timedelta(days=offset)
        if clock.weekday_name(day) in weekdays:
            return day
    raise ValueError(f"no delivery weekday in {weekdays}")


def history_public_id(seed: int, location_code: str, weeks_ago: int) -> str:
    return f"R-{fake_world.stable_seed(seed, 'history', location_code, weeks_ago):016x}"[:12]


def seed(session: Session, cfg: AppConfig, *, today: date | None = None) -> SeedSummary:
    if is_seeded(session):
        raise RuntimeError("The database already has data. Use `python run.py --reset` to start over.")
    today = today or clock.today()
    seed_value = cfg.app.random_seed
    summary = SeedSummary()

    sync_config_to_db(session, cfg)
    suppliers = {row.code: row for row in session.exec(select(models.Supplier)).all()}
    locations = {row.code: row for row in session.exec(select(models.Location)).all()}

    this_monday = clock.week_start(today)

    def snapshot_time(weeks_ago: int):
        return clock.at_boston(this_monday - timedelta(weeks=weeks_ago), time(6, 0))

    # Our ingredient list.
    products: dict[str, models.Product] = {}
    for fp in fake_world.PRODUCTS:
        products[fp.code] = models.Product(code=fp.code, name=fp.name, base_unit=models.BaseUnit(fp.base_unit),
                                           category=fp.category)
        session.add(products[fp.code])
    session.flush()
    summary.products = len(products)

    # Supplier catalogs (this week's prices) and price history.
    items: dict[tuple[str, str], models.SupplierItem] = {}
    for sc in cfg.suppliers:
        for offer in fake_world.offers_for(sc.code):
            item = models.SupplierItem(
                supplier_id=suppliers[sc.code].id,
                sku=offer.sku,
                description=offer.description,
                raw_pack_text=offer.raw_pack_text,
                case_price_cents=fake_world.case_price_cents(offer, 0, seed_value),
                in_stock=fake_world.in_stock(offer, 0),
                updated_at=snapshot_time(0),
            )
            session.add(item)
            items[(sc.code, offer.product_code)] = item
    session.flush()
    summary.supplier_items = len(items)

    for (supplier_code, product_code), item in items.items():
        offer = fake_world.offer(supplier_code, product_code)
        for weeks_ago in range(fake_world.HISTORY_WEEKS, -1, -1):
            session.add(models.PriceSnapshot(
                supplier_item_id=item.id,
                case_price_cents=fake_world.case_price_cents(offer, weeks_ago, seed_value),
                in_stock=fake_world.in_stock(offer, weeks_ago),
                captured_at=snapshot_time(weeks_ago),
                source="seed_history",
            ))
            summary.price_snapshots += 1

    # Known mappings: the main supplier only.
    main = cfg.main_supplier
    main_row = suppliers[main.code]
    for product_code, prod in products.items():
        item = items.get((main.code, product_code))
        if item is None:
            continue
        session.add(models.ItemMapping(
            product_id=prod.id,
            supplier_id=main_row.id,
            supplier_item_id=item.id,
            pack_size_base=fake_world.offer(main.code, product_code).pack_base_qty,
            confidence=1.0,
            source=models.MappingSource.person,
            needs_check=False,
            confirmed_by="seed: already ordered from this supplier",
            confirmed_at=snapshot_time(fake_world.HISTORY_WEEKS),
            created_at=snapshot_time(fake_world.HISTORY_WEEKS),
        ))
        summary.item_mappings += 1

    # Past orders: one request per location per week, all from the main supplier.
    for location_code, usual in fake_world.USUAL_WEEKLY_NEED.items():
        location = locations.get(location_code)
        if location is None or main.code not in suppliers or location_code not in main.serves_locations:
            continue
        for weeks_ago in range(fake_world.HISTORY_WEEKS, 0, -1):
            monday = this_monday - timedelta(weeks=weeks_ago)
            # Needed by Friday; the main supplier delivered on its first day after its lead time.
            # (A realistic needed-by date matters: the savings replay asks whether others could deliver.)
            needed_by = monday + timedelta(days=4)
            delivered = next_delivery_day(monday + timedelta(days=main.lead_time_days), main.delivery_weekdays)
            request = models.Request(
                public_id=history_public_id(seed_value, location_code, weeks_ago),
                location_id=location.id,
                needed_by=needed_by,
                original_text=f"Weekly order for {location.name} (fake history)",
                status=models.RequestStatus.ORDERED,
                source="seed_history",
                created_by="seed",
                created_at=clock.at_boston(monday, time(9, 0)),
                approved_by="owner (fake history)",
                approved_at=clock.at_boston(monday, time(10, 0)),
            )
            session.add(request)
            session.flush()
            summary.requests += 1

            subtotal = 0
            line_no = 0
            for product_code in sorted(usual):
                item = items.get((main.code, product_code))
                if item is None:
                    continue
                offer = fake_world.offer(main.code, product_code)
                need = fake_world.history_need(location_code, product_code, weeks_ago, seed_value)
                cases = cases_to_buy(need, offer.pack_base_qty)
                price = fake_world.case_price_cents(offer, weeks_ago, seed_value)
                line_no += 1
                session.add(models.RequestLine(
                    request_id=request.id,
                    line_no=line_no,
                    raw_text=f"{format_qty(need, products[product_code].base_unit.value)} {products[product_code].name}",
                    product_id=products[product_code].id,
                    qty_needed=need,
                    fill_status=models.LineFillStatus.ASSIGNED,
                    assigned_supplier_id=main_row.id,
                    supplier_item_id=item.id,
                    cases=cases,
                    case_price_cents=price,
                    line_cost_cents=cases * price,
                ))
                subtotal += cases * price
                summary.request_lines += 1

            order_key = make_order_id(request.public_id, main.code)
            session.add(models.Order(
                order_key=order_key,
                request_id=request.id,
                supplier_id=main_row.id,
                status=models.OrderStatus.CONFIRMED,
                confirmation_id=fake_world.confirmation_for(main.code, order_key),
                subtotal_cents=subtotal,
                delivery_fee_cents=main.delivery_fee_cents,
                total_cents=subtotal + main.delivery_fee_cents,
                deliver_on=delivered,
                attempts=1,
                created_at=clock.at_boston(monday, time(10, 0)),
                placed_at=clock.at_boston(monday, time(10, 1)),
            ))
            summary.orders += 1

    audit(session, "seed", "seed_fake_data", "database", None,
          random_seed=seed_value, seeded_for=today.isoformat(), **asdict(summary))
    session.flush()
    return summary


def seed_if_empty(engine: Engine, cfg: AppConfig, *, today: date | None = None) -> SeedSummary | None:
    with session_scope(engine) as session:
        if is_seeded(session):
            return None
        return seed(session, cfg, today=today)


def reset_data(cfg: AppConfig) -> None:
    """Delete the database (and its WAL files) and the fake suppliers' order ledgers."""
    dispose_engines()
    path = db_path(cfg)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    shutil.rmtree(ledger_dir(cfg), ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load fake test data.")
    parser.add_argument("--reset", action="store_true", help="delete the database first")
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.reset:
        reset_data(cfg)
    engine = get_engine(cfg)
    init_db(engine)
    summary = seed_if_empty(engine, cfg)
    if summary is None:
        print("Database already has data. Nothing to do (use --reset to start over).")
    else:
        print("Seeded fake test data:", ", ".join(f"{k}={v}" for k, v in asdict(summary).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
