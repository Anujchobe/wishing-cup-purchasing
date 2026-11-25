"""Database engine and helpers (SQLite file in the data folder).

SQLite settings applied to every connection:
  foreign_keys=ON   SQLite ignores foreign keys unless asked; we want them enforced.
  journal_mode=WAL  the UI can read while an order is being written.
  busy_timeout      wait up to 5 s for a lock instead of failing at once.

There are no migrations yet. The schema version is kept in SQLite's
user_version; if it doesn't match the code, startup stops and tells you to run
`python run.py --reset` rather than running new code on an old layout.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, inspect
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine, func, select

from app import models
from app.config import AppConfig, db_path


class SchemaMismatch(RuntimeError):
    """The database file was made by a different version of the code."""


_ENGINES: dict[str, Engine] = {}


def _set_sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


def make_engine(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    # as_posix(): forward slashes, so Windows paths like C:\... form a valid URL.
    engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _set_sqlite_pragmas)
    return engine


def get_engine(cfg: AppConfig) -> Engine:
    key = str(db_path(cfg))
    if key not in _ENGINES:
        _ENGINES[key] = make_engine(db_path(cfg))
    return _ENGINES[key]


def dispose_engines() -> None:
    """Close every pooled connection (needed before deleting the file on Windows)."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()


def init_db(engine: Engine) -> None:
    with engine.connect() as conn:
        existing = set(inspect(conn).get_table_names())
        version = conn.exec_driver_sql("PRAGMA user_version").scalar()
    if existing and version != models.SCHEMA_VERSION:
        raise SchemaMismatch(
            f"The database was made by a different version of the app (schema {version}, "
            f"this code needs {models.SCHEMA_VERSION}). Run `python run.py --reset` to rebuild it. "
            "That deletes the local database, including any orders made in it."
        )
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"PRAGMA user_version = {int(models.SCHEMA_VERSION)}")


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A session that commits on success and rolls back on any error."""
    with Session(engine, expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def sync_config_to_db(session: Session, cfg: AppConfig) -> None:
    """Make the locations and suppliers tables match config.yaml.

    Rows that disappeared from config are marked inactive, never deleted,
    because old requests and orders still point at them.
    """
    locations = {row.code: row for row in session.exec(select(models.Location)).all()}
    for lc in cfg.locations:
        row = locations.get(lc.code) or models.Location(code=lc.code, name=lc.name, kind=models.LocationKind(lc.kind))
        row.name, row.kind, row.active = lc.name, models.LocationKind(lc.kind), True
        session.add(row)
    for code, row in locations.items():
        if code not in {lc.code for lc in cfg.locations}:
            row.active = False
            session.add(row)

    suppliers = {row.code: row for row in session.exec(select(models.Supplier)).all()}
    for sc in cfg.suppliers:
        row = suppliers.get(sc.code) or models.Supplier(
            code=sc.code, name=sc.name, min_order_cents=0, delivery_fee_cents=0, lead_time_days=0
        )
        row.name = sc.name
        row.min_order_cents = sc.min_order_cents
        row.delivery_fee_cents = sc.delivery_fee_cents
        row.delivery_weekdays = list(sc.delivery_weekdays)
        row.lead_time_days = sc.lead_time_days
        row.serves_locations = list(sc.serves_locations)
        row.rep_name = sc.sales_rep.name
        row.rep_email = sc.sales_rep.email
        row.is_main = sc.code == cfg.purchasing.main_supplier
        row.is_placeholder = sc.placeholder
        row.data_is_fake = True
        row.active = True
        session.add(row)
    for code, row in suppliers.items():
        if code not in {sc.code for sc in cfg.suppliers}:
            row.active = False
            row.is_main = False
            session.add(row)
    session.flush()


def is_seeded(session: Session) -> bool:
    return session.exec(select(models.Product).limit(1)).first() is not None


def table_counts(session: Session) -> dict[str, int]:
    return {
        model.__tablename__: session.exec(select(func.count()).select_from(model)).one()
        for model in models.ALL_TABLES
    }


def audit(session: Session, actor: str, action: str, entity_type: str = "",
          entity_id: str | None = None, **details: Any) -> None:
    session.add(models.AuditLog(actor=actor, action=action, entity_type=entity_type,
                                entity_id=entity_id, details=details))
