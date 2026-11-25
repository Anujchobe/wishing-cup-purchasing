"""Start the Wishing Cup purchasing app.

    python run.py              set up the database and fake data if needed, then open the app
    python run.py --check      do the setup and print a status report, without starting the UI
    python run.py --reset      delete the local database and fake supplier ledgers first, then reseed
    python run.py --port 8502  use another port (default 8501)
    python run.py --no-browser don't open a browser tab

The UI listens on localhost only, so nobody else on the cafe Wi-Fi can reach it.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    sys.exit(f"This app needs Python 3.11 or newer. You have {sys.version.split()[0]}.")

import argparse
import subprocess
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app.config import ROOT as APP_ROOT
    from app.config import ConfigError, db_path, load_config, load_llm_settings
    from app.db import SchemaMismatch, get_engine, init_db, session_scope, sync_config_to_db, table_counts
    from app.llm.model_check import model_status
    from app.optimizer import solver_check
    from app.suppliers.mock import build_suppliers
    from app.workflow import recover_pending_orders
    from scripts.seed import reset_data, seed_if_empty
except ModuleNotFoundError as exc:  # usually: requirements not installed in this Python
    sys.exit(f"Missing library '{exc.name}'. Run:  {Path(sys.executable).name} -m pip install -r requirements.txt")


def prepare(reset: bool = False) -> int:
    """Validate config, build the database, seed fake data. Returns an exit code."""
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"Config problem. Fix config.yaml and try again.\n{exc}")
        return 2

    if reset:
        reset_data(cfg)
        print("Deleted the local database and fake supplier ledgers.")

    engine = get_engine(cfg)
    try:
        init_db(engine)
    except SchemaMismatch as exc:
        print(exc)
        return 3

    seeded = seed_if_empty(engine, cfg)
    with session_scope(engine) as session:
        sync_config_to_db(session, cfg)  # config edits take effect on every start
        counts = table_counts(session)
        ai = model_status(cfg, load_llm_settings())
    # Finish any order saved as PENDING when the app last stopped (same order ID, so never twice).
    recovered = recover_pending_orders(engine, build_suppliers(cfg), cfg)

    try:
        shown_path = db_path(cfg).relative_to(APP_ROOT)
    except ValueError:
        shown_path = db_path(cfg)
    print("Wishing Cup purchasing - setup check")
    print(f"  Config:     OK ({len(cfg.suppliers)} suppliers, {len(cfg.locations)} locations, "
          f"main supplier {cfg.main_supplier.name})")
    print(f"  Database:   {shown_path}")
    print("  Fake data:  " + ("loaded just now" if seeded else "already loaded")
          + f" ({counts['products']} products, {counts['supplier_items']} supplier items, "
            f"{counts['price_snapshots']} price snapshots, {counts['orders']} past orders)")
    print(f"  AI:         {ai.label}")
    print(f"  Orders:     {recovered.summary}")
    print(f"  Optimizer:  {solver_check()[1]}")
    switches = cfg.faults.describe_on()
    print("  Test switches: " + (", ".join(switches) if switches else "all off"))
    print("  NOTE: " + cfg.app.fake_data_label)
    return 0


def launch_ui(port: int, open_browser: bool) -> int:
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"),
        "--server.address", "localhost",
        "--server.port", str(port),
        "--server.headless", "true",          # skips Streamlit's first-run email prompt
        "--browser.gatherUsageStats", "false",
    ]
    url = f"http://localhost:{port}"
    print(f"Starting the app at {url}  (press Ctrl+C to stop)")
    if open_browser:
        timer = threading.Timer(2.0, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        return subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wishing Cup purchasing app")
    parser.add_argument("--check", action="store_true", help="set up and print status, don't start the UI")
    parser.add_argument("--reset", action="store_true", help="delete the local database and reseed fake data")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    code = prepare(reset=args.reset)
    if code != 0 or args.check:
        return code
    return launch_ui(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
