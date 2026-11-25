"""From nothing: the two commands work with no AI key. (pip install is done by then.)

1. `python run.py --check` builds the database, loads the fake data and reports no AI.
2. `python run.py` starts the app; we wait for it to answer, then stop it.
"""
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean_env(tmp_path):
    env = {**os.environ, "WISHING_CUP_DATA_DIR": str(tmp_path / "data"),
           "WISHING_CUP_ENV_FILE": str(tmp_path / "no.env")}   # no .env, no key
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "WISHING_CUP_CONFIG"):
        env.pop(name, None)
    return env


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def stop(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":  # run.py starts Streamlit as a child; stop both
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait(timeout=30)


def test_the_two_commands_from_scratch_with_no_ai_key(tmp_path):
    env = clean_env(tmp_path)
    assert not (tmp_path / "data").exists()
    check = subprocess.run([sys.executable, "run.py", "--check"], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=300)
    assert check.returncode == 0, check.stdout + check.stderr
    for text in ("loaded just now", "No AI key", "Using text matching", "PuLP/CBC is working",
                 "no pending orders", "FAKE TEST DATA"):
        assert text in check.stdout, (text, check.stdout)
    assert (tmp_path / "data" / "wishing_cup.db").exists()

    port = free_port()
    proc = subprocess.Popen([sys.executable, "run.py", "--no-browser", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=os.name != "nt")
    try:
        deadline, health = time.monotonic() + 120, None
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=5) as r:
                    health = r.read().decode().strip()
                    break
            except OSError:
                time.sleep(1)
        assert health == "ok", f"the app didn't start (exit code {proc.poll()})"
        with urllib.request.urlopen(f"http://localhost:{port}/", timeout=10) as r:
            assert r.status == 200
    finally:
        stop(proc)
