import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, *args):
    env = {**os.environ, "WISHING_CUP_DATA_DIR": str(tmp_path)}
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "WISHING_CUP_CONFIG"):
        env.pop(name, None)
    return subprocess.run([sys.executable, "run.py", "--check", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


def test_check_sets_everything_up(tmp_path):
    first = run(tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "loaded just now" in first.stdout and "FAKE TEST DATA" in first.stdout
    assert "PuLP/CBC is working" in first.stdout
    assert "  AI:" in first.stdout
    assert "Orders:     no pending orders" in first.stdout
    assert (tmp_path / "wishing_cup.db").exists()

    second = run(tmp_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "already loaded" in second.stdout


def test_reset_rebuilds(tmp_path):
    assert run(tmp_path).returncode == 0
    again = run(tmp_path, "--reset")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "loaded just now" in again.stdout


def test_bad_config_stops_with_a_plain_message(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text((ROOT / "config.yaml").read_text(encoding="utf-8").replace("main_supplier: sysco",
                                                                             "main_supplier: nobody"),
                   encoding="utf-8")
    env = {**os.environ, "WISHING_CUP_DATA_DIR": str(tmp_path), "WISHING_CUP_CONFIG": str(bad)}
    result = subprocess.run([sys.executable, "run.py", "--check"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "main_supplier 'nobody'" in result.stdout
