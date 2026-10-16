"""Regression tests for Item 3: Safe process lifecycle management without killing foreign processes."""

import asyncio
import socket
import subprocess
import sys
import time
import psutil
from fastapi.testclient import TestClient


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_foreign_service_survives_start_and_stop(monkeypatch):
    import aiforge.ui.app as ui_app

    port = _find_free_port()
    monkeypatch.setattr(ui_app, "LLM_SERVER_PORT", port)
    monkeypatch.setattr(ui_app, "llm_server_process", None)

    # Start an independent foreign TCP server on that port
    foreign_code = (
        f"import socket, time; s = socket.socket(); "
        f"s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
        f"s.bind(('127.0.0.1', {port})); s.listen(1); "
        f"print('READY', flush=True); time.sleep(30)"
    )
    foreign_proc = subprocess.Popen(
        [sys.executable, "-c", foreign_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_line = foreign_proc.stdout.readline().strip()
        assert ready_line == "READY"
        assert foreign_proc.poll() is None

        with TestClient(ui_app.app) as client:
            # Start should refuse because port is occupied by foreign process
            res_start = client.post("/api/server/start", json={"experiment_id": "BASE:openai-community/gpt2"})
            assert res_start.status_code == 409
            assert str(port) in res_start.json()["detail"]
            assert foreign_proc.poll() is None, "Foreign process was killed during start!"

            # Stop should report not running and leave foreign process untouched
            res_stop = client.post("/api/server/stop")
            assert res_stop.status_code == 200
            assert res_stop.json()["status"] == "not running"
            assert foreign_proc.poll() is None, "Foreign process was killed during stop!"
    finally:
        foreign_proc.terminate()
        try:
            foreign_proc.wait(timeout=2.0)
        except Exception:
            foreign_proc.kill()


def test_aiforge_subprocess_and_descendants_terminated_on_stop(monkeypatch, tmp_path):
    import aiforge.ui.app as ui_app

    port = _find_free_port()
    monkeypatch.setattr(ui_app, "LLM_SERVER_PORT", port)
    monkeypatch.setattr(ui_app, "LLM_SERVER_LOG", str(tmp_path / "llm.log"))
    monkeypatch.setattr(ui_app, "llm_server_process", None)

    child_pid_file = tmp_path / "child.pid"
    parent_script = (
        f"import subprocess, sys, time, pathlib\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
        f"time.sleep(60)\n"
    )
    monkeypatch.setattr(
        ui_app,
        "_build_serve_command",
        lambda exp_id: [sys.executable, "-c", parent_script],
    )

    with TestClient(ui_app.app) as client:
        res_start = client.post("/api/server/start", json={"experiment_id": "BASE:gpt2"})
        assert res_start.status_code == 200
        parent_pid = res_start.json()["pid"]

        # Wait for child to spawn
        deadline = time.time() + 5.0
        while not child_pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text().strip())

        assert psutil.pid_exists(parent_pid)
        assert psutil.pid_exists(child_pid)

        res_stop = client.post("/api/server/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["status"] == "stopped"

        assert not psutil.pid_exists(parent_pid)
        assert not psutil.pid_exists(child_pid)


def test_forge_subprocess_tree_closed_on_client_disconnect(monkeypatch, tmp_path):
    import aiforge.ui.app as ui_app

    pids_file = tmp_path / "forge_pids.txt"
    fake_forge_script = (
        f"import subprocess, sys, time, pathlib\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pids_file)!r}).write_text(f'{{os.getpid()}},{{child.pid}}' if 'os' in globals() else '')\n"
    )
    fake_forge_script = (
        f"import os, subprocess, sys, time, pathlib\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pids_file)!r}).write_text(f'{{os.getpid()}},{{child.pid}}')\n"
        f"print('PROGRESS: 10', flush=True)\n"
        f"time.sleep(60)\n"
    )

    orig_create = asyncio.create_subprocess_exec

    async def fake_create_subprocess_exec(*args, **kwargs):
        return await orig_create(sys.executable, "-c", fake_forge_script, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def _run_and_disconnect():
        req = ui_app.ForgeRequest(name="test", model="gpt2", topic="testing")
        resp = await ui_app.start_forge(req)
        gen = resp.body_iterator
        first_chunk = await gen.__anext__()
        assert "PROGRESS: 10" in first_chunk
        # Simulate client disconnect by closing the async generator early
        await gen.aclose()

    asyncio.run(_run_and_disconnect())

    assert pids_file.exists()
    parent_pid_str, child_pid_str = pids_file.read_text().strip().split(",")
    parent_pid, child_pid = int(parent_pid_str), int(child_pid_str)

    # Give OS a brief moment to reap
    time.sleep(0.1)
    assert not psutil.pid_exists(parent_pid)
    assert not psutil.pid_exists(child_pid)
