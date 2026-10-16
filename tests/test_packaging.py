"""Regression tests for Item 1: Packaging & Dependencies."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_wheel_includes_ui_assets_soup_and_dependencies(tmp_path):
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    build_dir = tmp_path / "build"
    subprocess.run(
        [
            sys.executable, "setup.py",
            "build", "--build-base", str(build_dir),
            "bdist_wheel", "--dist-dir", str(wheel_dir), "--bdist-dir", str(tmp_path / "bdist"),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(wheel_dir.glob("aiforge-*.whl"))
    assert len(wheels) == 1, f"Expected 1 wheel, found {wheels}"
    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path, "r") as zf:
        names = set(zf.namelist())
        assert "aiforge/ui/__init__.py" in names
        assert "aiforge/ui/app.py" in names
        assert "aiforge/ui/templates/index.html" in names
        assert "aiforge/ui/static/app.js" in names
        assert "aiforge/llm/backends/__init__.py" in names
        assert "aiforge/llm/backends/soup.py" in names
        assert "aiforge_cli.py" in names

        meta_files = [n for n in names if n.endswith(".dist-info/METADATA")]
        assert len(meta_files) == 1
        metadata_text = zf.read(meta_files[0]).decode("utf-8")
        assert "Requires-Dist: psutil" in metadata_text
        assert "Requires-Dist: pydantic<3.0,>=2.0" in metadata_text or "Requires-Dist: pydantic>=2.0" in metadata_text
        assert "openai" in metadata_text

    # Unpack wheel into an isolated site-packages dir and execute from an external cwd
    install_target = tmp_path / "installed_pkg"
    install_target.mkdir()
    with zipfile.ZipFile(wheel_path, "r") as zf:
        zf.extractall(install_target)

    external_cwd = tmp_path / "external_workdir"
    external_cwd.mkdir()

    # Ensure repository root is NOT on PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_target)
    env["AIFORGE_TESTING"] = "1"

    verify_code = """
import os, pathlib, aiforge, aiforge.ui.app, aiforge_cli
from aiforge.llm.backends.soup import SoupBackend
from fastapi.testclient import TestClient

pkg_file = pathlib.Path(aiforge.__file__).resolve()
assert str(pkg_file).startswith(os.environ["PYTHONPATH"]), f"Loaded from wrong path: {pkg_file}"

with TestClient(aiforge.ui.app.app) as client:
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "AIFORGE" in r_index.text
    r_js = client.get("/static/app.js")
    assert r_js.status_code == 200
"""
    res = subprocess.run(
        [sys.executable, "-I", "-c", f"import os, sys; sys.path.insert(0, {str(install_target)!r}); os.environ['PYTHONPATH']={str(install_target)!r}; os.environ['AIFORGE_TESTING']='1'\n" + verify_code],
        cwd=str(external_cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Isolated verification failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    cli_res = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(install_target)!r}); import aiforge_cli; sys.argv=['aiforge', '--help']; aiforge_cli.main()"],
        cwd=str(external_cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    assert cli_res.returncode == 0
    assert "AIFORGE" in cli_res.stdout
