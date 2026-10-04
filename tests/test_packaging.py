"""Packaging tests — non-Python assets ship inside the wheel and substrate
helpers require a caller-provided connection (no implicit env-var routing).

Widget HTML and the JSON Schema live inside the package and are reached through
``importlib.resources``; a substrate helper takes its connection as an argument
rather than reading the environment.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import entry_points, metadata
from pathlib import Path

import pytest

from osaa_metrics import core_table, discovery
from osaa_metrics.mcp.widget_loader import load_schema_text, load_widget


def test_chart_widget_ships_in_wheel() -> None:
    """``mcp/_widgets/chart.html`` is reachable via importlib.resources."""
    html = load_widget("chart.html")
    assert "<script" in html
    assert "@modelcontextprotocol/ext-apps" in html


def test_core_summary_widget_ships_in_wheel() -> None:
    """``mcp/_widgets/core_summary.html`` is reachable via importlib.resources."""
    html = load_widget("core_summary.html")
    assert "Core table — summary" in html


def test_discovery_widget_ships_in_discovery_subpackage() -> None:
    """``mcp_discovery/_widgets/discovery.html`` is reachable: widget_loader's
    name → package map routes ``discovery.html`` to the discovery sub-package,
    not to mcp/_widgets/."""
    html = load_widget("discovery.html")
    assert "function parseToolResult" in html  # shared.js was inlined
    assert ">Save<" in html


def test_bsl_schema_ships_in_wheel() -> None:
    """``_schemas/bsl_model.schema.json`` loads as valid JSON. The schema is
    substrate-owned so ``semantic.py`` validates without an MCP dependency;
    the MCP server re-serves the same bytes as ``schema://``."""
    raw = load_schema_text("bsl_model.schema.json")
    schema = json.loads(raw)
    assert "$schema" in schema
    assert "title" in schema or "description" in schema


def test_load_embedding_cache_requires_explicit_con() -> None:
    """``con`` is a required keyword-only argument, so the helper never builds
    a connection from the environment itself. Callers thread the connection
    ``build_connection`` returns in explicitly."""
    sig = inspect.signature(discovery.load_embedding_cache)
    assert "con" in sig.parameters
    assert sig.parameters["con"].kind.name == "KEYWORD_ONLY"
    assert sig.parameters["con"].default is inspect.Parameter.empty


def test_build_wide_table_requires_explicit_con() -> None:
    """``build_wide_table`` takes the session-scoped DuckDB/ibis backend as a
    required keyword-only argument; it opens no connection of its own."""
    sig = inspect.signature(core_table.build_wide_table)
    assert "con" in sig.parameters
    assert sig.parameters["con"].kind.name == "KEYWORD_ONLY"
    assert sig.parameters["con"].default is inspect.Parameter.empty


def test_console_scripts_are_the_two_servers():
    """The package ships the stdio and HTTP launchers and nothing else on
    PATH; install is a justfile, not a CLI."""
    ours = {
        ep.name
        for ep in entry_points(group="console_scripts")
        if ep.value.startswith("osaa_metrics.")
    }
    assert ours == {"osaa-mcp", "osaa-mcp-http"}


def test_no_cli_package():
    assert importlib.util.find_spec("osaa_metrics.cli") is None


def test_no_cli_extra():
    extras = set(metadata("osaa-metrics").get_all("Provides-Extra") or [])
    assert "cli" not in extras


def test_public_extras_name_the_encoder():
    """The optional local encoder installs as the `encoder` extra; `local`
    is the encoder mode value, not an extra."""
    extras = set(metadata("osaa-metrics").get_all("Provides-Extra") or [])
    assert "encoder" in extras
    assert "local" not in extras


_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


def _names(requirements: list[str]) -> set[str]:
    """Distribution names from requirement strings, extras and specifiers stripped."""
    out = set()
    for req in requirements:
        name = req.split("[")[0]
        for sep in (">", "<", "=", "!", "~", ";", " "):
            name = name.split(sep)[0]
        out.add(name.strip().lower())
    return out


def test_mcp_extra_declares_starlette() -> None:
    """``mcp/server.py`` imports ``starlette`` directly, so the ``mcp`` extra
    declares it rather than relying on fastmcp to pull it in."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "starlette" in _names(extras["mcp"])


def test_mcp_extra_declares_pydantic() -> None:
    """``mcp/server.py`` and ``mcp_discovery/server.py`` import ``pydantic``
    directly, so the ``mcp`` extra declares it rather than relying on
    fastmcp to pull it in."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "pydantic" in _names(extras["mcp"])


def test_encoder_extra_declares_huggingface_hub() -> None:
    """``_models.py`` imports ``huggingface_hub`` directly, so the
    ``encoder`` extra declares it rather than relying on
    sentence-transformers to pull it in."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "huggingface-hub" in _names(extras["encoder"])


def test_base_ships_altair_and_no_numpy() -> None:
    """``theme.py`` imports altair and the chart call renders through BSL's
    altair backend, so it is a base dependency. No module under ``src/``
    needs numpy at import time; it arrives with ibis's duckdb extra."""
    base = _names(_pyproject()["project"]["dependencies"])
    assert "altair" in base
    assert "numpy" not in base


def test_no_chart_rasterizers_declared() -> None:
    """BSL's altair backend reaches vl-convert, the image renderer, only when
    the chart call asks for ``png``/``svg`` output; ``static``/``json`` output
    never does. vegafusion is altair's data-transformer escape hatch past
    ``Chart.to_dict()``'s row cap, not an image renderer, and isn't declared
    either."""
    proj = _pyproject()["project"]
    declared = _names(proj["dependencies"])
    for reqs in proj["optional-dependencies"].values():
        declared |= _names(reqs)
    assert "vegafusion" not in declared
    assert "vl-convert-python" not in declared


def test_no_empty_viz_extra() -> None:
    """Altair is a base dependency and no chart rasterizers are declared, so
    there is no ``viz`` extra."""
    assert "viz" not in _pyproject()["project"]["optional-dependencies"]


def test_dev_extra_declares_numpy() -> None:
    """The discovery tests build fake embeddings with numpy directly."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "numpy" in _names(extras["dev"])


_REPO = Path(__file__).resolve().parents[1]

_ZIP_PROBE = """
import sys, osaa_metrics
assert osaa_metrics.__file__.startswith(sys.argv[1]), osaa_metrics.__file__
from osaa_metrics.semantic import _bsl_schema_validator
from osaa_metrics.session import _schema_validator
from osaa_metrics.mcp.widget_loader import load_schema_text
_bsl_schema_validator()
_schema_validator()
assert '"$schema"' in load_schema_text("bsl_model.schema.json")
print("zip-ok")
"""


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    out = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    (wheel,) = out.glob("*.whl")
    return wheel


def test_schemas_load_from_zipped_wheel(built_wheel: Path, tmp_path: Path) -> None:
    """Packaged schemas are read through the ``importlib.resources``
    traversable, so they load when the package runs from a still-zipped
    wheel. The probe imports the package from the wheel path itself and
    refuses the editable install."""
    env = {**os.environ, "PYTHONPATH": str(built_wheel)}
    result = subprocess.run(
        [sys.executable, "-c", _ZIP_PROBE, str(built_wheel)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "zip-ok"


def test_linux_torch_resolves_from_cpu_index() -> None:
    """The encoder extra declares torch directly, which is what lets the
    source override below apply to it; on Linux that override routes torch
    to the CPU-only index, so a Linux resolve of this repository does not
    pull the CUDA packages. The override is a uv source for this
    repository's own resolution; the published wheel's requirement is a
    bare torch, so an install from PyPI on Linux resolves torch from
    PyPI."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "torch" in _names(extras["encoder"])
    uv = _pyproject()["tool"]["uv"]
    (cpu_index,) = [i for i in uv["index"] if i["name"] == "pytorch-cpu"]
    assert cpu_index["url"] == "https://download.pytorch.org/whl/cpu"
    assert cpu_index["explicit"] is True
    (torch_src,) = uv["sources"]["torch"]
    assert torch_src == {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}
