"""The public worked examples under ``examples/`` stay runnable: each folder's
core table schema CSV names only known indicators, its core-model YAML loads
against the core table that CSV builds, its session.json restores through the
MCP ``load_session`` tool, and the example notebook runs end to end."""

import json
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
import yaml

from osaa_metrics import (
    build_connection,
    build_wide_table,
    load_embedding_cache,
    load_from_csv,
    load_settings,
    validate_and_load_model,
    validate_session,
)
from tests.conftest import DATA_AVAILABLE, NO_DATA_REASON

pytestmark = pytest.mark.skipif(not DATA_AVAILABLE, reason=NO_DATA_REASON)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# (folder, file stem shared by the folder's .csv / .yaml / .session.json)
EXAMPLE_SETS = [
    ("electricity_poverty", "electricity_poverty"),
    ("economic_transformation", "wdi_economy"),
]


def _paths(folder: str, stem: str) -> dict[str, Path]:
    base = EXAMPLES / folder
    return {
        "csv": base / f"{stem}.csv",
        "yaml": base / f"{stem}.yaml",
        "session": base / f"{stem}.session.json",
    }


@pytest.fixture(scope="module")
def con():
    return build_connection(load_settings())


@pytest.fixture(scope="module")
def catalogue(con):
    return load_embedding_cache(con=con)


@pytest.mark.parametrize(("folder", "stem"), EXAMPLE_SETS)
def test_csv_names_only_known_indicators(folder, stem, catalogue):
    _, unknown = load_from_csv(_paths(folder, stem)["csv"], cache_df=catalogue)
    assert unknown == []


@pytest.mark.parametrize(("folder", "stem"), EXAMPLE_SETS)
def test_yaml_loads_against_the_core_table_the_csv_builds(folder, stem, con, catalogue):
    paths = _paths(folder, stem)
    matched, _ = load_from_csv(paths["csv"], cache_df=catalogue)
    core_tbl = build_wide_table(matched, con=con)
    validate_and_load_model(paths["yaml"].read_text(), core_tbl)


@pytest.mark.parametrize(("folder", "stem"), EXAMPLE_SETS)
def test_session_carries_the_folder_csv_and_yaml(folder, stem):
    paths = _paths(folder, stem)
    session = json.loads(paths["session"].read_text())
    assert validate_session(session) == []
    csv_rows = pl.read_csv(paths["csv"]).to_dicts()
    assert session["schema"] == csv_rows
    assert session["yaml"] == paths["yaml"].read_text()


@pytest.mark.parametrize(("folder", "stem"), EXAMPLE_SETS)
def test_session_queries_name_only_fields_the_yaml_defines(folder, stem):
    paths = _paths(folder, stem)
    session = json.loads(paths["session"].read_text())
    model = next(iter(yaml.safe_load(paths["yaml"].read_text()).values()))
    fields = set(model["dimensions"]) | set(model["measures"])
    assert session["queries"]
    for q in session["queries"]:
        args = q["query_args"]
        named = set(args.get("dimensions") or []) | set(args.get("measures") or [])
        named |= {f["field"] for f in args.get("filters") or []}
        assert named <= fields, (q["id"], named - fields)


@pytest.mark.asyncio
@pytest.mark.parametrize(("folder", "stem"), EXAMPLE_SETS)
async def test_session_restores_through_load_session(folder, stem, client):
    text = _paths(folder, stem)["session"].read_text()
    await client.call_tool("load_session", {"json_text": text})


NOTEBOOK = EXAMPLES / "electricity_poverty" / "electricity_poverty.py"


def test_electricity_poverty_notebook_starts_from_the_session():
    """The notebook continues a chat session: it restores the session.json and
    queries the core semantic table, rather than rebuilding from the CSV."""
    text = NOTEBOOK.read_text()
    assert "electricity_poverty.session.json" in text
    assert "load_from_csv" not in text
    assert "core_semantic_table.query(" in text


def test_electricity_poverty_notebook_runs_end_to_end():
    notebook = NOTEBOOK
    result = subprocess.run(
        [shutil.which("uv") or "uv", "run", "--script", str(notebook)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
