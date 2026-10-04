"""The public playground notebook: its helpers, imported from the notebook
file itself, and an end-to-end rebuild of a real session against the local
data mirror."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "app" / "playground.py"
FIXTURES = REPO / "tests" / "fixtures" / "playground"


def _load_notebook():
    spec = importlib.util.spec_from_file_location("playground", NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Unregistered, the notebook's @app.class_definition classes are invisible
    # to its cells when app.run() executes them.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pg = _load_notebook()


def _upload(name, text):
    return SimpleNamespace(name=name, contents=text.encode("utf-8"))


def _wdi_upload():
    return pg.Upload.from_files(
        [
            _upload(
                "wdi_economy.session.json",
                (FIXTURES / "wdi_economy.session.json").read_text(),
            )
        ]
    )


def _csv_upload(csv_text):
    return pg.Upload.from_files([_upload("s.csv", csv_text)])


def _wdi_schema_csv():
    return pl.DataFrame(
        [
            {"code": r.code, "description": r.description, "var_name": r.var_name}
            for r in _wdi_upload().session.schema
        ]
    ).write_csv()


def test_notebook_imports_the_library_it_replaces():
    import osaa_metrics

    assert pg.load_session is osaa_metrics.load_session
    assert pg.dump_session is osaa_metrics.dump_session


def test_notebook_imports_only_the_public_surface():
    """Every osaa_metrics import names the package, never one of its modules."""
    source = NOTEBOOK.read_text()
    assert "from osaa_metrics import" in source
    assert "from osaa_metrics." not in source
    assert "import osaa_metrics." not in source


def test_the_osaa_theme_is_on_once_the_notebook_loads():
    """The setup block enables the osaa_metrics chart theme, once."""
    import altair as alt

    source = NOTEBOOK.read_text()
    setup = source.split("with app.setup:")[1].split("\n@app.")[0]
    assert "register_theme()" in setup
    assert source.count("register_theme()") == 1
    assert alt.theme.active == "osaa_default"


def test_notebook_carries_no_copy_of_the_session_contract():
    source = NOTEBOOK.read_text()
    assert "_SESSION_SCHEMA" not in source
    assert "def load_session" not in source
    assert "def validate_session" not in source
    assert "r2.dev" not in source


def test_upload_sorts_files_by_extension():
    upload = pg.Upload.from_files(
        [
            _upload("s.csv", "code,description,var_name\nA,a,a\n"),
            _upload("m.yml", "m: {}"),
            _upload("notes.txt", "hi"),
        ]
    )
    assert upload.schema_csv.startswith("code,")
    assert upload.model_yaml == "m: {}"
    assert upload.schema_row_count == 1
    assert upload.has_model and not upload.has_queries
    assert any("notes.txt" in m for m in upload.messages)


def test_a_session_takes_precedence_over_a_loose_schema():
    upload = pg.Upload.from_files(
        [
            _upload("s.csv", "code,description,var_name\nA,a,a\n"),
            _upload(
                "x.session.json", (FIXTURES / "wdi_economy.session.json").read_text()
            ),
        ]
    )
    assert upload.has_queries
    assert upload.schema_csv is None
    assert upload.schema_row_count == len(upload.session.schema)


def test_upload_skips_oversized_files():
    big = SimpleNamespace(name="big.json", contents=b"x" * (pg.Upload.MAX_BYTES + 1))
    upload = pg.Upload.from_files([big])
    assert upload.session is None and upload.error is None
    assert any("big.json" in m and "skipped" in m for m in upload.messages)


def test_upload_skips_files_that_are_not_utf8_text():
    upload = pg.Upload.from_files(
        [SimpleNamespace(name="bad.json", contents=b"\xff\xfe\x00bad")]
    )
    assert upload.session is None
    assert any("bad.json" in m and "UTF-8" in m for m in upload.messages)


def test_an_invalid_session_lists_each_problem():
    upload = pg.Upload.from_files(
        [_upload("s.json", json.dumps({"schema_version": "1"}))]
    )
    assert upload.session is None and not upload.has_schema
    assert upload.error.startswith("session.json failed validation:")
    assert "generated_at" in upload.error


def test_a_session_that_is_not_json_is_refused():
    upload = pg.Upload.from_files([_upload("s.json", "{not json")])
    assert "invalid JSON" in upload.error


def test_an_unreadable_schema_is_refused():
    upload = _csv_upload('code,description\n"unclosed')
    assert upload.error.startswith("core table schema could not be read")


@pytest.mark.parametrize(
    ("yaml_text", "expected"),
    [
        ("wdi_economy:\n  table: core_tbl\n", "wdi_economy.session.json"),
        (None, "session.session.json"),
        ("a: [unclosed", "session.session.json"),
        ("- a\n- b\n", "session.session.json"),
        ("just text", "session.session.json"),
    ],
)
def test_download_filename_falls_back_when_the_yaml_names_no_model(yaml_text, expected):
    assert pg.Upload(model_yaml=yaml_text).download_filename == expected


def test_replay_sql_runs_against_core():
    core_df = pl.DataFrame({"country": ["A", "B"], "gdp": [1.0, 2.0]})
    out = pg.replay_sql("SELECT sum(gdp) AS total FROM core", core_df)
    assert out["total"].to_list() == [3.0]


def test_replay_sql_cannot_read_files(tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    core_df = pl.DataFrame({"x": [1]})
    sql = f"SELECT * FROM read_csv('{secret}')"  # noqa: S608  # the test's own tmp path
    with pytest.raises(duckdb.Error, match="file system operations are disabled"):
        pg.replay_sql(sql, core_df)


def test_applying_annotations_leaves_the_base_spec_untouched():
    base = {
        "title": "Old",
        "encoding": {"x": {"field": "year"}, "tooltip": [{"field": "a"}]},
    }
    before = copy.deepcopy(base)
    merged = pg.ChartAnnotations(title="New", x_title="Year").apply_to(base)
    assert merged["title"] == {"text": "New"}
    assert merged["encoding"]["x"]["title"] == "Year"
    assert base == before


def test_annotations_round_trip_through_a_spec():
    spec = {"title": {"text": "T", "subtitle": "S"}, "encoding": {"y": {"title": "Y"}}}
    fields = pg.ChartAnnotations.from_spec(spec)
    assert fields == pg.ChartAnnotations(title="T", subtitle="S", y_title="Y")
    assert fields.apply_to(spec)["title"] == {"text": "T", "subtitle": "S"}


def test_annotations_take_the_form_value_as_is():
    """The Annotate form's batch keys are the field names, so its value
    builds the annotations directly."""
    value = {
        "title": "T",
        "subtitle": "",
        "x_title": "",
        "y_title": "",
        "description": "",
    }
    assert pg.ChartAnnotations(**value).title == "T"


def test_chart_from_spec_injects_records_and_takes_the_theme():
    chart = pg.chart_from_spec(
        {"mark": "bar", "encoding": {"x": {"field": "a", "type": "nominal"}}},
        pl.DataFrame({"a": ["x", "y"]}),
    )
    spec = chart.to_dict()
    assert spec["datasets"][spec["data"]["name"]] == [{"a": "x"}, {"a": "y"}]
    assert spec["config"]["title"]["anchor"] == "start"


def test_export_session_round_trips_with_edits():
    session = _wdi_upload().session
    qid = session.queries[0].id
    now = datetime(2026, 9, 28, tzinfo=UTC)
    again = pg.load_session(
        pg.export_session(session, {qid: pg.ChartAnnotations(title="Edited")}, now)
    )
    assert again.generated_at == now.isoformat(timespec="seconds")
    assert again.queries[0].chart_spec["title"] == {"text": "Edited"}
    assert again.queries[1].chart_spec == session.queries[1].chart_spec
    assert [q.id for q in again.queries] == [q.id for q in session.queries]


needs_data = pytest.mark.needs_data


def test_build_names_the_fix_when_files_are_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OSAA_DATA_MASTER_URL", str(tmp_path / "master.parquet"))
    monkeypatch.setenv("OSAA_DATA_META_URL", str(tmp_path / "meta.parquet"))
    with pytest.raises(ValueError, match="just data"):
        _wdi_upload().build()


@needs_data
def test_build_a_session_core_table():
    upload = _wdi_upload()
    _, core_df, unknown, indicators = upload.build()
    assert unknown == []
    assert core_df.height > 0
    assert indicators == [r.var_name for r in upload.session.schema]
    assert set(indicators) <= set(core_df.columns)


@needs_data
def test_a_session_with_an_unknown_code_is_refused():
    session = json.loads((FIXTURES / "wdi_economy.session.json").read_text())
    session["schema"].append(
        {"code": "NOT.A.CODE", "description": "made up", "var_name": "made_up"}
    )
    upload = pg.Upload.from_files([_upload("s.json", json.dumps(session))])
    with pytest.raises(ValueError, match="unknown indicator codes"):
        upload.build()


@needs_data
def test_a_csv_build_drops_and_reports_unknown_codes():
    upload = _csv_upload(_wdi_schema_csv() + "NOT.A.CODE,made up,made_up\n")
    _, core_df, unknown, _ = upload.build()
    assert unknown == ["NOT.A.CODE"]
    assert "made_up" not in core_df.columns


@needs_data
def test_a_csv_build_refuses_when_no_code_is_known():
    with pytest.raises(ValueError, match="none of the codes"):
        _csv_upload("code,description,var_name\nNOPE,x,x\n").build()


@needs_data
def test_every_saved_query_replays_on_the_rebuilt_core_table():
    upload = _wdi_upload()
    _, core_df, _, _ = upload.build()
    for q in upload.session.queries:
        records = pg.replay_sql(q.sql, core_df)
        assert records.height > 0, q.id
        pg.chart_from_spec(q.chart_spec, records).to_dict()


def test_the_notebook_runs_headless_with_nothing_uploaded():
    """With nothing uploaded the notebook runs and the build stops at its
    prompt."""
    _, defs = pg.app.run()
    assert "uploader" in defs
    assert "core_df" not in defs


def _run_with(upload, **extra):
    return pg.app.run(
        defs={
            "upload": upload,
            "build_button": SimpleNamespace(value=True),
            **extra,
        }
    )


@needs_data
def test_the_notebook_builds_an_uploaded_session_end_to_end():
    """The app's own cells, fed an uploaded session and a pressed Build button,
    rebuild the core table and offer the session back for download."""
    _, defs = _run_with(_wdi_upload())
    assert defs["core_df"].height > 0
    assert defs["unknown_codes"] == []
    grain = ["year", "iso3", "country"]
    indicators = [r.var_name for r in _wdi_upload().session.schema]
    assert defs["frozen_columns"] == grain
    assert defs["core_column_order"][: len(grain) + len(indicators)] == (
        grain + indicators
    )
    assert sorted(defs["core_column_order"]) == sorted(defs["core_df"].columns)
    assert defs["session_download"] is not None


@needs_data
def test_the_core_table_view_shows_the_core_table_summary():
    outputs, _ = _run_with(_wdi_upload())
    text = " ".join(str(getattr(o, "text", o)) for o in outputs)
    assert "core table — shape:" in text
    assert "null_percentage" in text


@needs_data
def test_the_semantic_model_view_lists_dimensions_and_measures():
    outputs, _ = _run_with(
        _wdi_upload(), nav=SimpleNamespace(value="Semantic Model"), uploader=None
    )
    text = " ".join(str(getattr(o, "text", o)) for o in outputs)
    assert "Semantic Model — <code>wdi_economy</code>" in text
    assert "Dimensions" in text and "Measures" in text


@needs_data
def test_the_semantic_model_view_reports_validation_errors():
    upload = _csv_upload(_wdi_schema_csv())
    upload = pg.Upload(
        schema_csv=upload.schema_csv,
        schema_row_count=upload.schema_row_count,
        model_yaml="m:\n  table: core_tbl\n  measures:\n    bad: _.no_such_column.sum()\n",
    )
    outputs, _ = _run_with(
        upload, nav=SimpleNamespace(value="Semantic Model"), uploader=None
    )
    text = " ".join(str(getattr(o, "text", o)) for o in outputs)
    assert "did not validate" in text
    assert "no_such_column" in text


def test_ui_strings_use_the_locked_terms():
    source = NOTEBOOK.read_text().lower()
    for banned in (
        "wide pivot",
        "schema.csv",
        "materialised table",
        "saved working set",
        "model summary",
    ):
        assert banned not in source, banned


def test_build_names_the_fix_when_the_data_is_missing(tmp_path, monkeypatch):
    """The app's Build cell, pointed at data files that do not exist, stops at
    a callout naming `just data` instead of raising."""
    monkeypatch.setenv("OSAA_DATA_MASTER_URL", str(tmp_path / "master.parquet"))
    monkeypatch.setenv("OSAA_DATA_META_URL", str(tmp_path / "meta.parquet"))
    outputs, defs = _run_with(_wdi_upload())
    assert "core_df" not in defs
    assert any("just data" in str(getattr(o, "text", o)) for o in outputs)


@needs_data
def test_a_failing_saved_query_does_not_break_the_page():
    session = json.loads((FIXTURES / "wdi_economy.session.json").read_text())
    session["queries"][0]["sql"] = "SELECT no_such_column FROM core"
    upload = pg.Upload.from_files([_upload("broken.session.json", json.dumps(session))])
    _, defs = _run_with(upload)
    assert "render_query_detail" in defs
    assert "session_download" in defs
    assert "did not run on this core table" in str(defs["render_query_detail"].text)
