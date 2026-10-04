"""osaa-metrics playground — rebuild a chat session's objects against the data.

Drop what a chat agent produced — a session.json, or a core table schema
(.csv) with an optional core-model YAML — and the app rebuilds the core table,
lists the core-model YAML's dimensions and measures, and, for a session,
replays each saved query and chart, so a person can explore, download, tune
chart titles and re-export the session.

Session parsing, the core table build and core-model YAML validation call
``osaa_metrics``'s public functions, and a session's core table is built the
way the MCP server builds it. The notebook holds the upload handling, the SQL
replay and the chart helpers, defined at the end of this file. The data is whatever ``osaa_metrics.config`` resolves: the
local ``data/`` sample by default (``just data``), or ``OSAA_DATA_MASTER_URL``
and ``OSAA_DATA_META_URL`` from ``.env``.

Run it with ``just app``.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="full",
    app_title="osaa-metrics playground",
    css_file="style.css",
)

with app.setup:
    import io
    import json
    from dataclasses import asdict, dataclass, replace
    from datetime import UTC, datetime
    from typing import Any, ClassVar, Self

    import altair as alt
    import duckdb
    import marimo as mo
    import polars as pl
    import yaml

    from osaa_metrics import (
        ModelValidationError,
        Session,
        SessionValidationError,
        build_connection,
        build_core_summary_payload_from_handle,
        build_wide_table,
        dump_session,
        enrich_core_table,
        load_embedding_cache,
        load_from_csv,
        load_session,
        load_settings,
        register_theme,
        validate_and_load_model,
    )

    # Every chart this notebook draws takes the osaa_metrics altair theme.
    register_theme()


@app.cell
def _():
    mo.md(r"""
    # osaa-metrics playground

    Drop a **session.json** from a chat session to rebuild its core table,
    its dimensions and measures, and each saved query and chart against the
    data, then explore, download, and re-export. A **core table schema
    (.csv)**, with an optional **core-model YAML**, rebuilds the core table
    and its dimensions and measures. The app starts empty.
    """)
    return


@app.cell
def _():
    uploader = mo.ui.file(
        filetypes=[".csv", ".yaml", ".yml", ".json"],
        multiple=True,
        kind="area",
        label="Drop core table schema (.csv) · core-model YAML · session.json",
    )
    nav = mo.ui.radio(
        options=["Core table", "Semantic Model", "Query & charts"],
        value="Core table",
        label="",
    )
    return nav, uploader


@app.cell
def _():
    build_button = mo.ui.run_button(
        label=f"{mo.icon('lucide:play')} Build",
        kind="success",
        full_width=True,
    )
    return (build_button,)


@app.cell
def _(uploader):
    upload = Upload.from_files(uploader.value or [])
    return (upload,)


@app.cell
def _(upload):
    # Status panel: what is loaded and which views it unlocks.
    if upload.error is not None:
        status = mo.md(f"**Load failed**\n\n```\n{upload.error}\n```").callout(
            kind="danger"
        )
    elif not upload.has_schema:
        status = mo.md("_Nothing loaded yet._").callout(kind="neutral")
    else:
        _stats = [mo.stat(str(upload.schema_row_count), "indicators")]
        if upload.has_queries:
            _stats.append(mo.stat(str(len(upload.session.queries)), "queries"))
        _views = ["✓ Core table"]
        if upload.has_model:
            _views.append("✓ Semantic Model")
        if upload.has_queries:
            _views.append("✓ Query & charts")
        status = mo.vstack(
            [
                mo.hstack(_stats, widths="equal", gap=0.5),
                mo.md("  \n".join(_views)),
                mo.md(f"<small>{' · '.join(upload.messages)}</small>"),
            ],
            gap=0.4,
        )
    return (status,)


@app.cell
def _(build_button, nav, session_download, status, uploader):
    def _section(icon, title, *body):
        # Tight internal grouping so the header reads as belonging to its body.
        return mo.vstack([mo.md(f"**{mo.icon(icon)} {title}**"), *body], gap=0.35)

    mo.sidebar(
        # One vstack with a wide between-section gap; each section groups tightly
        # internally, so what belongs with what is obvious.
        mo.vstack(
            [
                mo.md(
                    f"### {mo.icon('lucide:leaf')} osaa-metrics\n\n"
                    "<small>playground</small>"
                ),
                _section("lucide:upload", "Load", uploader, build_button),
                _section("lucide:activity", "Status", status),
                _section("lucide:layout-dashboard", "View", nav),
            ],
            gap=1.5,
        ),
        footer=_section("lucide:download", "Export", session_download),
        width=300,
    )
    return


@app.cell
def _(build_button, upload):
    # The build reads the data and runs the pivot, so it waits for an explicit
    # press of Build.
    mo.stop(not upload.has_schema, mo.md("Load a session, then press **Build**."))
    mo.stop(
        not build_button.value,
        mo.md("Loaded. Press **Build** to rebuild the core table from the data."),
    )
    try:
        core_tbl, core_df, unknown_codes, indicator_columns = upload.build()
    except ValueError as _exc:
        mo.stop(True, mo.md(str(_exc)).callout(kind="danger"))
    return core_df, core_tbl, indicator_columns, unknown_codes


@app.cell
def _(core_df, indicator_columns):
    # The core table view freezes the grain (year x country) on the left and
    # puts the indicators right after it, ahead of the other dimension columns.
    frozen_columns = [c for c in ("year", "iso3", "country") if c in core_df.columns]
    core_column_order = [
        *frozen_columns,
        *indicator_columns,
        *(
            c
            for c in core_df.columns
            if c not in frozen_columns and c not in indicator_columns
        ),
    ]
    return core_column_order, frozen_columns


@app.cell
def _(
    core_column_order,
    core_df,
    core_tbl,
    frozen_columns,
    nav,
    query_selector,
    render_query_detail,
    unknown_codes,
    upload,
):
    # Main pane: swaps on the sidebar's view choice. Each branch renders only
    # what the loaded files unlock.
    if nav.value == "Core table":
        _unknown_note = (
            mo.md(
                f"**{len(unknown_codes)} code(s) are not in the catalogue** and "
                f"were left out of the core table: `{', '.join(unknown_codes)}`"
            ).callout(kind="warn")
            if unknown_codes
            else mo.md("")
        )
        main_pane = mo.vstack(
            [
                mo.md(
                    f"## Core table\n\n**{core_df.height:,} rows × "
                    f"{core_df.width} columns**. Year, ISO3 code and country are "
                    f"frozen left, the indicators follow; sort and download are "
                    f"built into the table."
                ),
                _unknown_note,
                mo.ui.table(
                    core_df.select(core_column_order),
                    freeze_columns_left=frozen_columns,
                    show_download=True,
                    max_columns=core_df.width,
                    selection=None,
                ),
                mo.md("### Core table summary"),
                mo.md(build_core_summary_payload_from_handle(core_tbl)),
            ]
        )
    elif nav.value == "Semantic Model":
        if not upload.has_model:
            main_pane = mo.md(
                "## Semantic Model\n\n_Load a core-model YAML (or a session.json) "
                "to see the dimensions and measures you can query._"
            ).callout(kind="neutral")
        else:
            try:
                _, _, _summary, _name = validate_and_load_model(
                    upload.model_yaml, core_tbl
                )
            except ModelValidationError as _exc:
                main_pane = mo.md(
                    "## Semantic Model\n\n**The core-model YAML did not validate "
                    "against this core table**\n\n```\n"
                    + "\n".join(f"{e['path']}: {e['message']}" for e in _exc.errors)
                    + "\n```"
                ).callout(kind="danger")
            else:
                main_pane = mo.vstack(
                    [
                        mo.md(f"## Semantic Model — `{_name}`"),
                        mo.md(f"**Dimensions** · {len(_summary['dimensions'])}"),
                        mo.ui.table(
                            _summary["dimensions"],
                            selection=None,
                            show_download=False,
                            pagination=False,
                        ),
                        mo.md(f"**Measures** · {len(_summary['measures'])}"),
                        mo.ui.table(
                            _summary["measures"],
                            selection=None,
                            show_download=False,
                            pagination=False,
                        ),
                    ]
                )
    elif not upload.has_queries:
        main_pane = mo.md(
            "## Query & charts\n\n_Load a session.json to replay each "
            "query + chart combo._"
        ).callout(kind="neutral")
    else:
        main_pane = mo.vstack(
            [
                mo.md("## Query & charts"),
                mo.md(
                    "Every query + chart combo from the session. Pick one to "
                    "explore its chart, records, SQL, and annotations."
                ),
                mo.hstack(
                    [query_selector, render_query_detail],
                    widths=[1, 3],
                    gap=1.5,
                    align="start",
                ),
            ],
            gap=1.0,
        )
    main_pane
    return


@app.cell
def _(upload):
    # Selector listing every query + chart combo in the session.
    if upload.has_queries:
        _options = {
            f"{q.id} · {q.label or '(untitled)'}": q.id for q in upload.session.queries
        }
        query_selector = mo.ui.radio(
            options=_options,
            value=next(iter(_options)),
            label="Charts",
        )
    else:
        query_selector = mo.md("")
    return (query_selector,)


@app.cell
def _(query_selector, upload):
    # The selected query + chart pair, resolved from the selector.
    selected_query = None
    if upload.has_queries and getattr(query_selector, "value", None):
        selected_query = next(
            (q for q in upload.session.queries if q.id == query_selector.value), None
        )
    return (selected_query,)


@app.cell
def _(get_annotations, selected_query):
    # Annotation editor: a fielded form over vega-lite-native chart_spec fields.
    # Created here; its .value is read in the sibling render cell below (a UI
    # element's value can't be read in the cell that creates it).
    if selected_query is not None:
        _stored = get_annotations().get(selected_query.id)
        _current = ChartAnnotations.from_spec(
            _stored.apply_to(selected_query.chart_spec)
            if _stored
            else selected_query.chart_spec
        )
        annotation_form = (
            mo.md("{title}\n\n{subtitle}\n\n{x_title}\n\n{y_title}\n\n{description}")
            .batch(
                title=mo.ui.text(
                    value=_current.title, label="Chart title", full_width=True
                ),
                subtitle=mo.ui.text(
                    value=_current.subtitle, label="Subtitle", full_width=True
                ),
                x_title=mo.ui.text(
                    value=_current.x_title, label="X-axis title", full_width=True
                ),
                y_title=mo.ui.text(
                    value=_current.y_title, label="Y-axis title", full_width=True
                ),
                description=mo.ui.text_area(
                    value=_current.description,
                    label="Description",
                    rows=2,
                    full_width=True,
                ),
            )
            .form(submit_button_label="Apply", bordered=True)
        )
    else:
        annotation_form = mo.md("")
    return (annotation_form,)


@app.cell
def _(annotation_form, core_df, get_annotations, selected_query, set_annotations):
    # Detail view for the selected combo: Chart / Records / SQL / Annotate.
    # Reads annotation_form.value (its sibling cell created it) and stores the
    # fields in force for this query in the annotations state, which the
    # re-export reads.
    if selected_query is None:
        render_query_detail = mo.md("")
    elif not selected_query.sql:
        # The session schema allows a null sql; there is nothing to replay.
        render_query_detail = mo.md(
            f"**{selected_query.label or selected_query.id}** has no saved SQL, "
            "so there is nothing to replay. Its `query_args` are still in the "
            "uploaded session."
        )
    else:
        _sql_tab = mo.vstack(
            [
                mo.md("**Compiled SQL** (read-only), replayed as saved."),
                mo.ui.code_editor(
                    value=selected_query.sql,
                    language="sql",
                    disabled=True,
                ),
                mo.md("**query_args**"),
                mo.ui.code_editor(
                    value=json.dumps(selected_query.query_args, indent=2),
                    language="json",
                    disabled=True,
                ),
            ]
        )
        try:
            _records = replay_sql(selected_query.sql, core_df)
        except duckdb.Error as _exc:
            # One failing query shows its error; the rest of the page stands.
            render_query_detail = mo.vstack(
                [
                    mo.md(
                        f"**{selected_query.label or selected_query.id}**: the "
                        f"saved SQL did not run on this core table.\n\n"
                        f"```\n{_exc}\n```"
                    ).callout(kind="danger"),
                    _sql_tab,
                ]
            )
        else:
            # A form re-created on returning to a query has no value yet; the
            # edits applied earlier stand.
            _form_value = getattr(annotation_form, "value", None)
            _annotations = (
                ChartAnnotations(**_form_value)
                if _form_value
                else get_annotations().get(selected_query.id)
            )
            if (
                _annotations is not None
                and get_annotations().get(selected_query.id) != _annotations
            ):
                set_annotations(
                    lambda store: {**store, selected_query.id: _annotations}
                )
            _spec = (
                _annotations.apply_to(selected_query.chart_spec)
                if _annotations
                else selected_query.chart_spec
            )
            _chart_tab = mo.vstack(
                [
                    mo.ui.altair_chart(chart_from_spec(_spec, _records)),
                    mo.md(
                        "_Use the chart's native **⋯** menu (top-right) → "
                        "**Save as PNG / SVG**._"
                    ),
                ]
            )
            _records_tab = mo.ui.table(_records, selection=None, show_download=True)
            _annotate_tab = mo.vstack(
                [
                    mo.md(
                        "Edit the fields and click **Apply**: the **Chart** tab "
                        "re-renders and the edits are kept on re-export."
                    ),
                    annotation_form,
                ]
            )
            render_query_detail = mo.ui.tabs(
                {
                    "Chart": _chart_tab,
                    "Records": _records_tab,
                    "SQL": _sql_tab,
                    "Annotate": _annotate_tab,
                }
            )
    return (render_query_detail,)


@app.cell
def _():
    # Annotation edits, keyed by query id. mo.state because the map
    # accumulates across queries and marimo does not track dict mutation
    # across cells. The detail view writes it; the annotation form and the
    # re-export read it.
    get_annotations, set_annotations = mo.state({})
    return get_annotations, set_annotations


@app.cell
def _(get_annotations, upload):
    # Re-export session.json with the annotation edits folded into chart_spec.
    # The bytes are built when the cell runs, from the edits stored then.
    if upload.has_queries:
        session_download = mo.download(
            data=export_session(
                upload.session, get_annotations(), datetime.now(UTC)
            ).encode("utf-8"),
            filename=upload.download_filename,
            mimetype="application/json",
        )
    else:
        session_download = mo.md("_Load a session.json to re-export._")
    return (session_download,)


@app.class_definition
@dataclass(frozen=True)
class Upload:
    """The dropped files, sorted and parsed, and what they unlock.

    A session.json carries the core table schema and, when it has one, the
    core-model YAML; without a session, a core table schema (.csv) and an
    optional core-model YAML stand alone. ``error`` holds the reason the
    files cannot be used, when there is one.
    """

    # Each upload is decoded whole into memory, so larger files are refused.
    # A session.json, core table schema or core-model YAML from a chat session
    # is far below this.
    MAX_BYTES: ClassVar[int] = 4 * 1024 * 1024

    session: Session | None = None
    schema_csv: str | None = None
    model_yaml: str | None = None
    schema_row_count: int | None = None
    error: str | None = None
    messages: tuple[str, ...] = ()

    @classmethod
    def from_files(cls, files) -> Self:
        """Sort files by extension, one note per file in ``messages``, and
        parse what they hold; a session.json takes precedence over a
        separately dropped core table schema or core-model YAML."""
        texts: dict[str, str] = {}
        messages: list[str] = []
        for item in files:
            if len(item.contents) > cls.MAX_BYTES:
                messages.append(
                    f"{item.name}: skipped, larger than "
                    f"{cls.MAX_BYTES // (1024 * 1024)} MiB"
                )
                continue
            try:
                text = item.contents.decode("utf-8")
            except UnicodeDecodeError:
                messages.append(f"{item.name}: skipped, not UTF-8 text")
                continue
            lower = item.name.lower()
            if lower.endswith(".json"):
                texts["session"] = text
                messages.append(f"session: {item.name}")
            elif lower.endswith((".yaml", ".yml")):
                texts["model_yaml"] = text
                messages.append(f"core-model YAML: {item.name}")
            elif lower.endswith(".csv"):
                texts["schema_csv"] = text
                messages.append(f"core table schema: {item.name}")
            else:
                messages.append(f"ignored (unknown type): {item.name}")
        messages = tuple(messages)

        if "session" in texts:
            try:
                session = load_session(texts["session"])
            except SessionValidationError as exc:
                return cls(
                    error="session.json failed validation:\n"
                    + "\n".join(f"  {e['path']}: {e['message']}" for e in exc.errors),
                    messages=messages,
                )
            return cls(
                session=session,
                model_yaml=session.yaml,
                schema_row_count=len(session.schema),
                messages=messages,
            )
        if "schema_csv" in texts:
            try:
                rows = pl.read_csv(texts["schema_csv"].encode("utf-8")).height
            except pl.exceptions.PolarsError as exc:
                return cls(
                    error=f"core table schema could not be read: {exc}",
                    messages=messages,
                )
            return cls(
                schema_csv=texts["schema_csv"],
                model_yaml=texts.get("model_yaml"),
                schema_row_count=rows,
                messages=messages,
            )
        return cls(model_yaml=texts.get("model_yaml"), messages=messages)

    @property
    def has_schema(self) -> bool:
        return self.schema_row_count is not None and self.error is None

    @property
    def has_model(self) -> bool:
        return self.has_schema and self.model_yaml is not None

    @property
    def has_queries(self) -> bool:
        return self.has_schema and self.session is not None

    @property
    def download_filename(self) -> str:
        """``<model name>.session.json``, the model name being the core-model
        YAML's top-level key; ``session.session.json`` when there is no YAML
        or it names no model."""
        try:
            doc = yaml.safe_load(self.model_yaml) if self.model_yaml else None
        except yaml.YAMLError:
            doc = None
        name = next(iter(doc)) if isinstance(doc, dict) and doc else "session"
        return f"{name}.session.json"

    def build(self):
        """Build the core table on a connection ``osaa_metrics.config`` opens.

        Returns ``(core_tbl, core_df, unknown_codes, indicator_columns)``. A
        session's schema goes through ``enrich_core_table``, which refuses a
        code the catalogue lacks, as the MCP server's session load does; a
        core table schema (.csv) goes through ``load_from_csv``, whose unknown
        codes are left out of the build and returned. ``indicator_columns``
        are the built indicators' var_names, in schema order.
        Raises ValueError, with a message naming the fix, when the data cannot
        be opened or no code is in the catalogue.
        """
        settings = load_settings()
        try:
            con = build_connection(settings)
        except Exception as exc:
            raise ValueError(
                f"cannot open the data ({settings.data_master_url}, "
                f"{settings.data_meta_url}): {exc}. Run `just data` to download "
                "the sample, or point OSAA_DATA_MASTER_URL and OSAA_DATA_META_URL "
                "at your files in .env."
            ) from exc
        catalogue = load_embedding_cache(con=con)
        if self.session is not None:
            indicators = enrich_core_table(
                [asdict(row) for row in self.session.schema], cache_df=catalogue
            )
            unknown = []
        else:
            indicators, unknown = load_from_csv(
                io.BytesIO(self.schema_csv.encode("utf-8")), catalogue
            )
        if indicators.is_empty():
            raise ValueError(
                "none of the codes in the core table schema are in the catalogue: "
                f"{unknown}"
            )
        core_tbl = build_wide_table(indicators, con=con)
        core_df = core_tbl.to_polars()
        return core_tbl, core_df, unknown, indicators["var_name"].to_list()


@app.class_definition
@dataclass(frozen=True)
class ChartAnnotations:
    """The chart fields a person can edit, all vega-lite-native: the title
    and subtitle, the x- and y-axis titles, and the description."""

    title: str = ""
    subtitle: str = ""
    x_title: str = ""
    y_title: str = ""
    description: str = ""

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> Self:
        """Read the fields' current values out of a chart spec."""
        title_obj = spec.get("title")
        if isinstance(title_obj, dict):
            title, subtitle = title_obj.get("text", ""), title_obj.get("subtitle", "")
        elif isinstance(title_obj, str):
            title, subtitle = title_obj, ""
        else:
            title, subtitle = "", ""
        encoding = spec.get("encoding", {})
        return cls(
            title=title or "",
            subtitle=subtitle or "",
            x_title=(encoding.get("x") or {}).get("title") or "",
            y_title=(encoding.get("y") or {}).get("title") or "",
            description=spec.get("description") or "",
        )

    def apply_to(self, base_spec: dict[str, Any]) -> dict[str, Any]:
        """A copy of ``base_spec`` carrying these fields; the base is untouched.

        A blank chart title clears the title, and a blank description clears
        the description; blank axis titles keep the base spec's.
        """
        spec = dict(base_spec)

        title, subtitle = self.title.strip(), self.subtitle.strip()
        if subtitle:
            spec["title"] = {"text": title, "subtitle": subtitle}
        elif title:
            spec["title"] = {"text": title}
        else:
            spec.pop("title", None)

        # Copy each channel so x/y title edits don't mutate the base spec.
        # Channels are usually field-def dicts, but vega-lite also allows list
        # channels (tooltip, detail, order) — pass those through untouched
        # rather than dict()-ing them (a spec may carry tooltip as a list of
        # field defs, and dict() on a list of dicts raises).
        encoding = {
            channel: (dict(enc) if isinstance(enc, dict) else enc)
            for channel, enc in spec.get("encoding", {}).items()
        }
        if "x" in encoding and self.x_title.strip():
            encoding["x"]["title"] = self.x_title.strip()
        if "y" in encoding and self.y_title.strip():
            encoding["y"]["title"] = self.y_title.strip()
        if encoding:
            spec["encoding"] = encoding

        if self.description.strip():
            spec["description"] = self.description.strip()
        else:
            spec.pop("description", None)
        return spec


@app.function
def replay_sql(sql, core_df):
    """Re-run a query's saved SQL against the rebuilt ``core`` table.

    Registers ``core_df`` as the ``core`` relation on a fresh connection and
    executes the session's compiled SQL verbatim.
    """
    # The SQL arrives in an uploaded session.json, so it is untrusted input.
    # Without this, DuckDB's defaults let a query read arbitrary local files
    # (read_csv, read_text, ATTACH, COPY) and load extensions. The flag is
    # one-way for the life of the connection — a query cannot turn it back
    # on. core_df is registered from memory, so nothing legitimate here needs
    # file access.
    con = duckdb.connect(":memory:", config={"enable_external_access": False})
    try:
        con.register("core", core_df)
        records = con.execute(sql).pl()
    finally:
        con.close()
    return records


@app.function
def chart_from_spec(chart_spec, records):
    """Build an altair Chart from a vega-lite spec with records injected.

    The records replace any ``data`` the spec carries, as inline values.
    """
    spec = dict(chart_spec)
    spec["data"] = {"values": records.to_dicts()}
    spec.setdefault("width", "container")
    spec.setdefault("height", 340)
    return alt.Chart.from_dict(spec)


@app.function
def export_session(session, edits, now):
    """The session as JSON text, with each query's annotation edits folded
    into its chart spec and ``generated_at`` set to ``now``."""
    updated = replace(
        session,
        generated_at=now.isoformat(timespec="seconds"),
        queries=[
            replace(q, chart_spec=edits[q.id].apply_to(q.chart_spec))
            if q.id in edits
            else q
            for q in session.queries
        ],
    )
    return dump_session(updated) + "\n"


if __name__ == "__main__":
    app.run()
