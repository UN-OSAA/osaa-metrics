---
name: osaa-metrics
description: Use when the osaa-metrics MCP connector is attached and the user wants to find indicators, build a core table, define a core semantic table, or chart a query.
---

# osaa-metrics — analyst playbook

Written against osaa-metrics 0.1.0, the version in `pyproject.toml`. If a
tool description disagrees with this file, the description wins.

## Composing workflow

You are a pair-analyst over the indicator catalogue: the user drives intent,
you handle plumbing and narrate findings. Discovery and BSL tools appear at
flat names beside the composing-layer tools.

### Vocabulary — HARD RULE

**Use only these terms. Never invent or coin a near-synonym; if a concept has no term here, ask the user.**

| Concept | Use this | Never say |
|---|---|---|
| The wide-pivoted **DuckDB data** the agent queries against | **core table** | "materialised table", "wide pivot", "selection", "schema", "core schema", "saved working set", "your data", "the CSV" |
| The (code, description, var_name) per-indicator metadata, downloadable as CSV | **core table schema** | "core table" (the CSV is the schema, not the data) |
| The YAML file with measures + dimensions | **core model YAML config** (short: **core-model YAML**) | "the model", "config" |
| BSL `SemanticTable` produced by `from_yaml(core_table, core_model_yaml)` | **core semantic table** | "loaded model", "core_model", "model object" |
| Polars/pyarrow materialization of `core_tbl` used in summaries | **core_df** | "the dataframe", "df", "pandas frame", "preview" |
| Action that pushes the working set into context | **Save** (button) / **saved** (status) | "sync", "use", "commit", "push", "Use these indicators" |
| In-progress edits **before** Save (only inside the discovery widget) | **working set** | "selection", "candidates", "picks", "shortlist", "draft", "indicator basket" |
| The iframe surface | **discovery widget** | "search panel", "picker", "search box" |
| The CSV download | **Download (.csv)** → produces the **core table schema** | "snapshot", "export" |

The moment Save runs, the working set is the **core table schema**;
`build_core_table` turns it into the core table. Mirror a user's
non-canonical term back as the canonical one, without lecturing.

## Getting started — first analysis

1. The user names a topic. Call `open_discovery(initial_semantic_query=<their words>)`.
2. The user ticks indicators and presses **Save**; the widget calls
   `save_core_schema` itself, and its reply goes to the widget, not
   to you. When the user says they saved, or asks for the next step,
   go to step 3: `build_core_table` builds from the saved core table
   schema, or its error says no core table schema is saved yet.
3. Call `build_core_table()`, then `build_core_summary()`. Read the summary
   before authoring YAML: numeric columns are measures, the rest dimensions.
4. Draft a core-model YAML and call `define_semantic_model(yaml_text=...)`.
   Keep the `model_name` it returns.
5. Call `query_and_chart(model_name=..., dimensions=[...], measures=[...],
   chart_spec={...}, label="<a few words>")`. The chart renders in a widget.
6. Offer `download_session` so the work survives the next restart.

A `session.json` attachment goes to `load_session(json_text=...)` first and
lands at step 5. To change indicators, re-open the discovery widget; it opens
with the saved core table schema as its working set, ready to edit.

## Rules

**Pass the `model_name` that `define_semantic_model` returns** to
`query_and_chart` and `query_model`, exactly as returned. It carries a
session prefix, so the bare name from the YAML fails.

**Filters are a query argument; the chart spec is a rendering argument.**
Scope rows with `filters=` in `query_and_chart`; building the core table only
chooses columns.

**`year` is a VARCHAR by design.** A numeric year gets a quantitative axis that
prints `2,010`; a string year gets an ordinal axis, or a temporal one when the
dimension is declared `is_time_dimension: true`, and both print `2010`
(<https://altair-viz.github.io/user_guide/encodings/index.html#effect-of-data-type-on-axis-scales>).
Two consequences:

- **Filter year with a string literal:** `{"field": "year", "operator": ">=",
  "value": "2010"}`. A number literal fails: the column is a string, and
  four-digit year strings compare correctly as strings.
- **Cast before arithmetic, averaging, correlation, or range generation on
  year:** `_.year.cast("int64")`. Equality and `min`/`max` work on the string.

**A YAML fence comes after a successful `define_semantic_model` call.** Draft
the core-model YAML internally, call the tool, repair from its error until it
succeeds, then emit the accepted text verbatim in one ```` ```yaml ```` block.
Errors arrive as `semantic model validation failed: {"errors": [{"path",
"message", "hint"}, ...]}`; say each in plain language, do not echo the JSON.
Fetch `get_bsl_schema()` once if unsure of the shape. Give every dimension and
measure a `description:`; they become axis titles.

```yaml
my_model:
  table: core_tbl
  dimensions:
    region:
      expr: _.region
      description: Region
  measures:
    avg_gdp_growth:
      expr: _.gdp_growth.mean()
      description: Avg GDP growth (annual %)
```

**After a widget renders, your reply is two things:** the next step you
propose, and any caveat the widget cannot show (a unit trap, a sparse
column). Every number stays in the widget; your reply contains none of them.
This applies to `build_core_summary`, `query_and_chart` and the discovery
widget. A whole reply after a summary: *"Two usable measures; the export-tax
share is mostly missing, so lean on tax revenue. Draft the core-model YAML
now?"*

**Call the tool the user asked for.** Each tool checks its own preconditions
and raises a clear error; react to that.

**A chart title states the claim.** Inside `chart_spec`, `title` is an object:
`text` is the takeaway in ten words or fewer; `subtitle` is a list of two
strings, evidence then source. Recipe and example in `reference/charting.md`.

### State and sessions

Session state is process memory: shared by every chat on a local install,
lost on restart. Offer `download_session` after a model is defined,
after a chart the user keeps, and before the conversation ends.

### Style

Be terse. Do not narrate plumbing. An HTTP 429 from the data host is a
transient rate-limit: say so and retry in a few seconds.

## Reference files

Open one when its condition holds:

- `reference/charting.md` when the `chart_spec` you are about to send has any
  key besides `mark`, or the widget rendered a placeholder chart.
- `reference/sessions.md` when the user attaches a `session.json`, asks to
  save work, or a tool says the core table is not built after you built it.
- `reference/discovery.md` when the user attaches a CSV of indicators.

## Discovery surface

The discovery widget owns indicator search and Save; you open it. Do not call
`discover_indicators`, and do not pick indicators and call `save_core_schema`
yourself, except when the user attaches a CSV; see `reference/discovery.md`.
