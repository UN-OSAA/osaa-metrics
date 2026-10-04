# Charting via `query_and_chart`

Open this file when the `chart_spec` you are about to send has any key
besides `mark`, or when the widget rendered a placeholder chart.

## Grammar

`chart_spec` and `label` are both required; the tool errors without either.
Pass a plain vega-lite spec dict, or BSL's wrapper form
`{"backend": ..., "spec": {...}, "format": ...}` (the same grammar as
`query_model`). The widget always renders altair+json.

## What BSL does with your spec

BSL auto-detects a base spec (mark + encoding) from the query's dimensions
and measures, then fills in only the **top-level** keys `mark`, `encoding`
and `transform` that you left out. A key you supply replaces the
auto-detected one whole: once you send an `encoding` block, specify every
channel and put `sort` on the relevant one (`"-x"`, `null`, or
`{field, order}`). The server carries `title` through unchanged.

## Title, subtitle, source line

Pass all three as one `title` object:

- **`text`** is the takeaway: one sentence with a verb, true of the rows
  this query returned, ten words or fewer.
- **`subtitle`** is a list of lines. Line one is the evidence: indicator
  name, unit, breakdown, filter scope, year(s). Line two is the source:
  `Source: <source>, <indicator code(s)>` from the core table schema.

The server refuses `text` over 80 characters and any subtitle line over 120,
because vega title text does not wrap. Numbers below are illustrative:

```json
"title": {
  "text": "Tax revenue has stayed below 15% of GDP in most African regions",
  "subtitle": [
    "Tax revenue (% of GDP), regional mean by year, Africa, 2010-2022",
    "Source: World Bank WDI, GC.TAX.TOTL.GD.ZS"
  ]
}
```

`label` is a different thing: a noun phrase of a few words for the saved
chart list (`tax revenue by region`). `text` is the claim; `label` is not.

## Common shapes

- `{"mark": "line"}` pins the mark and auto-detects the encoding.
- `{"mark": "bar", "encoding": {...}}` takes full control of mark and encoding.
- `{"mark": "bar", "title": {...}}` puts a title on an auto-detected chart.

## Complex shapes

| Query shape | Chart you get | Minimal spec |
|---|---|---|
| 1 dimension, any measures | bar / grouped bar | `{"mark": "bar"}` |
| time series, 1 measure | line / multi-line | `{"mark": "line"}` |
| 2 non-time dimensions, 1 measure | heatmap | `{"mark": "rect"}` |

Anything else (2+ dimensions with 2+ measures, or 3+ non-time dimensions) has
no auto-detect rule and BSL returns a placeholder. Send an explicit `mark`
and a full `encoding` block with every channel.

## The widget's tabs

**Chart** (the render), **Query** (BSL operator graph), **SQL** (compiled
SQL), **Chart Spec** (the vega-lite sent to vega-embed).
