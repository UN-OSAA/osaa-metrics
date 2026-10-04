# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23",
#     "osaa-metrics",
#     "polars",
#     "pandas",
#     "altair",
#     "matplotlib",
#     "statsmodels",
#     "scipy",
# ]
#
# [tool.uv.sources]
# osaa-metrics = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From a chat session to a regression

    An analyst explored a question with the osaa-metrics agent: *is access to electricity
    associated with less extreme poverty?* The agent found the indicators, built a
    **core table**, wrote a **core-model YAML** (the dimensions and measures) and saved
    three charts. All of it was downloaded as `electricity_poverty.session.json`.

    This notebook picks the work up from that file:

    1. restore the core table and the **core semantic table** from the session
    2. check the core table and replay the saved charts, exactly as the chat drew them
    3. keep going: pull the regression data from the core semantic table, by the same
       measure names, and fit a panel regression with its standard diagnostics

    **Data.** World Bank World Development Indicators, from the osaa-metrics sample.

    **Run it** from the repository root: `just example electricity_poverty`.
    """)
    return


@app.cell
def _(mo):
    import json
    import logging
    import os

    import altair as alt
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import statsmodels.formula.api as smf
    from marimo._loggers import marimo_logger
    from scipy import stats

    from osaa_metrics import (
        build_connection,
        build_core_summary_payload_from_handle,
        build_wide_table,
        enrich_core_table,
        load_embedding_cache,
        load_session,
        load_settings,
        validate_and_load_model,
    )

    class _DatasetPanelNoise(logging.Filter):
        """Drop the error marimo's dataset panel logs when it inspects a core
        semantic table; the panel cannot read its schema, the notebook is unaffected."""

        def filter(self, record):
            return "get_datasets" not in record.pathname

    for _handler in marimo_logger().handlers:
        _handler.addFilter(_DatasetPanelNoise())

    EXAMPLE_DIR = mo.notebook_dir()
    REPO_DIR = EXAMPLE_DIR.parents[1]
    # OSAA_DATA_* from the environment win; otherwise the repository's local
    # data/ mirror (what `just data` downloads) is used.
    settings = load_settings(
        env={
            "OSAA_DATA_MASTER_URL": str(REPO_DIR / "data" / "master.parquet"),
            "OSAA_DATA_META_URL": str(REPO_DIR / "data" / "meta.parquet"),
            **os.environ,
        }
    )
    con = build_connection(settings)
    return (
        EXAMPLE_DIR,
        alt,
        build_core_summary_payload_from_handle,
        build_wide_table,
        con,
        enrich_core_table,
        json,
        load_embedding_cache,
        load_session,
        np,
        pl,
        plt,
        smf,
        stats,
        validate_and_load_model,
    )


@app.cell
def _(
    EXAMPLE_DIR,
    build_wide_table,
    con,
    enrich_core_table,
    load_embedding_cache,
    load_session,
    mo,
    pl,
    validate_and_load_model,
):
    session = load_session(
        (EXAMPLE_DIR / "electricity_poverty.session.json").read_text()
    )
    core_table_schema = pl.DataFrame(
        [
            {"code": r.code, "var_name": r.var_name, "description": r.description}
            for r in session.schema
        ]
    )
    core_tbl = build_wide_table(
        enrich_core_table(
            core_table_schema.to_dicts(), cache_df=load_embedding_cache(con=con)
        ),
        con=con,
    )
    core_semantic_table, _yaml_text, _model_summary, model_name = (
        validate_and_load_model(session.yaml, core_tbl)
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ## 1. Restore the session

    The session carries the **core table schema** ({core_table_schema.height} indicators),
    the core-model YAML and {len(session.queries)} saved charts. The core table is rebuilt
    from the schema, and the YAML loads into the core semantic table `{model_name}`.
    """
            ),
            core_table_schema,
        ]
    )
    return core_semantic_table, core_tbl, session


@app.cell
def _(build_core_summary_payload_from_handle, core_tbl, mo):
    mo.vstack(
        [
            mo.md(
                """
    ## 2. Check the core table

    The same summary the agent reads before it writes a core-model YAML: one row per
    column, with its range, spread and share of missing values. Poverty is observed
    far less often than the other indicators, because it comes from household surveys.
    """
            ),
            mo.md(build_core_summary_payload_from_handle(core_tbl)),
        ]
    )
    return


@app.cell
def _(alt, core_semantic_table, json, mo, session):
    def _replay(query):
        args = query.query_args
        rows = core_semantic_table.query(
            dimensions=args.get("dimensions"),
            measures=args.get("measures"),
            filters=args.get("filters") or None,
            order_by=[tuple(o) for o in args["order_by"]]
            if args.get("order_by")
            else None,
            limit=args.get("limit"),
        ).execute()
        values = json.loads(rows.to_json(orient="records"))
        return alt.Chart.from_dict({**query.chart_spec, "data": {"values": values}})

    mo.vstack(
        [
            mo.md(
                """
    ## 3. Replay the saved charts

    Each saved chart stores its query (dimensions, measures, filters) and its chart
    spec. Running the query against the core semantic table and drawing the spec gives
    back the chart from the chat.
    """
            ),
            *[
                mo.vstack([mo.md(f"**{q.label}**"), _replay(q)])
                for q in session.queries
            ],
        ]
    )
    return


@app.cell
def _(core_semantic_table, mo):
    MEASURES = [
        "avg_poverty_headcount",
        "avg_electricity_access",
        "avg_gdp_per_capita_ppp",
        "avg_rural_population",
    ]
    _by_country_year = (
        core_semantic_table.query(dimensions=["country", "year"], measures=MEASURES)
        .execute()
        .dropna(subset=MEASURES)
    )
    # A country with a single complete country-year carries no within-country
    # variation, so the fixed-effects model cannot use it; both models use the
    # same sample.
    _rows_per_country = _by_country_year.groupby("country")["year"].transform("size")
    regression_data = _by_country_year[_rows_per_country >= 2]
    mo.md(
        f"""
    ## 4. Keep going: from the core semantic table to a regression

    The regression data comes from the same core semantic table, queried by country and
    year with the measures the chat used: `{"`, `".join(MEASURES)}`. With one row per
    country and year, each average is the indicator's value for that country-year.

    {len(_by_country_year):,} country-years have all four measures. Countries with only
    one of them are dropped, leaving **{len(regression_data):,} rows** from
    {regression_data["country"].nunique()} countries.
    """
    )
    return (regression_data,)


@app.cell
def _(mo, np, pl, regression_data, smf):
    FORMULA = (
        "avg_poverty_headcount ~ avg_electricity_access"
        " + np.log(avg_gdp_per_capita_ppp) + avg_rural_population"
    )
    _clustered = {
        "cov_type": "cluster",
        "cov_kwds": {"groups": regression_data["country"]},
    }
    _env = {"np": np}
    pooled = smf.ols(FORMULA, regression_data, eval_env=_env).fit(**_clustered)
    fixed_effects = smf.ols(
        FORMULA + " + C(country) + C(year)", regression_data, eval_env=_env
    ).fit(**_clustered)

    def _row(name, fit):
        low, high = fit.conf_int().loc["avg_electricity_access"]
        return {
            "model": name,
            "electricity_access": fit.params["avg_electricity_access"],
            "ci_low": low,
            "ci_high": high,
            "p_value": fit.pvalues["avg_electricity_access"],
            "n": int(fit.nobs),
        }

    comparison = pl.DataFrame(
        [
            _row("Pooled OLS", pooled),
            _row("Country + year fixed effects", fixed_effects),
        ]
    )
    mo.vstack(
        [
            mo.md(
                """
    Both models regress the poverty headcount on electricity access, log GDP per capita
    and the rural share. **Pooled OLS** compares all country-years with each other.
    **Fixed effects** add a dummy for every country and every year, so each country is
    compared only with itself and shocks common to all countries in a year are removed.
    Standard errors are clustered by country in both.
    """
            ),
            comparison.with_columns(pl.selectors.float().round(3)),
        ]
    )
    return fixed_effects, pooled


@app.cell
def _(fixed_effects, mo, plt, stats):
    _fitted = fixed_effects.fittedvalues
    _resid = fixed_effects.resid
    _influence = fixed_effects.get_influence()
    _std_resid = _influence.resid_studentized_internal
    _leverage = _influence.hat_matrix_diag

    diagnostics_fig, _axes = plt.subplots(2, 2, figsize=(11, 8))
    _axes[0, 0].scatter(_fitted, _resid, s=8, alpha=0.4)
    _axes[0, 0].axhline(0, color="grey", linewidth=1)
    _axes[0, 0].set(title="Residuals vs fitted", xlabel="Fitted", ylabel="Residual")
    stats.probplot(_resid, dist="norm", plot=_axes[0, 1])
    _axes[0, 1].set_title("Normal Q-Q")
    _axes[1, 0].scatter(_fitted, abs(_std_resid) ** 0.5, s=8, alpha=0.4)
    _axes[1, 0].set(
        title="Scale-location", xlabel="Fitted", ylabel="sqrt(|standardised residual|)"
    )
    _axes[1, 1].scatter(_leverage, _std_resid, s=8, alpha=0.4)
    _axes[1, 1].set(
        title="Residuals vs leverage", xlabel="Leverage", ylabel="Standardised residual"
    )
    diagnostics_fig.suptitle("Diagnostics: country + year fixed effects model")
    diagnostics_fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                "### Diagnostics\nThe four standard residual plots for the fixed-effects model."
            ),
            diagnostics_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(fixed_effects, mo, pooled):
    def _summary(fit):
        low, high = fit.conf_int().loc["avg_electricity_access"]
        points = fit.params["avg_electricity_access"] * 10
        if high < 0:
            verdict = "less poverty, and the 95% interval excludes zero"
        elif low > 0:
            verdict = "more poverty, and the 95% interval excludes zero"
        else:
            verdict = "no clear direction: the 95% interval includes zero"
        return f"{points:+.1f} points of poverty per 10 points of access ({verdict})"

    mo.md(
        f"""
    ## Takeaway

    Holding log income and the rural share fixed:

    - all countries compared together: {_summary(pooled)}
    - each country compared only with itself: {_summary(fixed_effects)}

    The difference suggests that part of the pooled association reflects differences
    between countries or trends shared by all countries, which the fixed-effects model
    removes. These are associations, not causal effects: electrification and poverty
    reduction are both driven by growth and public investment that the controls capture
    only in part, and poverty is observed only in survey years.

    Every variable above was a named measure of the core semantic table: the definitions
    written in the chat carried into the regression unchanged.
    """
    )
    return


if __name__ == "__main__":
    app.run()
