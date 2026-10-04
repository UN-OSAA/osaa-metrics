# Worked examples

Each folder holds one analysis in the files osaa-metrics reads and writes:

- a **core table schema** (`.csv`): the indicators, one row each (`code`, `description`, `var_name`)
- a **core-model YAML** (`.yaml`): the dimensions and measures of the core semantic table
- a **session.json** (`.session.json`): the core table schema, the core-model YAML and a set of saved charts, which the `load_session` tool restores

Both examples use World Bank World Development Indicators from the sample data. Download it first with `just data`.

## electricity_poverty

Is higher access to electricity associated with less extreme poverty? The notebook
`electricity_poverty.py` continues a chat session from `electricity_poverty.session.json`:
it restores the core table and the core semantic table, replays the saved charts, then
queries the core semantic table by its measure names for the regression data and
compares a pooled regression with a country and year fixed-effects regression, with the
four standard residual plots.

Open it from the repository root:

```
just example electricity_poverty
```

The notebook declares its own dependencies in its header, so `just example` runs it in
an isolated environment and nothing is added to the project's.

## economic_transformation

Economic structure and its financing: sector shares of value added, value added per
worker, unemployment, investment, saving and external flows. The core table covers
every country; the saved charts follow Africa over time and compare it with the other
regions. It ships the core table schema, core-model YAML and session.json (`wdi_economy.*`) without a notebook.
