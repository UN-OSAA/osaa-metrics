# Agent instructions — osaa-metrics

A Python library plus an MCP server for analysing a catalogue of official statistics
from a chat client. Two skills under `skills/` carry the procedures; read the one
that matches the task before acting.

- `skills/osaa-setup/` — install and connect. Use it when asked to set up,
  install, or connect osaa-metrics.
- `skills/osaa-metrics/` — the analyst playbook: tool order, vocabulary, state
  rules. It applies whenever the osaa-metrics MCP server is attached.

## Install

From the repository root: `just setup`, `just data`, `just encoder`, `just check`.
`just encoder` downloads the model files (about 2.3 GB) once; discovery ranks results by
meaning from then on, and the server loads the model when it starts.

## Data

The sample is two parquet files published on Hugging Face (`spencerlima/osaa-metrics`,
CC-BY-4.0; the indicators come from the World Bank and UNESCO's Institute for
Statistics — credit them when you share the data). `just data` downloads them into
`data/`, where the server looks by default. Other files: `OSAA_DATA_MASTER_URL` and
`OSAA_DATA_META_URL` in `.env`. The `meta` file must carry a 1024-number `embedding`
per indicator; without it nothing is discoverable.

## Vocabulary

The terms in the playbook's vocabulary table are the only names for those concepts.
Do not coin new ones.
