# osaa-metrics

A chat-native pair-analyst over a catalogue of official statistics. The Python library
(`osaa_metrics`) pairs a long "One Big Table" of indicators with a
[Boring Semantic Layer (BSL)](https://github.com/boringdata/boring-semantic-layer) model.
The MCP server exposes that to Claude Desktop, where an analyst discovers indicators in
the discovery widget, Saves a working set, builds the core table, defines a core
semantic table, and composes queries and charts at runtime — a chart factory, not a
dashboard.

## What you need

- An Apple Silicon Mac on macOS 14 or later, or Linux, with a terminal. Intel Macs
  cannot install this version: the model runtime ships no build for them. Windows is
  untested.
- [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).
- Claude Desktop, for the analysis. It runs on macOS and Windows; there is no Linux
  build, so on Linux you can install and run the server but not analyse in this version.
- About 2.3 GB of disk for the model files, plus the sample data. On Linux without a
  GPU the install downloads several GB more, because the runtime's default build
  targets GPUs. It works; it is just large.
- No account, key, or token. A Hugging Face token is optional: export `HF_TOKEN`
  in your shell and `just encoder` downloads under a higher rate limit.

## Install

```bash
git clone https://github.com/UN-OSAA/osaa-metrics.git
cd osaa-metrics
just setup    # installs the server and the encoder runtime into .venv — runs: uv sync --frozen --extra mcp --extra encoder
just data     # downloads the sample data (about 75 MB) into data/
just encoder  # downloads the BGE-M3 model files (about 2.3 GB) into the model cache
just check    # confirms the data opens and the model files are cached
```

`just check` prints the two data files, the indicator count, and the model cache line
with the folder it looked in. A failure names the recipe that fixes it.

`just app` opens the playground in your browser. Upload a session.json and it rebuilds the
core table, lists the dimensions and measures, and replays each saved query and chart
against the data (`data/` unless `.env` points elsewhere). A core table schema CSV, with
an optional core-model YAML, rebuilds the core table and lists the dimensions and measures.

The repository ships a setup runbook at
`skills/osaa-setup/`. To use it as a Claude Skill, see
[Use Skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude).

### Connect Claude Desktop

Add the server to Claude Desktop's config file, with the absolute path of the clone:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "osaa-metrics": {
      "command": "uv",
      "args": ["run", "--frozen", "--no-sync", "--directory", "/absolute/path/to/osaa-metrics", "osaa-mcp"]
    }
  }
}
```

`--frozen` and `--no-sync` keep a Desktop launch from rewriting `uv.lock` or reshaping
`.venv`. Quit and reopen Claude Desktop; `osaa-metrics` appears among the connected
servers. Start a chat and ask to open indicator discovery. The repository ships an
analyst playbook at `skills/osaa-metrics/`. To use it as a Claude Skill, see
[Use Skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude).

## The data

The sample is two parquet files published on Hugging Face as
[`spencerlima/osaa-metrics`](https://huggingface.co/datasets/spencerlima/osaa-metrics)
(CC-BY-4.0). The indicators come from the World Bank and UNESCO's Institute for
Statistics — credit them when you share the data. `just data` puts the files in `data/`,
where the server looks by default; to use other files set `OSAA_DATA_MASTER_URL` and
`OSAA_DATA_META_URL` in `.env` (a local path or a URL each).

- `master` — one row per indicator × country × year. An `indicator_code` names exactly
  one indicator across the whole table — no two indicators share one.
- `meta` — one row per indicator, with a 1024-number `embedding` per row. **Every
  discovery path filters on that column, keyword search included**, and building the
  core table accepts only codes present there. Files without embeddings surface no
  indicators; the sample carries them.
- The core table starts at year 2000 by default. An indicator with no rows from 2000
  on still gets a column, holding only NULLs; the core table summary reports its
  `null_percentage` accordingly.

## Limits worth knowing

- **Run one analysis chat at a time.** Every chat talks to the same server, and they all
  share one core table. Picking indicators in a second chat replaces what you built in the
  first. Download the session before you switch.
- **State is process memory.** Restarting Claude Desktop restarts the server empty.
  `download_session` is the save.
- **Widgets need the network.** The discovery, summary, and chart widgets load their
  libraries from public CDNs; offline they do not render.
- **`session.json` is pre-1.0.** Its `schema_version` is `1`; a later version may
  change the shape without a migration path.
- **Claude Desktop is the analysis surface.** A client that renders no widgets cannot
  complete the flow in this version.

## Repository layout

```
AGENTS.md                          the vendor-neutral agent entry point
justfile                           setup · data · encoder · check · app
src/osaa_metrics/                  the library — discovery, core_table, semantic, summarize, session, theme
  encoders/                        Encoder Protocol + the local encoder
  mcp/                             FastMCP composing server (osaa-mcp / osaa-mcp-http)
  mcp_discovery/                   discovery sub-server + the discovery widget
  _schemas/                        bsl_model.schema.json, session.schema.json
app/playground.py                  the playground notebook (`just app`)
skills/osaa-metrics/               the analyst playbook
skills/osaa-setup/                 the setup runbook
.claude/skills/osaa-*              symlinks to the two skills above
.agents/skills/osaa-*              the same, for Codex · Cursor · OpenCode
data/                              the sample data, after `just data`
```

## Library use

`from osaa_metrics import ...` exposes the substrate without the server (`[mcp]` is
optional).

## Development

```bash
uv sync --frozen --extra dev --extra mcp
OSAA_SKIP_NET=1 uv run pytest -q
```

## Licence

Apache-2.0 — see `LICENSE` and `NOTICE`.
