---
name: osaa-setup
description: Install and connect osaa-metrics on this machine — dependencies, sample data, the model files, and the Claude Desktop connection. Use when asked to set up, install, or connect osaa-metrics, or when `just check` reports a problem.
---

# osaa-metrics — setup runbook

Follow the steps in order. Each names what success and failure look like. On a
failure, stop and show the user the output; do not improvise a workaround.

Run every command from the repository root — the directory that holds `justfile`.

## 1. Tools

This machine must be one the README's "What you need" section supports, `uv`
and `just` must both be present, and the Python floor in `pyproject.toml`
(`requires-python`) must be satisfiable here.

1. **Check the machine.** Read "What you need" in `README.md`. If it says the
   install cannot run here, stop and tell the user why. Otherwise, tell the
   user what this machine can and cannot do before continuing.
2. **Check the tools.** Run `uv --version` and `just --version`. Read
   `requires-python` from `pyproject.toml` rather than assuming a number.
3. **If a tool is missing**, tell the user what you would install and how,
   for this machine, and ask for approval before installing anything. The
   "how" follows the installation instructions on the official pages for
   `uv` and `just` that "What you need" in `README.md` links to.
4. **On approval**, install, then re-check. **On refusal**, stop and tell the
   user setup cannot continue.

If a tool reports a version but a later step cannot find it, that is a `PATH`
problem on this machine and the user resolves it.

## 2. Dependencies

`just setup` — checks that `uv` is installed, then runs
`uv sync --frozen --extra mcp --extra encoder`.

- Success: near the end of the output, a line reads `Installed N packages in ...`
  (a fresh or changed install — one `+ package==version` line per package follows it,
  so that line is not the last one on screen) or `Checked N packages in ...` (already
  up to date — nothing follows it).
- Failure: a uv error naming a package or a Python version. Show it.
- Linux without a GPU: the sync downloads several GB more than the model files,
  because the runtime's default build targets GPUs. Warn first; it installs and works.

## 3. Sample data

`just data` — downloads `master.parquet` and `meta.parquet` into `data/` (about 75 MB).

- Success: both files present; a second run prints `present — skipping` for each.
- Failure: a curl error. Retry once; if it fails again, show it.

## 4. Model files

`just encoder` — downloads the BGE-M3 model files (about 2.3 GB; minutes on a fast
connection) into the model cache.

- Success: progress bars, then one line printing the folder the files landed in. A
  second run prints the folder almost at once.
- Failure: a download error. Retry once; if it fails again, show it.

## 5. Check

`just check`

- Success: a `data:` line with two `OK`s and an indicator count, then an `encoder:` line
  ending `semantic search enabled`.
- Failure: the line names the fix — a recipe to run, or a setting to change in `.env`.
  Do that, then re-check.

## 6. Connect Claude Desktop

Open the config file — macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`;
Windows: `%APPDATA%\Claude\claude_desktop_config.json`. Add this entry, with the absolute path of the
repository:

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

`--frozen` and `--no-sync` are required: they keep a Desktop launch from rewriting
`uv.lock` or reshaping `.venv`.

Quit Claude Desktop fully and reopen it.

- Success: `osaa-metrics` is listed among the connected servers.
- Failure: open the server's own log in Desktop's log folder. The server prints
  `loading the encoder` when it starts the model load and `encoder loaded in N s`
  when it finishes, and it answers the host only after that second line.
  - Both lines present: the server started; the problem is elsewhere in the log.
  - Only the first line: the load is still running, or it failed. A failure prints a
    warning and a traceback right after the first line. Wait, then read the log again.
  - Neither line: the server crashed before the load. Show the user the last lines of
    the log.

## 7. First call

In a new chat, ask to open indicator discovery. The discovery widget appears. The
analyst playbook lives at `skills/osaa-metrics/`. To use it as a Claude Skill, see
[Use Skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude).

Claude Desktop is the analysis surface. A client that renders no widgets cannot
complete the flow in this version.
