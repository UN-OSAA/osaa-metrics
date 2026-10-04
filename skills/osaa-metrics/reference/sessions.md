# Sessions: restore, save, and state

Open this file when the user attaches a `session.json`, asks to save their
work, or a tool says the core table is not built after you built it.

## `load_session`: validate, then write

1. Call `load_session(json_text=<file contents>)` first; it bootstraps
   straight to querying.
2. On success you get `{loaded_at, schema_row_count, model_name,
   queries_loaded}`. The core table is built, the core semantic table is
   registered under `model_name`, and earlier query+chart pairs are back.
3. On failure the message either carries a `<name>:` prefix followed by JSON
   `{"errors": [{"path", "message", "hint"}, ...]}` (for example
   `session validation failed:`, `core table schema invalid:`,
   `session yaml validation failed:`) or is a plain-text refusal with no JSON
   body (for example `session yaml requires a core table schema`, or a field
   that is too long). Narrate the issue; do not echo the JSON. After
   `session yaml validation failed:` the restore's schema has already replaced
   the core table; this session's handle and model registration are cleared.

The validator enforces `schema_version == "1"`, `var_name` matching
`^[a-z_][a-z0-9_]*$` and unique, `QueryChartPair.id` matching `^q-\d{3,}$`,
unique and increasing, and `schema` at most 100 rows. The session schema is
fetchable via `get_session_schema()` or `session-schema://osaa-metrics/v1`.

## `download_session`

- To show the session inline, render `download_session`'s `content`; do not
  write scratch files.
- To save work, call `download_session` and hand its `content` to the user.

Downloads (`session.json`, core-model YAML) are re-executable recipes, not
computed results. A session that outgrows the server's 4 MiB reload ceiling
reloads only after its queries are trimmed by hand.

## State

On a local install, session state is shared per server process:
every chat against this server shares one core table and one core semantic
table. If state looks wrong (a core table you did not build, or one you built
now missing), ask whether another chat is open against this server before
debugging. Tell the user to run one analysis chat at a time and to call
`download_session` before switching. If a tool reports the core table is not
built, or a staleness guard fires, after you built it, the cached state may
have been evicted: re-run `build_core_table` (it rebuilds from the recipe) or
restore via `load_session`.
