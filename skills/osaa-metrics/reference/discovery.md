# Discovery surface

> separable — moves with `mcp-discovery`. When mcp-discovery is extracted to
> its own repo, this file becomes that repo's skill.

Open this file when the user attaches a CSV of indicators.

## Workflow

1. Call `open_discovery(initial_semantic_query, initial_keyword_query)` (both
   optional). The tool mounts an iframe; control returns to you at once.
2. The user picks indicators in the discovery widget and presses **Save**.
   The widget calls `save_core_schema(working_set)` itself; its reply
   goes to the widget, not to you.
3. When the user says they saved, or asks for the next step, call
   `build_core_table()`. It builds from the saved core table schema,
   or its error says no core table schema is saved yet.

## An attached CSV

A core table schema CSV has the columns `code`, `description` and
`var_name`. When the user attaches one, call
`save_core_schema(working_set=<its rows>)` yourself, then `build_core_table()`.

## Anti-patterns

- Do not call `discover_indicators(...)` yourself. The widget calls it
  internally; calling it from the agent bypasses the widget.
- Do not pick indicators and call `save_core_schema(...)` yourself, except
  when the user attaches a CSV (see above); otherwise the user picks them in
  the discovery widget.

## Errors

`save_core_schema` rejects rows that break its rules, for example a
`var_name` that does not match `^[a-z_][a-z0-9_]*$`, a duplicate `var_name`,
or a row with no `code` or `var_name`. A missing `description` is stored as
empty. Errors arrive as `ToolError` envelopes with a human-readable message.
