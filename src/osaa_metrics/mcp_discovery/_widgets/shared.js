// shared.js — tiny helpers used by every osaa-metrics widget.

/**
 * Parse a host CallToolResult into the structured payload, or null.
 * Prefers result.structuredContent (modern); falls back to JSON in
 * result.content[0].text. Used by every ontoolresult / callServerTool path.
 */
export function parseToolResult(result) {
  if (!result) return null;
  if (result.structuredContent) return result.structuredContent;
  const text = result.content?.[0]?.text;
  if (!text) return null;
  try { return JSON.parse(text); }
  catch { return null; }
}

/**
 * Same as parseToolResult, but throws when the host reports a tool error.
 * MCP tool errors come back as {isError: true, content: [{text: "..."}]} —
 * callServerTool returns normally on those, so callers without this helper
 * silently mistake errors for absent payloads. Use this at every
 * callServerTool site where the surrounding try/catch should surface the
 * server's error text. Keep parseToolResult for ontoolresult handlers that
 * should silently no-op on payloads not addressed to this widget.
 */
export function parseToolResultOrThrow(result) {
  if (result?.isError) {
    const text = result?.content?.[0]?.text ?? "tool returned an error";
    throw new Error(text);
  }
  return parseToolResult(result);
}

/** Escape `& < > " '` for safe inclusion in HTML innerHTML strings. */
export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/** Format Date to YYYY-MM-DD-HHMMSS for filenames. */
export function nowSlug() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-` +
         `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

/** Validate a var_name against the project's regex; return null if OK or an error message. */
export function validateVarName(name) {
  if (!/^[a-z_][a-z0-9_]*$/.test(name || "")) {
    return "must match ^[a-z_][a-z0-9_]*$";
  }
  return null;
}

/** CSV-encode an array of objects with the given column order. Returns a string. */
export function toCSV(rows, columns) {
  const esc = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const header = columns.map(esc).join(",");
  const body = rows.map((r) => columns.map((c) => esc(r[c])).join(",")).join("\n");
  return header + "\n" + body;
}
