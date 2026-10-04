"""Boring Semantic Layer wrappers: model loading, label resolution, summarize.

``validate_and_load_model`` is the validator behind the MCP
``define_semantic_model`` tool. BSL's own loader is one gate inside it, not
the whole of it: BSL accepts core model YAML this project rejects, so the
function wraps it in the phases described in its docstring."""

from __future__ import annotations

import functools
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


def display_label(
    model: Any, name: str, *, yaml_descriptions: dict | None = None
) -> str:
    """Return the description of a measure/dimension; prefers BSL, falls back to raw YAML map, then name."""
    measures = model.get_measures()
    if name in measures and measures[name].description:
        return measures[name].description
    dims = model.get_dimensions()
    if name in dims and dims[name].description:
        return dims[name].description
    if yaml_descriptions and name in yaml_descriptions:
        return yaml_descriptions[name]
    return name


def harvest_yaml_descriptions(yaml_path: str | Path) -> dict[str, str]:
    """Read a YAML model and return {field_name: description} for all dims and measures.

    BSL drops descriptions on calculated measures (BinOp expressions); this surfaces them
    directly from the YAML so we can reattach them to chart axis labels.
    """
    raw = yaml.safe_load(Path(yaml_path).read_text())
    block = next(iter(raw.values()))
    out: dict[str, str] = {}
    for section in ("dimensions", "measures"):
        for name, spec in (block.get(section) or {}).items():
            if isinstance(spec, dict) and "description" in spec:
                out[name] = spec["description"]
    return out


def summarize_model(model: Any) -> dict:
    """Return a JSON-serializable summary: {dimensions: [...], measures: [...]} for the agent."""
    dims = model.get_dimensions()
    base = model.get_measures()
    calc = model.get_calculated_measures()
    return {
        "dimensions": [
            {"name": k, "description": (d.description or None)} for k, d in dims.items()
        ],
        "measures": [
            {"name": k, "description": (m.description or None), "kind": "base"}
            for k, m in base.items()
        ]
        + [{"name": k, "description": None, "kind": "calculated"} for k in calc.keys()],
    }


# --- Phased semantic-model validation ---


# The schema lives in the substrate package (osaa_metrics/_schemas/) so this
# module does not depend on the MCP sub-package's asset layout. The MCP
# server serves these same bytes as the bsl-schema://osaa-metrics/v1 resource.
# Read through the traversable, never a filesystem path: the package may be
# running from a still-zipped wheel.
@functools.lru_cache(maxsize=1)
def _bsl_schema_validator() -> Draft202012Validator:
    """Validator for the core-model YAML, built once per process."""
    text = (
        files("osaa_metrics")
        .joinpath("_schemas", "bsl_model.schema.json")
        .read_text(encoding="utf-8")
    )
    return Draft202012Validator(json.loads(text))


class ModelValidationError(Exception):
    """Raised by validate_and_load_model with a normalized list of errors."""

    def __init__(self, errors: list[dict]):
        super().__init__(f"{len(errors)} validation error(s)")
        self.errors = errors


class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that raises on duplicate mapping keys and refuses YAML
    anchors / aliases.

    Standard ``yaml.safe_load`` silently keeps the last value when a mapping
    has duplicate keys, and BSL's ``SemanticModel.with_dimensions`` /
    ``with_measures`` overwrite just as silently, so a typo'd duplicate name
    would cost the user a definition with no warning anywhere. We surface
    duplicates as Phase-D validation errors instead.

    Anchors (``&name`` definitions) and aliases (``*name`` references) are
    refused outright: semantic models have no legitimate use for them, and
    they enable the "billion laughs" expansion DoS that the source-size cap
    can't prevent because the explosion happens during parsing, after the
    bytes have already passed the size check.
    """

    def compose_node(self, parent, index):
        # Refuse alias references — they appear as AliasEvent in the stream.
        if self.check_event(yaml.events.AliasEvent):
            event = self.peek_event()
            raise _AnchorRefusedError(
                kind="alias", name=event.anchor, mark=event.start_mark
            )
        # Refuse anchor definitions on scalar/sequence/mapping events.
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise _AnchorRefusedError(
                kind="anchor", name=event.anchor, mark=event.start_mark
            )
        return super().compose_node(parent, index)


class _AnchorRefusedError(Exception):
    def __init__(self, *, kind: str, name: str, mark: Any):
        self.kind = kind
        self.name = name
        self.mark = mark
        super().__init__(f"YAML {kind} {name!r} refused")


def _construct_mapping_strict(loader: yaml.Loader, node: yaml.MappingNode) -> dict:
    """Build the mapping in a single pass, detecting duplicates without
    re-invoking SafeLoader's construct_mapping (which would construct every
    scalar a second time)."""
    mapping: dict = {}
    duplicates: list = []
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            duplicates.append(key)
        mapping[key] = loader.construct_object(value_node, deep=True)
    if duplicates:
        raise _DuplicateKeyError(duplicates, node.start_mark)
    return mapping


class _DuplicateKeyError(Exception):
    def __init__(self, duplicates: list, mark: Any):
        self.duplicates = duplicates
        self.mark = mark
        super().__init__(f"duplicate keys: {duplicates}")


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_strict
)


def _normalize_jsonschema_error(err: Any) -> dict:
    path = (
        "/" + "/".join(str(p) for p in err.absolute_path) if err.absolute_path else "/"
    )
    return {
        "path": path,
        "message": err.message,
        "hint": "Fix the YAML structure; consult bsl-schema://osaa-metrics/v1.",
    }


def _unwrap_bsl_exception(exc: BaseException) -> BaseException:
    """BSL wraps SafeEvalError in returns' UnwrapFailedError, whose str() is
    empty; the original is hung off __cause__. Walk the chain to surface the
    actual root cause for the agent."""
    cur: BaseException = exc
    seen: set[int] = set()
    while cur.__cause__ is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.__cause__
    return cur


def _normalize_bsl_exception(exc: Exception, model_name: str) -> dict:
    """Map an exception raised while loading a model to the
    {path, message, hint} contract. BSL surfaces load failures as built-in
    exception types and as its own, often wrapped in returns'
    UnwrapFailedError; the root exception's message is preserved verbatim and
    a hint is added when the message matches a shape we recognise."""
    root = _unwrap_bsl_exception(exc)
    name = root.__class__.__name__
    msg = str(root) or root.__class__.__name__
    hint = ""
    if "Invalid Python syntax" in msg or name == "SafeEvalError":
        hint = "Check parentheses, quoting, and operator placement. Expressions must use only `_` and operators/method calls."
    elif name == "AttributeError":
        hint = "Column not in core_tbl. Check var_names in the saved core table or re-run discovery."
    elif "must specify 'expr'" in msg:
        hint = "Long-form dim/measure entries require an `expr:` field."
    return {"path": f"/{model_name}", "message": f"{name}: {msg}", "hint": hint}


def validate_and_load_model(
    yaml_text: str, core_tbl: Any
) -> tuple[Any, str, dict, str]:
    """Validate core model YAML and load it. Returns (model, yaml_text, summary, model_name) on success.

    ``model`` is the core semantic table BSL builds from the YAML.
    ``model_name`` is the YAML's top-level mapping key — BSL's convention for
    naming a semantic model. Callers register the returned ``model`` under a
    session-scoped key derived from this name.

    Phase A: parse YAML (strict against duplicates) + JSON Schema check.
    Phase B: BSL ``from_config(parsed, tables={"core_tbl": core_tbl})``, where
             BSL's safe-eval validator checks each expression against its
             positive AST allowlist and rejects anything outside it.
    Phase C: invoke each dimension/measure expr against ``core_tbl`` to
             surface deferred AttributeErrors (BSL only checks columns at
             query time, not at load).

    Phase D (duplicate-key detection) runs *during* Phase A via the strict
    YAML loader so the dimension/measure dict order survives untouched.

    Raises ``ModelValidationError`` with ``errors: list[{path,message,hint}]``.
    """
    errors: list[dict] = []

    # --- Phase A: parse YAML + schema check ---
    try:
        parsed = yaml.load(yaml_text, Loader=_DuplicateKeyLoader)  # noqa: S506  # a SafeLoader subclass that only adds duplicate-key and alias rejection
    except _DuplicateKeyError as e:
        errors.append(
            {
                "path": "/",
                "message": f"duplicate dimension/measure name(s): {e.duplicates}",
                "hint": "BSL silently overwrites duplicates; rename or remove the conflicts.",
            }
        )
        raise ModelValidationError(errors) from e
    except _AnchorRefusedError as e:
        errors.append(
            {
                "path": "/",
                "message": f"YAML {e.kind}s are not allowed (found {e.kind} {e.name!r})",
                "hint": "Remove the '&' / '*' references; semantic models don't need YAML anchors.",
            }
        )
        raise ModelValidationError(errors) from e
    except yaml.YAMLError as e:
        errors.append(
            {
                "path": "/",
                "message": f"YAML parse error: {e}",
                "hint": "Check indentation and quoting.",
            }
        )
        raise ModelValidationError(errors) from e

    if not isinstance(parsed, dict):
        errors.append(
            {
                "path": "/",
                "message": f"top level must be a mapping, got {type(parsed).__name__}",
                "hint": "The document is `<model_name>: { table: core_tbl, ... }`.",
            }
        )
        raise ModelValidationError(errors)

    validator = _bsl_schema_validator()
    schema_errs = sorted(
        validator.iter_errors(parsed), key=lambda e: list(e.absolute_path)
    )
    if schema_errs:
        errors.extend(_normalize_jsonschema_error(e) for e in schema_errs)
        raise ModelValidationError(errors)

    # The schema pins the document to a single top-level key, so the first
    # key is the model name.
    model_name = next(iter(parsed.keys()))

    # --- Phase B: BSL from_config (eager structural + safe-eval syntax check) ---
    try:
        # Imported here so importing this module does not load BSL.
        from boring_semantic_layer import from_config

        models = from_config(parsed, tables={"core_tbl": core_tbl})
    except Exception as e:
        errors.append(_normalize_bsl_exception(e, model_name))
        raise ModelValidationError(errors) from e

    if model_name not in models:
        errors.append(
            {
                "path": f"/{model_name}",
                "message": "BSL did not return the model under its declared name",
                "hint": "Internal error — please report.",
            }
        )
        raise ModelValidationError(errors)

    model = models[model_name]

    # --- Phase C: invoke each expr against core_tbl ---
    # Dimensions store the raw ibis Deferred (.resolve(tbl) forces column lookup).
    # Base measures wrap the deferred in a callable (`wrapped_expr(t)`) that
    # builds the aggregation against the live table — we just call it.
    dims = model.get_dimensions()
    base_meas = model.get_measures()
    for kind, name, dm in (
        *(("dimension", n, d) for n, d in dims.items()),
        *(("measure", n, m) for n, m in base_meas.items()),
    ):
        try:
            expr = dm.expr
            if hasattr(expr, "resolve"):
                expr.resolve(core_tbl)
            elif callable(expr):
                expr(core_tbl)
        except AttributeError as e:
            errors.append(
                {
                    "path": f"/{model_name}/{kind}s/{name}/expr",
                    "message": f"AttributeError: {e}",
                    "hint": (
                        f"`{name}` references a column not in the saved core table. "
                        "Check var_names in the discovery widget or load a different model."
                    ),
                }
            )
        except Exception as e:  # noqa: BLE001  # every expression failure is collected into the error list; none escapes
            root = _unwrap_bsl_exception(e)
            errors.append(
                {
                    "path": f"/{model_name}/{kind}s/{name}/expr",
                    "message": f"{root.__class__.__name__}: {root}",
                    "hint": "Expression failed to resolve against core_tbl.",
                }
            )

    if errors:
        raise ModelValidationError(errors)

    summary = summarize_model(model)
    return model, yaml_text, summary, model_name
