"""Adapters that re-export the local tool registry into LangGraph and
Pydantic AI shapes so the same set of tools can drive any of three runtimes:

  - the in-house qwen agent loop (scripts/agent.py)
  - Pydantic AI's `Agent` / `Tool` model (typed, validated args)
  - LangGraph's `tool` decorator + `ToolNode` graph node

The conversions are pure-data: each adapter walks the JSON-Schema in TOOLS
once at startup and produces the runtime-specific objects. No third-party
deps are imported until you actually call the corresponding `to_*` builder,
so the venv stays light unless you opt in.

Why both:

  * Pydantic AI is great when you want type-safe single-agent workflows with
    structured outputs and a clean retry/validation story; we use it as a
    "client SDK" for the local qwen server, pointed at the same chat API.
  * LangGraph is the right tool when you need explicit state machines —
    multi-agent supervision, branching tool execution, persistent
    checkpointing. Once exported, every tool here becomes a node.

Both adapters share a `wrap()` helper that keeps the cross-turn cache
semantics (so an external runtime gets the same dedup behavior as the CLI).
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_tools import (  # noqa: E402
    DISPATCH,
    TOOLS,
    CachedDispatcher,
    _arg_key,  # noqa: F401  (used by callers wrapping their own caches)
)


# --------------------------------------------------------------------------
# JSON-schema → Python annotation helpers
# --------------------------------------------------------------------------

_SCHEMA_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _annotation_from_prop(prop: dict) -> type:
    """Resolve a JSON-Schema property's `type` (or `enum`) to a Python type
    annotation. We don't try to model unions or pydantic Literals — for the
    purposes of these adapters, `str` is fine for enum-typed string fields,
    and any caller wanting stricter validation can subclass after."""
    t = prop.get("type")
    if isinstance(t, list):
        # type-list e.g. ["string", "null"] — pick the first non-null
        for tt in t:
            if tt and tt != "null":
                t = tt
                break
        else:
            t = "string"
    return _SCHEMA_TO_PY.get(t, str)


def _spec_for(name: str) -> dict:
    """Look up the OpenAI-shape schema for one tool by name."""
    for t in TOOLS:
        f = t.get("function", {}) or {}
        if f.get("name") == name:
            return f
    raise KeyError(f"no tool spec named {name!r}")


# --------------------------------------------------------------------------
# Pydantic AI adapter
# --------------------------------------------------------------------------

def to_pydantic_ai_tools(
    names: list[str] | None = None,
    cdisp: CachedDispatcher | None = None,
) -> list[Any]:
    """Return a list of `pydantic_ai.Tool` objects ready to attach to an Agent.

    Pass `names` to subset the registry. Pass `cdisp` to share a session
    cache across multiple Pydantic AI calls (recommended).
    """
    try:
        from pydantic_ai.tools import Tool  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pydantic-ai not installed. `pip install pydantic-ai` to use "
            "this adapter."
        ) from e
    cdisp = cdisp or CachedDispatcher()
    selected = names or list(DISPATCH.keys())
    out = []
    for name in selected:
        if name not in DISPATCH:
            continue
        spec = _spec_for(name)
        out.append(_pydantic_tool_from_spec(name, spec, cdisp, Tool))
    return out


def _pydantic_tool_from_spec(name: str, spec: dict,
                              cdisp: CachedDispatcher, Tool) -> Any:
    """Build one pydantic_ai.Tool from a spec, wired through CachedDispatcher.

    Pydantic AI introspects the wrapper function's signature, so we
    construct one with the right argument names + annotations. We delegate
    actual execution back to the cached dispatcher so users get the same
    cross-turn caching they'd get inside the CLI.
    """
    params = (spec.get("parameters") or {}).get("properties", {}) or {}
    required = set((spec.get("parameters") or {}).get("required", []))
    description = spec.get("description") or name

    # Build parameter list for the synthetic signature.
    args_list = []
    annotations: dict[str, type] = {}
    for arg_name, prop in params.items():
        annotations[arg_name] = _annotation_from_prop(prop)
        default = prop.get("default", inspect.Parameter.empty
                           if arg_name in required else None)
        args_list.append(
            inspect.Parameter(
                arg_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotations[arg_name],
            )
        )

    def runner(**kwargs):
        # Drop pydantic-injected None defaults so optional args don't override
        # the underlying tool's actual defaults.
        clean = {k: v for k, v in kwargs.items() if v is not None}
        result, _was_cached = cdisp.dispatch(name, clean)
        return result

    runner.__name__ = name
    runner.__doc__ = description
    runner.__signature__ = inspect.Signature(parameters=args_list,  # type: ignore[attr-defined]
                                              return_annotation=str)
    runner.__annotations__ = {**annotations, "return": str}

    return Tool(runner, name=name, description=description)


# --------------------------------------------------------------------------
# LangGraph adapter
# --------------------------------------------------------------------------

def to_langgraph_tools(
    names: list[str] | None = None,
    cdisp: CachedDispatcher | None = None,
) -> list[Any]:
    """Return a list of LangChain `BaseTool` objects (LangGraph compatible).

    LangGraph's `ToolNode` consumes any callable accepted by the LangChain
    runtime; the simplest way to make that bridge is via `langchain_core`'s
    `StructuredTool` factory, which validates args against a generated
    pydantic model.
    """
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
        from pydantic import BaseModel, create_model  # type: ignore
    except ImportError as e:
        raise ImportError(
            "langchain_core / pydantic not installed. `pip install "
            "langchain-core langgraph pydantic` to use this adapter."
        ) from e
    cdisp = cdisp or CachedDispatcher()
    selected = names or list(DISPATCH.keys())
    out = []
    for name in selected:
        if name not in DISPATCH:
            continue
        spec = _spec_for(name)
        out.append(_langgraph_tool_from_spec(name, spec, cdisp,
                                              StructuredTool, create_model,
                                              BaseModel))
    return out


def _langgraph_tool_from_spec(name: str, spec: dict, cdisp: CachedDispatcher,
                               StructuredTool, create_model, BaseModel) -> Any:
    params = (spec.get("parameters") or {}).get("properties", {}) or {}
    required = set((spec.get("parameters") or {}).get("required", []))
    description = spec.get("description") or name

    fields: dict[str, tuple[type, Any]] = {}
    for arg_name, prop in params.items():
        ann = _annotation_from_prop(prop)
        if arg_name in required:
            fields[arg_name] = (ann, ...)
        else:
            fields[arg_name] = (ann, prop.get("default", None))

    ArgsModel = create_model(f"{name}_args", __base__=BaseModel, **fields)

    def runner(**kwargs):
        clean = {k: v for k, v in kwargs.items() if v is not None}
        result, _was_cached = cdisp.dispatch(name, clean)
        return result

    runner.__name__ = name
    runner.__doc__ = description

    return StructuredTool.from_function(
        func=runner,
        name=name,
        description=description,
        args_schema=ArgsModel,
    )


# --------------------------------------------------------------------------
# OpenAI / generic schema export
# --------------------------------------------------------------------------

def to_openai_tools(names: list[str] | None = None) -> list[dict]:
    """Return the tool list in OpenAI Chat Completions shape (i.e. the same
    shape qwen_proxy.py already speaks). Use this when wiring in a model
    that wants the OpenAI tool format without the need for caching."""
    if names is None:
        return list(TOOLS)
    keep = set(names)
    return [t for t in TOOLS if t.get("function", {}).get("name") in keep]


def schema_summary() -> str:
    """Human-readable summary — handy for `python -m agent_adapters` output
    or for verifying the registry from a notebook."""
    out = [f"# agent_adapters — {len(TOOLS)} tools registered\n"]
    by_section: dict[str, list[str]] = {}
    for t in TOOLS:
        f = t.get("function", {}) or {}
        name = f.get("name", "?")
        # Cheap "section" inference from function name prefix.
        section = name.split("_", 1)[0] if "_" in name else "core"
        by_section.setdefault(section, []).append(name)
    for section in sorted(by_section):
        out.append(f"\n## {section}")
        for name in sorted(by_section[section]):
            spec = _spec_for(name)
            desc = (spec.get("description") or "").splitlines()[0][:100]
            out.append(f"  - {name}: {desc}")
    return "\n".join(out)


__all__ = [
    "to_pydantic_ai_tools",
    "to_langgraph_tools",
    "to_openai_tools",
    "schema_summary",
]


if __name__ == "__main__":
    print(schema_summary())
