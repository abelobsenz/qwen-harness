"""scripts/graph_designer.py — natural-language → AgentGraph.

User describes what they want in plain English. The designer asks the
local model to emit a structured graph spec (AGFMT), validates it, then
renders it as a Python file the existing executor can load.

The model never writes raw Python that we eval — it produces structured
fields that we lower into a fixed template. Safer than letting it free-
form code, and far easier for a small/quantized model to do reliably.

Three entry points:

    design_preview(description)
        → {"ok", "spec", "code"}  — design WITHOUT saving. Used by the UI
        for the user-controllable "review the architecture before
        committing" flow. The user can hand-edit the returned spec.

    save_spec(spec, description)
        → {"ok", "name", "path"}  — persist a (possibly edited) spec.
        Re-validates, renders, writes the file, then ensures the graph
        loads cleanly AND has no cycles (raises 422-equivalent if it does).

    design_and_save(description)
        → end-to-end: design + save in one shot. Kept for the CLI and
        for direct programmatic use; the UI now goes through preview/save.

CLI usage:
    python scripts/graph_designer.py "describe your graph here"
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from typing import Any


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
if HOST in ("0.0.0.0", ""):
    HOST = "127.0.0.1"
PORT = os.environ.get("QWEN_PORT", "8000")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
MODELS_URL = f"http://{HOST}:{PORT}/v1/models"


# Tools the designer is allowed to assign. Stay strict — don't expose
# write tools to user-built graphs by default; they can be added by hand
# later if the user really wants them.
_DESIGNER_ALLOWED_TOOLS = (
    "web_search", "web_fetch", "now", "make_table",
    "read_file", "list_files", "grep", "csv_summary",
    "github_repo", "arxiv_search",
)

_TAGS = ("t", "j", "l", "n", "b", "kv")


_SYSTEM_PROMPT = """\
Graph architect. Turn a workflow description into an AgentGraph spec — a
DAG of specialized agents whose outputs flow as AGFMT. Design the smallest
correct DAG.

# Rules
- One responsibility per node. Decompose.
- First node(s) declare the graph's INPUTS (what the user supplies at run
  time). Short snake_case names.
- Output names match what downstream nodes consume — data flows by NAME.
- List input processed per-element: set `map_over` to the input name and
  `map_item_key` to the singular. With `batch_map=true`, tools must be null.
- Allowed tools: web_search, web_fetch, now, make_table, read_file,
  list_files, grep, csv_summary, github_repo, arxiv_search. Anything else
  is not permitted; synthesis-only nodes have tools=null.
- Edges are directed; a node with multiple incoming edges sees the union
  of their outputs.
- Conditional edges: `when` is a Python expression on the predecessor's
  output dict. Keep simple (`category=='X'`).

# Output (AGFMT)
@name:t
<snake_case graph id>

@nodes:j
[
  {
    "name": "<snake_case>",
    "role": "<short noun phrase>",
    "goal": "<2-3 imperative sentences>",
    "inputs": ["<name>", ...],
    "outputs": [["<name>", "<tag>"], ...],
    "tools": ["<tool>", ...] | null,
    "max_steps": 2 | 6 | 8,
    "map_over": "<input_name>" | null,
    "map_item_key": "<singular>" | null,
    "batch_map": false | true,
    "extra_instructions": ""
  },
  ...
]

@edges:j
[
  {"src": "<node>", "dst": "<node>", "when": null},
  ...
]
@END

Tags: t=text, j=JSON, l=list, n=number, b=bool. Use j for nested objects,
l for short identifier lists, t for prose.

# Style
- 2-5 nodes typical. More only if responsibilities truly split.
- max_steps: 2 for synthesis; 6-8 for tool-using nodes.
- Specific goals — the goal IS the node's prompt; vague in, vague out.
"""


def _resolve_model_id() -> str:
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=4) as r:
            data = json.loads(r.read())
        items = data.get("data") or []
        if items:
            return items[0].get("id") or "qwen3.6"
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("QWEN_MODEL_NAME", "qwen3.6")


def _post(messages: list[dict]) -> dict:
    body = {
        "model": _resolve_model_id(),
        "messages": messages,
        "stream": False,
        # 8192 leaves room for a thinking pass + the full AGFMT block on
        # complex multi-node designs. The previous 4096 cap let qwen3.6's
        # thinking trace eat the entire budget on a 5-node design.
        "max_tokens": 8192,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def design_spec(description: str) -> dict:
    """Run the designer model and return the validated spec dict."""
    from agfmt import decode

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Design a graph for this workflow:\n\n"
            f"{description.strip()}\n\n"
            "Begin your response with `@name:t` on the first line. Emit "
            "ONLY the AGFMT block. No node-by-node prose, no commentary, "
            "no self-review — the format spec is your output template, "
            "fill it in directly. /no_think"
        )},
    ]
    resp = _post(messages)
    content = (resp["choices"][0]["message"].get("content") or "").strip()
    decoded = decode(content)
    spec = _validate_spec(decoded)
    return spec


# Identifier regexes used for name + node validation. Restrictive on
# purpose: model output gets normalized rather than crashing the renderer.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _slugify(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", (s or "").strip().lower()).strip("_")
    return s or "graph"


def _validate_spec(spec: Any) -> dict:
    """Coerce the designer's output into a valid spec or raise.

    We don't try to be heroic: if a required field is missing or a node
    references an unknown predecessor, raise loud and let the caller surface
    the error. Easier to debug than silently producing a broken graph file.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"designer output is not a dict: {type(spec).__name__}")
    raw_name = str(spec.get("name") or "").strip()
    name = _slugify(raw_name) if raw_name else ""
    if not name or name == "graph":
        # Model fell back to a placeholder. Derive from any keyword nodes
        # present, else a generic timestamp suffix; better than colliding
        # on a meaningless `graph_graph.py`.
        nodes = spec.get("nodes") or []
        first = (nodes[0].get("name") if nodes and isinstance(nodes[0], dict) else "")
        if first:
            name = f"{_slugify(first)}_pipeline"
        else:
            import datetime as _dt
            name = "graph_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    nodes = spec.get("nodes") or []
    edges = spec.get("edges") or []
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("designer output: `nodes` must be a non-empty list")
    if not isinstance(edges, list):
        raise ValueError("designer output: `edges` must be a list (may be empty)")

    seen_names: set[str] = set()
    clean_nodes: list[dict] = []
    for raw in nodes:
        if not isinstance(raw, dict):
            raise ValueError(f"node entry is not a dict: {raw!r}")
        nname = _slugify(str(raw.get("name") or ""))
        if not nname:
            raise ValueError(f"node missing `name`: {raw!r}")
        if nname in seen_names:
            raise ValueError(f"duplicate node name: {nname!r}")
        seen_names.add(nname)
        role = str(raw.get("role") or "specialist").strip()[:80]
        goal = str(raw.get("goal") or "").strip()
        if not goal:
            raise ValueError(f"node {nname!r} missing `goal`")
        inputs = raw.get("inputs") or []
        if not isinstance(inputs, list):
            raise ValueError(f"node {nname!r}: inputs must be a list")
        inputs = [str(x).strip() for x in inputs if str(x).strip()]
        outputs = raw.get("outputs") or []
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"node {nname!r}: outputs must be non-empty list")
        out_pairs: list[tuple[str, str]] = []
        for o in outputs:
            if isinstance(o, list) and len(o) >= 2:
                onm, otag = str(o[0]).strip(), str(o[1]).strip()
            elif isinstance(o, str) and ":" in o:
                onm, _, otag = o.partition(":")
                onm, otag = onm.strip(), otag.strip()
            elif isinstance(o, str):
                onm, otag = o.strip(), "t"
            else:
                raise ValueError(f"node {nname!r}: output entry not parseable: {o!r}")
            if otag not in _TAGS:
                otag = "t"
            out_pairs.append((onm, otag))
        tools = raw.get("tools")
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError(f"node {nname!r}: tools must be a list or null")
            tools = [t for t in (str(x).strip() for x in tools)
                     if t in _DESIGNER_ALLOWED_TOOLS]
            if not tools:
                tools = None
        max_steps = int(raw.get("max_steps") or (6 if tools else 2))
        max_steps = max(1, min(max_steps, 12))
        map_over = raw.get("map_over") or None
        map_item_key = raw.get("map_item_key") or None
        batch_map = bool(raw.get("batch_map"))
        extra = str(raw.get("extra_instructions") or "")
        clean_nodes.append({
            "name": nname, "role": role, "goal": goal,
            "inputs": inputs, "outputs": out_pairs,
            "tools": tools, "max_steps": max_steps,
            "map_over": map_over, "map_item_key": map_item_key,
            "batch_map": batch_map, "extra_instructions": extra,
        })

    clean_edges: list[dict] = []
    for raw in edges:
        if not isinstance(raw, dict):
            continue
        src = _slugify(str(raw.get("src") or ""))
        dst = _slugify(str(raw.get("dst") or ""))
        if src not in seen_names or dst not in seen_names:
            raise ValueError(f"edge {src} → {dst} references unknown node")
        when = raw.get("when")
        if when is not None and not isinstance(when, str):
            when = None
        clean_edges.append({"src": src, "dst": dst, "when": when})

    return {"name": name, "nodes": clean_nodes, "edges": clean_edges}


def render_python(spec: dict, *, description: str = "") -> str:
    """Render the validated spec as a Python file the executor can load."""
    name = spec["name"]
    lines = [
        f'"""Auto-generated graph: {name}.',
        "",
        "Designed from description:",
        f"    {description.strip() or '(no description provided)'}",
        "",
        "Edit by hand if needed; the executor only requires a top-level",
        "`graph` AgentGraph instance.",
        '"""',
        "from __future__ import annotations",
        "import os, sys",
        'sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))',
        "from agent_graph import AgentGraph",
        "",
        f'graph = AgentGraph({name!r})',
        "",
    ]
    for node in spec["nodes"]:
        outs_repr = "[" + ", ".join(f"({n!r}, {t!r})" for n, t in node["outputs"]) + "]"
        kw = [
            f"role={node['role']!r}",
            f"goal={node['goal']!r}",
            f"inputs={node['inputs']!r}",
            f"outputs={outs_repr}",
            f"max_steps={node['max_steps']!r}",
        ]
        if node["tools"]:
            kw.append(f"tools={node['tools']!r}")
        if node["map_over"]:
            kw.append(f"map_over={node['map_over']!r}")
        if node["map_item_key"]:
            kw.append(f"map_item_key={node['map_item_key']!r}")
        if node["batch_map"]:
            kw.append("batch_map=True")
        if node["extra_instructions"]:
            kw.append(f"extra_instructions={node['extra_instructions']!r}")
        lines.append(f"graph.add_node({node['name']!r},")
        for k in kw:
            lines.append(f"    {k},")
        lines.append(")")
        lines.append("")
    for e in spec["edges"]:
        if e.get("when"):
            lines.append(f"graph.add_edge({e['src']!r}, {e['dst']!r}, when={e['when']!r})")
        else:
            lines.append(f"graph.add_edge({e['src']!r}, {e['dst']!r})")
    lines.append("")
    return "\n".join(lines)


def design_preview(description: str) -> dict:
    """Design a graph and return the spec + rendered code WITHOUT saving.

    Used by the UI to let the user review/edit the auto-generated architecture
    before committing it to the examples/ directory. Returns:
        {"ok": True, "spec": {...}, "code": "...", "description": "..."}
    or {"ok": False, "error": "..."} on validation failure.
    """
    try:
        spec = design_spec(description)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "stage": "design"}
    try:
        code = render_python(spec, description=description)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"render failed: {type(e).__name__}: {e}",
                "stage": "render", "spec": spec}
    return {"ok": True, "spec": spec, "code": code,
            "description": description}


def save_spec(spec: Any, *, description: str = "",
              examples_dir: str | None = None,
              overwrite: bool = False) -> dict:
    """Persist a (possibly user-edited) spec to a new examples/<name>_graph.py.

    Re-runs validation so the user can't silently inject malformed JSON
    (the renderer relies on shape invariants the validator enforces).
    Returns the same shape as design_and_save.
    """
    if examples_dir is None:
        examples_dir = os.path.join(os.path.dirname(_HERE), "examples")
    os.makedirs(examples_dir, exist_ok=True)
    try:
        clean = _validate_spec(spec)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"validation failed: {type(e).__name__}: {e}",
                "stage": "validate"}
    code = render_python(clean, description=description)
    base = clean["name"]
    fname = f"{base}_graph.py"
    path = os.path.join(examples_dir, fname)
    if os.path.exists(path) and not overwrite:
        i = 2
        while os.path.exists(os.path.join(examples_dir, f"{base}{i}_graph.py")):
            i += 1
        fname = f"{base}{i}_graph.py"
        path = os.path.join(examples_dir, fname)
        clean["name"] = f"{base}{i}"
        # Re-render so the in-file `graph = AgentGraph('name')` matches the
        # disambiguated filename (otherwise the file's graph.name would be
        # the original colliding name, confusing the executor's name lookup).
        code = render_python(clean, description=description)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        from agent_graph import _load_graph_module
        loaded = _load_graph_module(path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "name": clean["name"], "path": path,
                "error": f"{type(e).__name__}: {e}",
                "spec": clean, "code": code, "stage": "load"}
    # Cycle detection: topo_order() raises on cycles. Without this, a user
    # could save a cyclic graph and only discover the bug when they tried
    # to run it. Best to fail loud at save-time.
    try:
        loaded.topo_order()
    except ValueError as e:
        # Remove the file we just wrote so the user doesn't see a phantom
        # graph in the list that explodes on every run attempt.
        try:
            os.remove(path)
        except OSError:
            pass
        return {"ok": False, "name": clean["name"], "path": path,
                "error": f"graph contains a cycle: {e}",
                "spec": clean, "code": code, "stage": "topology"}
    return {"ok": True, "name": clean["name"], "path": path,
            "spec": clean, "code": code}


def design_and_save(description: str, *, examples_dir: str | None = None,
                    overwrite: bool = False) -> dict:
    """End-to-end: design, validate, render, write file. Returns metadata.

    Kept for backwards compatibility / CLI use. The UI now goes through
    design_preview → user-edits → save_spec for explicit user approval.
    """
    if examples_dir is None:
        examples_dir = os.path.join(os.path.dirname(_HERE), "examples")
    try:
        spec = design_spec(description)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"design failed: {type(e).__name__}: {e}",
                "stage": "design"}
    return save_spec(spec, description=description, examples_dir=examples_dir,
                     overwrite=overwrite)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: graph_designer.py \"<description>\"", file=sys.stderr)
        return 2
    desc = " ".join(sys.argv[1:])
    print(f"# designing graph for:\n  {desc}\n")
    out = design_and_save(desc)
    print(json.dumps(out, indent=2, default=str)[:1500])
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
