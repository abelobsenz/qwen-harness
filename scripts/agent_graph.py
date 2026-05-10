"""scripts/agent_graph.py — compartmentalized multi-agent graph executor.

Define a DAG of small specialized agents. Each agent runs in its own context
with an auto-generated, tailored system prompt and only the tools and inputs
it needs. Outputs flow between nodes in AGFMT (see scripts/agfmt.py) so no
single agent ever sees the full conversation history — keeping every node's
context well under the 60k compaction threshold.

# Why compartmentalize?

A monolithic agent doing "research the market and recommend trades" needs to
hold (a) all search snippets, (b) all reasoning, and (c) the final synthesis
in one context. After a dozen tool calls that's 30-50k tokens. Splitting
into (researcher) → (analyst) → (recommender) means:

  - researcher sees: the topic + web tools + its own searches.
  - analyst sees: the researcher's compact AGFMT output + reasoning.
  - recommender sees: the analyst's themes + risk-framing instruction.

Each node tops out at a few thousand tokens. Total tokens across nodes is
typically smaller than the monolith because each node ditches scratch work
the next one doesn't need.

# Quick example

    from agent_graph import AgentGraph, NodeSpec

    g = AgentGraph("market_research")
    g.add_node("researcher",
        role="market researcher",
        goal="Find 3 concrete data points (numbers/named entities) about the topic.",
        inputs=["topic"],
        outputs=[("facts", "j"), ("sources", "l")],
        tools=["web_search", "now"],
        max_steps=8)
    g.add_node("analyst",
        role="market analyst",
        goal="Identify the 2-3 dominant themes from the facts.",
        inputs=["facts"],
        outputs=[("themes", "l"), ("commentary", "t")],
        max_steps=2)  # no tools — pure synthesis
    g.add_edge("researcher", "analyst")
    result = g.run({"topic": "US equities today"})
    print(result["analyst"]["commentary"])

# CLI

    python scripts/agent_graph.py examples/market_research.py "US equities"

The first arg is a Python file that defines a top-level `graph` variable
(an AgentGraph instance). The remaining args become the initial input —
either as `key=value` pairs or, if the graph has a single declared entry
input, a bare positional value.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

# Local module — load lazily so this file is importable in test environments.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
if HOST in ("0.0.0.0", ""):
    HOST = "127.0.0.1"
PORT = os.environ.get("QWEN_PORT", "8000")
MODEL_ENV = os.environ.get("QWEN_MODEL_NAME", "qwen3.6")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
MODELS_URL = f"http://{HOST}:{PORT}/v1/models"


_resolved_model: str | None = None


# Inference coordination: dflash-serve is single-stream by design — even when
# graph-level parallelism (parallel nodes, map_over, batch_map) submits
# concurrent HTTP POSTs, dflash queues them and processes one at a time.
# Without coordination, the queueing happens inside dflash where TCP timeouts
# can fire, GPU memory peaks unpredictably, and the model's prompt cache gets
# thrashed by interleaved requests.
#
# A small Python-side semaphore turns those concurrent posts into a clean FIFO
# that respects keep-alive: only N inferences are in flight at once, so the
# unified-memory pressure stays bounded and the prompt-cache hit rate improves
# (consecutive requests with overlapping prefixes hit cache instead of
# evicting each other).
#
# Default 1 (strict serialize) matches dflash's actual capacity. Override with
# AGENT_GRAPH_MAX_CONCURRENT_INFERENCE if a future inference backend supports
# real continuous batching.
import threading as _threading

_INFERENCE_SEM = _threading.Semaphore(
    max(1, int(os.environ.get("AGENT_GRAPH_MAX_CONCURRENT_INFERENCE", "1"))))


def _model_id() -> str:
    """Resolve the actual loaded model id (vs the alias). The local
    dflash-serve rejects alias names like 'qwen3.6' — it expects the
    exact id from /v1/models. Cached after first lookup."""
    global _resolved_model
    if _resolved_model is not None:
        return _resolved_model
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=4) as r:
            data = json.loads(r.read())
        items = data.get("data") or []
        if items:
            _resolved_model = items[0].get("id") or MODEL_ENV
        else:
            _resolved_model = MODEL_ENV
    except Exception:  # noqa: BLE001
        _resolved_model = MODEL_ENV
    return _resolved_model


# ---------- node spec ------------------------------------------------------


# A single output may be either "name" (default tag :t) or ("name", "tag")
OutputSpec = str | tuple[str, str]


def _split_output(spec: OutputSpec) -> tuple[str, str]:
    if isinstance(spec, str):
        if ":" in spec:
            n, _, t = spec.partition(":")
            return n.strip(), t.strip() or "t"
        return spec.strip(), "t"
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return str(spec[0]).strip(), str(spec[1]).strip() or "t"
    raise TypeError(f"output spec must be str or (name, tag); got {spec!r}")


@dataclass
class NodeSpec:
    """Specification for one node (one specialized agent) in the graph."""
    name: str
    role: str                            # e.g. "market researcher"
    goal: str                            # 1-2 sentences: what this node achieves
    inputs: list[str] = field(default_factory=list)
    outputs: list[OutputSpec] = field(default_factory=list)
    tools: list[str] | None = None       # subset of tools; None = no tools
    max_steps: int = 12
    extra_instructions: str = ""         # appended verbatim to the system prompt
    tool_result_max: int = 4000          # per-tool truncation inside this node
    # Map mode: if set, this node runs once per item in `inputs[map_over]`.
    # The named input must be a list. Each invocation sees the item under the
    # singular key derived from map_over (or `item` if no obvious singular).
    # All other inputs are passed unchanged to every invocation. Outputs are
    # collected into lists; the node's downstream consumers see lists.
    map_over: str | None = None
    map_item_key: str | None = None      # override the per-item key
    # batch_map=True: bundle all N items into ONE inference call instead of N.
    # Trades compartmentalization (the model sees all items at once) for an
    # ~N× wall-time reduction on small short-item maps. Only valid with
    # tools=None — the model must produce all outputs in one shot.
    batch_map: bool = False
    # Retry: if AGFMT parsing fails or required outputs are missing, re-prompt
    # the model with the validation error up to N times. Defaults to 1 retry.
    max_output_retries: int = 1


@dataclass
class EdgeSpec:
    src: str
    dst: str
    # Optional gate. Callable form: `when(predecessor_outputs) -> bool`.
    # String form: a Python expression evaluated against the predecessor's
    # output dict (keys exposed as locals). If False, the edge is inactive
    # for this run — no data flows along it. A node with no active incoming
    # edges (or whose required inputs aren't provided by any active edge or
    # initial input) is skipped.
    when: "Any" = None


# ---------- graph ----------------------------------------------------------


class AgentGraph:
    """Directed acyclic graph of specialized agents."""

    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: list[EdgeSpec] = []

    # --- builder API ---

    def add_node(self, name: str, *, role: str, goal: str,
                 inputs: Sequence[str] = (),
                 outputs: Sequence[OutputSpec] = (),
                 tools: Sequence[str] | None = None,
                 max_steps: int = 12,
                 extra_instructions: str = "",
                 tool_result_max: int = 4000,
                 map_over: str | None = None,
                 map_item_key: str | None = None,
                 batch_map: bool = False,
                 max_output_retries: int = 1) -> NodeSpec:
        if name in self.nodes:
            raise ValueError(f"node {name!r} already defined")
        node = NodeSpec(
            name=name, role=role, goal=goal,
            inputs=list(inputs), outputs=list(outputs),
            tools=list(tools) if tools is not None else None,
            max_steps=max_steps,
            extra_instructions=extra_instructions,
            tool_result_max=tool_result_max,
            map_over=map_over,
            map_item_key=map_item_key,
            batch_map=batch_map,
            max_output_retries=max_output_retries,
        )
        self.nodes[name] = node
        return node

    def add_edge(self, src: str, dst: str, *, when: "Any" = None) -> None:
        """Add an edge. `when` is an optional gate: callable
        `(pred_outputs) -> bool`, or a Python expression string evaluated
        against the predecessor's output dict (keys as locals)."""
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"edge {src!r} → {dst!r} references unknown node")
        self.edges.append(EdgeSpec(src, dst, when=when))

    def _edge_active(self, edge: EdgeSpec, pred_outputs: dict[str, Any]) -> bool:
        """Evaluate an edge's gate against the predecessor's outputs."""
        if edge.when is None:
            return True
        if callable(edge.when):
            try:
                return bool(edge.when(pred_outputs))
            except Exception:  # noqa: BLE001
                return False
        if isinstance(edge.when, str):
            try:
                return bool(eval(edge.when, {"__builtins__": {}}, dict(pred_outputs)))
            except Exception:  # noqa: BLE001
                return False
        return bool(edge.when)

    # --- topology ---

    def topo_order(self) -> list[str]:
        """Kahn's algorithm. Raises on cycle."""
        in_deg = {n: 0 for n in self.nodes}
        for e in self.edges:
            in_deg[e.dst] += 1
        ready = [n for n, d in in_deg.items() if d == 0]
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for e in self.edges:
                if e.src == n:
                    in_deg[e.dst] -= 1
                    if in_deg[e.dst] == 0:
                        ready.append(e.dst)
        if len(order) != len(self.nodes):
            raise ValueError(f"cycle detected; ordered {len(order)}/{len(self.nodes)}")
        return order

    def predecessors(self, node: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == node]

    # --- system-prompt generation ---

    # Static prefix shared by EVERY graph node's system prompt. By placing
    # this block FIRST (before any per-node variable text), the dflash prompt
    # cache can match the prefix on every node-call after the first one and
    # skip its ~250-token prefill. Audit across all 8 example graphs: 25
    # unique system prompts → all now share these ~480 leading chars.
    _GRAPH_STATIC_PREFIX = (
        "Specialized agent in a multi-step graph. Handle ONE well-scoped "
        "task and emit results in AGFMT.\n"
        "\n"
        "# Rules\n"
        "- Single-task focus. Don't branch or pre-empt downstream steps.\n"
        "- `[cached…]` / `[REFUSED…]` are final — don't repeat the call.\n"
        "- Decision discipline: make one short planning pass, then act. If "
        "you're uncertain, do one concrete check and then either proceed, "
        "answer, or stop. Repeated self-questioning is a stop signal.\n"
        "- For web_search, each new query for the same data point must change "
        "a real dimension: entity, period, metric, source type, site/domain, "
        "filetype, geography, or exact phrase. Synonyms, word order changes, "
        "and filler words are not new searches.\n"
        "- After a near-duplicate, cached, empty, or refused web result, pivot "
        "to a different source/tool, fetch a known URL, use find_in_url, or "
        "synthesize from the evidence already gathered.\n"
        "- `@<name>:<tag>` values are the FINAL answer only — no thinking "
        "trace, no numbered steps, no scratchwork. Count words yourself if "
        "the answer needs N words; write only the final N.\n"
        "\n"
    )

    def _system_prompt(self, node: NodeSpec) -> str:
        """Auto-generate a tailored system prompt for ONE node.

        Layout, deliberately ordered for prompt-cache friendliness:
          1. Static prefix (operating rules) — IDENTICAL per node, cacheable.
          2. AGFMT format spec block — IDENTICAL per node, cacheable.
          3. Per-node tail: role, goal, inputs, required outputs, tool list,
             extra instructions. Variable.

        After this refactor, switching between nodes (or between graphs) only
        re-prefills the variable tail; the static head stays warm in dflash's
        LRUPromptCache.
        """
        from agfmt import AGFMT_STATIC_FORMAT_BLOCK, output_template_required_only

        out_specs = [f"{n}:{t}" for n, t in (_split_output(o) for o in node.outputs)]
        parts = [AgentGraph._GRAPH_STATIC_PREFIX, AGFMT_STATIC_FORMAT_BLOCK, ""]
        # ---- variable tail starts here ----
        parts.append("# Your role")
        parts.append(f"You are a {node.role}.")
        parts.append("")
        parts.append("# Your goal")
        parts.append(node.goal)
        parts.append("")
        if node.inputs:
            parts.append("# Inputs")
            parts.append("Provided in the user message as AGFMT sections "
                          f"(@<name>:<tag>): {', '.join(node.inputs)}.")
            parts.append("")
        parts.append(output_template_required_only(out_specs))
        parts.append("")
        if node.tools:
            parts.append(f"# Tools\nAvailable: {', '.join(node.tools)}. Call them only when essential.")
        else:
            parts.append("# Tools\nNone — produce the outputs from the inputs alone.")
        if node.extra_instructions.strip():
            parts.append("")
            parts.append("# Extra instructions")
            parts.append(node.extra_instructions.rstrip())
        return "\n".join(parts)

    def _user_message(self, node: NodeSpec, ctx: dict[str, Any]) -> str:
        """Build the single user message that feeds the node its inputs."""
        from agfmt import encode

        if not node.inputs:
            return "Begin."
        # Keep just the inputs the node declares (don't leak others).
        payload = {k: ctx[k] for k in node.inputs if k in ctx}
        return encode(payload)

    # --- execution ---

    def run(self, initial_inputs: dict[str, Any], *,
            verbose: bool = True,
            log_path: str | None = None,
            max_parallel: int = 4,
            event_cb: "Any" = None) -> dict[str, dict[str, Any]]:
        """Execute the graph.

        Independent nodes (whose ready predecessors are all complete) run in
        parallel up to `max_parallel`. Edges with falsy gates are inactive
        for that run; downstream nodes that lose all their inputs are
        skipped. `event_cb`, if given, is called for each lifecycle event
        with a small dict — handy for streaming progress to a UI.

        Returns: {node_name: {output_name: value, ...}, ...}
        Skipped nodes appear with sentinel `{"_skipped": True}` so callers
        can distinguish "ran and produced null" from "didn't run".
        """
        from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

        order = self.topo_order()  # validates no cycles
        results: dict[str, dict[str, Any]] = {}
        log: list[dict[str, Any]] = []
        skipped: set[str] = set()
        completed: set[str] = set()
        t_run0 = time.monotonic()

        def emit(kind: str, **fields: Any) -> None:
            if event_cb is not None:
                try:
                    event_cb({"kind": kind, **fields})
                except Exception:  # noqa: BLE001
                    pass

        emit("graph_start", graph=self.name, nodes=order)

        def ready(nname: str) -> bool:
            """A node is ready when all its predecessors are completed/skipped."""
            preds = self.predecessors(nname)
            return all((p in completed) or (p in skipped) for p in preds)

        def collect_ctx(nname: str) -> dict[str, Any] | None:
            """Build the input context, returning None if the node should be
            skipped.

            Skip rules:
              - If the node has no predecessors, it's a source — always runs
                from initial_inputs (subject to required-inputs check).
              - If the node has predecessors but NONE of its incoming edges
                are active (gate-false or pred skipped), the node is skipped.
                This is the conditional-branch contract: a node behind a gate
                doesn't sneak in via initial_inputs.
              - If at least one edge is active but a declared input is still
                missing (not provided by any active edge or initial input),
                the node is skipped — it can't run without its inputs.
            """
            node = self.nodes[nname]
            incoming = [e for e in self.edges if e.dst == nname]
            ctx: dict[str, Any] = {}
            if incoming:
                active = 0
                for e in incoming:
                    pred_out = results.get(e.src)
                    if pred_out is None or pred_out.get("_skipped"):
                        continue
                    if not self._edge_active(e, pred_out):
                        continue
                    active += 1
                    for k, v in pred_out.items():
                        if k.startswith("_"):
                            continue
                        ctx[k] = v
                if active == 0:
                    return None  # gate-skipped
            for k, v in initial_inputs.items():
                ctx.setdefault(k, v)
            for inp in node.inputs:
                if inp not in ctx:
                    return None
            return ctx

        with ThreadPoolExecutor(max_workers=max(1, int(max_parallel))) as pool:
            in_flight: dict[Any, str] = {}
            remaining = set(self.nodes)
            while remaining or in_flight:
                # Schedule any newly-ready nodes
                for nname in list(remaining):
                    if not ready(nname):
                        continue
                    ctx = collect_ctx(nname)
                    if ctx is None:
                        skipped.add(nname)
                        results[nname] = {"_skipped": True,
                                          "_reason": "no active edge supplies a required input"}
                        remaining.remove(nname)
                        emit("node_skipped", node=nname,
                             reason=results[nname]["_reason"])
                        log.append({"node": nname, "skipped": True,
                                    "reason": results[nname]["_reason"]})
                        continue
                    if verbose:
                        seen = [k for k in self.nodes[nname].inputs if k in ctx]
                        extra = [k for k in ctx if k not in self.nodes[nname].inputs]
                        print(f"\n=== node: {nname} ({self.nodes[nname].role}) ===")
                        if extra:
                            print(f"inputs: {seen}  (also in ctx, hidden from model: {extra})")
                        else:
                            print(f"inputs: {seen}")
                    emit("node_start", node=nname, role=self.nodes[nname].role,
                         inputs=list(self.nodes[nname].inputs))
                    fut = pool.submit(self._dispatch_node, nname, ctx, verbose)
                    in_flight[fut] = nname
                    remaining.remove(nname)
                if not in_flight:
                    # Nothing scheduled and nothing running — must mean some
                    # remaining node has unfulfillable predecessors. Mark them.
                    for nname in list(remaining):
                        skipped.add(nname)
                        results[nname] = {"_skipped": True,
                                          "_reason": "predecessors all skipped"}
                        emit("node_skipped", node=nname,
                             reason=results[nname]["_reason"])
                        log.append({"node": nname, "skipped": True,
                                    "reason": "predecessors all skipped"})
                        remaining.remove(nname)
                    break

                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for fut in done:
                    nname = in_flight.pop(fut)
                    try:
                        output, stats = fut.result()
                    except Exception as e:  # noqa: BLE001
                        output = {n: f"[node error: {type(e).__name__}: {e}]"
                                  for n, _ in (_split_output(o)
                                               for o in self.nodes[nname].outputs)}
                        stats = {"error": f"{type(e).__name__}: {e}"}
                    results[nname] = output
                    completed.add(nname)
                    if verbose:
                        outline = ", ".join(f"{k}={_preview(v)}" for k, v in output.items())
                        print(f"outputs[{nname}]: {outline}")
                        print(f"stats[{nname}]: {stats}")
                    emit("node_end", node=nname, outputs=output, stats=stats)
                    log.append({"node": nname, "stats": stats, "outputs": output})

        wall_s = round(time.monotonic() - t_run0, 2)
        emit("graph_end", wall_s=wall_s)
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({"graph": self.name, "wall_s": wall_s, "log": log},
                          f, indent=2, default=str)
        return results

    def _dispatch_node(self, nname: str, ctx: dict[str, Any],
                       verbose: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        """Wrapper that handles map-over and per-call timing for one node."""
        node = self.nodes[nname]
        if node.map_over:
            if node.batch_map:
                return self._run_batched_map_node(node, ctx, verbose=verbose)
            return self._run_map_node(node, ctx, verbose=verbose)
        t0 = time.monotonic()
        out, stats = self._run_node(node, ctx, verbose=verbose)
        stats.setdefault("wall_s", round(time.monotonic() - t0, 2))
        return out, stats

    def _run_batched_map_node(self, node: NodeSpec, ctx: dict[str, Any],
                              *, verbose: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a map-over node as a SINGLE inference call producing all N
        outputs at once.

        Trade-offs vs the per-item path: ~N× faster wall when items are short
        (one prompt prefill instead of N), but the model sees all items in one
        context — items can subtly leak between iterations. Use only when
        items are independent and short. Tools must be empty (one inference
        call has no time for a tool loop across N items).
        """
        from agfmt import encode, decode
        from agent_tools import real_tokens  # lazy — real Qwen tokens
        items = ctx.get(node.map_over)
        if not isinstance(items, list):
            err = (f"[map_over input {node.map_over!r} is not a list "
                   f"(got {type(items).__name__})]")
            return ({n: err for n, _ in (_split_output(o) for o in node.outputs)},
                    {"map_error": err})
        if node.tools:
            return ({n: "[batch_map=True is not allowed with tools]"
                     for n, _ in (_split_output(o) for o in node.outputs)},
                    {"map_error": "batch_map_with_tools_disallowed"})

        item_key = node.map_item_key or _singular(node.map_over)
        n = len(items)

        # Build a tailored system prompt that asks for N indexed outputs per
        # declared output name. e.g. for outputs=[("paragraph","t")] and N=3:
        #   @paragraph_0:t, @paragraph_1:t, @paragraph_2:t
        out_specs = [_split_output(o) for o in node.outputs]
        indexed_specs: list[tuple[str, str, int]] = [
            (name, tag, i) for i in range(n) for (name, tag) in out_specs
        ]
        sys_lines = [
            f"{node.role}. Batch task — same operation per item.",
            "",
            "# Goal",
            f"For EACH of {n} items in the `{node.map_over}` list:",
            node.goal.rstrip(),
            "",
            "# Inputs",
            f"`{node.map_over}` is a list of {n} items (indices 0..{n - 1}); "
            f"each item is referred to as `{item_key}_<index>`.",
            "",
            "# Output (AGFMT)",
            f"Produce EXACTLY {n * len(out_specs)} sections in this order, "
            "as `<name>_<i>:<tag>`:",
        ]
        for name, tag, i in indexed_specs:
            sys_lines.append(f"  @{name}_{i}:{tag}")
        sys_lines.append("@END")
        sys_lines.append("")
        sys_lines.append("# Rules")
        sys_lines.append("- One section per (name, index). Don't skip or merge.")
        sys_lines.append("- Items are independent — no state carries across.")
        if node.extra_instructions.strip():
            sys_lines.append("")
            sys_lines.append(node.extra_instructions.rstrip())
        system_prompt = "\n".join(sys_lines)

        # User message: all inputs encoded as AGFMT, including the LIST.
        payload = {k: ctx[k] for k in node.inputs if k in ctx}
        user_msg = encode(payload)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        t0 = time.monotonic()
        try:
            resp = self._post(messages, [])
        except Exception as e:  # noqa: BLE001
            return ({nm: f"[batched_map error: {type(e).__name__}: {e}]"
                     for nm, _ in out_specs},
                    {"batched_map_error": str(e)})
        asst = resp["choices"][0]["message"]
        content = (asst.get("content") or "").strip()
        decoded = {}
        try:
            from agfmt import decode as _agdecode
            decoded = _agdecode(content)
        except Exception:  # noqa: BLE001
            pass

        # Aggregate: for each declared output, collect the N indexed values.
        aggregated: dict[str, Any] = {}
        for name, _tag in out_specs:
            aggregated[name] = [decoded.get(f"{name}_{i}") for i in range(n)]
        wall = round(time.monotonic() - t0, 2)
        return aggregated, {
            "map_n": n,
            "batched_map": True,
            "max_msgs_tokens": real_tokens(messages + [asst]),
            "input_tokens": real_tokens(messages),
            "wall_s": wall,
        }

    def _run_map_node(self, node: NodeSpec, ctx: dict[str, Any],
                      *, verbose: bool,
                      map_max_parallel: int = 4) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a map-over node: invoke the spec once per item in `ctx[map_over]`.

        Iterations run concurrently in a thread pool. The inference server may
        serialize them depending on its batching strategy, but Python-level
        concurrency at least overlaps tool-call latency across iterations
        (e.g. parallel web_search) and prevents tool I/O from gating the
        critical path. For pure-LLM map nodes the benefit is bounded by
        upstream throughput.
        """
        from concurrent.futures import ThreadPoolExecutor
        items = ctx.get(node.map_over)
        if not isinstance(items, list):
            err = (f"[map_over input {node.map_over!r} is not a list "
                   f"(got {type(items).__name__})]")
            return ({n: err for n, _ in (_split_output(o) for o in node.outputs)},
                    {"map_error": err})
        item_key = node.map_item_key or _singular(node.map_over)
        t0 = time.monotonic()

        def _one(i: int, item: Any) -> tuple[int, dict, dict]:
            sub_ctx = {k: v for k, v in ctx.items() if k != node.map_over}
            sub_ctx[item_key] = item
            sub_node = NodeSpec(
                name=f"{node.name}#{i}",
                role=node.role, goal=node.goal,
                inputs=[item_key] + [k for k in node.inputs if k != node.map_over],
                outputs=node.outputs, tools=node.tools,
                max_steps=node.max_steps,
                extra_instructions=node.extra_instructions,
                tool_result_max=node.tool_result_max,
                max_output_retries=node.max_output_retries,
            )
            if verbose:
                print(f"  [map {i + 1}/{len(items)}] {item_key}={_preview(item)}")
            out_i, stats_i = self._run_node(sub_node, sub_ctx, verbose=False)
            return i, out_i, stats_i

        outputs_by_idx: dict[int, dict] = {}
        stats_by_idx: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, map_max_parallel)) as pool:
            futs = [pool.submit(_one, i, item) for i, item in enumerate(items)]
            for f in futs:
                i, out_i, stats_i = f.result()
                outputs_by_idx[i] = out_i
                stats_by_idx[i] = stats_i

        outputs_list = [outputs_by_idx[i] for i in range(len(items))]
        per_call_stats = [stats_by_idx[i] for i in range(len(items))]
        aggregated: dict[str, Any] = {}
        for name, _tag in (_split_output(o) for o in node.outputs):
            aggregated[name] = [o.get(name) for o in outputs_list]
        agg_stats = {
            "map_n": len(items),
            "map_parallel": map_max_parallel,
            "map_per_call": per_call_stats,
            "map_total_tool_calls": sum(s.get("n_tool_calls", 0)
                                         for s in per_call_stats),
            "wall_s": round(time.monotonic() - t0, 2),
        }
        return aggregated, agg_stats

    def _run_node(self, node: NodeSpec, inputs: dict[str, Any],
                  *, verbose: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one node's inference loop until it emits its AGFMT block.

        Returns (parsed_outputs, stats). Stats include input_tokens,
        n_tool_calls, max_messages_tokens — useful for verifying the
        compartmentalization claim.

        Failure modes handled:
          - Upstream error in any input → short-circuit (don't cascade made-up output)
          - 3 consecutive turns where every tool result was cached → inject
            "synthesize and stop" nudge (mirrors agent.py + qwen_ui)
          - Penultimate step → inject "final turn — emit AGFMT now" so a
            looping model still produces a parseable answer instead of [step-cap]
        """
        from agent_tools import (  # lazy
            _filtered_tools, CachedDispatcher, real_tokens,
        )

        # --- short-circuit on upstream error ---
        out_names = [n for n, _ in (_split_output(o) for o in node.outputs)]
        bad_inputs = {
            k: v[:120] for k, v in inputs.items()
            if isinstance(v, str) and (
                v.startswith("[node error") or v.startswith("[step-cap")
                or v.startswith("[skipped"))
        }
        if bad_inputs:
            err = (f"[skipped — upstream error(s): "
                   + "; ".join(f"{k}={v}" for k, v in bad_inputs.items()) + "]")
            return ({n: err for n in out_names},
                    {"n_tool_calls": 0, "max_msgs_tokens": 0,
                     "input_tokens": 0, "steps": 0,
                     "skipped_upstream_error": True})

        system_prompt = self._system_prompt(node)
        user_msg = self._user_message(node, inputs)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        if node.tools:
            all_tools = _filtered_tools()
            tools = [
                t for t in all_tools
                if t.get("type") == "function"
                and (t.get("function") or {}).get("name") in set(node.tools)
            ]
        else:
            tools = []

        cdisp = CachedDispatcher()
        # Seed the URL guard with anything URL-shaped that's already in the
        # node's context: the system prompt (often mentions docs sites),
        # the goal text, and any inputs the upstream nodes have produced.
        # Otherwise the model can't legitimately fetch a URL its predecessor
        # just handed it.
        try:
            for m in messages:
                content = m.get("content")
                if isinstance(content, str):
                    cdisp.note_text(content)
        except Exception:  # noqa: BLE001
            pass
        n_tool_calls = 0
        cache_hits = 0
        consecutive_all_cached = 0
        max_msgs_tokens = real_tokens(messages)
        finalize_nudged = False
        loop_guard_nudged = False  # single-fire per node — see _check below
        # Use a LOCAL retry counter rather than mutating node.max_output_retries.
        # Mutating the shared NodeSpec would carry exhausted retries across runs
        # of the same graph (and across map-iterations using the same node).
        retries_left = node.max_output_retries

        for step in range(node.max_steps):
            # On the penultimate step, force the model to emit AGFMT now —
            # without this, a model that keeps tool-calling silently runs out
            # of steps and we return [step-cap] which downstream nodes can't
            # use. The nudge trades a possibly-incomplete answer for a
            # PARSEABLE one.
            if (not finalize_nudged
                    and step == max(0, node.max_steps - 2)
                    and (n_tool_calls > 0 or step > 0)):
                messages.append({
                    "role": "user",
                    "content": (
                        "[FINAL TURN] No more tool calls. Emit your @<output>:<tag> "
                        "AGFMT block now with the best answer you can give based on "
                        "what's already gathered. End with @END."
                    ),
                })
                finalize_nudged = True

            try:
                resp = self._post(messages, tools)
            except Exception as e:  # noqa: BLE001
                return (
                    {n: f"[node error: {type(e).__name__}: {e}]" for n in out_names},
                    {"error": f"{type(e).__name__}: {e}",
                     "n_tool_calls": n_tool_calls,
                     "max_msgs_tokens": max_msgs_tokens,
                     "input_tokens": real_tokens(messages[:2]),
                     "steps": step},
                )
            asst = resp["choices"][0]["message"]
            messages.append(asst)
            max_msgs_tokens = max(max_msgs_tokens, real_tokens(messages))

            # Loop-guard surfacing: if the proxy aborted this turn with
            # the `[loop-guard:]` marker, prepend a course-correction
            # nudge BEFORE the AGFMT-retry path runs. Without this, the
            # AGFMT parser would see the truncated content, fail to find
            # required outputs, and fire the OUTPUT INVALID retry —
            # which doesn't tell the model that its previous turn was
            # loop-aborted, so the next turn often resumes the loop.
            # Single-fire per node to avoid spamming. See
            # scripts/loop_guard_marker.py for the false-positive-
            # resistant detector.
            from loop_guard_marker import (
                is_proxy_abort_marker, extract_reason, harness_nudge_message,
            )
            asst_content = asst.get("content") or ""
            if (not loop_guard_nudged
                    and is_proxy_abort_marker(asst_content)):
                loop_guard_nudged = True
                reason = extract_reason(asst_content)
                if verbose:
                    print(f"  [loop-guard fired in graph node — "
                          f"injecting course-correction (reason={reason})]")
                messages.append(harness_nudge_message(reason))
                continue  # let the next step act on the nudge

            tool_calls = asst.get("tool_calls") or []
            if not tool_calls:
                content = (asst.get("content") or "").strip()
                parsed, missing = self._parse_outputs_strict(node, content)
                if missing and retries_left > 0:
                    # Retry: tell the model exactly what was wrong
                    missing_list = ", ".join(missing)
                    messages.append({
                        "role": "user",
                        "content": (
                            "[OUTPUT INVALID] Your previous answer did not parse "
                            "as valid AGFMT or was missing required outputs: "
                            f"{missing_list}. Re-emit ONLY the AGFMT block now. "
                            "Use exactly these section headers, in this order:\n"
                            + "\n".join(f"@{n}:{t}" for n, t in
                                        (_split_output(o) for o in node.outputs))
                            + "\n@END"
                        ),
                    })
                    retries_left -= 1
                    continue
                return parsed, {
                    "n_tool_calls": n_tool_calls,
                    "cache_hits": cache_hits,
                    "max_msgs_tokens": max_msgs_tokens,
                    "input_tokens": real_tokens(messages[:2]),
                    "steps": step + 1,
                    "finalize_nudged": finalize_nudged,
                    "missing_outputs": missing,
                }

            turn_cached = 0
            for tc in tool_calls:
                fname = (tc.get("function") or {}).get("name") or ""
                fargs_raw = (tc.get("function") or {}).get("arguments") or "{}"
                try:
                    fargs = json.loads(fargs_raw) if isinstance(fargs_raw, str) else fargs_raw
                except json.JSONDecodeError:
                    fargs = {}
                if verbose:
                    print(f"  → {fname}({json.dumps(fargs, ensure_ascii=False)[:80]})")
                result, was_cached = cdisp.dispatch(fname, fargs)
                n_tool_calls += 1
                if was_cached:
                    cache_hits += 1
                    turn_cached += 1
                if isinstance(result, str) and len(result) > node.tool_result_max:
                    result = (result[:node.tool_result_max]
                              + f"\n…[truncated {len(result) - node.tool_result_max} chars]")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": fname,
                    "content": str(result),
                })
                max_msgs_tokens = max(max_msgs_tokens, real_tokens(messages))

            if turn_cached == len(tool_calls) and tool_calls:
                consecutive_all_cached += 1
            else:
                consecutive_all_cached = 0
            if consecutive_all_cached >= 3:
                messages.append({
                    "role": "user",
                    "content": (
                        "[LOOP DETECTED] The last 3 turns produced only cached results. "
                        "Stop searching. Emit the @<output>:<tag> AGFMT block now "
                        "with your best synthesis from what's above. End with @END."
                    ),
                })
                consecutive_all_cached = 0

        last = messages[-1]
        content = last.get("content") if last.get("role") == "assistant" else ""
        if content and content.strip():
            parsed = self._parse_outputs(node, content)
            # If parsing yielded no real outputs, still surface step-cap
            if all(v in (None, "") for v in parsed.values()):
                parsed = {n: "[step-cap reached without final answer]" for n in out_names}
        else:
            parsed = {n: "[step-cap reached without final answer]" for n in out_names}
        return parsed, {
            "n_tool_calls": n_tool_calls,
            "cache_hits": cache_hits,
            "max_msgs_tokens": max_msgs_tokens,
            "input_tokens": real_tokens(messages[:2]),
            "steps": node.max_steps,
            "step_cap_hit": True,
            "finalize_nudged": finalize_nudged,
        }

    @staticmethod
    def _parse_outputs(node: NodeSpec, content: str) -> dict[str, Any]:
        """Decode and return outputs (without the missing-list)."""
        out, _missing = AgentGraph._parse_outputs_strict(node, content)
        return out

    @staticmethod
    def _parse_outputs_strict(node: NodeSpec,
                               content: str) -> tuple[dict[str, Any], list[str]]:
        """Decode the assistant's final content per the node's output spec.

        Returns (outputs_dict, missing_names). missing_names is empty when
        every required output parsed successfully. Used by the retry loop to
        decide whether to ask the model to try again.
        """
        from agfmt import decode

        try:
            decoded = decode(content) if content.strip() else {}
        except ValueError:
            decoded = {}
        out: dict[str, Any] = {}
        missing: list[str] = []
        specs = [_split_output(o) for o in node.outputs]
        for name, _tag in specs:
            if name in decoded and decoded[name] is not None:
                out[name] = decoded[name]
            elif len(specs) == 1 and content.strip():
                # Single-output node: take the whole content if AGFMT failed.
                out[name] = content.strip()
            else:
                out[name] = None
                missing.append(name)
        return out, missing

    def _post(self, messages: list[dict], tools: list[dict]) -> dict:
        body: dict[str, Any] = {
            "model": _model_id(),
            "messages": messages,
            "stream": False,
            "max_tokens": int(os.environ.get("AGENT_GRAPH_MAX_TOKENS", "4096")),
            "temperature": 0.0,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        # Disable thinking by default for graph nodes. Their system prompts
        # are tightly scoped (one role, fixed inputs, fixed output schema), so
        # the visible thinking block adds little decision value and CAN go
        # wrong: when the model emits its scratch reasoning INSIDE the AGFMT
        # output (real bug observed: summarizer dumped its `Here's a thinking
        # process: 1. Analyze...` into @summary:t until the token budget ran
        # out). Override per-call by setting AGENT_GRAPH_THINKING=1.
        if os.environ.get("AGENT_GRAPH_THINKING", "0") not in ("1", "true", "True"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        # Acquire the inference semaphore around the actual HTTP call only —
        # request building (above) is cheap and parallel. With max=1 this
        # turns N concurrent map iterations into a clean FIFO at dflash
        # without changing the per-call wall (each still pays its own
        # prefill + decode).
        with _INFERENCE_SEM:
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:400]
                except Exception:  # noqa: BLE001
                    detail = ""
                raise RuntimeError(f"upstream HTTP {e.code}: {detail}") from None


def _preview(v: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    s = " ".join(str(s).split())
    return s if len(s) <= 70 else s[:70] + "…"


def _singular(name: str) -> str:
    """Best-effort English singular for map item keys. `concerns` → `concern`,
    `topics` → `topic`, `data` → `item`. Tunable later if names get exotic."""
    n = name
    if n.endswith("ies") and len(n) > 3:
        return n[:-3] + "y"
    if n.endswith("ses") and len(n) > 3:
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss") and len(n) > 1:
        return n[:-1]
    return "item"


# ---------- CLI ------------------------------------------------------------


def _load_graph_module(path: str) -> AgentGraph:
    spec = importlib.util.spec_from_file_location("graph_def", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load graph from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "graph"):
        raise AttributeError(f"{path}: module must define a top-level `graph` AgentGraph instance")
    g = mod.graph
    # Duck-type: when the CLI runs as __main__ and the example imports
    # `from agent_graph`, Python creates two distinct class objects with the
    # same name. isinstance() would falsely reject one of them. Check the
    # contract instead.
    required = ("nodes", "edges", "run", "topo_order", "add_node", "add_edge")
    missing = [a for a in required if not hasattr(g, a)]
    if missing:
        raise TypeError(
            f"{path}: `graph` is not an AgentGraph (missing: {missing}); "
            f"got {type(g).__name__}"
        )
    return g


def _parse_initial_inputs(args: list[str], graph: AgentGraph) -> dict[str, Any]:
    """Accept either `key=value` pairs or, if the entry node has exactly one
    input not produced by any predecessor, a single bare positional value.
    """
    if not args:
        return {}
    if len(args) == 1 and "=" not in args[0]:
        # Find a single missing input across all topo-entry nodes
        order = graph.topo_order()
        provided: set[str] = set()
        for n in order:
            for p in graph.predecessors(n):
                provided.update((graph.nodes[p].outputs and
                                 [_split_output(o)[0] for o in graph.nodes[p].outputs]) or [])
        needed: list[str] = []
        for n in order:
            for inp in graph.nodes[n].inputs:
                if inp not in provided and inp not in needed:
                    needed.append(inp)
            for o in graph.nodes[n].outputs:
                provided.add(_split_output(o)[0])
        if len(needed) == 1:
            return {needed[0]: args[0]}
        raise ValueError(f"bare positional input ambiguous; declared inputs: {needed}. "
                         f"Use key=value form.")
    out: dict[str, Any] = {}
    for a in args:
        if "=" not in a:
            raise ValueError(f"input must be key=value, got {a!r}")
        k, _, v = a.partition("=")
        out[k.strip()] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a compartmentalized agent graph.")
    ap.add_argument("graph_file", help="Python file defining a top-level `graph` AgentGraph")
    ap.add_argument("inputs", nargs="*", help="key=value pairs (or one bare value)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log", default=None,
                    help="Optional JSON log path: per-node stats + outputs")
    args = ap.parse_args()

    g = _load_graph_module(args.graph_file)
    inputs = _parse_initial_inputs(args.inputs, g)
    print(f"# graph: {g.name}  nodes={list(g.nodes)}  inputs={list(inputs)}")
    t0 = time.monotonic()
    out = g.run(inputs, verbose=not args.quiet, log_path=args.log)
    wall = time.monotonic() - t0
    print(f"\n# total wall: {wall:.1f}s")
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
