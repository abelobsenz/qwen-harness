"""Example agent graph: market research → analysis → trade ideation.

Three small specialized agents, each with the smallest possible context for
its job. Outputs flow as AGFMT between them.

Run:
    python scripts/agent_graph.py examples/market_research_graph.py "US equities today"

Or with explicit input keys:
    python scripts/agent_graph.py examples/market_research_graph.py topic="oil markets"
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph("market_research")

# Node 1: researcher — pulls 3 concrete data points using web tools.
# Tightly scoped: it does not analyze, it does not opine. Just facts + sources.
graph.add_node(
    "researcher",
    role="market researcher",
    goal=("Find exactly 3 concrete data points about the topic. Each must include "
          "a number or named entity (e.g. index level, %change, ticker, headline). "
          "Use web_search and now. Stop as soon as you have 3 — do not over-collect."),
    inputs=["topic"],
    outputs=[("facts", "j"), ("sources", "l")],
    tools=["web_search", "now"],
    max_steps=8,
    extra_instructions=(
        "facts shape: a list of objects, each with `claim` (string), `value` "
        "(number or short string), `as_of` (date or 'today'). "
        "sources: one URL per line."
    ),
)

# Node 2: analyst — pure synthesis, no tools. Sees only the researcher's
# AGFMT output, not the raw search snippets, so context stays tiny.
graph.add_node(
    "analyst",
    role="market analyst",
    goal=("Identify 2-3 dominant THEMES from the facts and write a 2-3 sentence "
          "commentary that names specific numbers/entities from the input. Do not "
          "introduce facts that aren't in the input."),
    inputs=["facts"],
    outputs=[("themes", "l"), ("commentary", "t")],
    tools=None,  # no tools — pure reasoning
    max_steps=2,
)

# Node 3: ideator — proposes one concrete trade with a thesis and risk caveat.
# Sees only themes + commentary, never the raw research.
graph.add_node(
    "ideator",
    role="trade ideator",
    goal=("Propose ONE concrete trade idea (instrument, direction, brief thesis "
          "in <=30 words, one risk caveat in <=20 words). Make sure the thesis "
          "ties directly to one of the input themes."),
    inputs=["themes", "commentary"],
    outputs=[("trade", "j"), ("disclaimer", "t")],
    tools=None,
    max_steps=2,
    extra_instructions=(
        "trade shape: {\"instrument\": str, \"direction\": \"long\"|\"short\", "
        "\"thesis\": str, \"risk\": str}. "
        "disclaimer: one short line reminding this isn't financial advice."
    ),
)

graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "ideator")
