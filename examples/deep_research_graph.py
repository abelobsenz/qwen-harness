"""Hard test: deep research graph.

aspect_lister → map(researcher with web_search) → critic → synthesizer

This is the real "deep research" pattern the user pays a token tax for in
naive monolithic agents:
  - List 3 distinct angles to investigate
  - Research each angle in parallel-ish (sequential within map, but each
    invocation has its OWN minimal context — no cross-pollution of search
    snippets)
  - Critic surfaces gaps from the structured findings
  - Synthesizer produces the final answer

Each researcher iteration sees ONLY its angle, the topic, and its own
search results — never the other researchers' raw snippets. That's the
context-compartmentalization win in action.

Run:
    python scripts/agent_graph.py examples/deep_research_graph.py topic="state of speculative decoding on Apple Silicon in 2026"
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph("deep_research")

graph.add_node(
    "aspect_lister",
    role="research planner",
    goal=("Given the topic, propose exactly 3 distinct, NON-OVERLAPPING angles "
          "to investigate. Each angle is a short imperative phrase (under 12 "
          "words) that names a specific question. Avoid generic angles like "
          "'history' or 'overview' — be concrete."),
    inputs=["topic"],
    outputs=[("angles", "l")],
    max_steps=2,
)

graph.add_node(
    "researcher",
    role="domain researcher",
    goal=("Research the assigned angle on the topic. Run 1-3 web_search calls. "
          "Extract 2-3 concrete findings (numbers, names, dates, project URLs, "
          "or quoted claims). DO NOT analyze or speculate beyond what sources "
          "say. If results are thin, state that honestly."),
    inputs=["angles", "topic"],
    outputs=[("findings", "j"), ("citations", "l")],
    tools=["web_search"],
    max_steps=6,
    map_over="angles",
    map_item_key="angle",
    extra_instructions=(
        "findings shape: a JSON list of {\"claim\": str, \"detail\": str} pairs. "
        "citations: one URL per line."
    ),
)

graph.add_node(
    "critic",
    role="research critic",
    goal=("Review the per-angle findings. Identify (a) the SINGLE biggest gap "
          "or unanswered question across the corpus, (b) any apparent "
          "contradictions between findings. Be terse and specific."),
    inputs=["findings"],
    outputs=[("gap", "t"), ("contradictions", "l")],
    max_steps=2,
)

graph.add_node(
    "synthesizer",
    role="lead analyst",
    goal=("Produce the final 4-6 sentence answer to the topic, weaving in "
          "specific findings from the research. Acknowledge the critic's gap "
          "in one closing sentence. Cite at least 2 of the URLs from `citations` "
          "inline as (source: <domain>) — the reader needs traceability."),
    inputs=["topic", "findings", "citations", "gap"],
    outputs=[("answer", "t")],
    max_steps=2,
)

graph.add_edge("aspect_lister", "researcher")
graph.add_edge("aspect_lister", "synthesizer")  # synthesizer also wants angles list as context
graph.add_edge("researcher", "critic")
graph.add_edge("researcher", "synthesizer")
graph.add_edge("critic", "synthesizer")
