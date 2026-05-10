"""Smoke test: conditional edges + parallel siblings.

classify → (tech_path | finance_path) → final

Either tech_path or finance_path runs depending on classify's output.
The other is skipped (sentinel `_skipped=True`). final synthesizes
whichever one ran.

Run:
    python scripts/agent_graph.py examples/branching_graph.py topic="Apple Vision Pro"
    python scripts/agent_graph.py examples/branching_graph.py topic="Federal Reserve rates"
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph("branching")

graph.add_node(
    "classify",
    role="topic classifier",
    goal=("Classify the topic into exactly ONE of these categories: 'tech', "
          "'finance', or 'other'. Output the lowercase word ONLY."),
    inputs=["topic"],
    outputs=[("category", "t")],
    max_steps=2,
)

graph.add_node(
    "tech_path",
    role="tech analyst",
    goal=("In 2 sentences, name the most important recent development for the topic, "
          "with a specific company or product name."),
    inputs=["topic"],
    outputs=[("note", "t")],
    max_steps=2,
)

graph.add_node(
    "finance_path",
    role="finance analyst",
    goal=("In 2 sentences, name the most important recent finance angle for the topic, "
          "with a specific number or named institution."),
    inputs=["topic"],
    outputs=[("note", "t")],
    max_steps=2,
)

graph.add_node(
    "final",
    role="editor",
    goal=("Write a one-sentence summary that incorporates the analyst's note. "
          "If no note was provided, say 'No analyst was applicable.'"),
    inputs=["topic", "note"],
    outputs=[("summary", "t")],
    max_steps=2,
)

graph.add_edge("classify", "tech_path",
               when='category.lower().startswith("tech")')
graph.add_edge("classify", "finance_path",
               when='category.lower().startswith("fin")')
graph.add_edge("tech_path", "final")
graph.add_edge("finance_path", "final")
