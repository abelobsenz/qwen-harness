"""Example: 4-node code-review graph.

surveyor → reviewer → prioritizer → fixer-stub

Exercises a longer chain, mixed tool subsets, and AGFMT round-trip with
nested JSON. Each node sees only its predecessor's compact output.

Run:
    python scripts/agent_graph.py examples/code_review_graph.py path=scripts/agfmt.py

Or just:
    python scripts/agent_graph.py examples/code_review_graph.py scripts/agfmt.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph("code_review")

# 1. SURVEYOR — read the file, build a structural map (functions, sizes).
graph.add_node(
    "surveyor",
    role="code surveyor",
    goal=("Read the target file and build a compact structural map. List "
          "every top-level def/class, its line range, and a 1-sentence "
          "summary. Do not deep-read every function body."),
    inputs=["path"],
    outputs=[("file_path", "t"), ("structure", "j"), ("loc", "n")],
    tools=["read_file", "grep", "list_files"],
    max_steps=6,
    extra_instructions=(
        "structure shape: a JSON list, each item like "
        "{\"name\": str, \"kind\": \"def\"|\"class\", "
        "\"line_start\": int, \"summary\": str}."
    ),
)

# 2. REVIEWER — given the structure, identify up to 5 concerns.
# Sees only the structure JSON, not the raw file content.
graph.add_node(
    "reviewer",
    role="senior reviewer",
    goal=("From the structural map alone, produce up to 5 concrete review "
          "concerns. Each must reference a specific item by name. If the "
          "map looks fine, output an empty list — do NOT invent issues."),
    inputs=["file_path", "structure"],
    outputs=[("concerns", "j")],
    tools=None,
    max_steps=2,
    extra_instructions=(
        "concerns shape: JSON list of "
        "{\"target\": str (name), \"severity\": \"low\"|\"med\"|\"high\", "
        "\"issue\": str (1 sentence), \"suggestion\": str (1 sentence)}. "
        "Empty list [] is a valid answer if the code looks clean."
    ),
)

# 3. PRIORITIZER — order concerns and pick the top item.
graph.add_node(
    "prioritizer",
    role="release manager",
    goal=("Order the concerns by impact (severity × surface). Identify the "
          "single highest-priority concern as `top` (or null if list empty)."),
    inputs=["concerns"],
    outputs=[("ranked", "j"), ("top", "j")],
    tools=None,
    max_steps=2,
    extra_instructions=(
        "ranked: same list ordered most → least impactful. "
        "top: the first item from ranked, or the JSON literal null if empty."
    ),
)

# 4. FIXER-STUB — sketch a minimal patch description for the top item.
# Sees ONLY the top concern and the original path, not the full structure.
graph.add_node(
    "fixer_stub",
    role="patch sketcher",
    goal=("Given the top concern, sketch a minimal patch as a 3-step plan. "
          "Do NOT actually edit the file. If `top` is null, return an empty "
          "plan list and a one-line note that no fix is needed."),
    inputs=["file_path", "top"],
    outputs=[("plan", "l"), ("rationale", "t")],
    tools=None,
    max_steps=2,
)

graph.add_edge("surveyor", "reviewer")
graph.add_edge("reviewer", "prioritizer")
graph.add_edge("prioritizer", "fixer_stub")
graph.add_edge("surveyor", "fixer_stub")  # fixer also needs file_path
