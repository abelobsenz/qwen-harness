"""Variant: writer pipeline with batch_map=True on the writer node.

Compare wall time vs writer_pipeline_graph.py: batch_map should produce all
N paragraphs in a single inference call instead of N parallel calls.

Run:
    python scripts/agent_graph.py examples/writer_batched_graph.py topic="why context compartmentalization wins"
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph("writer_batched")

graph.add_node(
    "outliner",
    role="content strategist",
    goal=("Produce exactly 3 short section titles for a 600-word post on the "
          "topic. Each title under 6 words. CONTENT sections only — no intro, "
          "no conclusion."),
    inputs=["topic"],
    outputs=[("sections", "l")],
    max_steps=2,
)

graph.add_node(
    "writer",
    role="staff writer",
    goal=("Write ONE paragraph (4-6 sentences) for the given section. "
          "Mention the post's overall topic at least once. Each paragraph "
          "should stand alone."),
    inputs=["sections", "topic"],
    outputs=[("paragraph", "t")],
    max_steps=2,
    map_over="sections",
    map_item_key="section",
    batch_map=True,        # <-- the new feature: one call for all sections
)

graph.add_node(
    "editor",
    role="editor",
    goal=("Produce the FULL final blog post text. One hook sentence, then "
          "for each i: '## ' + sections[i] + blank line + paragraph[i]. End "
          "with one takeaway line. No placeholders."),
    inputs=["topic", "sections", "paragraph"],
    outputs=[("post", "t"), ("word_count", "n")],
    max_steps=2,
)

graph.add_edge("outliner", "writer")
graph.add_edge("outliner", "editor")
graph.add_edge("writer", "editor")
