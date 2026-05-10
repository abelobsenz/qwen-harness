"""Hard test: outliner → writer (map_over sections, parallel) → editor.

Exercises the map_over feature: one node spec is invoked once per item in a
list input. All invocations run in the inner loop (sequential within the map
node, but the map node itself is one DAG node that runs in parallel with
any siblings).

Run:
    python scripts/agent_graph.py examples/writer_pipeline_graph.py topic="why agent graphs beat monolithic agents"
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph("writer_pipeline")

graph.add_node(
    "outliner",
    role="content strategist",
    goal=("Produce exactly 3 short section titles for a 600-word post on the "
          "topic. Each title must be under 6 words. No introduction, no "
          "conclusion sections — these are CONTENT sections only."),
    inputs=["topic"],
    outputs=[("sections", "l")],
    max_steps=2,
)

graph.add_node(
    "writer",
    role="staff writer",
    goal=("Write ONE paragraph (4-6 sentences) for the given section. The "
          "paragraph must mention the post's overall topic at least once "
          "and be self-contained — assume the reader hasn't seen the others."),
    # The node receives a list `sections` from the outliner; map_over tells
    # the runtime to iterate over its items, exposing each as `section`.
    inputs=["sections", "topic"],
    outputs=[("paragraph", "t")],
    max_steps=2,
    map_over="sections",
    map_item_key="section",
    extra_instructions=(
        "Produce ONLY the paragraph text in @paragraph:t — no heading, no "
        "framing, no commentary."
    ),
)

graph.add_node(
    "editor",
    role="editor",
    goal=("Produce the FULL final blog post text (not a template). Open with "
          "ONE hook sentence about the topic, then each section as: a `## ` "
          "Markdown heading using the section title, blank line, then the "
          "matching paragraph verbatim from the input. Close with ONE takeaway "
          "line. Output the entire post as plain text in the @post:t section — "
          "do NOT use placeholders, do NOT abbreviate, do NOT skip paragraphs."),
    inputs=["topic", "sections", "paragraph"],
    outputs=[("post", "t"), ("word_count", "n")],
    max_steps=2,
    extra_instructions=(
        "Inputs: `sections` is a list of N titles. `paragraph` is a list of "
        "N paragraph bodies in the same order. Pair them by index. "
        "@word_count must be the actual word count of your final @post text."
    ),
)

graph.add_edge("outliner", "writer")
graph.add_edge("outliner", "editor")
graph.add_edge("writer", "editor")
