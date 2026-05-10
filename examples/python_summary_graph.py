"""Auto-generated graph: graph.

Designed from description:
    a graph that takes a python file path, lists its top-level functions, and writes a 200 word summary about what the file does

Edit by hand if needed; the executor only requires a top-level
`graph` AgentGraph instance.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph('python_summary')

graph.add_node('read_and_parse',
    role='File Reader & Parser',
    goal='Read the Python file at the given path. Extract all top-level function names into a list. Return the list and the raw file content.',
    inputs=['file_path'],
    outputs=[('function_names', 'l'), ('file_content', 't')],
    max_steps=6,
    tools=['read_file'],
    extra_instructions="Return only a JSON object with keys 'function_names' (list of strings) and 'file_content' (string).",
)

graph.add_node('summarize',
    role='Summarizer',
    goal='Analyze the provided Python file content and list of top-level functions. Write a concise, exactly 200-word summary explaining what the file does, its purpose, and how the functions interact.',
    inputs=['file_content', 'function_names'],
    outputs=[('summary', 't')],
    max_steps=2,
)

graph.add_edge('read_and_parse', 'summarize')
