"""Auto-generated graph: fetch_readme_pipeline.

Designed from description:
    a graph that takes a github repo URL, fetches its README, and writes a 100-word summary

Edit by hand if needed; the executor only requires a top-level
`graph` AgentGraph instance.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph('fetch_readme_pipeline')

graph.add_node('fetch_readme',
    role='README Fetcher',
    goal='Fetch the README content from the provided GitHub repository URL using the github_repo tool.',
    inputs=['github_url'],
    outputs=[('readme_content', 't')],
    max_steps=6,
    tools=['github_repo'],
)

graph.add_node('summarize_readme',
    role='README Summarizer',
    goal="Read the provided README content and write a concise, exactly 100-word summary capturing the project's purpose, key features, and tech stack.",
    inputs=['readme_content'],
    outputs=[('summary', 't')],
    max_steps=2,
)

graph.add_edge('fetch_readme', 'summarize_readme')
