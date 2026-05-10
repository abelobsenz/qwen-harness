"""Auto-generated graph: ai_search_pipeline.

Designed from description:
    A graph that finds novel tech updates in the AI sector and notifies me via a system message. Prioritize big LLM drops like new frontier models AND local llm optimizations that could be useful in optimizing my local system.

Edit by hand if needed; the executor only requires a top-level
`graph` AgentGraph instance.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph('ai_search_pipeline')

graph.add_node('ai_search',
    role='Search Agent',
    goal='Search the web for recent AI technology updates, focusing on new model releases and optimization techniques.',
    inputs=['search_query'],
    outputs=[('search_results', 'l')],
    max_steps=6,
    tools=['web_search'],
    extra_instructions='Use broad queries to capture both frontier models and local optimizations.',
)

graph.add_node('ai_filter',
    role='Filter Agent',
    goal='Filter the search results to identify and prioritize two categories: 1) New frontier LLM model drops, and 2) Local LLM optimizations useful for personal systems. Discard irrelevant news.',
    inputs=['search_results'],
    outputs=[('relevant_updates', 'l')],
    max_steps=2,
    extra_instructions='Return a list of updates. Each update should have a title, summary, and category (frontier or local).',
)

graph.add_node('format_notification',
    role='Notification Agent',
    goal='Format the relevant updates into a concise, readable notification message suitable for a system alert. Group by category.',
    inputs=['relevant_updates'],
    outputs=[('notification', 't')],
    max_steps=2,
    extra_instructions='Keep it brief. Highlight key details.',
)

graph.add_edge('ai_search', 'ai_filter')
graph.add_edge('ai_filter', 'format_notification')
