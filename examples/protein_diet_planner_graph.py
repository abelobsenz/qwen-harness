"""Auto-generated graph: protein_diet_planner.

Designed from description:
    A graph that searches the web for high-protein products and compares their macro and micro nutrient profiles, then builds a dietary plan around them accounting for the weight, height, sex, and activity level of the user.

Edit by hand if needed; the executor only requires a top-level
`graph` AgentGraph instance.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph('protein_diet_planner')

graph.add_node('search_products',
    role='Product Searcher',
    goal='Search the web for high-protein products matching the query. Extract and return a list of URLs pointing to product detail pages.',
    inputs=['query'],
    outputs=[('product_links', 'l')],
    max_steps=6,
    tools=['web_search'],
)

graph.add_node('fetch_nutrition',
    role='Nutrition Extractor',
    goal='Fetch the content of each product URL and extract detailed macro and micro nutrient profiles. Return structured nutrition data for each product.',
    inputs=['product_links'],
    outputs=[('nutrition_data', 'j')],
    max_steps=6,
    tools=['web_fetch'],
    map_over='product_links',
)

graph.add_node('compare_nutrition',
    role='Nutrition Analyst',
    goal='Compare the extracted nutrition data across all products. Rank them by protein content and overall nutrient density. Return a ranked list of products.',
    inputs=['nutrition_data'],
    outputs=[('ranked_products', 'j')],
    max_steps=2,
)

graph.add_node('calculate_requirements',
    role='Metabolic Calculator',
    goal="Calculate daily caloric and macronutrient requirements based on the user's weight, height, sex, and activity level. Return a JSON object with daily targets.",
    inputs=['weight', 'height', 'sex', 'activity_level'],
    outputs=[('daily_macros', 'j')],
    max_steps=2,
)

graph.add_node('build_diet_plan',
    role='Diet Planner',
    goal='Combine the ranked products with the calculated daily requirements to construct a personalized dietary plan. Return the final plan as text.',
    inputs=['ranked_products', 'daily_macros'],
    outputs=[('diet_plan', 't')],
    max_steps=2,
)

graph.add_edge('search_products', 'fetch_nutrition')
graph.add_edge('fetch_nutrition', 'compare_nutrition')
graph.add_edge('calculate_requirements', 'build_diet_plan')
graph.add_edge('compare_nutrition', 'build_diet_plan')
