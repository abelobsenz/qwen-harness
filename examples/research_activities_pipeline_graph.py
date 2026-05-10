"""Auto-generated graph: research_activities_pipeline.

Designed from description:
    Plan, budget, and schedule a 5-day team offsite for 18 people in Austin.

Edit by hand if needed; the executor only requires a top-level
`graph` AgentGraph instance.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from agent_graph import AgentGraph

graph = AgentGraph('research_activities_pipeline')

graph.add_node('research_activities',
    role='Activity Researcher',
    goal='Search for diverse, engaging activities in the specified location suitable for a group of the given size. Find options for team building, dining, and leisure. Return a list of activity options with details.',
    inputs=['location', 'attendees'],
    outputs=[('activity_options', 'l')],
    max_steps=6,
    tools=['web_search'],
    extra_instructions='Focus on Austin. Ensure activities can accommodate 18 people. Include estimated costs and duration for each activity.',
)

graph.add_node('draft_schedule',
    role='Scheduler',
    goal='Create a 5-day itinerary using the provided activity options. Distribute activities across days to balance intensity and variety. Include morning, afternoon, and evening slots.',
    inputs=['activity_options', 'duration'],
    outputs=[('itinerary', 'j')],
    max_steps=2,
    extra_instructions='Ensure the schedule is realistic and accounts for travel time between venues. Include downtime.',
)

graph.add_node('calculate_budget',
    role='Budget Analyst',
    goal='Estimate the total cost for the itinerary based on the activities and number of attendees. Include estimated costs for activities, meals, and any venue fees.',
    inputs=['itinerary', 'attendees'],
    outputs=[('budget_estimate', 'j')],
    max_steps=2,
    extra_instructions='Provide a breakdown by category (e.g., activities, food) and a total.',
)

graph.add_edge('research_activities', 'draft_schedule')
graph.add_edge('draft_schedule', 'calculate_budget')
