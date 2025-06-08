# agents/agent.py

from google.adk.agents import ParallelAgent, SequentialAgent, LlmAgent
from google.adk.tools.crewai_tool import CrewaiTool
from crewai_tools import ScrapeWebsiteTool
import requests
from CustomSerperTool import CustomSerperDevTool
from google.adk.models.lite_llm import LiteLlm
from tools.plans_tool import plans_tool
from tools.plan_change_tool import plan_change_tool
import os

from agents.sub_agents.label.label_agent import create_label_agent
from agents.sub_agents.scroll_text.scroll_text_agent import create_scroll_text_agent
from agents.sub_agents.button.button_agent import create_button_agent
from agents.sub_agents.graph.graph_agent import create_graph_agent
from agents.sub_agents.pie_chart.pie_chart_agent import create_pie_chart_agent
from agents.sub_agents.plans.plans_agent import create_plans_agent
from agents.sub_agents.plan_change.plan_change_agent import create_plan_change_agent
from agents.sub_agents.ui_planner.ui_planner_agent import create_ui_planner_agent
from agents.sub_agents.ui_info.ui_info_agent import create_ui_info_agent
from agents.sub_agents.image.image_agent import create_image_agent
from agents.sub_agents.image_grid.image_grid_agent import create_image_grid_agent
from agents.sub_agents.link_card.link_card_agent import create_link_card_agent
from agents.sub_agents.map.map_agent import create_map_agent
from agents.sub_agents.composite_card.composite_card_agent import create_composite_card_agent
from agents.sub_agents.detail_card.detail_card_agent import create_detail_card_agent

def BSS_TOOL():
    """
    Retrieves the user current mobile plan, usage, and account information.

    Args:
        None

    Returns:
        str: The data, usage, and account information, or None if an error occurs.
    """
    url = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/usageConsumption/v4/queryUsageConsumption"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIrOTE4MDc2NjI0MjQyIiwiYWNjb3VudF9udW1iZXIiOiJTUjIwMjUwNTMwMjA0NzI1IiwibG9jYWxlIjoiZW4tVVMiLCJleHAiOjE3NDk4OTY0NjJ9.1IXJdjcercKM0_MxEDCnEuYnpXvxDmRSbp4O3DOjino",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    try:
        response.raise_for_status()
        result = response.json()
        print(result)
        return (f"result: {result}")
    except requests.exceptions.HTTPError as err:
        print(f"test:Status Code: {response.status_code}, Error: {err}")
        return (f"test:Status Code: {response.status_code}, Error: {err}")

GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
FAST_MODEL = LiteLlm("openai/gemma2-9b-it") if os.getenv("OPENAI_API_BASE") == "https://api.groq.com/openai/v1/" else GEMINI_MODEL

# Create all agents
label_agent = create_label_agent(GEMINI_MODEL)
scroll_text_agent = create_scroll_text_agent(GEMINI_MODEL)
button_agent = create_button_agent(GEMINI_MODEL)
graph_agent = create_graph_agent(GEMINI_MODEL)
pie_chart_agent = create_pie_chart_agent(GEMINI_MODEL)
plans_agent = create_plans_agent(GEMINI_MODEL)
plan_change_agent = create_plan_change_agent(GEMINI_MODEL)
ui_planner_agent = create_ui_planner_agent(GEMINI_MODEL)
ui_info_agent = create_ui_info_agent(GEMINI_MODEL)
image_agent = create_image_agent(GEMINI_MODEL)
image_grid_agent = create_image_grid_agent(GEMINI_MODEL)
link_card_agent = create_link_card_agent(GEMINI_MODEL)
map_agent = create_map_agent(GEMINI_MODEL)
composite_card_agent = create_composite_card_agent(GEMINI_MODEL)
detail_card_agent = create_detail_card_agent(GEMINI_MODEL)

# Create the parallel agent
parallel_ui_agent = ParallelAgent(
    name="ParallelUIComponentGenerator",
    sub_agents=[
        label_agent,
        scroll_text_agent,
        button_agent,
        graph_agent,
        pie_chart_agent,
        plans_agent,
        plan_change_agent,
        image_agent,
        image_grid_agent,
        link_card_agent,
        map_agent,
        composite_card_agent,
        detail_card_agent
    ],
    description="Executes only the required UI component agents in parallel."
)

# Create the merge agent
merge_agent = LlmAgent(
    name="UIOutputMergerAgent",
    model=GEMINI_MODEL,
    instruction="""
You will receive components from session state and combine them into one valid JSON response.
Focus on organizing telecom service information in a logical flow:

1. Current Plan & Usage
2. Available Plans
3. Usage Statistics
4. Action Buttons

Combine components into this structure:
{
  "components": [
    merged_components
  ]
}

You MUST:
- Only include components relevant to telecom services
- Arrange components in a logical order
- Follow the same JSON structure as provided
- Return only the final JSON
""",
    description="Final merge and sorting agent for telecom service UI components."
)

# Create the sequential agent pipeline
ui_planner_formatter_pipeline = SequentialAgent(
    name="UIPlannerFormatterPipeline",
    sub_agents=[ui_planner_agent, ui_info_agent],
    description="Pipeline: plans UI layout, then formats components."
)

# Set the root agent
root_agent = ui_planner_formatter_pipeline

# Export the root agent
__all__ = ['root_agent']
