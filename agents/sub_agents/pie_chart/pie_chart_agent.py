from google.adk.agents import LlmAgent
from .prompt import PIE_CHART_AGENT_PROMPT

def create_pie_chart_agent(model):
    return LlmAgent(
        name="PieChartComponentAgent",
        model=model,
        instruction=PIE_CHART_AGENT_PROMPT,
        description="Creates pie chart components for telecom service distribution.",
        output_key="pie_chart_component"
    ) 