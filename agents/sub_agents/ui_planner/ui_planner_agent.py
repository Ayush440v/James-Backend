from google.adk.agents import LlmAgent
from .prompt import UI_PLANNER_AGENT_PROMPT

def create_ui_planner_agent(model):
    return LlmAgent(
        name="ComponentFormatterAgent",
        model=model,
        instruction=UI_PLANNER_AGENT_PROMPT,
        description="Formats UI components for telecom service information.",
        output_key="plan"
    ) 