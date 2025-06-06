from google.adk.agents import LlmAgent
from .prompt import BUTTON_AGENT_PROMPT

def create_button_agent(model):
    return LlmAgent(
        name="ButtonComponentAgent",
        model=model,
        instruction=BUTTON_AGENT_PROMPT,
        description="Creates JSON button components for telecom service actions.",
        output_key="button_component"
    ) 