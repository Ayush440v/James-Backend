from google.adk.agents import LlmAgent
from .prompt import LABEL_AGENT_PROMPT

def create_label_agent(model):
    return LlmAgent(
        name="LabelComponentAgent",
        model=model,
        instruction=LABEL_AGENT_PROMPT,
        description="Creates JSON label components for telecom information.",
        output_key="label_component"
    ) 