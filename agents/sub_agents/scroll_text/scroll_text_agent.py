from google.adk.agents import LlmAgent
from .prompt import SCROLL_TEXT_AGENT_PROMPT

def create_scroll_text_agent(model):
    return LlmAgent(
        name="ScrollTextComponentAgent",
        model=model,
        instruction=SCROLL_TEXT_AGENT_PROMPT,
        description="Creates scrollable text components for detailed telecom information.",
        output_key="scroll_text_component"
    ) 