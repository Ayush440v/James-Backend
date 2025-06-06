from google.adk.agents import LlmAgent
from .prompt import DETAIL_CARD_AGENT_PROMPT

def create_detail_card_agent(model):
    return LlmAgent(
        name="DetailCardComponentAgent",
        model=model,
        instruction=DETAIL_CARD_AGENT_PROMPT,
        description="Creates detail cards.",
        output_key="detail_card_component"
    ) 