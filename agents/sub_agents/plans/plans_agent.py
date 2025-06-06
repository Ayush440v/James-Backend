from google.adk.agents import LlmAgent
from .prompt import PLANS_AGENT_PROMPT
from tools.plans_tool import plans_tool

def create_plans_agent(model):
    return LlmAgent(
        name="PlansAgent",
        model=model,
        instruction=PLANS_AGENT_PROMPT,
        description="Handles queries about available mobile plans",
        output_key="plans_response",
        tools=[plans_tool]
    ) 