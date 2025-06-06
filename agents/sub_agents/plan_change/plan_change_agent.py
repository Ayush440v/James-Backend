from google.adk.agents import LlmAgent
from .prompt import PLAN_CHANGE_AGENT_PROMPT
from tools.plan_change_tool import plan_change_tool

def create_plan_change_agent(model):
    return LlmAgent(
        name="PlanChangeAgent",
        model=model,
        instruction=PLAN_CHANGE_AGENT_PROMPT,
        description="Handles plan change requests",
        output_key="plan_change_response",
        tools=[plan_change_tool]
    ) 