from google.adk.agents import LlmAgent
from agents.sub_agents.plan_change.prompt import PLAN_CHANGE_AGENT_PROMPT
from tools.plans_tool import plans_tool
from tools.plan_change_tool import plan_change_tool

def create_plan_change_agent(model):
    """
    Creates a plan change agent that processes plan change requests.
    
    The agent:
    1. Uses plans_tool to get available plans and find the selected plan's externalIdentifier
    2. Uses plan_change_tool to process the plan change request
    3. Returns the result of the plan change operation
    """
    # Create the agent
    return LlmAgent(
        name="PlanChangeAgent",
        model=model,
        instruction=PLAN_CHANGE_AGENT_PROMPT,
        description="Processes plan change requests by verifying plans and executing changes",
        tools=[plans_tool, plan_change_tool],
        output_key="plan_change_result"
    ) 