from google.adk.agents import LlmAgent
from .prompt import UI_INFO_AGENT_PROMPT
from tools.bss_tool import BSS_TOOL
from tools.plans_tool import plans_tool
from tools.plan_change_tool import plan_change_tool

def create_ui_info_agent(model):
    return LlmAgent(
        name="UIInfoAgent",
        model=model,
        instruction=UI_INFO_AGENT_PROMPT,
        description="Provides dynamic information about telecom services.",
        output_key="output",
        tools=[BSS_TOOL, plans_tool, plan_change_tool]
    ) 