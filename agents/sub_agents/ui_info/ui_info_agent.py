from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext, FunctionTool
from .prompt import UI_INFO_AGENT_PROMPT
import requests
from typing import Dict, Any, List, Union, Optional
import json
from requests.exceptions import RequestException, Timeout, JSONDecodeError
from tools.plans_tool import plans_tool
from tools.plan_change_tool import plan_change_tool
from tools.balance_top_up_tool import balance_top_up_tool


def get_usage_consumption(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieves usage consumption data from the Totogi API using a JWT token.

    Args:
        tool_context (ToolContext): The context object providing access to session state.

    Returns:
        dict: The JSON response from the API.
    """
    jwt_token = tool_context.state.get("jwt_token")
    if not jwt_token:
        raise ValueError("JWT token not found in session state.")

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }

    endpoint = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/usageConsumption/v4/queryUsageConsumption"

    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()
    return {"success": True, "data": response.json()}

def create_ui_info_agent(model):
    return LlmAgent(
        name="UIInfoAgent",
        model=model,
        instruction=UI_INFO_AGENT_PROMPT,
        description="Generates clean JSON UI components for telecom service information",
        output_key="ui_components",
        tools=[FunctionTool(get_usage_consumption), plans_tool, plan_change_tool, balance_top_up_tool]
    ) 