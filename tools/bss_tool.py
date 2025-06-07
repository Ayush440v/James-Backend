from google.adk.tools import FunctionTool, ToolContext
import requests

def get_usage_consumption(input_data: dict, tool_context: ToolContext) -> dict:
    """
    Retrieves usage consumption data from the Totogi API using a JWT token.

    Args:
        input_data (dict): The input data required by the API.
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

    response = requests.post(endpoint, headers=headers, json=input_data)
    response.raise_for_status()
    return response.json()

# Initialize the FunctionTool without the 'name' parameter
usage_tool = FunctionTool(func=get_usage_consumption)