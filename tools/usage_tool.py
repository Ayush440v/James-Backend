from google.adk.tools import FunctionTool, ToolContext
from google.adk.auth.auth_schemes import HTTPBearer
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, HTTPAuth
from google.adk.auth import AuthConfig
import requests

# Define the authentication scheme and credential (Bearer token, placeholder values)
auth_scheme = HTTPBearer(bearer_format="JWT")
auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HTTPAuth(scheme="bearer", bearer_format="JWT")
)
auth_config = AuthConfig(
    auth_scheme=auth_scheme,
    raw_auth_credential=auth_credential
)

def usage_consumption_tool(input_data: dict, tool_context: ToolContext) -> dict:
    """
    Retrieves the user's current mobile plan, usage, and account information.
    Requires a valid JWT token in the session state.

    Args:
        input_data (dict): (Unused, for AFC compatibility)
        tool_context (ToolContext): Context containing session state and other information

    Returns:
        dict: The data, usage, and account information, or error if an error occurs.
        
    Raises:
        ValueError: If JWT token is not found in session state
    """
    # Step 1: Check for exchanged credential (Bearer token)
    exchanged_credential = tool_context.get_auth_response(auth_config)
    jwt_token = None
    if exchanged_credential and hasattr(exchanged_credential, "http") and exchanged_credential.http and exchanged_credential.http.bearer_token:
        jwt_token = exchanged_credential.http.bearer_token
    else:
        # Step 2: If not present, request credential
        tool_context.request_credential(auth_config)
        return {"pending": True, "message": "Awaiting user authentication."}

    # Step 3: Make the authenticated API call
    url = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/usageConsumption/v4/queryUsageConsumption"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    request_body = {
        "query": {
            "filter": {"type": "ALL"}
        }
    }
    try:
        response = requests.post(url, headers=headers, json=request_body)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Create the FunctionTool
usage_tool = FunctionTool(func=usage_consumption_tool)