from google.adk.tools import FunctionTool,ToolContext
import requests

def query_usage_consumption(token: str,tool_context:ToolContext):
    url = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/usageConsumption/v4/queryUsageConsumption"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIrOTE4MDc2NjI0MjQyIiwiYWNjb3VudF9udW1iZXIiOiJTUjIwMjUwNTI3MTEzMDA5IiwibG9jYWxlIjoiZW4tVVMiLCJleHAiOjE3NDk2MjkxNDF9.PhlrhIltM2ia1zvEBURARQ6Md8cSrIqg1zEUomsGCME",
        "Content-Type": "application/json"
    }
    

    response = requests.post(url, headers=headers)
    
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        return {"error": str(err), "status_code": response.status_code, "response": response.text}
    

def my_name():
    """
    Retrieves the users name.

    Args:
        None

    Returns:
        str: The users name, or None if an error occurs.
    """
    return "Adhip Kashyap"
