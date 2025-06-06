import requests

def BSS_TOOL():
    """
    Retrieves the user current mobile plan, usage, and account information.

    Args:
        None

    Returns:
        str: The data, usage, and account information, or None if an error occurs.
    """
    url = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/usageConsumption/v4/queryUsageConsumption"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIrOTE4MDc2NjI0MjQyIiwiYWNjb3VudF9udW1iZXIiOiJTUjIwMjUwNTMwMjA0NzI1IiwibG9jYWxlIjoiZW4tVVMiLCJleHAiOjE3NDk4OTY0NjJ9.1IXJdjcercKM0_MxEDCnEuYnpXvxDmRSbp4O3DOjino",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    try:
        response.raise_for_status()
        result = response.json()
        print(result)
        return (f"result: {result}")
    except requests.exceptions.HTTPError as err:
        print(f"test:Status Code: {response.status_code}, Error: {err}")
        return (f"test:Status Code: {response.status_code}, Error: {err}") 