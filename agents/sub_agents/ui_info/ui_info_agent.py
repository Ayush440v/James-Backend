from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext, FunctionTool
from .prompt import UI_INFO_AGENT_PROMPT
import requests
from typing import Dict, Any, List, Union, Optional
import json
from requests.exceptions import RequestException, Timeout, JSONDecodeError


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

def get_available_plans() -> Dict[str, Any]:
    """
    Retrieves available mobile plans from the Totogi API.

    Args:
       None
    Returns:
        Dict[str, Any]: A dictionary containing either:
            - 'success': True and 'data': The processed plans data
            - 'success': False and 'error': An error message

    Raises:
        None: All exceptions are caught and returned as error messages
    """
    endpoint = "https://assets.ontology.bss-magic.totogi.solutions/du/totogi-ontology/productCatalogManagement/v4/productOffering-en.json"
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        
        plans_data = response.json()
        
        # Handle both list and dictionary responses
        if isinstance(plans_data, list):
            processed_plans = {
                'success': True,
                'data': {
                    'plans': plans_data,
                    'total_plans': len(plans_data),
                    'last_updated': None  # List response doesn't have lastModified
                }
            }
        else:
            processed_plans = {
                'success': True,
                'data': {
                    'plans': plans_data.get('productOffering', []),
                    'total_plans': len(plans_data.get('productOffering', [])),
                    'last_updated': plans_data.get('lastModified', '')
                }
            }
        
        return processed_plans

    except Timeout:
        return {'success': False, 'error': 'Request timed out while fetching plans'}
    except JSONDecodeError:
        return {'success': False, 'error': 'Invalid response format from plans API'}
    except RequestException as e:
        return {'success': False, 'error': f'Failed to fetch plans: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}


def change_plan(input_data: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Changes the user's mobile plan using the Totogi API.

    Args:
        input_data (Dict[str, Any]): Dictionary containing:
            - plan_id (str): The ID of the plan to switch to
            - base_type (str): The base type of the plan (default: "ProductPrice")
            - type (str): The type of the plan (default: "ProductPrice")
            - referred_type (str): The referred type of the plan (default: "ProductPrice")
        tool_context (Optional[ToolContext]): The context object providing access to session state.

    Returns:
        Dict[str, Any]: A dictionary containing either:
            - 'success': True and 'data': The order confirmation data
            - 'success': False and 'error': An error message
    """
    endpoint = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/serviceOrdering/v4/serviceOrder"
    
    # Extract plan details from input_data
    plan_id = input_data.get('plan_id')
    if not plan_id:
        return {'success': False, 'error': 'Plan ID is required'}

    base_type = input_data.get('base_type', 'ProductPrice')
    type_value = input_data.get('type', 'ProductPrice')
    referred_type = input_data.get('referred_type', 'ProductPrice')

    # Get JWT token from tool context
    jwt_token = tool_context.state.get("jwt_token")
    if not jwt_token:
        return {'success': False, 'error': 'JWT token not found in session state'}

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "serviceOrderItem": [
            {
                "quantity": 1,
                "service": {
                    "supportingResource": [],
                    "relatedEntity": [
                        {
                            "id": plan_id,
                            "baseType": base_type,
                            "type": type_value,
                            "referredType": referred_type
                        }
                    ]
                }
            }
        ]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        order_data = response.json()
        return {
            'success': True,
            'data': {
                'order': order_data,
                'message': 'Plan change request submitted successfully'
            }
        }

    except Timeout:
        return {'success': False, 'error': 'Request timed out while changing plan'}
    except JSONDecodeError:
        return {'success': False, 'error': 'Invalid response format from plan change API'}
    except RequestException as e:
        return {'success': False, 'error': f'Failed to change plan: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}

# Initialize the FunctionTool

def create_ui_info_agent(model):
    return LlmAgent(
        name="UIInfoAgent",
        model=model,
        instruction=UI_INFO_AGENT_PROMPT,
        description="Generates clean JSON UI components for telecom service information",
        output_key="ui_components",
        tools=[FunctionTool(get_usage_consumption), FunctionTool(get_available_plans), FunctionTool(change_plan)]
    ) 