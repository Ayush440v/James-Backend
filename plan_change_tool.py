from google.adk.tools import FunctionTool, ToolContext
import requests
from typing import Dict, Any, Optional
from requests.exceptions import RequestException, Timeout, JSONDecodeError

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
    if not tool_context or not hasattr(tool_context, 'jwt_token'):
        return {'success': False, 'error': 'JWT token not available in tool context'}

    headers = {
        "Authorization": f"Bearer {tool_context.jwt_token}",
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
    except RequestException as e:
        return {'success': False, 'error': f'Failed to change plan: {str(e)}'}
    except JSONDecodeError:
        return {'success': False, 'error': 'Invalid response format from plan change API'}
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}

# Initialize the FunctionTool
plan_change_tool = FunctionTool(func=change_plan) 