from google.adk.tools import FunctionTool, ToolContext
import requests
from typing import Dict, Any, List, Union, Optional
import json
from requests.exceptions import RequestException, Timeout, JSONDecodeError

def get_available_plans(input_data: Optional[Dict[str, Any]] = None, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Retrieves available mobile plans from the Totogi API.

    Args:
        input_data (Optional[Dict[str, Any]]): The input data required by the API (can be empty for this endpoint).
        tool_context (Optional[ToolContext]): The context object providing access to session state.

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
    except RequestException as e:
        return {'success': False, 'error': f'Failed to fetch plans: {str(e)}'}
    except JSONDecodeError:
        return {'success': False, 'error': 'Invalid response format from plans API'}
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}

# Initialize the FunctionTool
plans_tool = FunctionTool(func=get_available_plans) 