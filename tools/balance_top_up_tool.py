from google.adk.tools import FunctionTool, ToolContext
import requests
from typing import Dict, Any, Optional
from requests.exceptions import RequestException, Timeout, JSONDecodeError

def balance_top_up(input_data: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Tops up the user's account balance using the Totogi payment API.

    Args:
        input_data (Dict[str, Any]): Dictionary containing:
            - paymentAmount (str or float): The amount in AED to top up
        tool_context (Optional[ToolContext]): The context object providing access to session state.

    Returns:
        Dict[str, Any]: A dictionary containing either:
            - 'success': True and 'data': The payment confirmation data
            - 'success': False and 'error': An error message
    """
    endpoint = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/customerManagement/v4/payment"

    payment_amount = input_data.get('paymentAmount')
    if not payment_amount:
        return {'success': False, 'error': 'Payment amount (AED) is required'}

    # Get JWT token from tool context
    jwt_token = tool_context.state.get("jwt_token") if tool_context else None
    if not jwt_token:
        return {'success': False, 'error': 'JWT token not found in session state'}

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "paymentAmount": str(payment_amount)
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        payment_data = response.json()
        return {
            'success': True,
            'data': payment_data,
            'message': 'Balance top-up request submitted successfully'
        }
    except Timeout:
        return {'success': False, 'error': 'Request timed out while topping up balance'}
    except RequestException as e:
        return {'success': False, 'error': f'Failed to top up balance: {str(e)}'}
    except JSONDecodeError:
        return {'success': False, 'error': 'Invalid response format from balance top-up API'}
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}

balance_top_up_tool = FunctionTool(func=balance_top_up) 