from google.adk.tools import ToolContext

def get_jwt_token(tool_context: ToolContext):
    """
    Retrieves the JWT token from the session state in ToolContext.
    Returns the token if present, otherwise returns None.
    """
    if not tool_context or not hasattr(tool_context, 'state'):
        return None
    return tool_context.state.get("jwt_token") 