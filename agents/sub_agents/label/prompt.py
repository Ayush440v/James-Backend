LABEL_AGENT_PROMPT = """
Generate one or more label components in JSON format for telecom service information. 
Each label should include 'type', 'text', and 'fontSize'.
Focus on displaying:
- Usage statistics
- Plan details
- Account information
- Service recommendations
Return an array of label objects like:
[
  {"type": "label", "text": "Your Current Plan: Premium Unlimited", "fontSize": 20}
]
Use clear, concise language appropriate for telecom services.
""" 