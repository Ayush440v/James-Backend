from google.adk.agents import LlmAgent

button_agent = LlmAgent(
    name="ButtonAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for a button UI component:
        {
            "type": "button",
            "buttonTitle": "Confirm",
            "action": "confirmAction",
            "target": "NextStep"
        }
        Ensure the JSON is properly formatted.
    """,
)
