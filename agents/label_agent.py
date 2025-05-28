from google.adk.agents import LlmAgent

label_agent = LlmAgent(
    name="LabelAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for a label UI component:
        {
            "type": "label",
            "text": "Welcome to the Generative UI!",
            "fontSize": 20
        }
        Ensure the JSON is properly formatted.
    """,
)
