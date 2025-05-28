from google.adk.agents import LlmAgent

scroll_text_agent = LlmAgent(
    name="ScrollTextAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for a scrollText UI component:
        {
            "type": "scrollText",
            "text": "This is a long description that should be scrollable. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
        }
        Ensure the JSON is properly formatted.
    """,
)
