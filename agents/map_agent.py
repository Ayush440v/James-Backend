from google.adk.agents import LlmAgent

map_agent = LlmAgent(
    name="MapAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for a map UI component:
        {
            "type": "map",
            "latitude": 37.7749,
            "longitude": -122.4194
        }
        Ensure the JSON is properly formatted.
    """,
)
