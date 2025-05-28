from google.adk.agents import LlmAgent

link_card_agent = LlmAgent(
    name="LinkCardAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for a linkCard UI component:
        {
            "type": "linkCard",
            "text": "https://www.apple.com"
        }
        Ensure the JSON is properly formatted.
    """,
)
