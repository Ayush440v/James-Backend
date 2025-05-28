from google.adk.agents import LlmAgent

image_agent = LlmAgent(
    name="ImageAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for an image UI component:
        {
            "type": "image",
            "text": "https://via.placeholder.com/600x200.png?text=Header+Image"
        }
        Ensure the JSON is properly formatted.
    """,
)
