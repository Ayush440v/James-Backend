from google.adk.agents import LlmAgent

image_grid_agent = LlmAgent(
    name="ImageGridAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for an imageGrid UI component:
        {
            "type": "imageGrid",
            "items": [
                "https://via.placeholder.com/100x100.png?text=1",
                "https://via.placeholder.com/100x100.png?text=2",
                "https://via.placeholder.com/100x100.png?text=3"
            ]
        }
        Ensure the JSON is properly formatted.
    """,
)
