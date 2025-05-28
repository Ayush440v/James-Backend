from google.adk.agents import LlmAgent

composite_card_agent = LlmAgent(
    name="CompositeCardAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for a compositeCard UI component:
        {
            "type": "compositeCard",
            "text": "Order your favorite meals now!",
            "image": "https://via.placeholder.com/300x150.png?text=Food+Promo",
            "buttonTitle": "Browse Menu"
        }
        Ensure the JSON is properly formatted.
    """,
)
