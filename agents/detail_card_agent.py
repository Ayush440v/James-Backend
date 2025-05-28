from google.adk.agents import LlmAgent

detail_card_agent = LlmAgent(
    name="DetailCardAgent",
    model="gemini-2.0-flash",
    instruction="""
        Generate a JSON object for a detailCard UI component:
        {
            "type": "detailCard",
            "title": "Doctor Appointment",
            "text": "You have an upcoming consultation with Dr. Smith.",
            "date": "2025-06-20",
            "time": "10:30 AM",
            "image": "https://via.placeholder.com/600x200.png?text=Doctor+Visit",
            "buttonTitle": "Join Call"
        }
        Ensure the JSON is properly formatted.
    """,
)
