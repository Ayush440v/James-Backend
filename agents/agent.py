# agents/agent.py

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

GEMINI_MODEL = "gemini-2.0-flash"

# --- 1. Define UI Component Sub-Agents ---

label_agent = LlmAgent(
    name="LabelComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate one or more label components in JSON format. 
Each label should include 'type', 'text', and 'fontSize'.
Return an array of label objects like:
[
  {"type": "label", "text": "Welcome to the Generative UI!", "fontSize": 20}
]
Do not include placeholder values; use mock content if needed.
""",
    description="Creates JSON label components.",
    output_key="label_component"
)

image_agent = LlmAgent(
    name="ImageComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate one or more image components in JSON format. Each should include a valid image URL.
Return an array like:
[
  {"type": "image", "text": "https://via.placeholder.com/600x200.png?text=Header+Image"}
]
Do not use placeholders; use realistic image URLs.
""",
    description="Creates JSON image components.",
    output_key="image_component"
)

image_grid_agent = LlmAgent(
    name="ImageGridComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Create an 'imageGrid' component with 3-6 image URLs in JSON format:
{
  "type": "imageGrid",
  "items": ["url1", "url2", "url3"]
}
Generate mock images but avoid placeholders.
""",
    description="Creates image grid components.",
    output_key="image_grid_component"
)

link_card_agent = LlmAgent(
    name="LinkCardComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a link card component with a valid URL:
{
  "type": "linkCard",
  "text": "https://www.apple.com"
}
Do not use placeholder URLs.
""",
    description="Creates link card components.",
    output_key="link_card_component"
)

map_agent = LlmAgent(
    name="MapComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a map component showing a location:
{
  "type": "map",
  "latitude": 37.7749,
  "longitude": -122.4194
}
Use real coordinates for a known city or place.
""",
    description="Creates map components.",
    output_key="map_component"
)

composite_card_agent = LlmAgent(
    name="CompositeCardComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a composite card that includes 'text', 'image', and 'buttonTitle':
{
  "type": "compositeCard",
  "text": "Order your favorite meals now!",
  "image": "https://via.placeholder.com/300x150.png?text=Food+Promo",
  "buttonTitle": "Browse Menu"
}
Generate realistic content for each field.
""",
    description="Creates composite card components.",
    output_key="composite_card_component"
)

scroll_text_agent = LlmAgent(
    name="ScrollTextComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a scrollable text component with long paragraph content:
{
  "type": "scrollText",
  "text": "Long content here..."
}
Create mock article-style text, no placeholders.
""",
    description="Creates scrollable text components.",
    output_key="scroll_text_component"
)

detail_card_agent = LlmAgent(
    name="DetailCardComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Create a detail card with title, text, date, time, image, and buttonTitle:
{
  "type": "detailCard",
  "title": "Doctor Appointment",
  "text": "You have an upcoming consultation with Dr. Smith.",
  "date": "2025-06-20",
  "time": "10:30 AM",
  "image": "https://via.placeholder.com/600x200.png?text=Doctor+Visit",
  "buttonTitle": "Join Call"
}
Fill all fields with mock but realistic data.
""",
    description="Creates detail cards.",
    output_key="detail_card_component"
)

button_agent = LlmAgent(
    name="ButtonComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a button component with 'buttonTitle', 'action', and 'target':
{
  "type": "button",
  "buttonTitle": "Confirm",
  "action": "confirmAction",
  "target": "NextStep"
}
Use realistic button logic.
""",
    description="Creates JSON button components.",
    output_key="button_component"
)

# --- 2. Create the ParallelAgent (Executes All Component Agents) ---

parallel_ui_agent = ParallelAgent(
    name="ParallelUIComponentGenerator",
    sub_agents=[
        label_agent,
        image_agent,
        image_grid_agent,
        link_card_agent,
        map_agent,
        composite_card_agent,
        scroll_text_agent,
        detail_card_agent,
        button_agent
    ],
    description="Executes all UI component agents in parallel."
)

# --- 3. Define the Final Merge Agent (Controls Ordering and Output) ---

merge_agent = LlmAgent(
    name="UIOutputMergerAgent",
    model=GEMINI_MODEL,
    instruction="""
You will receive the following components from session state:
- label_component
- image_component
- image_grid_component
- link_card_component
- map_component
- composite_card_component
- scroll_text_component
- detail_card_component
- button_component

Combine these into one valid JSON response with this structure:
{
  "components": [
    {...}, {...}, ...
  ]
}

Ensure logical ordering for user experience:
1. It should have a text first
2. Rest of the components listing can be different based on the contextual requirements.

You MUST:
- Merge components into one JSON
- Preserve formatting and nesting
- Return only final JSON (no explanation, no prefix)
""",
    description="Final merge and sorting agent to return one valid UI component response."
)

# --- 4. Create the SequentialAgent (Executes in Order) ---

ui_generation_pipeline = SequentialAgent(
    name="DynamicUIGenerationPipeline",
    sub_agents=[parallel_ui_agent, merge_agent],
    description="Generates and merges UI components into final JSON layout."
)

# --- 5. Expose Root Agent ---

root_agent = ui_generation_pipeline
