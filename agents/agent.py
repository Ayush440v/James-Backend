# agents/agent.py

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools.crewai_tool import CrewaiTool
from CustomSerperTool import CustomSerperDevTool
from google.adk.tools import google_search
GEMINI_MODEL = "gemini-2.0-flash"
root_agent = None
# --- 1. Define UI Component Sub-Agents ---

# Instantiate the CrewAI tool
serper_image_tool_instance = CustomSerperDevTool(
    n_results=10,
    save_file=False,
    search_type="images"
)
serper_maps_tool_instance = CustomSerperDevTool(
    n_results=1,
    save_file=False,
    search_type="search"
)
google_search_tool_instance = CustomSerperDevTool(
    n_results=10,
    save_file=False,
    search_type="search"
)
serper_image_tool = CrewaiTool(
    name="InternetImageSearch",
    description="Searches the internet specifically for recent images using Serper.",
    tool=serper_image_tool_instance
)
serper_maps_tool = CrewaiTool(
    name="InternetMapsSearch",
    description="Searches the internet specifically for recent maps using Serper.",
    tool=serper_maps_tool_instance
)
google_search_tool = CrewaiTool(
    name="InternetSearch",
    description="Searches the internet specifically for recent data using Google.",
    tool=google_search_tool_instance
)
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
  {"type": "image", "text": imageUrl}
]
Use the serper_image_tool tool to find the image URL. Use the imageURL as the text.
DO NOT MAKE UP IMAGE URLS.
""",
    description="Creates JSON image components.",
    output_key="image_component",
    tools=[serper_image_tool]
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
se the serper_image_tool tool to find the image URL.
DO NOT MAKE UP IMAGE URLS.
""",
    description="Creates image grid components.",
    output_key="image_grid_component",
    tools=[serper_image_tool]
)

link_card_agent = LlmAgent(
    name="LinkCardComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a link card component with a valid URL:
{
  "type": "linkCard",
  "text": imageUrl
}

Use the serper_image_tool tool to find the image URL. Use the imageURL as the text.
DO NOT MAKE UP URLS.
""",
    description="Creates link card components.",
    output_key="link_card_component",
    tools=[serper_image_tool]
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
Use the serper_maps_tool tool to find the coordinates.
DO NOT MAKE UP COORDINATES.
""",
    description="Creates map components.",
    output_key="map_component",
    tools=[serper_maps_tool]
)

composite_card_agent = LlmAgent(
    name="CompositeCardComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a composite card that includes 'text', 'image', and 'buttonTitle':
{
  "type": "compositeCard",
  "text": "Order your favorite meals now!",
  "image": images.imageURL,
  "buttonTitle": "Browse Menu"
}
Generate realistic content for each field.

Use the serper_image_tool tool to find the image URL. Use the imageURL as the text.
DO NOT MAKE UP IMAGE URLS.
""",
    description="Creates composite card components.",
    output_key="composite_card_component",
    tools=[serper_image_tool]
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
Fill only those fields with mock but realistic data which are relevant to the user's request.
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

component_formatter_agent = LlmAgent(
    name="ComponentFormatterAgent",
    model=GEMINI_MODEL,
    instruction="""
You are an assistant that dynamically generates a structured JSON response of UI component objects based on the plan.

Instructions:

Read the {state.plan} to get the list of component types.

Use the user query to generate relevant and realistic content for each component.

Always return the output in the following strict format:
```json
\{
  "components": [
    \{ componentObject1 \},
    \{ componentObject2 \},
    ...
  ]
\}
```
Component types and required fields:

label: \{ "type": "label", "text": string, "fontSize": integer \}

image: \{ "type": "image", "text": imageUrl \}

image_grid: \{ "type": "imageGrid", "items": [imageUrl, ...] \}

link_card: \{ "type": "linkCard", "text": url \}

map: \{ "type": "map", "latitude": float, "longitude": float \}

composite_card: \{ "type": "compositeCard", "text": string, "image": imageUrl, "buttonTitle": string \}

scroll_text: \{ "type": "scrollText", "text": string \}

detail_card: \{ "type": "detailCard", "title": string, "text": string, "date": string, "time": string, "image": imageUrl, "buttonTitle": string \}

button: \{ "type": "button", "buttonTitle": string, "action": string, "target": string \}

Output rules:

Include one fully-formed component for each type in the plan.

Generate multiple relevant results where applicable (e.g. imageGrid, linkCard).

Only return the JSON object—no extra text, no comments, no explanation.

Use realistic, query-specific data.


The output must always follow this schema exactly.
""",
    description="Formats a list of component types into fully specified UI component objects based on the user's query.",
    output_key="formatted_components",
    tools=[google_search_tool, serper_image_tool, serper_maps_tool]
)

ui_planner_agent = LlmAgent(
    name="UIPlannerAgent",
    model=GEMINI_MODEL,
    instruction="""
You are an assistant that provides dynamic, real-time information tailored to the user's query.
Generate the text for a dynamic UI component based on the user's query.

Include multiple relevant results when appropriate to give the user options or broader context.


Use google_search_tool to search the internet for relevant links.
use serper_image_tool to search the internet for relevant images.
use serper_maps_tool to search the internet for relevant maps.
Include any urls(links, images, etc.) that are relevant to the user's query


Always return actual URLs from the tools. Do not use placeholder or invented URLs.

Do not include prefix text like "Here is" or "The following"; just return the results.

Ensure all content is accurate, relevant, and verifiable.
""",
    description="Provides dynamic information that is relevant to the user's query.",
    output_key="plan",
    tools=[google_search_tool, serper_image_tool, serper_maps_tool]
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
    description="Executes only the required UI component agents in parallel."
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
    merged_components
  ]
}

You COULD:
- Experiment with the order of components to show a range of different UI layouts that can be built.
- Arrange same components in different order, you can set label before and then after image or any other component to give a separate in elements and make the ui more dynamic and engaging.

You MUST:
- Only merge the components that seems most relevant to the user's request. If a component is not relevant to the user's request, do not include it in the final JSON.
- Keep the component list tight and precise, you don't need to include all the component.
- Follow and use the same JSON reponse and structure as provided by component agents. Make no changes on your own
- Should ensure that component list is not too heavy and as precise as possible.
- Merge components into one JSON
- Preserve formatting and nesting
- Return only final JSON (no explanation, no prefix)
""",
    description="Final merge and sorting agent to return one valid UI component response."
)

# --- 4. Create the SequentialAgent (Executes in Order) ---

ui_planner_formatter_pipeline = SequentialAgent(
    name="UIPlannerFormatterPipeline",
    sub_agents=[ui_planner_agent, component_formatter_agent],
    description="Pipeline: plans UI layout, then formats components."
)

root_agent = ui_planner_formatter_pipeline
