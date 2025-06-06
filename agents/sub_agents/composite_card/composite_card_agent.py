from google.adk.agents import LlmAgent
from .prompt import COMPOSITE_CARD_AGENT_PROMPT
from google.adk.tools.crewai_tool import CrewaiTool
from CustomSerperTool import CustomSerperDevTool

def create_composite_card_agent(model):
    serper_image_tool_instance = CustomSerperDevTool(
        n_results=10,
        save_file=False,
        search_type="images"
    )
    serper_image_tool = CrewaiTool(
        name="InternetImageSearch",
        description="Searches the internet specifically for recent images using Serper.",
        tool=serper_image_tool_instance
    )

    return LlmAgent(
        name="CompositeCardComponentAgent",
        model=model,
        instruction=COMPOSITE_CARD_AGENT_PROMPT,
        description="Creates composite card components.",
        output_key="composite_card_component",
        tools=[serper_image_tool]
    ) 