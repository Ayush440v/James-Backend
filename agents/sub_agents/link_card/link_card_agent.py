from google.adk.agents import LlmAgent
from .prompt import LINK_CARD_AGENT_PROMPT
from google.adk.tools.crewai_tool import CrewaiTool
from CustomSerperTool import CustomSerperDevTool

def create_link_card_agent(model):
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
        name="LinkCardComponentAgent",
        model=model,
        instruction=LINK_CARD_AGENT_PROMPT,
        description="Creates link card components.",
        output_key="link_card_component",
        tools=[serper_image_tool]
    ) 