from google.adk.agents import LlmAgent
from .prompt import IMAGE_GRID_AGENT_PROMPT
from google.adk.tools.crewai_tool import CrewaiTool
from CustomSerperTool import CustomSerperDevTool

def create_image_grid_agent(model):
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
        name="ImageGridComponentAgent",
        model=model,
        instruction=IMAGE_GRID_AGENT_PROMPT,
        description="Creates image grid components.",
        output_key="image_grid_component",
        tools=[serper_image_tool]
    ) 