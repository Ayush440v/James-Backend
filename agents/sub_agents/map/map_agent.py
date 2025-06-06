from google.adk.agents import LlmAgent
from .prompt import MAP_AGENT_PROMPT
from google.adk.tools.crewai_tool import CrewaiTool
from CustomSerperTool import CustomSerperDevTool

def create_map_agent(model):
    serper_maps_tool_instance = CustomSerperDevTool(
        n_results=1,
        save_file=False,
        search_type="maps"
    )
    serper_maps_tool = CrewaiTool(
        name="InternetMapsSearch",
        description="Searches the internet specifically for recent maps using Serper.",
        tool=serper_maps_tool_instance
    )

    return LlmAgent(
        name="MapComponentAgent",
        model=model,
        instruction=MAP_AGENT_PROMPT,
        description="Creates map components.",
        output_key="map_component",
        tools=[serper_maps_tool]
    ) 