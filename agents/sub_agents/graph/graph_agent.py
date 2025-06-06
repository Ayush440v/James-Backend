from google.adk.agents import LlmAgent
from .prompt import GRAPH_AGENT_PROMPT

def create_graph_agent(model):
    return LlmAgent(
        name="GraphComponentAgent",
        model=model,
        instruction=GRAPH_AGENT_PROMPT,
        description="Creates graph components for telecom usage visualization.",
        output_key="graph_component"
    ) 