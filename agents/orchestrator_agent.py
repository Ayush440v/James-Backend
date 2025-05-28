from google.adk.agents import BaseAgent
from agents.label_agent import label_agent
from agents.image_agent import image_agent
from agents.image_grid_agent import image_grid_agent
from agents.link_card_agent import link_card_agent
from agents.map_agent import map_agent
from agents.composite_card_agent import composite_card_agent
from agents.scroll_text_agent import scroll_text_agent
from agents.detail_card_agent import detail_card_agent
from agents.button_agent import button_agent

class OrchestratorAgent(BaseAgent):
    async def _run_async_impl(self, context):
        components = []

        label_response = await label_agent.run_async(context)
        components.append(label_response)

        image_response = await image_agent.run_async(context)
        components.append(image_response)

        image_grid_response = await image_grid_agent.run_async(context)
        components.append(image_grid_response)

        link_card_response = await link_card_agent.run_async(context)
        components.append(link_card_response)

        map_response = await map_agent.run_async(context)
        components.append(map_response)

        composite_card_response = await composite_card_agent.run_async(context)
        components.append(composite_card_response)

        scroll_text_response = await scroll_text_agent.run_async(context)
        components.append(scroll_text_response)

        detail_card_response = await detail_card_agent.run_async(context)
        components.append(detail_card_response)

        button_response = await button_agent.run_async(context)
        components.append(button_response)

        return {
            "components": components
        }
