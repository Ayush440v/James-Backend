import asyncio
from agents.orchestrator_agent import OrchestratorAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from CustomSerperTool import CustomSerperDevTool

async def main():
    # session_service = InMemorySessionService()
    # runner = Runner(
    #     app_name="adk_ui_orchestrator",
    #     session_service=session_service
    # )

    # orchestrator = OrchestratorAgent(name="UIOrchestrator")

    # response = await runner.run(
    #     agent=orchestrator,
    #     user_id="user_123",
    #     session_id="session_abc",
    #     input="Generate a dynamic UI based on the conversation context."
    # )

    # print(response)

    # Instantiate the CrewAI tool
    serper_image_tool_instance = CustomSerperDevTool(
        n_results=10,
        save_file=False,
        search_type="maps"
    )
    print(serper_image_tool_instance.run(search_query="cat pictures"))
if __name__ == "__main__":
    asyncio.run(main())
