# agents/agent.py

from google.adk.agents import ParallelAgent, SequentialAgent, LlmAgent
from google.adk.tools.crewai_tool import CrewaiTool
from crewai_tools import ScrapeWebsiteTool
import requests
from CustomSerperTool import CustomSerperDevTool
from google.adk.models.lite_llm import LiteLlm
from tools.plans_tool import plans_tool
from tools.plan_change_tool import plan_change_tool
import os

from agents.sub_agents.ui_info.ui_info_agent import create_ui_info_agent

GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"

ui_info_agent = create_ui_info_agent(GEMINI_MODEL)

root_agent = ui_info_agent

__all__ = ['root_agent']
