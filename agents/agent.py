# agents/agent.py

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools.crewai_tool import CrewaiTool
from crewai_tools import ScrapeWebsiteTool
import requests
from CustomSerperTool import CustomSerperDevTool
from google.adk.models.lite_llm import LiteLlm
from plans_tool import plans_tool
from plan_change_tool import plan_change_tool
import os

def BSS_TOOL():
    """
    Retrieves the user current mobile plan, usage, and account information.

    Args:
        None

    Returns:
        str: The data, usage, and account information, or None if an error occurs.
    """
    url = "https://ingress.ontology.bss-magic.totogi.solutions/du/totogi-ontology/usageConsumption/v4/queryUsageConsumption"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIrOTE4MDc2NjI0MjQyIiwiYWNjb3VudF9udW1iZXIiOiJTUjIwMjUwNTMwMjA0NzI1IiwibG9jYWxlIjoiZW4tVVMiLCJleHAiOjE3NDk4OTY0NjJ9.1IXJdjcercKM0_MxEDCnEuYnpXvxDmRSbp4O3DOjino",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    try:
        response.raise_for_status()
        result = response.json()
        print(result)
        return (f"result: {result}")
    except requests.exceptions.HTTPError as err:
        print(f"test:Status Code: {response.status_code}, Error: {err}")
        return (f"test:Status Code: {response.status_code}, Error: {err}")

GEMINI_MODEL = "gemini-2.0-flash"
FAST_MODEL = LiteLlm("openai/gemma2-9b-it") if os.getenv("OPENAI_API_BASE") == "https://api.groq.com/openai/v1/" else GEMINI_MODEL

# --- 1. Define UI Component Sub-Agents ---

label_agent = LlmAgent(
    name="LabelComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate one or more label components in JSON format for telecom service information. 
Each label should include 'type', 'text', and 'fontSize'.
Focus on displaying:
- Usage statistics
- Plan details
- Account information
- Service recommendations
Return an array of label objects like:
[
  {"type": "label", "text": "Your Current Plan: Premium Unlimited", "fontSize": 20}
]
Use clear, concise language appropriate for telecom services.
""",
    description="Creates JSON label components for telecom information.",
    output_key="label_component"
)

scroll_text_agent = LlmAgent(
    name="ScrollTextComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate a scrollable text component with detailed telecom service information:
{
  "type": "scrollText",
  "text": "Detailed service information here..."
}
Focus on:
- Detailed plan descriptions
- Terms and conditions
- Usage history
- Service agreements
Create clear, structured content for telecom services.
""",
    description="Creates scrollable text components for detailed telecom information.",
    output_key="scroll_text_component"
)

button_agent = LlmAgent(
    name="ButtonComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate button components for telecom service actions:
{
  "type": "button",
  "buttonTitle": "View Usage",
  "action": "viewUsage",
  "target": "usageDetails"
}
Common actions:
- View Usage
- Change Plan
- View History
- Contact Support
- Upgrade Plan
Use clear, action-oriented button titles.
""",
    description="Creates JSON button components for telecom service actions.",
    output_key="button_component"
)

graph_agent = LlmAgent(
    name="GraphComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate graph components for telecom usage visualization:
{
  "type": "graph",
  "graphType": "line",
  "title": "Data Usage Trend",
  "xAxisLabels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
  "yAxisLabels": ["0GB", "2GB", "4GB", "6GB", "8GB", "10GB"],
  "dataPoints": [
    { "x": 0, "y": 2 },
    { "x": 1, "y": 3 },
    { "x": 2, "y": 4 }
  ]
}
Focus on:
- Data usage trends
- Call minutes usage
- SMS usage
- Plan consumption
Generate realistic usage data visualization.
""",
    description="Creates graph components for telecom usage visualization.",
    output_key="graph_component"
)

pie_chart_agent = LlmAgent(
    name="PieChartComponentAgent",
    model=GEMINI_MODEL,
    instruction="""
Generate pie chart components for telecom service distribution:
{
  "type": "pieChart",
  "centerText": "Usage Distribution",
  "entries": [
    { "label": "Data", "value": 40 },
    { "label": "Voice", "value": 35 },
    { "label": "SMS", "value": 25 }
  ]
}
Focus on:
- Usage type distribution
- Plan feature distribution
- Service category breakdown
Generate meaningful distribution data for telecom services.
""",
    description="Creates pie chart components for telecom service distribution.",
    output_key="pie_chart_component"
)

plans_agent = LlmAgent(
    name="PlansAgent",
    model=GEMINI_MODEL,
    instruction="""
    When asked about available and other mobile plans to upgrade or switch to, use the plans_tool to fetch and display the current available plans from the API.
    Format the response in a clear, user-friendly way, highlighting key features and pricing of each plan.
    Focus on:
    - Plan features and benefits
    - Pricing details
    - Data allowances
    - Voice and SMS limits
    - Additional services
    """,
    description="Handles queries about available mobile plans",
    output_key="plans_response",
    tools=[plans_tool]
)

plan_change_agent = LlmAgent(
    name="PlanChangeAgent",
    model=GEMINI_MODEL,
    instruction="""
    When a user wants to change their plan, use the plan_change_tool to submit the plan change request.
    The tool requires a plan_id from the available plans.
    Format the response to clearly indicate the success or failure of the plan change request.
    Include:
    - Confirmation of plan change
    - Effective date of change
    - New plan details
    - Next steps
    """,
    description="Handles plan change requests",
    output_key="plan_change_response",
    tools=[plan_change_tool]
)

ui_planner_agent = LlmAgent(
    name="ComponentFormatterAgent",
    model=GEMINI_MODEL,
    instruction="""
You are a Telecom Service Assistant that generates dynamic UI components for mobile service information.
Focus on displaying:
- Current plan details
- Usage statistics
- Available plans
- Service recommendations
- Usage history
- Plan comparison

Leave all the fields empty for the components. Only generate the Schema of the components.
NEVER directly answer the user's query, only generate the UI structure.

Component types and required fields:

label: { "type": "label", "text": string, "fontSize": integer }

scroll_text: { "type": "scrollText", "text": string }

button: { "type": "button", "buttonTitle": string, "action": string, "target": string, "buttonUrl": url }

graph: { 
  "type": "graph",
  "graphType": "line" | "bar",
  "title": string,
  "xAxisLabels": [string],
  "yAxisLabels": [string],
  "dataPoints": [{ "x": number, "y": number }]
}

pieChart: {
  "type": "pieChart",
  "centerText": string,
  "entries": [{ "label": string, "value": number }]
}

Output rules:
- Generate components relevant to telecom services
- Focus on user's current plan and usage
- Include plan comparison when relevant
- Show usage trends and statistics
- Provide clear action buttons
- Return only the JSON object—no extra text or comments
""",
    description="Formats UI components for telecom service information.",
    output_key="plan"
)

ui_info_agent = LlmAgent(
    name="UIInfoAgent",
    model=GEMINI_MODEL,
    instruction="""
You are a Telecom Service Assistant that provides dynamic, real-time information about mobile services.

For usage-related queries:
1. Use BSS_TOOL to fetch current usage data
2. Format the response to show:
   - Data usage
   - Voice minutes
   - SMS usage
   - Remaining limits
   - Usage trends

For plan-related queries:
1. If the user wants to see available plans:
   - Use plans_tool to fetch available plans
   - Format plans with features and pricing
   - Highlight benefits of each plan
2. If the user wants to change their plan:
   - First use plans_tool to show available plans
   - Then use plan_change_tool with the selected plan_id
   - Show confirmation and next steps

For account-related queries:
1. Use BSS_TOOL to fetch account information
2. Display:
   - Current plan details
   - Billing information
   - Service status
   - Account settings

General rules:
- Focus only on telecom services
- Use clear, concise language
- Provide information
- Show relevant statistics and trends
- Include appropriate action buttons but without any mock data or placeholder text, these should come only if any real data is available.
- Maintain the original component structure

You will be provided a Scaffolded UI object in {state.plan}.
Your responsibility is to replace placeholder content with relevant telecom service information.
""",
    description="Provides dynamic information about telecom services.",
    output_key="output",
    tools=[BSS_TOOL, plans_tool, plan_change_tool]
)

# --- 2. Create the ParallelAgent (Executes All Component Agents) ---

parallel_ui_agent = ParallelAgent(
    name="ParallelUIComponentGenerator",
    sub_agents=[
        label_agent,
        scroll_text_agent,
        button_agent,
        graph_agent,
        pie_chart_agent,
        plans_agent,
        plan_change_agent
    ],
    description="Executes only the required UI component agents in parallel."
)

# --- 3. Define the Final Merge Agent (Controls Ordering and Output) ---

merge_agent = LlmAgent(
    name="UIOutputMergerAgent",
    model=GEMINI_MODEL,
    instruction="""
You will receive components from session state and combine them into one valid JSON response.
Focus on organizing telecom service information in a logical flow:

1. Current Plan & Usage
2. Available Plans
3. Usage Statistics
4. Action Buttons

Combine components into this structure:
{
  "components": [
    merged_components
  ]
}

You MUST:
- Only include components relevant to telecom services
- Arrange components in a logical order
- Keep the component list focused and precise
- Follow the same JSON structure as provided
- Return only the final JSON
""",
    description="Final merge and sorting agent for telecom service UI components."
)

# --- 4. Create the SequentialAgent (Executes in Order) ---

ui_planner_formatter_pipeline = SequentialAgent(
    name="UIPlannerFormatterPipeline",
    sub_agents=[ui_planner_agent, ui_info_agent],
    description="Pipeline: plans UI layout, then formats components."
)

root_agent = ui_planner_formatter_pipeline
