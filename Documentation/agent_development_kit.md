LLM Agent
The LlmAgent (often aliased simply as Agent) is a core component in ADK, acting as the "thinking" part of your application. It leverages the power of a Large Language Model (LLM) for reasoning, understanding natural language, making decisions, generating responses, and interacting with tools.

Unlike deterministic Workflow Agents that follow predefined execution paths, LlmAgent behavior is non-deterministic. It uses the LLM to interpret instructions and context, deciding dynamically how to proceed, which tools to use (if any), or whether to transfer control to another agent.

Building an effective LlmAgent involves defining its identity, clearly guiding its behavior through instructions, and equipping it with the necessary tools and capabilities.

Defining the Agent's Identity and Purpose¶
First, you need to establish what the agent is and what it's for.

name (Required): Every agent needs a unique string identifier. This name is crucial for internal operations, especially in multi-agent systems where agents need to refer to or delegate tasks to each other. Choose a descriptive name that reflects the agent's function (e.g., customer_support_router, billing_inquiry_agent). Avoid reserved names like user.

description (Optional, Recommended for Multi-Agent): Provide a concise summary of the agent's capabilities. This description is primarily used by other LLM agents to determine if they should route a task to this agent. Make it specific enough to differentiate it from peers (e.g., "Handles inquiries about current billing statements," not just "Billing agent").

model (Required): Specify the underlying LLM that will power this agent's reasoning. This is a string identifier like "gemini-2.0-flash". The choice of model impacts the agent's capabilities, cost, and performance. See the Models page for available options and considerations.


Python

# Example: Defining the basic identity
``` python
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country."
    # instruction and tools will be added next
)
```

Guiding the Agent: Instructions (instruction)¶
The instruction parameter is arguably the most critical for shaping an LlmAgent's behavior. It's a string (or a function returning a string) that tells the agent:

Its core task or goal.
Its personality or persona (e.g., "You are a helpful assistant," "You are a witty pirate").
Constraints on its behavior (e.g., "Only answer questions about X," "Never reveal Y").
How and when to use its tools. You should explain the purpose of each tool and the circumstances under which it should be called, supplementing any descriptions within the tool itself.
The desired format for its output (e.g., "Respond in JSON," "Provide a bulleted list").
Tips for Effective Instructions:

Be Clear and Specific: Avoid ambiguity. Clearly state the desired actions and outcomes.
Use Markdown: Improve readability for complex instructions using headings, lists, etc.
Provide Examples (Few-Shot): For complex tasks or specific output formats, include examples directly in the instruction.
Guide Tool Use: Don't just list tools; explain when and why the agent should use them.
State:

The instruction is a string template, you can use the {var} syntax to insert dynamic values into the instruction.
{var} is used to insert the value of the state variable named var.
{artifact.var} is used to insert the text content of the artifact named var.
If the state variable or artifact does not exist, the agent will raise an error. If you want to ignore the error, you can append a ? to the variable name as in {var?}.

Python
Java

# Example: Adding instructions
``` python
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country.
When a user asks for the capital of a country:
1. Identify the country name from the user's query.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city.
Example Query: "What's the capital of {country}?"
Example Response: "The capital of France is Paris."
""",
    # tools will be added next
)
```

(Note: For instructions that apply to all agents in a system, consider using global_instruction on the root agent, detailed further in the Multi-Agents section.)

Equipping the Agent: Tools (tools)¶
Tools give your LlmAgent capabilities beyond the LLM's built-in knowledge or reasoning. They allow the agent to interact with the outside world, perform calculations, fetch real-time data, or execute specific actions.

tools (Optional): Provide a list of tools the agent can use. Each item in the list can be:
A native function or method (wrapped as a FunctionTool). Python ADK automatically wraps the native function into a FuntionTool whereas, you must explicitly wrap your Java methods using FunctionTool.create(...)
An instance of a class inheriting from BaseTool.
An instance of another agent (AgentTool, enabling agent-to-agent delegation - see Multi-Agents).
The LLM uses the function/tool names, descriptions (from docstrings or the description field), and parameter schemas to decide which tool to call based on the conversation and its instructions.


Python

# Define a tool function
``` python
def get_capital_city(country: str) -> str:
  """Retrieves the capital city for a given country."""
  # Replace with actual logic (e.g., API call, database lookup)
  capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
  return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

# Add the tool to the agent
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country... (previous instruction text)""",
    tools=[get_capital_city] # Provide the function directly
)
```

Learn more about Tools in the Tools section.

Advanced Configuration & Control¶
Beyond the core parameters, LlmAgent offers several options for finer control:

Fine-Tuning LLM Generation (generate_content_config)¶
You can adjust how the underlying LLM generates responses using generate_content_config.

generate_content_config (Optional): Pass an instance of google.genai.types.GenerateContentConfig to control parameters like temperature (randomness), max_output_tokens (response length), top_p, top_k, and safety settings.

Python

``` python
from google.genai import types

agent = LlmAgent(
    # ... other params
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2, # More deterministic output
        max_output_tokens=250
    )
)
```

Structuring Data (input_schema, output_schema, output_key)¶
For scenarios requiring structured data exchange with an LLM Agent, the ADK provides mechanisms to define expected input and desired output formats using schema definitions.

input_schema (Optional): Define a schema representing the expected input structure. If set, the user message content passed to this agent must be a JSON string conforming to this schema. Your instructions should guide the user or preceding agent accordingly.

output_schema (Optional): Define a schema representing the desired output structure. If set, the agent's final response must be a JSON string conforming to this schema.

Constraint: Using output_schema enables controlled generation within the LLM but disables the agent's ability to use tools or transfer control to other agents. Your instructions must guide the LLM to produce JSON matching the schema directly.
output_key (Optional): Provide a string key. If set, the text content of the agent's final response will be automatically saved to the session's state dictionary under this key. This is useful for passing results between agents or steps in a workflow.

In Python, this might look like: session.state[output_key] = agent_response_text
In Java: session.state().put(outputKey, agentResponseText)

Python
The input and output schema is typically a Pydantic BaseModel.

``` python
from pydantic import BaseModel, Field

class CapitalOutput(BaseModel):
    capital: str = Field(description="The capital of the country.")

structured_capital_agent = LlmAgent(
    # ... name, model, description
    instruction="""You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {"capital": "capital_name"}""",
    output_schema=CapitalOutput, # Enforce JSON output
    output_key="found_capital"  # Store result in state['found_capital']
    # Cannot use tools=[get_capital_city] effectively here
)
```

Managing Context (include_contents)¶
Control whether the agent receives the prior conversation history.

include_contents (Optional, Default: 'default'): Determines if the contents (history) are sent to the LLM.
'default': The agent receives the relevant conversation history.
'none': The agent receives no prior contents. It operates based solely on its current instruction and any input provided in the current turn (useful for stateless tasks or enforcing specific contexts).

Python

stateless_agent = LlmAgent(
    # ... other params
    include_contents='none'
)

Planning & Code Execution¶
python_only

For more complex reasoning involving multiple steps or executing code:

planner (Optional): Assign a BasePlanner instance to enable multi-step reasoning and planning before execution. (See Multi-Agents patterns).
code_executor (Optional): Provide a BaseCodeExecutor instance to allow the agent to execute code blocks (e.g., Python) found in the LLM's response. (See Tools/Built-in tools).
Putting It Together: Example¶
Code
Here's the complete basic capital_agent:


Python
Java

# --- Full example code demonstrating LlmAgent with Tools vs. Output Schema ---
import json # Needed for pretty printing dicts

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. Define Constants ---
APP_NAME = "agent_comparison_app"
USER_ID = "test_user_456"
SESSION_ID_TOOL_AGENT = "session_tool_agent_xyz"
SESSION_ID_SCHEMA_AGENT = "session_schema_agent_xyz"
MODEL_NAME = "gemini-2.0-flash"

# --- 2. Define Schemas ---

# Input schema used by both agents
class CountryInput(BaseModel):
    country: str = Field(description="The country to get information about.")

# Output schema ONLY for the second agent
class CapitalInfoOutput(BaseModel):
    capital: str = Field(description="The capital city of the country.")
    # Note: Population is illustrative; the LLM will infer or estimate this
    # as it cannot use tools when output_schema is set.
    population_estimate: str = Field(description="An estimated population of the capital city.")

# --- 3. Define the Tool (Only for the first agent) ---
def get_capital_city(country: str) -> str:
    """Retrieves the capital city of a given country."""
    print(f"\n-- Tool Call: get_capital_city(country='{country}') --")
    country_capitals = {
        "united states": "Washington, D.C.",
        "canada": "Ottawa",
        "france": "Paris",
        "japan": "Tokyo",
    }
    result = country_capitals.get(country.lower(), f"Sorry, I couldn't find the capital for {country}.")
    print(f"-- Tool Result: '{result}' --")
    return result

# --- 4. Configure Agents ---

# Agent 1: Uses a tool and output_key
capital_agent_with_tool = LlmAgent(
    model=MODEL_NAME,
    name="capital_agent_tool",
    description="Retrieves the capital city using a specific tool.",
    instruction="""You are a helpful agent that provides the capital city of a country using a tool.
The user will provide the country name in a JSON format like {"country": "country_name"}.
1. Extract the country name.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city found by the tool.
""",
    tools=[get_capital_city],
    input_schema=CountryInput,
    output_key="capital_tool_result", # Store final text response
)

# Agent 2: Uses output_schema (NO tools possible)
structured_info_agent_schema = LlmAgent(
    model=MODEL_NAME,
    name="structured_info_agent_schema",
    description="Provides capital and estimated population in a specific JSON format.",
    instruction=f"""You are an agent that provides country information.
The user will provide the country name in a JSON format like {{"country": "country_name"}}.
Respond ONLY with a JSON object matching this exact schema:
{json.dumps(CapitalInfoOutput.model_json_schema(), indent=2)}
Use your knowledge to determine the capital and estimate the population. Do not use any tools.
""",
    # *** NO tools parameter here - using output_schema prevents tool use ***
    input_schema=CountryInput,
    output_schema=CapitalInfoOutput, # Enforce JSON output structure
    output_key="structured_info_result", # Store final JSON response
)

# --- 5. Set up Session Management and Runners ---
session_service = InMemorySessionService()

# Create separate sessions for clarity, though not strictly necessary if context is managed
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_TOOL_AGENT)
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_SCHEMA_AGENT)

# Create a runner for EACH agent
capital_runner = Runner(
    agent=capital_agent_with_tool,
    app_name=APP_NAME,
    session_service=session_service
)
structured_runner = Runner(
    agent=structured_info_agent_schema,
    app_name=APP_NAME,
    session_service=session_service
)

# --- 6. Define Agent Interaction Logic ---
``` python
async def call_agent_and_print(
    runner_instance: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    query_json: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query_json}")

    user_content = types.Content(role='user', parts=[types.Part(text=query_json)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = session_service.get_session(app_name=APP_NAME,
                                                  user_id=USER_ID,
                                                  session_id=session_id)
    stored_output = current_session.state.get(agent_instance.output_key)

    # Pretty print if the stored output looks like JSON (likely from output_schema)
    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        # Attempt to parse and pretty print if it's JSON
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
         # Otherwise, print as string
        print(stored_output)
    print("-" * 30)


# --- 7. Run Interactions ---
async def main():
    print("--- Testing Agent with Tool ---")
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "France"}')
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "Canada"}')

    print("\n\n--- Testing Agent with Output Schema (No Tool Use) ---")
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "France"}')
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "Japan"}')

if __name__ == "__main__":
    await main()
```

(This example demonstrates the core concepts. More complex agents might incorporate schemas, context control, planning, etc.)

# Sequential Agents
The SequentialAgent is a workflow agent that executes its sub-agents in the order they are specified in the list.

Use the SequentialAgent when you want the execution to occur in a fixed, strict order.

Example¶
You want to build an agent that can summarize any webpage, using two tools: Get Page Contents and Summarize Page. Because the agent must always call Get Page Contents before calling Summarize Page (you can't summarize from nothing!), you should build your agent using a SequentialAgent.
As with other workflow agents, the SequentialAgent is not powered by an LLM, and is thus deterministic in how it executes. That being said, workflow agents are concerned only with their execution (i.e. in sequence), and not their internal logic; the tools or sub-agents of a workflow agent may or may not utilize LLMs.

How it works¶
When the SequentialAgent's Run Async method is called, it performs the following actions:

Iteration: It iterates through the sub agents list in the order they were provided.
Sub-Agent Execution: For each sub-agent in the list, it calls the sub-agent's Run Async method.
Sequential Agent

Full Example: Code Development Pipeline¶
Consider a simplified code development pipeline:

Code Writer Agent: An LLM Agent that generates initial code based on a specification.
Code Reviewer Agent: An LLM Agent that reviews the generated code for errors, style issues, and adherence to best practices. It receives the output of the Code Writer Agent.
Code Refactorer Agent: An LLM Agent that takes the reviewed code (and the reviewer's comments) and refactors it to improve quality and address issues.
A SequentialAgent is perfect for this:


SequentialAgent(sub_agents=[CodeWriterAgent, CodeReviewerAgent, CodeRefactorerAgent])
This ensures the code is written, then reviewed, and finally refactored, in a strict, dependable order. The output from each sub-agent is passed to the next by storing them in state via Output Key.

Code

Python
Java

# Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup

# --- 1. Define Sub-Agents for Each Pipeline Stage ---

# Code Writer Agent
# Takes the initial specification (from user query) and writes code.
code_writer_agent = LlmAgent(
    name="CodeWriterAgent",
    model=GEMINI_MODEL,
    # Change 3: Improved instruction
    instruction="""You are a Python Code Generator.
Based *only* on the user's request, write Python code that fulfills the requirement.
Output *only* the complete Python code block, enclosed in triple backticks (```python ... ```). 
Do not add any other text before or after the code block.
""",
    description="Writes initial Python code based on a specification.",
    output_key="generated_code" # Stores output in state['generated_code']
)

# Code Reviewer Agent
# Takes the code generated by the previous agent (read from state) and provides feedback.
code_reviewer_agent = LlmAgent(
    name="CodeReviewerAgent",
    model=GEMINI_MODEL,
    # Change 3: Improved instruction, correctly using state key injection
    instruction="""You are an expert Python Code Reviewer. 
    Your task is to provide constructive feedback on the provided code.

    **Code to Review:**
    ```python
    {generated_code}
    ```

**Review Criteria:**
1.  **Correctness:** Does the code work as intended? Are there logic errors?
2.  **Readability:** Is the code clear and easy to understand? Follows PEP 8 style guidelines?
3.  **Efficiency:** Is the code reasonably efficient? Any obvious performance bottlenecks?
4.  **Edge Cases:** Does the code handle potential edge cases or invalid inputs gracefully?
5.  **Best Practices:** Does the code follow common Python best practices?

**Output:**
Provide your feedback as a concise, bulleted list. Focus on the most important points for improvement.
If the code is excellent and requires no changes, simply state: "No major issues found."
Output *only* the review comments or the "No major issues" statement.
""",
    description="Reviews code and provides feedback.",
    output_key="review_comments", # Stores output in state['review_comments']
)


# Code Refactorer Agent
# Takes the original code and the review comments (read from state) and refactors the code.
code_refactorer_agent = LlmAgent(
    name="CodeRefactorerAgent",
    model=GEMINI_MODEL,
    # Change 3: Improved instruction, correctly using state key injection
    instruction="""You are a Python Code Refactoring AI.
Your goal is to improve the given Python code based on the provided review comments.

  **Original Code:**
  ```python
  {generated_code}
  ```

  **Review Comments:**
  {review_comments}

**Task:**
Carefully apply the suggestions from the review comments to refactor the original code.
If the review comments state "No major issues found," return the original code unchanged.
Ensure the final code is complete, functional, and includes necessary imports and docstrings.

**Output:**
Output *only* the final, refactored Python code block, enclosed in triple backticks (```python ... ```). 
Do not add any other text before or after the code block.
""",
    description="Refactors code based on review comments.",
    output_key="refactored_code", # Stores output in state['refactored_code']
)


# --- 2. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub_agents in order.
code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[code_writer_agent, code_reviewer_agent, code_refactorer_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
    # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = code_pipeline_agent

# Loop agents¶
The LoopAgent¶
The LoopAgent is a workflow agent that executes its sub-agents in a loop (i.e. iteratively). It repeatedly runs a sequence of agents for a specified number of iterations or until a termination condition is met.

Use the LoopAgent when your workflow involves repetition or iterative refinement, such as like revising code.

Example¶
You want to build an agent that can generate images of food, but sometimes when you want to generate a specific number of items (e.g. 5 bananas), it generates a different number of those items in the image (e.g. an image of 7 bananas). You have two tools: Generate Image, Count Food Items. Because you want to keep generating images until it either correctly generates the specified number of items, or after a certain number of iterations, you should build your agent using a LoopAgent.
As with other workflow agents, the LoopAgent is not powered by an LLM, and is thus deterministic in how it executes. That being said, workflow agents are only concerned only with their execution (i.e. in a loop), and not their internal logic; the tools or sub-agents of a workflow agent may or may not utilize LLMs.

How it Works¶
When the LoopAgent's Run Async method is called, it performs the following actions:

Sub-Agent Execution: It iterates through the Sub Agents list in order. For each sub-agent, it calls the agent's Run Async method.
Termination Check:

Crucially, the LoopAgent itself does not inherently decide when to stop looping. You must implement a termination mechanism to prevent infinite loops. Common strategies include:

Max Iterations: Set a maximum number of iterations in the LoopAgent. The loop will terminate after that many iterations.
Escalation from sub-agent: Design one or more sub-agents to evaluate a condition (e.g., "Is the document quality good enough?", "Has a consensus been reached?"). If the condition is met, the sub-agent can signal termination (e.g., by raising a custom event, setting a flag in a shared context, or returning a specific value).
Loop Agent

Full Example: Iterative Document Improvement¶
Imagine a scenario where you want to iteratively improve a document:

Writer Agent: An LlmAgent that generates or refines a draft on a topic.
Critic Agent: An LlmAgent that critiques the draft, identifying areas for improvement.


LoopAgent(sub_agents=[WriterAgent, CriticAgent], max_iterations=5)
In this setup, the LoopAgent would manage the iterative process. The CriticAgent could be designed to return a "STOP" signal when the document reaches a satisfactory quality level, preventing further iterations. Alternatively, the max iterations parameter could be used to limit the process to a fixed number of cycles, or external logic could be implemented to make stop decisions. The loop would run at most five times, ensuring the iterative refinement doesn't continue indefinitely.

Full Code

Python
Java

# Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup

import asyncio
import os
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent, SequentialAgent
from google.genai import types
from google.adk.runners import InMemoryRunner
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.tool_context import ToolContext
from typing import AsyncGenerator, Optional
from google.adk.events import Event, EventActions

# --- Constants ---
APP_NAME = "doc_writing_app_v3" # New App Name
USER_ID = "dev_user_01"
SESSION_ID_BASE = "loop_exit_tool_session" # New Base Session ID
GEMINI_MODEL = "gemini-2.0-flash"
STATE_INITIAL_TOPIC = "initial_topic"

# --- State Keys ---
STATE_CURRENT_DOC = "current_document"
STATE_CRITICISM = "criticism"
# Define the exact phrase the Critic should use to signal completion
COMPLETION_PHRASE = "No major issues found."

# --- Tool Definition ---
def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}

# --- Agent Definitions ---

# STEP 1: Initial Writer Agent (Runs ONCE at the beginning)
initial_writer_agent = LlmAgent(
    name="InitialWriterAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    # MODIFIED Instruction: Ask for a slightly more developed start
    instruction=f"""You are a Creative Writing Assistant tasked with starting a story.
    Write the *first draft* of a short story (aim for 2-4 sentences).
    Base the content *only* on the topic provided below. Try to introduce a specific element (like a character, a setting detail, or a starting action) to make it engaging.
    Topic: {{initial_topic}}

    Output *only* the story/document text. Do not add introductions or explanations.
""",
    description="Writes the initial document draft based on the topic, aiming for some initial substance.",
    output_key=STATE_CURRENT_DOC
)

# STEP 2a: Critic Agent (Inside the Refinement Loop)
critic_agent_in_loop = LlmAgent(
    name="CriticAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    # MODIFIED Instruction: More nuanced completion criteria, look for clear improvement paths.
    instruction=f"""You are a Constructive Critic AI reviewing a short document draft (typically 2-6 sentences). Your goal is balanced feedback.

    **Document to Review:**
    ```
    {{current_document}}
    ```

    **Task:**
    Review the document for clarity, engagement, and basic coherence according to the initial topic (if known).

    IF you identify 1-2 *clear and actionable* ways the document could be improved to better capture the topic or enhance reader engagement (e.g., "Needs a stronger opening sentence", "Clarify the character's goal"):
    Provide these specific suggestions concisely. Output *only* the critique text.

    ELSE IF the document is coherent, addresses the topic adequately for its length, and has no glaring errors or obvious omissions:
    Respond *exactly* with the phrase "{COMPLETION_PHRASE}" and nothing else. It doesn't need to be perfect, just functionally complete for this stage. Avoid suggesting purely subjective stylistic preferences if the core is sound.

    Do not add explanations. Output only the critique OR the exact completion phrase.
""",
    description="Reviews the current draft, providing critique if clear improvements are needed, otherwise signals completion.",
    output_key=STATE_CRITICISM
)


# STEP 2b: Refiner/Exiter Agent (Inside the Refinement Loop)
refiner_agent_in_loop = LlmAgent(
    name="RefinerAgent",
    model=GEMINI_MODEL,
    # Relies solely on state via placeholders
    include_contents='none',
    instruction=f"""You are a Creative Writing Assistant refining a document based on feedback OR exiting the process.
    **Current Document:**
    ```
    {{current_document}}
    ```
    **Critique/Suggestions:**
    {{criticism}}

    **Task:**
    Analyze the 'Critique/Suggestions'.
    IF the critique is *exactly* "{COMPLETION_PHRASE}":
    You MUST call the 'exit_loop' function. Do not output any text.
    ELSE (the critique contains actionable feedback):
    Carefully apply the suggestions to improve the 'Current Document'. Output *only* the refined document text.

    Do not add explanations. Either output the refined document OR call the exit_loop function.
""",
    description="Refines the document based on critique, or calls exit_loop if critique indicates completion.",
    tools=[exit_loop], # Provide the exit_loop tool
    output_key=STATE_CURRENT_DOC # Overwrites state['current_document'] with the refined version
)


# STEP 2: Refinement Loop Agent
refinement_loop = LoopAgent(
    name="RefinementLoop",
    # Agent order is crucial: Critique first, then Refine/Exit
    sub_agents=[
        critic_agent_in_loop,
        refiner_agent_in_loop,
    ],
    max_iterations=5 # Limit loops
)

# STEP 3: Overall Sequential Pipeline
# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = SequentialAgent(
    name="IterativeWritingPipeline",
    sub_agents=[
        initial_writer_agent, # Run first to create initial doc
        refinement_loop       # Then run the critique/refine loop
    ],
    description="Writes an initial document and then iteratively refines it with critique using an exit tool."
)

# Parallel agents¶
The ParallelAgent is a workflow agent that executes its sub-agents concurrently. This dramatically speeds up workflows where tasks can be performed independently.

Use ParallelAgent when: For scenarios prioritizing speed and involving independent, resource-intensive tasks, a ParallelAgent facilitates efficient parallel execution. When sub-agents operate without dependencies, their tasks can be performed concurrently, significantly reducing overall processing time.

As with other workflow agents, the ParallelAgent is not powered by an LLM, and is thus deterministic in how it executes. That being said, workflow agents are only concerned with their execution (i.e. executing sub-agents in parallel), and not their internal logic; the tools or sub-agents of a workflow agent may or may not utilize LLMs.

Example¶
This approach is particularly beneficial for operations like multi-source data retrieval or heavy computations, where parallelization yields substantial performance gains. Importantly, this strategy assumes no inherent need for shared state or direct information exchange between the concurrently executing agents.

How it works¶
When the ParallelAgent's run_async() method is called:

Concurrent Execution: It initiates the run_async() method of each sub-agent present in the sub_agents list concurrently. This means all the agents start running at (approximately) the same time.
Independent Branches: Each sub-agent operates in its own execution branch. There is no automatic sharing of conversation history or state between these branches during execution.
Result Collection: The ParallelAgent manages the parallel execution and, typically, provides a way to access the results from each sub-agent after they have completed (e.g., through a list of results or events). The order of results may not be deterministic.
Independent Execution and State Management¶
It's crucial to understand that sub-agents within a ParallelAgent run independently. If you need communication or data sharing between these agents, you must implement it explicitly. Possible approaches include:

Shared InvocationContext: You could pass a shared InvocationContext object to each sub-agent. This object could act as a shared data store. However, you'd need to manage concurrent access to this shared context carefully (e.g., using locks) to avoid race conditions.
External State Management: Use an external database, message queue, or other mechanism to manage shared state and facilitate communication between agents.
Post-Processing: Collect results from each branch, and then implement logic to coordinate data afterwards.
Parallel Agent

Full Example: Parallel Web Research¶
Imagine researching multiple topics simultaneously:

Researcher Agent 1: An LlmAgent that researches "renewable energy sources."
Researcher Agent 2: An LlmAgent that researches "electric vehicle technology."
Researcher Agent 3: An LlmAgent that researches "carbon capture methods."


ParallelAgent(sub_agents=[ResearcherAgent1, ResearcherAgent2, ResearcherAgent3])
These research tasks are independent. Using a ParallelAgent allows them to run concurrently, potentially reducing the total research time significantly compared to running them sequentially. The results from each agent would be collected separately after they finish.

Full Code

Python
Java

 # Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup
 # --- 1. Define Researcher Sub-Agents (to run in parallel) ---

 # Researcher 1: Renewable Energy
 researcher_agent_1 = LlmAgent(
     name="RenewableEnergyResearcher",
     model=GEMINI_MODEL,
     instruction="""You are an AI Research Assistant specializing in energy.
 Research the latest advancements in 'renewable energy sources'.
 Use the Google Search tool provided.
 Summarize your key findings concisely (1-2 sentences).
 Output *only* the summary.
 """,
     description="Researches renewable energy sources.",
     tools=[google_search],
     # Store result in state for the merger agent
     output_key="renewable_energy_result"
 )

 # Researcher 2: Electric Vehicles
 researcher_agent_2 = LlmAgent(
     name="EVResearcher",
     model=GEMINI_MODEL,
     instruction="""You are an AI Research Assistant specializing in transportation.
 Research the latest developments in 'electric vehicle technology'.
 Use the Google Search tool provided.
 Summarize your key findings concisely (1-2 sentences).
 Output *only* the summary.
 """,
     description="Researches electric vehicle technology.",
     tools=[google_search],
     # Store result in state for the merger agent
     output_key="ev_technology_result"
 )

 # Researcher 3: Carbon Capture
 researcher_agent_3 = LlmAgent(
     name="CarbonCaptureResearcher",
     model=GEMINI_MODEL,
     instruction="""You are an AI Research Assistant specializing in climate solutions.
 Research the current state of 'carbon capture methods'.
 Use the Google Search tool provided.
 Summarize your key findings concisely (1-2 sentences).
 Output *only* the summary.
 """,
     description="Researches carbon capture methods.",
     tools=[google_search],
     # Store result in state for the merger agent
     output_key="carbon_capture_result"
 )

 # --- 2. Create the ParallelAgent (Runs researchers concurrently) ---
 # This agent orchestrates the concurrent execution of the researchers.
 # It finishes once all researchers have completed and stored their results in state.
 parallel_research_agent = ParallelAgent(
     name="ParallelWebResearchAgent",
     sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
     description="Runs multiple research agents in parallel to gather information."
 )

 # --- 3. Define the Merger Agent (Runs *after* the parallel agents) ---
 # This agent takes the results stored in the session state by the parallel agents
 # and synthesizes them into a single, structured response with attributions.
 merger_agent = LlmAgent(
     name="SynthesisAgent",
     model=GEMINI_MODEL,  # Or potentially a more powerful model if needed for synthesis
     instruction="""You are an AI Assistant responsible for combining research findings into a structured report.

 Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.

 **Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not present in these specific summaries.**

 **Input Summaries:**

 *   **Renewable Energy:**
     {renewable_energy_result}

 *   **Electric Vehicles:**
     {ev_technology_result}

 *   **Carbon Capture:**
     {carbon_capture_result}

 **Output Format:**

 ## Summary of Recent Sustainable Technology Advancements

 ### Renewable Energy Findings
 (Based on RenewableEnergyResearcher's findings)
 [Synthesize and elaborate *only* on the renewable energy input summary provided above.]

 ### Electric Vehicle Findings
 (Based on EVResearcher's findings)
 [Synthesize and elaborate *only* on the EV input summary provided above.]

 ### Carbon Capture Findings
 (Based on CarbonCaptureResearcher's findings)
 [Synthesize and elaborate *only* on the carbon capture input summary provided above.]

 ### Overall Conclusion
 [Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

 Output *only* the structured report following this format. Do not include introductory or concluding phrases outside this structure, and strictly adhere to using only the provided input summary content.
 """,
     description="Combines research findings from parallel agents into a structured, cited report, strictly grounded on provided inputs.",
     # No tools needed for merging
     # No output_key needed here, as its direct response is the final output of the sequence
 )


 # --- 4. Create the SequentialAgent (Orchestrates the overall flow) ---
 # This is the main agent that will be run. It first executes the ParallelAgent
 # to populate the state, and then executes the MergerAgent to produce the final output.
 sequential_pipeline_agent = SequentialAgent(
     name="ResearchAndSynthesisPipeline",
     # Run parallel research first, then merge
     sub_agents=[parallel_research_agent, merger_agent],
     description="Coordinates parallel research and synthesizes the results."
 )

 root_agent = sequential_pipeline_agent

 Advanced Concept

Building custom agents by directly implementing _run_async_impl (or its equivalent in other languages) provides powerful control but is more complex than using the predefined LlmAgent or standard WorkflowAgent types. We recommend understanding those foundational agent types first before tackling custom orchestration logic.

Custom agents¶
Custom agents provide the ultimate flexibility in ADK, allowing you to define arbitrary orchestration logic by inheriting directly from BaseAgent and implementing your own control flow. This goes beyond the predefined patterns of SequentialAgent, LoopAgent, and ParallelAgent, enabling you to build highly specific and complex agentic workflows.

Introduction: Beyond Predefined Workflows¶
What is a Custom Agent?¶
A Custom Agent is essentially any class you create that inherits from google.adk.agents.BaseAgent and implements its core execution logic within the _run_async_impl asynchronous method. You have complete control over how this method calls other agents (sub-agents), manages state, and handles events.

Note

The specific method name for implementing an agent's core asynchronous logic may vary slightly by SDK language (e.g., runAsyncImpl in Java, _run_async_impl in Python). Refer to the language-specific API documentation for details.

Why Use Them?¶
While the standard Workflow Agents (SequentialAgent, LoopAgent, ParallelAgent) cover common orchestration patterns, you'll need a Custom agent when your requirements include:

Conditional Logic: Executing different sub-agents or taking different paths based on runtime conditions or the results of previous steps.
Complex State Management: Implementing intricate logic for maintaining and updating state throughout the workflow beyond simple sequential passing.
External Integrations: Incorporating calls to external APIs, databases, or custom libraries directly within the orchestration flow control.
Dynamic Agent Selection: Choosing which sub-agent(s) to run next based on dynamic evaluation of the situation or input.
Unique Workflow Patterns: Implementing orchestration logic that doesn't fit the standard sequential, parallel, or loop structures.
intro_components.png

Implementing Custom Logic:¶
The core of any custom agent is the method where you define its unique asynchronous behavior. This method allows you to orchestrate sub-agents and manage the flow of execution.


Python
Java
The heart of any custom agent is the _run_async_impl method. This is where you define its unique behavior.

Signature: async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
Asynchronous Generator: It must be an async def function and return an AsyncGenerator. This allows it to yield events produced by sub-agents or its own logic back to the runner.
ctx (InvocationContext): Provides access to crucial runtime information, most importantly ctx.session.state, which is the primary way to share data between steps orchestrated by your custom agent.

Key Capabilities within the Core Asynchronous Method:


Python
Java
Calling Sub-Agents: You invoke sub-agents (which are typically stored as instance attributes like self.my_llm_agent) using their run_async method and yield their events:


async for event in self.some_sub_agent.run_async(ctx):
    # Optionally inspect or log the event
    yield event # Pass the event up
Managing State: Read from and write to the session state dictionary (ctx.session.state) to pass data between sub-agent calls or make decisions:


# Read data set by a previous agent
previous_result = ctx.session.state.get("some_key")

# Make a decision based on state
if previous_result == "some_value":
    # ... call a specific sub-agent ...
else:
    # ... call another sub-agent ...

# Store a result for a later step (often done via a sub-agent's output_key)
# ctx.session.state["my_custom_result"] = "calculated_value"
Implementing Control Flow: Use standard Python constructs (if/elif/else, for/while loops, try/except) to create sophisticated, conditional, or iterative workflows involving your sub-agents.


Managing Sub-Agents and State¶
Typically, a custom agent orchestrates other agents (like LlmAgent, LoopAgent, etc.).

Initialization: You usually pass instances of these sub-agents into your custom agent's constructor and store them as instance fields/attributes (e.g., this.story_generator = story_generator_instance or self.story_generator = story_generator_instance). This makes them accessible within the custom agent's core asynchronous execution logic (such as: _run_async_impl method).
Sub Agents List: When initializing the BaseAgent using it's super() constructor, you should pass a sub agents list. This list tells the ADK framework about the agents that are part of this custom agent's immediate hierarchy. It's important for framework features like lifecycle management, introspection, and potentially future routing capabilities, even if your core execution logic (_run_async_impl) calls the agents directly via self.xxx_agent. Include the agents that your custom logic directly invokes at the top level.
State: As mentioned, ctx.session.state is the standard way sub-agents (especially LlmAgents using output key) communicate results back to the orchestrator and how the orchestrator passes necessary inputs down.
Design Pattern Example: StoryFlowAgent¶
Let's illustrate the power of custom agents with an example pattern: a multi-stage content generation workflow with conditional logic.

Goal: Create a system that generates a story, iteratively refines it through critique and revision, performs final checks, and crucially, regenerates the story if the final tone check fails.

Why Custom? The core requirement driving the need for a custom agent here is the conditional regeneration based on the tone check. Standard workflow agents don't have built-in conditional branching based on the outcome of a sub-agent's task. We need custom logic (if tone == "negative": ...) within the orchestrator.

Part 1: Simplified custom agent Initialization¶

Python
Java
We define the StoryFlowAgent inheriting from BaseAgent. In __init__, we store the necessary sub-agents (passed in) as instance attributes and tell the BaseAgent framework about the top-level agents this custom agent will directly orchestrate.


class StoryFlowAgent(BaseAgent):
    """
    Custom agent for a story generation and refinement workflow.

    This agent orchestrates a sequence of LLM agents to generate a story,
    critique it, revise it, check grammar and tone, and potentially
    regenerate the story if the tone is negative.
    """

    # --- Field Declarations for Pydantic ---
    # Declare the agents passed during initialization as class attributes with type hints
    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent

    loop_agent: LoopAgent
    sequential_agent: SequentialAgent

    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        story_generator: LlmAgent,
        critic: LlmAgent,
        reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
    ):
        """
        Initializes the StoryFlowAgent.

        Args:
            name: The name of the agent.
            story_generator: An LlmAgent to generate the initial story.
            critic: An LlmAgent to critique the story.
            reviser: An LlmAgent to revise the story based on criticism.
            grammar_check: An LlmAgent to check the grammar.
            tone_check: An LlmAgent to analyze the tone.
        """
        # Create internal agents *before* calling super().__init__
        loop_agent = LoopAgent(
            name="CriticReviserLoop", sub_agents=[critic, reviser], max_iterations=2
        )
        sequential_agent = SequentialAgent(
            name="PostProcessing", sub_agents=[grammar_check, tone_check]
        )

        # Define the sub_agents list for the framework
        sub_agents_list = [
            story_generator,
            loop_agent,
            sequential_agent,
        ]

        # Pydantic will validate and assign them based on the class annotations.
        super().__init__(
            name=name,
            story_generator=story_generator,
            critic=critic,
            reviser=reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,
            loop_agent=loop_agent,
            sequential_agent=sequential_agent,
            sub_agents=sub_agents_list, # Pass the sub_agents list directly
        )

Part 2: Defining the Custom Execution Logic¶

Python
Java
This method orchestrates the sub-agents using standard Python async/await and control flow.


@override
async def _run_async_impl(
    self, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    """
    Implements the custom orchestration logic for the story workflow.
    Uses the instance attributes assigned by Pydantic (e.g., self.story_generator).
    """
    logger.info(f"[{self.name}] Starting story generation workflow.")

    # 1. Initial Story Generation
    logger.info(f"[{self.name}] Running StoryGenerator...")
    async for event in self.story_generator.run_async(ctx):
        logger.info(f"[{self.name}] Event from StoryGenerator: {event.model_dump_json(indent=2, exclude_none=True)}")
        yield event

    # Check if story was generated before proceeding
    if "current_story" not in ctx.session.state or not ctx.session.state["current_story"]:
         logger.error(f"[{self.name}] Failed to generate initial story. Aborting workflow.")
         return # Stop processing if initial story failed

    logger.info(f"[{self.name}] Story state after generator: {ctx.session.state.get('current_story')}")


    # 2. Critic-Reviser Loop
    logger.info(f"[{self.name}] Running CriticReviserLoop...")
    # Use the loop_agent instance attribute assigned during init
    async for event in self.loop_agent.run_async(ctx):
        logger.info(f"[{self.name}] Event from CriticReviserLoop: {event.model_dump_json(indent=2, exclude_none=True)}")
        yield event

    logger.info(f"[{self.name}] Story state after loop: {ctx.session.state.get('current_story')}")

    # 3. Sequential Post-Processing (Grammar and Tone Check)
    logger.info(f"[{self.name}] Running PostProcessing...")
    # Use the sequential_agent instance attribute assigned during init
    async for event in self.sequential_agent.run_async(ctx):
        logger.info(f"[{self.name}] Event from PostProcessing: {event.model_dump_json(indent=2, exclude_none=True)}")
        yield event

    # 4. Tone-Based Conditional Logic
    tone_check_result = ctx.session.state.get("tone_check_result")
    logger.info(f"[{self.name}] Tone check result: {tone_check_result}")

    if tone_check_result == "negative":
        logger.info(f"[{self.name}] Tone is negative. Regenerating story...")
        async for event in self.story_generator.run_async(ctx):
            logger.info(f"[{self.name}] Event from StoryGenerator (Regen): {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
    else:
        logger.info(f"[{self.name}] Tone is not negative. Keeping current story.")
        pass

    logger.info(f"[{self.name}] Workflow finished.")
Explanation of Logic:
The initial story_generator runs. Its output is expected to be in ctx.session.state["current_story"].
The loop_agent runs, which internally calls the critic and reviser sequentially for max_iterations times. They read/write current_story and criticism from/to the state.
The sequential_agent runs, calling grammar_check then tone_check, reading current_story and writing grammar_suggestions and tone_check_result to the state.
Custom Part: The if statement checks the tone_check_result from the state. If it's "negative", the story_generator is called again, overwriting the current_story in the state. Otherwise, the flow ends.

Part 3: Defining the LLM Sub-Agents¶
These are standard LlmAgent definitions, responsible for specific tasks. Their output key parameter is crucial for placing results into the session.state where other agents or the custom orchestrator can access them.


Python
Java

GEMINI_2_FLASH = "gemini-2.0-flash" # Define model constant
# --- Define the individual LLM agents ---
story_generator = LlmAgent(
    name="StoryGenerator",
    model=GEMINI_2_FLASH,
    instruction="""You are a story writer. Write a short story (around 100 words) about a cat,
based on the topic provided in session state with key 'topic'""",
    input_schema=None,
    output_key="current_story",  # Key for storing output in session state
)

critic = LlmAgent(
    name="Critic",
    model=GEMINI_2_FLASH,
    instruction="""You are a story critic. Review the story provided in
session state with key 'current_story'. Provide 1-2 sentences of constructive criticism
on how to improve it. Focus on plot or character.""",
    input_schema=None,
    output_key="criticism",  # Key for storing criticism in session state
)

reviser = LlmAgent(
    name="Reviser",
    model=GEMINI_2_FLASH,
    instruction="""You are a story reviser. Revise the story provided in
session state with key 'current_story', based on the criticism in
session state with key 'criticism'. Output only the revised story.""",
    input_schema=None,
    output_key="current_story",  # Overwrites the original story
)

grammar_check = LlmAgent(
    name="GrammarCheck",
    model=GEMINI_2_FLASH,
    instruction="""You are a grammar checker. Check the grammar of the story
provided in session state with key 'current_story'. Output only the suggested
corrections as a list, or output 'Grammar is good!' if there are no errors.""",
    input_schema=None,
    output_key="grammar_suggestions",
)

tone_check = LlmAgent(
    name="ToneCheck",
    model=GEMINI_2_FLASH,
    instruction="""You are a tone analyzer. Analyze the tone of the story
provided in session state with key 'current_story'. Output only one word: 'positive' if
the tone is generally positive, 'negative' if the tone is generally negative, or 'neutral'
otherwise.""",
    input_schema=None,
    output_key="tone_check_result", # This agent's output determines the conditional flow
)

Part 4: Instantiating and Running the custom agent¶
Finally, you instantiate your StoryFlowAgent and use the Runner as usual.


Python
Java

# --- Create the custom agent instance ---
story_flow_agent = StoryFlowAgent(
    name="StoryFlowAgent",
    story_generator=story_generator,
    critic=critic,
    reviser=reviser,
    grammar_check=grammar_check,
    tone_check=tone_check,
)

# --- Setup Runner and Session ---
session_service = InMemorySessionService()
initial_state = {"topic": "a brave kitten exploring a haunted house"}
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initial_state # Pass initial state here
)
logger.info(f"Initial session state: {session.state}")

runner = Runner(
    agent=story_flow_agent, # Pass the custom orchestrator agent
    app_name=APP_NAME,
    session_service=session_service
)

# --- Function to Interact with the Agent ---
def call_agent(user_input_topic: str):
    """
    Sends a new topic to the agent (overwriting the initial one if needed)
    and runs the workflow.
    """
    current_session = session_service.get_session(app_name=APP_NAME, 
                                                  user_id=USER_ID, 
                                                  session_id=SESSION_ID)
    if not current_session:
        logger.error("Session not found!")
        return

    current_session.state["topic"] = user_input_topic
    logger.info(f"Updated session state topic to: {user_input_topic}")

    content = types.Content(role='user', parts=[types.Part(text=f"Generate a story about: {user_input_topic}")])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    final_response = "No final response captured."
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            logger.info(f"Potential final response from [{event.author}]: {event.content.parts[0].text}")
            final_response = event.content.parts[0].text

    print("\n--- Agent Interaction Result ---")
    print("Agent Final Response: ", final_response)

    final_session = session_service.get_session(app_name=APP_NAME, 
                                                user_id=USER_ID, 
                                                session_id=SESSION_ID)
    print("Final Session State:")
    import json
    print(json.dumps(final_session.state, indent=2))
    print("-------------------------------\n")

# --- Run the Agent ---
call_agent("a lonely robot finding a friend in a junkyard")

(Note: The full runnable code, including imports and execution logic, can be found linked below.)

Full Code Example¶
Storyflow Agent

Python
Java

# Full runnable code for the StoryFlowAgent example
import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from pydantic import BaseModel, Field

# --- Constants ---
APP_NAME = "story_app"
USER_ID = "12345"
SESSION_ID = "123344"
GEMINI_2_FLASH = "gemini-2.0-flash"

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Custom Orchestrator Agent ---
class StoryFlowAgent(BaseAgent):
    """
    Custom agent for a story generation and refinement workflow.

    This agent orchestrates a sequence of LLM agents to generate a story,
    critique it, revise it, check grammar and tone, and potentially
    regenerate the story if the tone is negative.
    """

    # --- Field Declarations for Pydantic ---
    # Declare the agents passed during initialization as class attributes with type hints
    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent

    loop_agent: LoopAgent
    sequential_agent: SequentialAgent

    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        story_generator: LlmAgent,
        critic: LlmAgent,
        reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
    ):
        """
        Initializes the StoryFlowAgent.

        Args:
            name: The name of the agent.
            story_generator: An LlmAgent to generate the initial story.
            critic: An LlmAgent to critique the story.
            reviser: An LlmAgent to revise the story based on criticism.
            grammar_check: An LlmAgent to check the grammar.
            tone_check: An LlmAgent to analyze the tone.
        """
        # Create internal agents *before* calling super().__init__
        loop_agent = LoopAgent(
            name="CriticReviserLoop", sub_agents=[critic, reviser], max_iterations=2
        )
        sequential_agent = SequentialAgent(
            name="PostProcessing", sub_agents=[grammar_check, tone_check]
        )

        # Define the sub_agents list for the framework
        sub_agents_list = [
            story_generator,
            loop_agent,
            sequential_agent,
        ]

        # Pydantic will validate and assign them based on the class annotations.
        super().__init__(
            name=name,
            story_generator=story_generator,
            critic=critic,
            reviser=reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,
            loop_agent=loop_agent,
            sequential_agent=sequential_agent,
            sub_agents=sub_agents_list, # Pass the sub_agents list directly
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the story workflow.
        Uses the instance attributes assigned by Pydantic (e.g., self.story_generator).
        """
        logger.info(f"[{self.name}] Starting story generation workflow.")

        # 1. Initial Story Generation
        logger.info(f"[{self.name}] Running StoryGenerator...")
        async for event in self.story_generator.run_async(ctx):
            logger.info(f"[{self.name}] Event from StoryGenerator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # Check if story was generated before proceeding
        if "current_story" not in ctx.session.state or not ctx.session.state["current_story"]:
             logger.error(f"[{self.name}] Failed to generate initial story. Aborting workflow.")
             return # Stop processing if initial story failed

        logger.info(f"[{self.name}] Story state after generator: {ctx.session.state.get('current_story')}")


        # 2. Critic-Reviser Loop
        logger.info(f"[{self.name}] Running CriticReviserLoop...")
        # Use the loop_agent instance attribute assigned during init
        async for event in self.loop_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from CriticReviserLoop: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Story state after loop: {ctx.session.state.get('current_story')}")

        # 3. Sequential Post-Processing (Grammar and Tone Check)
        logger.info(f"[{self.name}] Running PostProcessing...")
        # Use the sequential_agent instance attribute assigned during init
        async for event in self.sequential_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from PostProcessing: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # 4. Tone-Based Conditional Logic
        tone_check_result = ctx.session.state.get("tone_check_result")
        logger.info(f"[{self.name}] Tone check result: {tone_check_result}")

        if tone_check_result == "negative":
            logger.info(f"[{self.name}] Tone is negative. Regenerating story...")
            async for event in self.story_generator.run_async(ctx):
                logger.info(f"[{self.name}] Event from StoryGenerator (Regen): {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
        else:
            logger.info(f"[{self.name}] Tone is not negative. Keeping current story.")
            pass

        logger.info(f"[{self.name}] Workflow finished.")

# --- Define the individual LLM agents ---
story_generator = LlmAgent(
    name="StoryGenerator",
    model=GEMINI_2_FLASH,
    instruction="""You are a story writer. Write a short story (around 100 words) about a cat,
based on the topic provided in session state with key 'topic'""",
    input_schema=None,
    output_key="current_story",  # Key for storing output in session state
)

critic = LlmAgent(
    name="Critic",
    model=GEMINI_2_FLASH,
    instruction="""You are a story critic. Review the story provided in
session state with key 'current_story'. Provide 1-2 sentences of constructive criticism
on how to improve it. Focus on plot or character.""",
    input_schema=None,
    output_key="criticism",  # Key for storing criticism in session state
)

reviser = LlmAgent(
    name="Reviser",
    model=GEMINI_2_FLASH,
    instruction="""You are a story reviser. Revise the story provided in
session state with key 'current_story', based on the criticism in
session state with key 'criticism'. Output only the revised story.""",
    input_schema=None,
    output_key="current_story",  # Overwrites the original story
)

grammar_check = LlmAgent(
    name="GrammarCheck",
    model=GEMINI_2_FLASH,
    instruction="""You are a grammar checker. Check the grammar of the story
provided in session state with key 'current_story'. Output only the suggested
corrections as a list, or output 'Grammar is good!' if there are no errors.""",
    input_schema=None,
    output_key="grammar_suggestions",
)

tone_check = LlmAgent(
    name="ToneCheck",
    model=GEMINI_2_FLASH,
    instruction="""You are a tone analyzer. Analyze the tone of the story
provided in session state with key 'current_story'. Output only one word: 'positive' if
the tone is generally positive, 'negative' if the tone is generally negative, or 'neutral'
otherwise.""",
    input_schema=None,
    output_key="tone_check_result", # This agent's output determines the conditional flow
)

# --- Create the custom agent instance ---
story_flow_agent = StoryFlowAgent(
    name="StoryFlowAgent",
    story_generator=story_generator,
    critic=critic,
    reviser=reviser,
    grammar_check=grammar_check,
    tone_check=tone_check,
)

# --- Setup Runner and Session ---
session_service = InMemorySessionService()
initial_state = {"topic": "a brave kitten exploring a haunted house"}
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initial_state # Pass initial state here
)
logger.info(f"Initial session state: {session.state}")

runner = Runner(
    agent=story_flow_agent, # Pass the custom orchestrator agent
    app_name=APP_NAME,
    session_service=session_service
)

# --- Function to Interact with the Agent ---
def call_agent(user_input_topic: str):
    """
    Sends a new topic to the agent (overwriting the initial one if needed)
    and runs the workflow.
    """
    current_session = session_service.get_session(app_name=APP_NAME, 
                                                  user_id=USER_ID, 
                                                  session_id=SESSION_ID)
    if not current_session:
        logger.error("Session not found!")
        return

    current_session.state["topic"] = user_input_topic
    logger.info(f"Updated session state topic to: {user_input_topic}")

    content = types.Content(role='user', parts=[types.Part(text=f"Generate a story about: {user_input_topic}")])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    final_response = "No final response captured."
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            logger.info(f"Potential final response from [{event.author}]: {event.content.parts[0].text}")
            final_response = event.content.parts[0].text

    print("\n--- Agent Interaction Result ---")
    print("Agent Final Response: ", final_response)

    final_session = session_service.get_session(app_name=APP_NAME, 
                                                user_id=USER_ID, 
                                                session_id=SESSION_ID)
    print("Final Session State:")
    import json
    print(json.dumps(final_session.state, indent=2))
    print("-------------------------------\n")

# --- Run the Agent ---
call_agent("a lonely robot finding a friend in a junkyard")

Multi-Agent Systems in ADK¶
As agentic applications grow in complexity, structuring them as a single, monolithic agent can become challenging to develop, maintain, and reason about. The Agent Development Kit (ADK) supports building sophisticated applications by composing multiple, distinct BaseAgent instances into a Multi-Agent System (MAS).

In ADK, a multi-agent system is an application where different agents, often forming a hierarchy, collaborate or coordinate to achieve a larger goal. Structuring your application this way offers significant advantages, including enhanced modularity, specialization, reusability, maintainability, and the ability to define structured control flows using dedicated workflow agents.

You can compose various types of agents derived from BaseAgent to build these systems:

LLM Agents: Agents powered by large language models. (See LLM Agents)
Workflow Agents: Specialized agents (SequentialAgent, ParallelAgent, LoopAgent) designed to manage the execution flow of their sub-agents. (See Workflow Agents)
Custom agents: Your own agents inheriting from BaseAgent with specialized, non-LLM logic. (See Custom Agents)
The following sections detail the core ADK primitives—such as agent hierarchy, workflow agents, and interaction mechanisms—that enable you to construct and manage these multi-agent systems effectively.

1. ADK Primitives for Agent Composition¶
ADK provides core building blocks—primitives—that enable you to structure and manage interactions within your multi-agent system.

Note

The specific parameters or method names for the primitives may vary slightly by SDK language (e.g., sub_agents in Python, subAgents in Java). Refer to the language-specific API documentation for details.

1.1. Agent Hierarchy (Parent agent, Sub Agents)¶
The foundation for structuring multi-agent systems is the parent-child relationship defined in BaseAgent.

Establishing Hierarchy: You create a tree structure by passing a list of agent instances to the sub_agents argument when initializing a parent agent. ADK automatically sets the parent_agent attribute on each child agent during initialization.
Single Parent Rule: An agent instance can only be added as a sub-agent once. Attempting to assign a second parent will result in a ValueError.
Importance: This hierarchy defines the scope for Workflow Agents and influences the potential targets for LLM-Driven Delegation. You can navigate the hierarchy using agent.parent_agent or find descendants using agent.find_agent(name).

Python
Java

# Conceptual Example: Defining Hierarchy
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="Greeter", model="gemini-2.0-flash")
task_doer = BaseAgent(name="TaskExecutor") # Custom non-LLM agent

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
    description="I coordinate greetings and tasks.",
    sub_agents=[ # Assign sub_agents here
        greeter,
        task_doer
    ]
)

# Framework automatically sets:
# assert greeter.parent_agent == coordinator
# assert task_doer.parent_agent == coordinator

1.2. Workflow Agents as Orchestrators¶
ADK includes specialized agents derived from BaseAgent that don't perform tasks themselves but orchestrate the execution flow of their sub_agents.

SequentialAgent: Executes its sub_agents one after another in the order they are listed.
Context: Passes the same InvocationContext sequentially, allowing agents to easily pass results via shared state.

Python
Java

# Conceptual Example: Sequential Pipeline
from google.adk.agents import SequentialAgent, LlmAgent

step1 = LlmAgent(name="Step1_Fetch", output_key="data") # Saves output to state['data']
step2 = LlmAgent(name="Step2_Process", instruction="Process data from state key 'data'.")

pipeline = SequentialAgent(name="MyPipeline", sub_agents=[step1, step2])
# When pipeline runs, Step2 can access the state['data'] set by Step1.

ParallelAgent: Executes its sub_agents in parallel. Events from sub-agents may be interleaved.
Context: Modifies the InvocationContext.branch for each child agent (e.g., ParentBranch.ChildName), providing a distinct contextual path which can be useful for isolating history in some memory implementations.
State: Despite different branches, all parallel children access the same shared session.state, enabling them to read initial state and write results (use distinct keys to avoid race conditions).

Python
Java

# Conceptual Example: Parallel Execution
from google.adk.agents import ParallelAgent, LlmAgent

fetch_weather = LlmAgent(name="WeatherFetcher", output_key="weather")
fetch_news = LlmAgent(name="NewsFetcher", output_key="news")

gatherer = ParallelAgent(name="InfoGatherer", sub_agents=[fetch_weather, fetch_news])
# When gatherer runs, WeatherFetcher and NewsFetcher run concurrently.
# A subsequent agent could read state['weather'] and state['news'].

LoopAgent: Executes its sub_agents sequentially in a loop.
Termination: The loop stops if the optional max_iterations is reached, or if any sub-agent returns an Event with escalate=True in it's Event Actions.
Context & State: Passes the same InvocationContext in each iteration, allowing state changes (e.g., counters, flags) to persist across loops.

Python
Java

# Conceptual Example: Loop with Condition
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

class CheckCondition(BaseAgent): # Custom agent to check state
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        status = ctx.session.state.get("status", "pending")
        is_done = (status == "completed")
        yield Event(author=self.name, actions=EventActions(escalate=is_done)) # Escalate if done

process_step = LlmAgent(name="ProcessingStep") # Agent that might update state['status']

poller = LoopAgent(
    name="StatusPoller",
    max_iterations=10,
    sub_agents=[process_step, CheckCondition(name="Checker")]
)
# When poller runs, it executes process_step then Checker repeatedly
# until Checker escalates (state['status'] == 'completed') or 10 iterations pass.

1.3. Interaction & Communication Mechanisms¶
Agents within a system often need to exchange data or trigger actions in one another. ADK facilitates this through:

a) Shared Session State (session.state)¶
The most fundamental way for agents operating within the same invocation (and thus sharing the same Session object via the InvocationContext) to communicate passively.

Mechanism: One agent (or its tool/callback) writes a value (context.state['data_key'] = processed_data), and a subsequent agent reads it (data = context.state.get('data_key')). State changes are tracked via CallbackContext.
Convenience: The output_key property on LlmAgent automatically saves the agent's final response text (or structured output) to the specified state key.
Nature: Asynchronous, passive communication. Ideal for pipelines orchestrated by SequentialAgent or passing data across LoopAgent iterations.
See Also: State Management

Python
Java

# Conceptual Example: Using output_key and reading state
from google.adk.agents import LlmAgent, SequentialAgent

agent_A = LlmAgent(name="AgentA", instruction="Find the capital of France.", output_key="capital_city")
agent_B = LlmAgent(name="AgentB", instruction="Tell me about the city stored in state key 'capital_city'.")

pipeline = SequentialAgent(name="CityInfo", sub_agents=[agent_A, agent_B])
# AgentA runs, saves "Paris" to state['capital_city'].
# AgentB runs, its instruction processor reads state['capital_city'] to get "Paris".

b) LLM-Driven Delegation (Agent Transfer)¶
Leverages an LlmAgent's understanding to dynamically route tasks to other suitable agents within the hierarchy.

Mechanism: The agent's LLM generates a specific function call: transfer_to_agent(agent_name='target_agent_name').
Handling: The AutoFlow, used by default when sub-agents are present or transfer isn't disallowed, intercepts this call. It identifies the target agent using root_agent.find_agent() and updates the InvocationContext to switch execution focus.
Requires: The calling LlmAgent needs clear instructions on when to transfer, and potential target agents need distinct descriptions for the LLM to make informed decisions. Transfer scope (parent, sub-agent, siblings) can be configured on the LlmAgent.
Nature: Dynamic, flexible routing based on LLM interpretation.

Python
Java

# Conceptual Setup: LLM Transfer
from google.adk.agents import LlmAgent

booking_agent = LlmAgent(name="Booker", description="Handles flight and hotel bookings.")
info_agent = LlmAgent(name="Info", description="Provides general information and answers questions.")

coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction="You are an assistant. Delegate booking tasks to Booker and info requests to Info.",
    description="Main coordinator.",
    # AutoFlow is typically used implicitly here
    sub_agents=[booking_agent, info_agent]
)
# If coordinator receives "Book a flight", its LLM should generate:
# FunctionCall(name='transfer_to_agent', args={'agent_name': 'Booker'})
# ADK framework then routes execution to booking_agent.

c) Explicit Invocation (AgentTool)¶
Allows an LlmAgent to treat another BaseAgent instance as a callable function or Tool.

Mechanism: Wrap the target agent instance in AgentTool and include it in the parent LlmAgent's tools list. AgentTool generates a corresponding function declaration for the LLM.
Handling: When the parent LLM generates a function call targeting the AgentTool, the framework executes AgentTool.run_async. This method runs the target agent, captures its final response, forwards any state/artifact changes back to the parent's context, and returns the response as the tool's result.
Nature: Synchronous (within the parent's flow), explicit, controlled invocation like any other tool.
(Note: AgentTool needs to be imported and used explicitly).

Python
Java

# Conceptual Setup: Agent as a Tool
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import agent_tool
from pydantic import BaseModel

# Define a target agent (could be LlmAgent or custom BaseAgent)
class ImageGeneratorAgent(BaseAgent): # Example custom agent
    name: str = "ImageGen"
    description: str = "Generates an image based on a prompt."
    # ... internal logic ...
    async def _run_async_impl(self, ctx): # Simplified run logic
        prompt = ctx.session.state.get("image_prompt", "default prompt")
        # ... generate image bytes ...
        image_bytes = b"..."
        yield Event(author=self.name, content=types.Content(parts=[types.Part.from_bytes(image_bytes, "image/png")]))

image_agent = ImageGeneratorAgent()
image_tool = agent_tool.AgentTool(agent=image_agent) # Wrap the agent

# Parent agent uses the AgentTool
artist_agent = LlmAgent(
    name="Artist",
    model="gemini-2.0-flash",
    instruction="Create a prompt and use the ImageGen tool to generate the image.",
    tools=[image_tool] # Include the AgentTool
)
# Artist LLM generates a prompt, then calls:
# FunctionCall(name='ImageGen', args={'image_prompt': 'a cat wearing a hat'})
# Framework calls image_tool.run_async(...), which runs ImageGeneratorAgent.
# The resulting image Part is returned to the Artist agent as the tool result.

These primitives provide the flexibility to design multi-agent interactions ranging from tightly coupled sequential workflows to dynamic, LLM-driven delegation networks.

2. Common Multi-Agent Patterns using ADK Primitives¶
By combining ADK's composition primitives, you can implement various established patterns for multi-agent collaboration.

Coordinator/Dispatcher Pattern¶
Structure: A central LlmAgent (Coordinator) manages several specialized sub_agents.
Goal: Route incoming requests to the appropriate specialist agent.
ADK Primitives Used:
Hierarchy: Coordinator has specialists listed in sub_agents.
Interaction: Primarily uses LLM-Driven Delegation (requires clear descriptions on sub-agents and appropriate instruction on Coordinator) or Explicit Invocation (AgentTool) (Coordinator includes AgentTool-wrapped specialists in its tools).

Python
Java

# Conceptual Code: Coordinator using LLM Transfer
from google.adk.agents import LlmAgent

billing_agent = LlmAgent(name="Billing", description="Handles billing inquiries.")
support_agent = LlmAgent(name="Support", description="Handles technical support requests.")

coordinator = LlmAgent(
    name="HelpDeskCoordinator",
    model="gemini-2.0-flash",
    instruction="Route user requests: Use Billing agent for payment issues, Support agent for technical problems.",
    description="Main help desk router.",
    # allow_transfer=True is often implicit with sub_agents in AutoFlow
    sub_agents=[billing_agent, support_agent]
)
# User asks "My payment failed" -> Coordinator's LLM should call transfer_to_agent(agent_name='Billing')
# User asks "I can't log in" -> Coordinator's LLM should call transfer_to_agent(agent_name='Support')

Sequential Pipeline Pattern¶
Structure: A SequentialAgent contains sub_agents executed in a fixed order.
Goal: Implement a multi-step process where the output of one step feeds into the next.
ADK Primitives Used:
Workflow: SequentialAgent defines the order.
Communication: Primarily uses Shared Session State. Earlier agents write results (often via output_key), later agents read those results from context.state.

Python
Java

# Conceptual Code: Sequential Data Pipeline
from google.adk.agents import SequentialAgent, LlmAgent

validator = LlmAgent(name="ValidateInput", instruction="Validate the input.", output_key="validation_status")
processor = LlmAgent(name="ProcessData", instruction="Process data if state key 'validation_status' is 'valid'.", output_key="result")
reporter = LlmAgent(name="ReportResult", instruction="Report the result from state key 'result'.")

data_pipeline = SequentialAgent(
    name="DataPipeline",
    sub_agents=[validator, processor, reporter]
)
# validator runs -> saves to state['validation_status']
# processor runs -> reads state['validation_status'], saves to state['result']
# reporter runs -> reads state['result']

Parallel Fan-Out/Gather Pattern¶
Structure: A ParallelAgent runs multiple sub_agents concurrently, often followed by a later agent (in a SequentialAgent) that aggregates results.
Goal: Execute independent tasks simultaneously to reduce latency, then combine their outputs.
ADK Primitives Used:
Workflow: ParallelAgent for concurrent execution (Fan-Out). Often nested within a SequentialAgent to handle the subsequent aggregation step (Gather).
Communication: Sub-agents write results to distinct keys in Shared Session State. The subsequent "Gather" agent reads multiple state keys.

Python
Java

# Conceptual Code: Parallel Information Gathering
from google.adk.agents import SequentialAgent, ParallelAgent, LlmAgent

fetch_api1 = LlmAgent(name="API1Fetcher", instruction="Fetch data from API 1.", output_key="api1_data")
fetch_api2 = LlmAgent(name="API2Fetcher", instruction="Fetch data from API 2.", output_key="api2_data")

gather_concurrently = ParallelAgent(
    name="ConcurrentFetch",
    sub_agents=[fetch_api1, fetch_api2]
)

synthesizer = LlmAgent(
    name="Synthesizer",
    instruction="Combine results from state keys 'api1_data' and 'api2_data'."
)

overall_workflow = SequentialAgent(
    name="FetchAndSynthesize",
    sub_agents=[gather_concurrently, synthesizer] # Run parallel fetch, then synthesize
)
# fetch_api1 and fetch_api2 run concurrently, saving to state.
# synthesizer runs afterwards, reading state['api1_data'] and state['api2_data'].

Hierarchical Task Decomposition¶
Structure: A multi-level tree of agents where higher-level agents break down complex goals and delegate sub-tasks to lower-level agents.
Goal: Solve complex problems by recursively breaking them down into simpler, executable steps.
ADK Primitives Used:
Hierarchy: Multi-level parent_agent/sub_agents structure.
Interaction: Primarily LLM-Driven Delegation or Explicit Invocation (AgentTool) used by parent agents to assign tasks to subagents. Results are returned up the hierarchy (via tool responses or state).

Python
Java

# Conceptual Code: Hierarchical Research Task
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

# Low-level tool-like agents
web_searcher = LlmAgent(name="WebSearch", description="Performs web searches for facts.")
summarizer = LlmAgent(name="Summarizer", description="Summarizes text.")

# Mid-level agent combining tools
research_assistant = LlmAgent(
    name="ResearchAssistant",
    model="gemini-2.0-flash",
    description="Finds and summarizes information on a topic.",
    tools=[agent_tool.AgentTool(agent=web_searcher), agent_tool.AgentTool(agent=summarizer)]
)

# High-level agent delegating research
report_writer = LlmAgent(
    name="ReportWriter",
    model="gemini-2.0-flash",
    instruction="Write a report on topic X. Use the ResearchAssistant to gather information.",
    tools=[agent_tool.AgentTool(agent=research_assistant)]
    # Alternatively, could use LLM Transfer if research_assistant is a sub_agent
)
# User interacts with ReportWriter.
# ReportWriter calls ResearchAssistant tool.
# ResearchAssistant calls WebSearch and Summarizer tools.
# Results flow back up.

Review/Critique Pattern (Generator-Critic)¶
Structure: Typically involves two agents within a SequentialAgent: a Generator and a Critic/Reviewer.
Goal: Improve the quality or validity of generated output by having a dedicated agent review it.
ADK Primitives Used:
Workflow: SequentialAgent ensures generation happens before review.
Communication: Shared Session State (Generator uses output_key to save output; Reviewer reads that state key). The Reviewer might save its feedback to another state key for subsequent steps.

Python
Java

# Conceptual Code: Generator-Critic
from google.adk.agents import SequentialAgent, LlmAgent

generator = LlmAgent(
    name="DraftWriter",
    instruction="Write a short paragraph about subject X.",
    output_key="draft_text"
)

reviewer = LlmAgent(
    name="FactChecker",
    instruction="Review the text in state key 'draft_text' for factual accuracy. Output 'valid' or 'invalid' with reasons.",
    output_key="review_status"
)

# Optional: Further steps based on review_status

review_pipeline = SequentialAgent(
    name="WriteAndReview",
    sub_agents=[generator, reviewer]
)
# generator runs -> saves draft to state['draft_text']
# reviewer runs -> reads state['draft_text'], saves status to state['review_status']

Iterative Refinement Pattern¶
Structure: Uses a LoopAgent containing one or more agents that work on a task over multiple iterations.
Goal: Progressively improve a result (e.g., code, text, plan) stored in the session state until a quality threshold is met or a maximum number of iterations is reached.
ADK Primitives Used:
Workflow: LoopAgent manages the repetition.
Communication: Shared Session State is essential for agents to read the previous iteration's output and save the refined version.
Termination: The loop typically ends based on max_iterations or a dedicated checking agent setting escalate=True in the Event Actions when the result is satisfactory.

Python
Java

# Conceptual Code: Iterative Code Refinement
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

# Agent to generate/refine code based on state['current_code'] and state['requirements']
code_refiner = LlmAgent(
    name="CodeRefiner",
    instruction="Read state['current_code'] (if exists) and state['requirements']. Generate/refine Python code to meet requirements. Save to state['current_code'].",
    output_key="current_code" # Overwrites previous code in state
)

# Agent to check if the code meets quality standards
quality_checker = LlmAgent(
    name="QualityChecker",
    instruction="Evaluate the code in state['current_code'] against state['requirements']. Output 'pass' or 'fail'.",
    output_key="quality_status"
)

# Custom agent to check the status and escalate if 'pass'
class CheckStatusAndEscalate(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        status = ctx.session.state.get("quality_status", "fail")
        should_stop = (status == "pass")
        yield Event(author=self.name, actions=EventActions(escalate=should_stop))

refinement_loop = LoopAgent(
    name="CodeRefinementLoop",
    max_iterations=5,
    sub_agents=[code_refiner, quality_checker, CheckStatusAndEscalate(name="StopChecker")]
)
# Loop runs: Refiner -> Checker -> StopChecker
# State['current_code'] is updated each iteration.
# Loop stops if QualityChecker outputs 'pass' (leading to StopChecker escalating) or after 5 iterations.

Human-in-the-Loop Pattern¶
Structure: Integrates human intervention points within an agent workflow.
Goal: Allow for human oversight, approval, correction, or tasks that AI cannot perform.
ADK Primitives Used (Conceptual):
Interaction: Can be implemented using a custom Tool that pauses execution and sends a request to an external system (e.g., a UI, ticketing system) waiting for human input. The tool then returns the human's response to the agent.
Workflow: Could use LLM-Driven Delegation (transfer_to_agent) targeting a conceptual "Human Agent" that triggers the external workflow, or use the custom tool within an LlmAgent.
State/Callbacks: State can hold task details for the human; callbacks can manage the interaction flow.
Note: ADK doesn't have a built-in "Human Agent" type, so this requires custom integration.

Python
Java

# Conceptual Code: Using a Tool for Human Approval
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

# --- Assume external_approval_tool exists ---
# This tool would:
# 1. Take details (e.g., request_id, amount, reason).
# 2. Send these details to a human review system (e.g., via API).
# 3. Poll or wait for the human response (approved/rejected).
# 4. Return the human's decision.
# async def external_approval_tool(amount: float, reason: str) -> str: ...
approval_tool = FunctionTool(func=external_approval_tool)

# Agent that prepares the request
prepare_request = LlmAgent(
    name="PrepareApproval",
    instruction="Prepare the approval request details based on user input. Store amount and reason in state.",
    # ... likely sets state['approval_amount'] and state['approval_reason'] ...
)

# Agent that calls the human approval tool
request_approval = LlmAgent(
    name="RequestHumanApproval",
    instruction="Use the external_approval_tool with amount from state['approval_amount'] and reason from state['approval_reason'].",
    tools=[approval_tool],
    output_key="human_decision"
)

# Agent that proceeds based on human decision
process_decision = LlmAgent(
    name="ProcessDecision",
    instruction="Check state key 'human_decision'. If 'approved', proceed. If 'rejected', inform user."
)

approval_workflow = SequentialAgent(
    name="HumanApprovalWorkflow",
    sub_agents=[prepare_request, request_approval, process_decision]
)

These patterns provide starting points for structuring your multi-agent systems. You can mix and match them as needed to create the most effective architecture for your specific application.

Function tools¶
What are function tools?¶
When out-of-the-box tools don't fully meet specific requirements, developers can create custom function tools. This allows for tailored functionality, such as connecting to proprietary databases or implementing unique algorithms.

For example, a function tool, "myfinancetool", might be a function that calculates a specific financial metric. ADK also supports long running functions, so if that calculation takes a while, the agent can continue working on other tasks.

ADK offers several ways to create functions tools, each suited to different levels of complexity and control:

Function Tool
Long Running Function Tool
Agents-as-a-Tool
1. Function Tool¶
Transforming a function into a tool is a straightforward way to integrate custom logic into your agents. In fact, when you assign a function to an agent’s tools list, the framework will automatically wrap it as a Function Tool for you. This approach offers flexibility and quick integration.

Parameters¶
Define your function parameters using standard JSON-serializable types (e.g., string, integer, list, dictionary). It's important to avoid setting default values for parameters, as the language model (LLM) does not currently support interpreting them.

Return Type¶
The preferred return type for a Function Tool is a dictionary in Python or Map in Java. This allows you to structure the response with key-value pairs, providing context and clarity to the LLM. If your function returns a type other than a dictionary, the framework automatically wraps it into a dictionary with a single key named "result".

Strive to make your return values as descriptive as possible. For example, instead of returning a numeric error code, return a dictionary with an "error_message" key containing a human-readable explanation. Remember that the LLM, not a piece of code, needs to understand the result. As a best practice, include a "status" key in your return dictionary to indicate the overall outcome (e.g., "success", "error", "pending"), providing the LLM with a clear signal about the operation's state.

Docstring / Source code comments¶
The docstring (or comments above) your function serve as the tool's description and is sent to the LLM. Therefore, a well-written and comprehensive docstring is crucial for the LLM to understand how to use the tool effectively. Clearly explain the purpose of the function, the meaning of its parameters, and the expected return values.

Example

Python
Java
This tool is a python function which obtains the Stock price of a given Stock ticker/ symbol.

Note: You need to pip install yfinance library before using this tool.


from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import yfinance as yf


APP_NAME = "stock_app"
USER_ID = "1234"
SESSION_ID = "session1234"

def get_stock_price(symbol: str):
    """
    Retrieves the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price, or None if an error occurs.
    """
    try:
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period="1d")
        if not historical_data.empty:
            current_price = historical_data['Close'].iloc[-1]
            return current_price
        else:
            return None
    except Exception as e:
        print(f"Error retrieving stock price for {symbol}: {e}")
        return None


stock_price_agent = Agent(
    model='gemini-2.0-flash',
    name='stock_agent',
    instruction= 'You are an agent who retrieves stock prices. If a ticker symbol is provided, fetch the current price. If only a company name is given, first perform a Google search to find the correct ticker symbol before retrieving the stock price. If the provided ticker symbol is invalid or data cannot be retrieved, inform the user that the stock price could not be found.',
    description='This agent specializes in retrieving real-time stock prices. Given a stock ticker symbol (e.g., AAPL, GOOG, MSFT) or the stock name, use the tools and reliable data sources to provide the most up-to-date price.',
    tools=[get_stock_price], # You can add Python functions directly to the tools list; they will be automatically wrapped as FunctionTools.
)


# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=stock_price_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

call_agent("stock price of GOOG")
The return value from this tool will be wrapped into a dictionary.


{"result": "$123"}

Best Practices¶
While you have considerable flexibility in defining your function, remember that simplicity enhances usability for the LLM. Consider these guidelines:

Fewer Parameters are Better: Minimize the number of parameters to reduce complexity.
Simple Data Types: Favor primitive data types like str and int over custom classes whenever possible.
Meaningful Names: The function's name and parameter names significantly influence how the LLM interprets and utilizes the tool. Choose names that clearly reflect the function's purpose and the meaning of its inputs. Avoid generic names like do_stuff() or beAgent().
2. Long Running Function Tool¶
Designed for tasks that require a significant amount of processing time without blocking the agent's execution. This tool is a subclass of FunctionTool.

When using a LongRunningFunctionTool, your function can initiate the long-running operation and optionally return an initial result** (e.g. the long-running operation id). Once a long running function tool is invoked the agent runner will pause the agent run and let the agent client to decide whether to continue or wait until the long-running operation finishes. The agent client can query the progress of the long-running operation and send back an intermediate or final response. The agent can then continue with other tasks. An example is the human-in-the-loop scenario where the agent needs human approval before proceeding with a task.

How it Works¶
In Python, you wrap a function with LongRunningFunctionTool. In Java, you pass a Method name to LongRunningFunctionTool.create().

Initiation: When the LLM calls the tool, your function starts the long-running operation.

Initial Updates: Your function should optionally return an initial result (e.g. the long-running operaiton id). The ADK framework takes the result and sends it back to the LLM packaged within a FunctionResponse. This allows the LLM to inform the user (e.g., status, percentage complete, messages). And then the agent run is ended / paused.

Continue or Wait: After each agent run is completed. Agent client can query the progress of the long-running operation and decide whether to continue the agent run with an intermediate response (to update the progress) or wait until a final response is retrieved. Agent client should send the intermediate or final response back to the agent for the next run.

Framework Handling: The ADK framework manages the execution. It sends the intermediate or final FunctionResponse sent by agent client to the LLM to generate a user friendly message.

Creating the Tool¶
Define your tool function and wrap it using the LongRunningFunctionTool class:


Python
Java

# 1. Define the long running function
def ask_for_approval(
    purpose: str, amount: float
) -> dict[str, Any]:
    """Ask for approval for the reimbursement."""
    # create a ticket for the approval
    # Send a notification to the approver with the link of the ticket
    return {'status': 'pending', 'approver': 'Sean Zhou', 'purpose' : purpose, 'amount': amount, 'ticket-id': 'approval-ticket-1'}

def reimburse(purpose: str, amount: float) -> str:
    """Reimburse the amount of money to the employee."""
    # send the reimbrusement request to payment vendor
    return {'status': 'ok'}

# 2. Wrap the function with LongRunningFunctionTool
long_running_tool = LongRunningFunctionTool(func=ask_for_approval)

Intermediate / Final result Updates¶
Agent client received an event with long running function calls and check the status of the ticket. Then Agent client can send the intermediate or final response back to update the progress. The framework packages this value (even if it's None) into the content of the FunctionResponse sent back to the LLM.

Applies to only Java ADK

When passing ToolContext with Function Tools, ensure that one of the following is true:

The Schema is passed with the ToolContext parameter in the function signature, like:


@com.google.adk.tools.Annotations.Schema(name = "toolContext") ToolContext toolContext
OR
The following -parameters flag is set to the mvn compiler plugin


<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.14.0</version> <!-- or newer -->
            <configuration>
                <compilerArgs>
                    <arg>-parameters</arg>
                </compilerArgs>
            </configuration>
        </plugin>
    </plugins>
</build>
This constraint is temporary and will be removed.

Python
Java

# Agent Interaction
async def call_agent(query):

    def get_long_running_function_call(event: Event) -> types.FunctionCall:
        # Get the long running function call from the event
        if not event.long_running_tool_ids or not event.content or not event.content.parts:
            return
        for part in event.content.parts:
            if (
                part
                and part.function_call
                and event.long_running_tool_ids
                and part.function_call.id in event.long_running_tool_ids
            ):
                return part.function_call

    def get_function_response(event: Event, function_call_id: str) -> types.FunctionResponse:
        # Get the function response for the fuction call with specified id.
        if not event.content or not event.content.parts:
            return
        for part in event.content.parts:
            if (
                part
                and part.function_response
                and part.function_response.id == function_call_id
            ):
                return part.function_response

    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    print("\nRunning agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id=USER_ID, new_message=content
    )


    long_running_function_call, long_running_function_response, ticket_id = None, None, None
    async for event in events_async:
        # Use helper to check for the specific auth request event
        if not long_running_function_call:
            long_running_function_call = get_long_running_function_call(event)
        else:
            long_running_function_response = get_function_response(event, long_running_function_call.id)
            if long_running_function_response:
                ticket_id = long_running_function_response.response['ticket-id']
        if event.content and event.content.parts:
            if text := ''.join(part.text or '' for part in event.content.parts):
                print(f'[{event.author}]: {text}')


    if long_running_function_response:
        # query the status of the correpsonding ticket via tciket_id
        # send back an intermediate / final response
        updated_response = long_running_function_response.model_copy(deep=True)
        updated_response.response = {'status': 'approved'}
        async for event in runner.run_async(
          session_id=session.id, user_id=USER_ID, new_message=types.Content(parts=[types.Part(function_response = updated_response)], role='user')
        ):
            if event.content and event.content.parts:
                if text := ''.join(part.text or '' for part in event.content.parts):
                    print(f'[{event.author}]: {text}')

Python complete example: File Processing Simulation

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Any
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.tools import LongRunningFunctionTool
from google.adk.sessions import InMemorySessionService
from google.genai import types


# 1. Define the long running function
def ask_for_approval(
    purpose: str, amount: float
) -> dict[str, Any]:
    """Ask for approval for the reimbursement."""
    # create a ticket for the approval
    # Send a notification to the approver with the link of the ticket
    return {'status': 'pending', 'approver': 'Sean Zhou', 'purpose' : purpose, 'amount': amount, 'ticket-id': 'approval-ticket-1'}

def reimburse(purpose: str, amount: float) -> str:
    """Reimburse the amount of money to the employee."""
    # send the reimbrusement request to payment vendor
    return {'status': 'ok'}

# 2. Wrap the function with LongRunningFunctionTool
long_running_tool = LongRunningFunctionTool(func=ask_for_approval)


# 3. Use the tool in an Agent
file_processor_agent = Agent(
    # Use a model compatible with function calling
    model="gemini-2.0-flash",
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. If the amount is less than $100, you will automatically
      approve the reimbursement.

      If the amount is greater than $100, you will
      ask for approval from the manager. If the manager approves, you will
      call reimburse() to reimburse the amount to the employee. If the manager
      rejects, you will inform the employee of the rejection.
    """,
    tools=[reimburse, long_running_tool]
)


APP_NAME = "human_in_the_loop"
USER_ID = "1234"
SESSION_ID = "session1234"

# Session and Runner
session_service = InMemorySessionService()
session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=file_processor_agent, app_name=APP_NAME, session_service=session_service)



# Agent Interaction
async def call_agent(query):

    def get_long_running_function_call(event: Event) -> types.FunctionCall:
        # Get the long running function call from the event
        if not event.long_running_tool_ids or not event.content or not event.content.parts:
            return
        for part in event.content.parts:
            if (
                part
                and part.function_call
                and event.long_running_tool_ids
                and part.function_call.id in event.long_running_tool_ids
            ):
                return part.function_call

    def get_function_response(event: Event, function_call_id: str) -> types.FunctionResponse:
        # Get the function response for the fuction call with specified id.
        if not event.content or not event.content.parts:
            return
        for part in event.content.parts:
            if (
                part
                and part.function_response
                and part.function_response.id == function_call_id
            ):
                return part.function_response

    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    print("\nRunning agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id=USER_ID, new_message=content
    )


    long_running_function_call, long_running_function_response, ticket_id = None, None, None
    async for event in events_async:
        # Use helper to check for the specific auth request event
        if not long_running_function_call:
            long_running_function_call = get_long_running_function_call(event)
        else:
            long_running_function_response = get_function_response(event, long_running_function_call.id)
            if long_running_function_response:
                ticket_id = long_running_function_response.response['ticket-id']
        if event.content and event.content.parts:
            if text := ''.join(part.text or '' for part in event.content.parts):
                print(f'[{event.author}]: {text}')


    if long_running_function_response:
        # query the status of the correpsonding ticket via tciket_id
        # send back an intermediate / final response
        updated_response = long_running_function_response.model_copy(deep=True)
        updated_response.response = {'status': 'approved'}
        async for event in runner.run_async(
          session_id=session.id, user_id=USER_ID, new_message=types.Content(parts=[types.Part(function_response = updated_response)], role='user')
        ):
            if event.content and event.content.parts:
                if text := ''.join(part.text or '' for part in event.content.parts):
                    print(f'[{event.author}]: {text}')


# reimbursement that doesn't require approval
asyncio.run(call_agent("Please reimburse 50$ for meals"))
# await call_agent("Please reimburse 50$ for meals") # For Notebooks, uncomment this line and comment the above line
# reimbursement that requires approval
asyncio.run(call_agent("Please reimburse 200$ for meals"))
# await call_agent("Please reimburse 200$ for meals") # For Notebooks, uncomment this line and comment the above line
Key aspects of this example¶
LongRunningFunctionTool: Wraps the supplied method/function; the framework handles sending yielded updates and the final return value as sequential FunctionResponses.

Agent instruction: Directs the LLM to use the tool and understand the incoming FunctionResponse stream (progress vs. completion) for user updates.

Final return: The function returns the final result dictionary, which is sent in the concluding FunctionResponse to indicate completion.

3. Agent-as-a-Tool¶
This powerful feature allows you to leverage the capabilities of other agents within your system by calling them as tools. The Agent-as-a-Tool enables you to invoke another agent to perform a specific task, effectively delegating responsibility. This is conceptually similar to creating a Python function that calls another agent and uses the agent's response as the function's return value.

Key difference from sub-agents¶
It's important to distinguish an Agent-as-a-Tool from a Sub-Agent.

Agent-as-a-Tool: When Agent A calls Agent B as a tool (using Agent-as-a-Tool), Agent B's answer is passed back to Agent A, which then summarizes the answer and generates a response to the user. Agent A retains control and continues to handle future user input.

Sub-agent: When Agent A calls Agent B as a sub-agent, the responsibility of answering the user is completely transferred to Agent B. Agent A is effectively out of the loop. All subsequent user input will be answered by Agent B.

Usage¶
To use an agent as a tool, wrap the agent with the AgentTool class.


Python
Java

tools=[AgentTool(agent=agent_b)]

Customization¶
The AgentTool class provides the following attributes for customizing its behavior:

skip_summarization: bool: If set to True, the framework will bypass the LLM-based summarization of the tool agent's response. This can be useful when the tool's response is already well-formatted and requires no further processing.
Example

Python
Java

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

APP_NAME="summary_agent"
USER_ID="user1234"
SESSION_ID="1234"

summary_agent = Agent(
    model="gemini-2.0-flash",
    name="summary_agent",
    instruction="""You are an expert summarizer. Please read the following text and provide a concise summary.""",
    description="Agent to summarize text",
)

root_agent = Agent(
    model='gemini-2.0-flash',
    name='root_agent',
    instruction="""You are a helpful assistant. When the user provides a text, use the 'summarize' tool to generate a summary. Always forward the user's message exactly as received to the 'summarize' tool, without modifying or summarizing it yourself. Present the response from the tool to the user.""",
    tools=[AgentTool(agent=summary_agent)]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)


long_text = """Quantum computing represents a fundamentally different approach to computation, 
leveraging the bizarre principles of quantum mechanics to process information. Unlike classical computers 
that rely on bits representing either 0 or 1, quantum computers use qubits which can exist in a state of superposition - effectively 
being 0, 1, or a combination of both simultaneously. Furthermore, qubits can become entangled, 
meaning their fates are intertwined regardless of distance, allowing for complex correlations. This parallelism and 
interconnectedness grant quantum computers the potential to solve specific types of incredibly complex problems - such 
as drug discovery, materials science, complex system optimization, and breaking certain types of cryptography - far 
faster than even the most powerful classical supercomputers could ever achieve, although the technology is still largely in its developmental stages."""


call_agent(long_text)

How it works¶
When the main_agent receives the long text, its instruction tells it to use the 'summarize' tool for long texts.
The framework recognizes 'summarize' as an AgentTool that wraps the summary_agent.
Behind the scenes, the main_agent will call the summary_agent with the long text as input.
The summary_agent will process the text according to its instruction and generate a summary.
The response from the summary_agent is then passed back to the main_agent.
The main_agent can then take the summary and formulate its final response to the user (e.g., "Here's a summary of the text: ...")

Built-in tools¶
These built-in tools provide ready-to-use functionality such as Google Search or code executors that provide agents with common capabilities. For instance, an agent that needs to retrieve information from the web can directly use the google_search tool without any additional setup.

How to Use¶
Import: Import the desired tool from the tools module. This is agents.tools in Python or com.google.adk.tools in Java.
Configure: Initialize the tool, providing required parameters if any.
Register: Add the initialized tool to the tools list of your Agent.
Once added to an agent, the agent can decide to use the tool based on the user prompt and its instructions. The framework handles the execution of the tool when the agent calls it. Important: check the Limitations section of this page.

Available Built-in tools¶
Note: Java only supports Google Search and Code Execution tools currently.

Google Search¶
The google_search tool allows the agent to perform web searches using Google Search. The google_search tool is only compatible with Gemini 2 models.

Additional requirements when using the google_search tool

When you use grounding with Google Search, and you receive Search suggestions in your response, you must display the Search suggestions in production and in your applications. For more information on grounding with Google Search, see Grounding with Google Search documentation for Google AI Studio or Vertex AI. The UI code (HTML) is returned in the Gemini response as renderedContent, and you will need to show the HTML in your app, in accordance with the policy.


Python
Java

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

APP_NAME="google_search_agent"
USER_ID="user1234"
SESSION_ID="1234"


root_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    # google_search is a pre-built tool which allows the agent to perform Google searches.
    tools=[google_search]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    """
    Helper function to call the agent with a query.
    """
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

call_agent("what's the latest ai news?")

Code Execution¶
The built_in_code_execution tool enables the agent to execute code, specifically when using Gemini 2 models. This allows the model to perform tasks like calculations, data manipulation, or running small scripts.


Python
Java

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types

AGENT_NAME = "calculator_agent"
APP_NAME = "calculator"
USER_ID = "user1234"
SESSION_ID = "session_code_exec_async"
GEMINI_MODEL = "gemini-2.0-flash"

# Agent Definition
code_agent = LlmAgent(
    name=AGENT_NAME,
    model=GEMINI_MODEL,
    executor=[BuiltInCodeExecutor],
    instruction="""You are a calculator agent.
    When given a mathematical expression, write and execute Python code to calculate the result.
    Return only the final numerical result as plain text, without markdown or code blocks.
    """,
    description="Executes Python code to perform calculations.",
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
)
runner = Runner(agent=code_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction (Async)
async def call_agent_async(query):
    content = types.Content(role="user", parts=[types.Part(text=query)])
    print(f"\n--- Running Query: {query} ---")
    final_response_text = "No final text response captured."
    try:
        # Use run_async
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content
        ):
            print(f"Event ID: {event.id}, Author: {event.author}")

            # --- Check for specific parts FIRST ---
            has_specific_part = False
            if event.content and event.content.parts:
                for part in event.content.parts:  # Iterate through all parts
                    if part.executable_code:
                        # Access the actual code string via .code
                        print(
                            f"  Debug: Agent generated code:\n```python\n{part.executable_code.code}\n```"
                        )
                        has_specific_part = True
                    elif part.code_execution_result:
                        # Access outcome and output correctly
                        print(
                            f"  Debug: Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}"
                        )
                        has_specific_part = True
                    # Also print any text parts found in any event for debugging
                    elif part.text and not part.text.isspace():
                        print(f"  Text: '{part.text.strip()}'")
                        # Do not set has_specific_part=True here, as we want the final response logic below

            # --- Check for final response AFTER specific parts ---
            # Only consider it final if it doesn't have the specific code parts we just handled
            if not has_specific_part and event.is_final_response():
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    final_response_text = event.content.parts[0].text.strip()
                    print(f"==> Final Agent Response: {final_response_text}")
                else:
                    print("==> Final Agent Response: [No text content in final event]")

    except Exception as e:
        print(f"ERROR during agent run: {e}")
    print("-" * 30)


# Main async function to run the examples
async def main():
    await call_agent_async("Calculate the value of (5 + 7) * 3")
    await call_agent_async("What is 10 factorial?")


# Execute the main async function
try:
    asyncio.run(main())
except RuntimeError as e:
    # Handle specific error when running asyncio.run in an already running loop (like Jupyter/Colab)
    if "cannot be called from a running event loop" in str(e):
        print("\nRunning in an existing event loop (like Colab/Jupyter).")
        print("Please run `await main()` in a notebook cell instead.")
        # If in an interactive environment like a notebook, you might need to run:
        # await main()
    else:
        raise e  # Re-raise other runtime errors

Vertex AI Search¶
The vertex_ai_search_tool uses Google Cloud's Vertex AI Search, enabling the agent to search across your private, configured data stores (e.g., internal documents, company policies, knowledge bases). This built-in tool requires you to provide the specific data store ID during configuration.


import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import VertexAiSearchTool

# Replace with your actual Vertex AI Search Datastore ID
# Format: projects/<PROJECT_ID>/locations/<LOCATION>/collections/default_collection/dataStores/<DATASTORE_ID>
# e.g., "projects/12345/locations/us-central1/collections/default_collection/dataStores/my-datastore-123"
YOUR_DATASTORE_ID = "YOUR_DATASTORE_ID_HERE"

# Constants
APP_NAME_VSEARCH = "vertex_search_app"
USER_ID_VSEARCH = "user_vsearch_1"
SESSION_ID_VSEARCH = "session_vsearch_1"
AGENT_NAME_VSEARCH = "doc_qa_agent"
GEMINI_2_FLASH = "gemini-2.0-flash"

# Tool Instantiation
# You MUST provide your datastore ID here.
vertex_search_tool = VertexAiSearchTool(data_store_id=YOUR_DATASTORE_ID)

# Agent Definition
doc_qa_agent = LlmAgent(
    name=AGENT_NAME_VSEARCH,
    model=GEMINI_2_FLASH, # Requires Gemini model
    tools=[vertex_search_tool],
    instruction=f"""You are a helpful assistant that answers questions based on information found in the document store: {YOUR_DATASTORE_ID}.
    Use the search tool to find relevant information before answering.
    If the answer isn't in the documents, say that you couldn't find the information.
    """,
    description="Answers questions using a specific Vertex AI Search datastore.",
)

# Session and Runner Setup
session_service_vsearch = InMemorySessionService()
runner_vsearch = Runner(
    agent=doc_qa_agent, app_name=APP_NAME_VSEARCH, session_service=session_service_vsearch
)
session_vsearch = session_service_vsearch.create_session(
    app_name=APP_NAME_VSEARCH, user_id=USER_ID_VSEARCH, session_id=SESSION_ID_VSEARCH
)

# Agent Interaction Function
async def call_vsearch_agent_async(query):
    print("\n--- Running Vertex AI Search Agent ---")
    print(f"Query: {query}")
    if "YOUR_DATASTORE_ID_HERE" in YOUR_DATASTORE_ID:
        print("Skipping execution: Please replace YOUR_DATASTORE_ID_HERE with your actual datastore ID.")
        print("-" * 30)
        return

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "No response received."
    try:
        async for event in runner_vsearch.run_async(
            user_id=USER_ID_VSEARCH, session_id=SESSION_ID_VSEARCH, new_message=content
        ):
            # Like Google Search, results are often embedded in the model's response.
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()
                print(f"Agent Response: {final_response_text}")
                # You can inspect event.grounding_metadata for source citations
                if event.grounding_metadata:
                    print(f"  (Grounding metadata found with {len(event.grounding_metadata.grounding_attributions)} attributions)")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure your datastore ID is correct and the service account has permissions.")
    print("-" * 30)

# --- Run Example ---
async def run_vsearch_example():
    # Replace with a question relevant to YOUR datastore content
    await call_vsearch_agent_async("Summarize the main points about the Q2 strategy document.")
    await call_vsearch_agent_async("What safety procedures are mentioned for lab X?")

# Execute the example
# await run_vsearch_example()

# Running locally due to potential colab asyncio issues with multiple awaits
try:
    asyncio.run(run_vsearch_example())
except RuntimeError as e:
    if "cannot be called from a running event loop" in str(e):
        print("Skipping execution in running event loop (like Colab/Jupyter). Run locally.")
    else:
        raise e
Use Built-in tools with other tools¶
The following code sample demonstrates how to use multiple built-in tools or how to use built-in tools with other tools by using multiple agents:


Python
Java

from google.adk.tools import agent_tool
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.code_executors import BuiltInCodeExecutor


search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="""
    You're a specialist in Google Search
    """,
    tools=[google_search],
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="""
    You're a specialist in Code Execution
    """,
    code_executor=[BuiltInCodeExecutor],
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    tools=[agent_tool.AgentTool(agent=search_agent), agent_tool.AgentTool(agent=coding_agent)],
)

Limitations¶
Warning

Currently, for each root agent or single agent, only one built-in tool is supported. No other tools of any type can be used in the same agent.

For example, the following approach that uses a built-in tool along with other tools within a single agent is not currently supported:


Python
Java

root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    tools=[custom_function], 
    executor=[BuiltInCodeExecutor] # <-- not supported when used with tools
)

Warning

Built-in tools cannot be used within a sub-agent.

For example, the following approach that uses built-in tools within sub-agents is not currently supported:


Python
Java

search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="""
    You're a specialist in Google Search
    """,
    tools=[google_search],
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="""
    You're a specialist in Code Execution
    """,
    executor=[BuiltInCodeExecutor],
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    sub_agents=[
        search_agent,
        coding_agent
    ],
)

Third Party Tools¶
python_only

ADK is designed to be highly extensible, allowing you to seamlessly integrate tools from other AI Agent frameworks like CrewAI and LangChain. This interoperability is crucial because it allows for faster development time and allows you to reuse existing tools.

1. Using LangChain Tools¶
ADK provides the LangchainTool wrapper to integrate tools from the LangChain ecosystem into your agents.

Example: Web Search using LangChain's Tavily tool¶
Tavily provides a search API that returns answers derived from real-time search results, intended for use by applications like AI agents.

Follow ADK installation and setup guide.

Install Dependencies: Ensure you have the necessary LangChain packages installed. For example, to use the Tavily search tool, install its specific dependencies:


pip install langchain_community tavily-python
Obtain a Tavily API KEY and export it as an environment variable.


export TAVILY_API_KEY=<REPLACE_WITH_API_KEY>
Import: Import the LangchainTool wrapper from ADK and the specific LangChain tool you wish to use (e.g, TavilySearchResults).


from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults
Instantiate & Wrap: Create an instance of your LangChain tool and pass it to the LangchainTool constructor.


# Instantiate the LangChain tool
tavily_tool_instance = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# Wrap it with LangchainTool for ADK
adk_tavily_tool = LangchainTool(tool=tavily_tool_instance)
Add to Agent: Include the wrapped LangchainTool instance in your agent's tools list during definition.


from google.adk import Agent

# Define the ADK agent, including the wrapped tool
my_agent = Agent(
    name="langchain_tool_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using TavilySearch.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[adk_tavily_tool] # Add the wrapped tool here
)
Full Example: Tavily Search¶
Here's the full code combining the steps above to create and run an agent using the LangChain Tavily search tool.


import os
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.langchain_tool import LangchainTool
from google.genai import types
from langchain_community.tools import TavilySearchResults

# Ensure TAVILY_API_KEY is set in your environment
if not os.getenv("TAVILY_API_KEY"):
    print("Warning: TAVILY_API_KEY environment variable not set.")

APP_NAME = "news_app"
USER_ID = "1234"
SESSION_ID = "session1234"

# Instantiate LangChain tool
tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# Wrap with LangchainTool
adk_tavily_tool = LangchainTool(tool=tavily_search)

# Define Agent with the wrapped tool
my_agent = Agent(
    name="langchain_tool_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using TavilySearch.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[adk_tavily_tool] # Add the wrapped tool here
)

session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=my_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

call_agent("stock price of GOOG")
2. Using CrewAI tools¶
ADK provides the CrewaiTool wrapper to integrate tools from the CrewAI library.

Example: Web Search using CrewAI's Serper API¶
Serper API provides access to Google Search results programmatically. It allows applications, like AI agents, to perform real-time Google searches (including news, images, etc.) and get structured data back without needing to scrape web pages directly.

Follow ADK installation and setup guide.

Install Dependencies: Install the necessary CrewAI tools package. For example, to use the SerperDevTool:


pip install crewai-tools
Obtain a Serper API KEY and export it as an environment variable.


export SERPER_API_KEY=<REPLACE_WITH_API_KEY>
Import: Import CrewaiTool from ADK and the desired CrewAI tool (e.g, SerperDevTool).


from google.adk.tools.crewai_tool import CrewaiTool
from crewai_tools import SerperDevTool
Instantiate & Wrap: Create an instance of the CrewAI tool. Pass it to the CrewaiTool constructor. Crucially, you must provide a name and description to the ADK wrapper, as these are used by ADK's underlying model to understand when to use the tool.


# Instantiate the CrewAI tool
serper_tool_instance = SerperDevTool(
    n_results=10,
    save_file=False,
    search_type="news",
)

# Wrap it with CrewaiTool for ADK, providing name and description
adk_serper_tool = CrewaiTool(
    name="InternetNewsSearch",
    description="Searches the internet specifically for recent news articles using Serper.",
    tool=serper_tool_instance
)
Add to Agent: Include the wrapped CrewaiTool instance in your agent's tools list.


from google.adk import Agent

# Define the ADK agent
my_agent = Agent(
    name="crewai_search_agent",
    model="gemini-2.0-flash",
    description="Agent to find recent news using the Serper search tool.",
    instruction="I can find the latest news for you. What topic are you interested in?",
    tools=[adk_serper_tool] # Add the wrapped tool here
)
Full Example: Serper API¶
Here's the full code combining the steps above to create and run an agent using the CrewAI Serper API search tool.


import os
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.crewai_tool import CrewaiTool
from google.genai import types
from crewai_tools import SerperDevTool


# Constants
APP_NAME = "news_app"
USER_ID = "user1234"
SESSION_ID = "1234"

# Ensure SERPER_API_KEY is set in your environment
if not os.getenv("SERPER_API_KEY"):
    print("Warning: SERPER_API_KEY environment variable not set.")

serper_tool_instance = SerperDevTool(
    n_results=10,
    save_file=False,
    search_type="news",
)

adk_serper_tool = CrewaiTool(
    name="InternetNewsSearch",
    description="Searches the internet specifically for recent news articles using Serper.",
    tool=serper_tool_instance
)

serper_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    # Add the Serper tool
    tools=[adk_serper_tool]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=serper_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

call_agent("what's the latest news on AI Agents?")

Model Context Protocol Tools¶
This guide walks you through two ways of integrating Model Context Protocol (MCP) with ADK.

What is Model Context Protocol (MCP)?¶
The Model Context Protocol (MCP) is an open standard designed to standardize how Large Language Models (LLMs) like Gemini and Claude communicate with external applications, data sources, and tools. Think of it as a universal connection mechanism that simplifies how LLMs obtain context, execute actions, and interact with various systems.

MCP follows a client-server architecture, defining how data (resources), interactive templates (prompts), and actionable functions (tools) are exposed by an MCP server and consumed by an MCP client (which could be an LLM host application or an AI agent).

This guide covers two primary integration patterns:

Using Existing MCP Servers within ADK: An ADK agent acts as an MCP client, leveraging tools provided by external MCP servers.
Exposing ADK Tools via an MCP Server: Building an MCP server that wraps ADK tools, making them accessible to any MCP client.
Prerequisites¶
Before you begin, ensure you have the following set up:

Set up ADK: Follow the standard ADK setup instructions in the quickstart.
Install/update Python/Java: MCP requires Python version of 3.9 or higher for Python or Java 17+.
Setup Node.js and npx: (Python only) Many community MCP servers are distributed as Node.js packages and run using npx. Install Node.js (which includes npx) if you haven't already. For details, see https://nodejs.org/en.
Verify Installations: (Python only) Confirm adk and npx are in your PATH within the activated virtual environment:

# Both commands should print the path to the executables.
which adk
which npx
1. Using MCP servers with ADK agents (ADK as an MCP client) in adk web¶
This section demonstrates how to integrate tools from external MCP (Model Context Protocol) servers into your ADK agents. This is the most common integration pattern when your ADK agent needs to use capabilities provided by an existing service that exposes an MCP interface. You will see how the MCPToolset class can be directly added to your agent's tools list, enabling seamless connection to an MCP server, discovery of its tools, and making them available for your agent to use. These examples primarily focus on interactions within the adk web development environment.

MCPToolset class¶
The MCPToolset class is ADK's primary mechanism for integrating tools from an MCP server. When you include an MCPToolset instance in your agent's tools list, it automatically handles the interaction with the specified MCP server. Here's how it works:

Connection Management: On initialization, MCPToolset establishes and manages the connection to the MCP server. This can be a local server process (using StdioServerParameters for communication over standard input/output) or a remote server (using SseServerParams for Server-Sent Events). The toolset also handles the graceful shutdown of this connection when the agent or application terminates.
Tool Discovery & Adaptation: Once connected, MCPToolset queries the MCP server for its available tools (via the list_tools MCP method). It then converts the schemas of these discovered MCP tools into ADK-compatible BaseTool instances.
Exposure to Agent: These adapted tools are then made available to your LlmAgent as if they were native ADK tools.
Proxying Tool Calls: When your LlmAgent decides to use one of these tools, MCPToolset transparently proxies the call (using the call_tool MCP method) to the MCP server, sends the necessary arguments, and returns the server's response back to the agent.
Filtering (Optional): You can use the tool_filter parameter when creating an MCPToolset to select a specific subset of tools from the MCP server, rather than exposing all of them to your agent.
The following examples demonstrate how to use MCPToolset within the adk web development environment. For scenarios where you need more fine-grained control over the MCP connection lifecycle or are not using adk web, refer to the "Using MCP Tools in your own Agent out of adk web" section later in this page.

Example 1: File System MCP Server¶
This example demonstrates connecting to a local MCP server that provides file system operations.

Step 1: Define your Agent with MCPToolset¶
Create an agent.py file (e.g., in ./adk_agent_samples/mcp_agent/agent.py). The MCPToolset is instantiated directly within the tools list of your LlmAgent.

Important: Replace "/path/to/your/folder" in the args list with the absolute path to an actual folder on your local system that the MCP server can access.
Important: Place the .env file in the parent directory of the ./adk_agent_samples directory.

# ./adk_agent_samples/mcp_agent/agent.py
import os # Required for path operations
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# It's good practice to define paths dynamically if possible,
# or ensure the user understands the need for an ABSOLUTE path.
# For this example, we'll construct a path relative to this file,
# assuming '/path/to/your/folder' is in the same directory as agent.py.
# REPLACE THIS with an actual absolute path if needed for your setup.
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/path/to/your/folder")
# Ensure TARGET_FOLDER_PATH is an absolute path for the MCP server.
# If you created ./adk_agent_samples/mcp_agent/your_folder,

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='filesystem_assistant_agent',
    instruction='Help the user manage their files. You can list files, read files, etc.',
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=[
                    "-y",  # Argument for npx to auto-confirm install
                    "@modelcontextprotocol/server-filesystem",
                    # IMPORTANT: This MUST be an ABSOLUTE path to a folder the
                    # npx process can access.
                    # Replace with a valid absolute path on your system.
                    # For example: "/Users/youruser/accessible_mcp_files"
                    # or use a dynamically constructed absolute path:
                    os.path.abspath(TARGET_FOLDER_PATH),
                ],
            ),
            # Optional: Filter which tools from the MCP server are exposed
            # tool_filter=['list_directory', 'read_file']
        )
    ],
)
Step 2: Create an __init__.py file¶
Ensure you have an __init__.py in the same directory as agent.py to make it a discoverable Python package for ADK.


# ./adk_agent_samples/mcp_agent/__init__.py
from . import agent
Step 3: Run adk web and Interact¶
Navigate to the parent directory of mcp_agent (e.g., adk_agent_samples) in your terminal and run:


cd ./adk_agent_samples # Or your equivalent parent directory
adk web
Note for Windows users

When hitting the _make_subprocess_transport NotImplementedError, consider using adk web --no-reload instead.

Once the ADK Web UI loads in your browser:

Select the filesystem_assistant_agent from the agent dropdown.
Try prompts like:
"List files in the current directory."
"Can you read the file named sample.txt?" (assuming you created it in TARGET_FOLDER_PATH).
"What is the content of another_file.md?"
You should see the agent interacting with the MCP file system server, and the server's responses (file listings, file content) relayed through the agent. The adk web console (terminal where you ran the command) might also show logs from the npx process if it outputs to stderr.

MCP with ADK Web - FileSystem Example

Example 2: Google Maps MCP Server¶
This example demonstrates connecting to the Google Maps MCP server.

Step 1: Get API Key and Enable APIs¶
Google Maps API Key: Follow the directions at Use API keys to obtain a Google Maps API Key.
Enable APIs: In your Google Cloud project, ensure the following APIs are enabled:
Directions API
Routes API For instructions, see the Getting started with Google Maps Platform documentation.
Step 2: Define your Agent with MCPToolset for Google Maps¶
Modify your agent.py file (e.g., in ./adk_agent_samples/mcp_agent/agent.py). Replace YOUR_GOOGLE_MAPS_API_KEY with the actual API key you obtained.


# ./adk_agent_samples/mcp_agent/agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Retrieve the API key from an environment variable or directly insert it.
# Using an environment variable is generally safer.
# Ensure this environment variable is set in the terminal where you run 'adk web'.
# Example: export GOOGLE_MAPS_API_KEY="YOUR_ACTUAL_KEY"
google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")

if not google_maps_api_key:
    # Fallback or direct assignment for testing - NOT RECOMMENDED FOR PRODUCTION
    google_maps_api_key = "YOUR_GOOGLE_MAPS_API_KEY_HERE" # Replace if not using env var
    if google_maps_api_key == "YOUR_GOOGLE_MAPS_API_KEY_HERE":
        print("WARNING: GOOGLE_MAPS_API_KEY is not set. Please set it as an environment variable or in the script.")
        # You might want to raise an error or exit if the key is crucial and not found.

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='maps_assistant_agent',
    instruction='Help the user with mapping, directions, and finding places using Google Maps tools.',
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='npx',
                args=[
                    "-y",
                    "@modelcontextprotocol/server-google-maps",
                ],
                # Pass the API key as an environment variable to the npx process
                # This is how the MCP server for Google Maps expects the key.
                env={
                    "GOOGLE_MAPS_API_KEY": google_maps_api_key
                }
            ),
            # You can filter for specific Maps tools if needed:
            # tool_filter=['get_directions', 'find_place_by_id']
        )
    ],
)
Step 3: Ensure __init__.py Exists¶
If you created this in Example 1, you can skip this. Otherwise, ensure you have an __init__.py in the ./adk_agent_samples/mcp_agent/ directory:


# ./adk_agent_samples/mcp_agent/__init__.py
from . import agent
Step 4: Run adk web and Interact¶
Set Environment Variable (Recommended): Before running adk web, it's best to set your Google Maps API key as an environment variable in your terminal:


export GOOGLE_MAPS_API_KEY="YOUR_ACTUAL_GOOGLE_MAPS_API_KEY"
Replace YOUR_ACTUAL_GOOGLE_MAPS_API_KEY with your key.
Run adk web: Navigate to the parent directory of mcp_agent (e.g., adk_agent_samples) and run:


cd ./adk_agent_samples # Or your equivalent parent directory
adk web
Interact in the UI:

Select the maps_assistant_agent.
Try prompts like:
"Get directions from GooglePlex to SFO."
"Find coffee shops near Golden Gate Park."
"What's the route from Paris, France to Berlin, Germany?"
You should see the agent use the Google Maps MCP tools to provide directions or location-based information.

MCP with ADK Web - Google Maps Example

2. Building an MCP server with ADK tools (MCP server exposing ADK)¶
This pattern allows you to wrap existing ADK tools and make them available to any standard MCP client application. The example in this section exposes the ADK load_web_page tool through a custom-built MCP server.

Summary of steps¶
You will create a standard Python MCP server application using the mcp library. Within this server, you will:

Instantiate the ADK tool(s) you want to expose (e.g., FunctionTool(load_web_page)).
Implement the MCP server's @app.list_tools() handler to advertise the ADK tool(s). This involves converting the ADK tool definition to the MCP schema using the adk_to_mcp_tool_type utility from google.adk.tools.mcp_tool.conversion_utils.
Implement the MCP server's @app.call_tool() handler. This handler will:
Receive tool call requests from MCP clients.
Identify if the request targets one of your wrapped ADK tools.
Execute the ADK tool's .run_async() method.
Format the ADK tool's result into an MCP-compliant response (e.g., mcp.types.TextContent).
Prerequisites¶
Install the MCP server library in the same Python environment as your ADK installation:


pip install mcp
Step 1: Create the MCP Server Script¶
Create a new Python file for your MCP server, for example, my_adk_mcp_server.py.

Step 2: Implement the Server Logic¶
Add the following code to my_adk_mcp_server.py. This script sets up an MCP server that exposes the ADK load_web_page tool.


# my_adk_mcp_server.py
import asyncio
import json
import os
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types # Use alias to avoid conflict
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio # For running as a stdio server

# ADK Tool Imports
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.load_web_page import load_web_page # Example ADK tool
# ADK <-> MCP Conversion Utility
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# --- Load Environment Variables (If ADK tools need them, e.g., API keys) ---
load_dotenv() # Create a .env file in the same directory if needed

# --- Prepare the ADK Tool ---
# Instantiate the ADK tool you want to expose.
# This tool will be wrapped and called by the MCP server.
print("Initializing ADK load_web_page tool...")
adk_tool_to_expose = FunctionTool(load_web_page)
print(f"ADK tool '{adk_tool_to_expose.name}' initialized and ready to be exposed via MCP.")
# --- End ADK Tool Prep ---

# --- MCP Server Setup ---
print("Creating MCP Server instance...")
# Create a named MCP Server instance using the mcp.server library
app = Server("adk-tool-exposing-mcp-server")

# Implement the MCP server's handler to list available tools
@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    print("MCP Server: Received list_tools request.")
    # Convert the ADK tool's definition to the MCP Tool schema format
    mcp_tool_schema = adk_to_mcp_tool_type(adk_tool_to_expose)
    print(f"MCP Server: Advertising tool: {mcp_tool_schema.name}")
    return [mcp_tool_schema]

# Implement the MCP server's handler to execute a tool call
@app.call_tool()
async def call_mcp_tool(
    name: str, arguments: dict
) -> list[mcp_types.Content]: # MCP uses mcp_types.Content
    """MCP handler to execute a tool call requested by an MCP client."""
    print(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    # Check if the requested tool name matches our wrapped ADK tool
    if name == adk_tool_to_expose.name:
        try:
            # Execute the ADK tool's run_async method.
            # Note: tool_context is None here because this MCP server is
            # running the ADK tool outside of a full ADK Runner invocation.
            # If the ADK tool requires ToolContext features (like state or auth),
            # this direct invocation might need more sophisticated handling.
            adk_tool_response = await adk_tool_to_expose.run_async(
                args=arguments,
                tool_context=None,
            )
            print(f"MCP Server: ADK tool '{name}' executed. Response: {adk_tool_response}")

            # Format the ADK tool's response (often a dict) into an MCP-compliant format.
            # Here, we serialize the response dictionary as a JSON string within TextContent.
            # Adjust formatting based on the ADK tool's output and client needs.
            response_text = json.dumps(adk_tool_response, indent=2)
            # MCP expects a list of mcp_types.Content parts
            return [mcp_types.TextContent(type="text", text=response_text)]

        except Exception as e:
            print(f"MCP Server: Error executing ADK tool '{name}': {e}")
            # Return an error message in MCP format
            error_text = json.dumps({"error": f"Failed to execute tool '{name}': {str(e)}"})
            return [mcp_types.TextContent(type="text", text=error_text)]
    else:
        # Handle calls to unknown tools
        print(f"MCP Server: Tool '{name}' not found/exposed by this server.")
        error_text = json.dumps({"error": f"Tool '{name}' not implemented by this server."})
        return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output."""
    # Use the stdio_server context manager from the mcp.server.stdio library
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        print("MCP Stdio Server: Starting handshake with client...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name, # Use the server name defined above
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    # Define server capabilities - consult MCP docs for options
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        print("MCP Stdio Server: Run loop finished or client disconnected.")

if __name__ == "__main__":
    print("Launching MCP Server to expose ADK tools via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        print("\nMCP Server (stdio) stopped by user.")
    except Exception as e:
        print(f"MCP Server (stdio) encountered an error: {e}")
    finally:
        print("MCP Server (stdio) process exiting.")
# --- End MCP Server ---
Step 3: Test your Custom MCP Server with an ADK Agent¶
Now, create an ADK agent that will act as a client to the MCP server you just built. This ADK agent will use MCPToolset to connect to your my_adk_mcp_server.py script.

Create an agent.py (e.g., in ./adk_agent_samples/mcp_client_agent/agent.py):


# ./adk_agent_samples/mcp_client_agent/agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters

# IMPORTANT: Replace this with the ABSOLUTE path to your my_adk_mcp_server.py script
PATH_TO_YOUR_MCP_SERVER_SCRIPT = "/path/to/your/my_adk_mcp_server.py" # <<< REPLACE

if PATH_TO_YOUR_MCP_SERVER_SCRIPT == "/path/to/your/my_adk_mcp_server.py":
    print("WARNING: PATH_TO_YOUR_MCP_SERVER_SCRIPT is not set. Please update it in agent.py.")
    # Optionally, raise an error if the path is critical

root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='web_reader_mcp_client_agent',
    instruction="Use the 'load_web_page' tool to fetch content from a URL provided by the user.",
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command='python3', # Command to run your MCP server script
                args=[PATH_TO_YOUR_MCP_SERVER_SCRIPT], # Argument is the path to the script
            )
            # tool_filter=['load_web_page'] # Optional: ensure only specific tools are loaded
        )
    ],
)
And an __init__.py in the same directory:


# ./adk_agent_samples/mcp_client_agent/__init__.py
from . import agent
To run the test:

Start your custom MCP server (optional, for separate observation): You can run your my_adk_mcp_server.py directly in one terminal to see its logs:


python3 /path/to/your/my_adk_mcp_server.py
It will print "Launching MCP Server..." and wait. The ADK agent (run via adk web) will then connect to this process if the command in StdioServerParameters is set up to execute it. (Alternatively, MCPToolset will start this server script as a subprocess automatically when the agent initializes).
Run adk web for the client agent: Navigate to the parent directory of mcp_client_agent (e.g., adk_agent_samples) and run:


cd ./adk_agent_samples # Or your equivalent parent directory
adk web
Interact in the ADK Web UI:

Select the web_reader_mcp_client_agent.
Try a prompt like: "Load the content from https://example.com"
The ADK agent (web_reader_mcp_client_agent) will use MCPToolset to start and connect to your my_adk_mcp_server.py. Your MCP server will receive the call_tool request, execute the ADK load_web_page tool, and return the result. The ADK agent will then relay this information. You should see logs from both the ADK Web UI (and its terminal) and potentially from your my_adk_mcp_server.py terminal if you ran it separately.

This example demonstrates how ADK tools can be encapsulated within an MCP server, making them accessible to a broader range of MCP-compliant clients, not just ADK agents.

Refer to the documentation, to try it out with Claude Desktop.

Using MCP Tools in your own Agent out of adk web¶
This section is relevant to you if:

You are developing your own Agent using ADK
And, you are NOT using adk web,
And, you are exposing the agent via your own UI
Using MCP Tools requires a different setup than using regular tools, due to the fact that specs for MCP Tools are fetched asynchronously from the MCP Server running remotely, or in another process.

The following example is modified from the "Example 1: File System MCP Server" example above. The main differences are:

Your tool and agent are created asynchronously
You need to properly manage the exit stack, so that your agents and tools are destructed properly when the connection to MCP Server is closed.

# agent.py (modify get_tools_async and other parts as needed)
# ./adk_agent_samples/mcp_agent/agent.py
import os
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters

# Load environment variables from .env file in the parent directory
# Place this near the top, before using env vars like API keys
load_dotenv('../.env')

# Ensure TARGET_FOLDER_PATH is an absolute path for the MCP server.
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/path/to/your/folder")

# --- Step 1: Agent Definition ---
async def get_agent_async():
  """Creates an ADK Agent equipped with tools from the MCP Server."""
  toolset = MCPToolset(
      # Use StdioServerParameters for local process communication
      connection_params=StdioServerParameters(
          command='npx', # Command to run the server
          args=["-y",    # Arguments for the command
                "@modelcontextprotocol/server-filesystem",
                TARGET_FOLDER_PATH],
      ),
      tool_filter=['read_file', 'list_directory'] # Optional: filter specific tools
      # For remote servers, you would use SseServerParams instead:
      # connection_params=SseServerParams(url="http://remote-server:port/path", headers={...})
  )

  # Use in an agent
  root_agent = LlmAgent(
      model='gemini-2.0-flash', # Adjust model name if needed based on availability
      name='enterprise_assistant',
      instruction='Help user accessing their file systems',
      tools=[toolset], # Provide the MCP tools to the ADK agent
  )
  return root_agent, toolset

# --- Step 2: Main Execution Logic ---
async def async_main():
  session_service = InMemorySessionService()
  # Artifact service might not be needed for this example
  artifacts_service = InMemoryArtifactService()

  session = await session_service.create_session(
      state={}, app_name='mcp_filesystem_app', user_id='user_fs'
  )

  # TODO: Change the query to be relevant to YOUR specified folder.
  # e.g., "list files in the 'documents' subfolder" or "read the file 'notes.txt'"
  query = "list files in the tests folder"
  print(f"User Query: '{query}'")
  content = types.Content(role='user', parts=[types.Part(text=query)])

  root_agent, toolset = await get_agent_async()

  runner = Runner(
      app_name='mcp_filesystem_app',
      agent=root_agent,
      artifact_service=artifacts_service, # Optional
      session_service=session_service,
  )

  print("Running agent...")
  events_async = runner.run_async(
      session_id=session.id, user_id=session.user_id, new_message=content
  )

  async for event in events_async:
    print(f"Event received: {event}")

  # Cleanup is handled automatically by the agent framework
  # But you can also manually close if needed:
  print("Closing MCP server connection...")
  await toolset.close()
  print("Cleanup complete.")

if __name__ == '__main__':
  try:
    asyncio.run(async_main())
  except Exception as e:
    print(f"An error occurred: {e}")
Key considerations¶
When working with MCP and ADK, keep these points in mind:

Protocol vs. Library: MCP is a protocol specification, defining communication rules. ADK is a Python library/framework for building agents. MCPToolset bridges these by implementing the client side of the MCP protocol within the ADK framework. Conversely, building an MCP server in Python requires using the model-context-protocol library.

ADK Tools vs. MCP Tools:

ADK Tools (BaseTool, FunctionTool, AgentTool, etc.) are Python objects designed for direct use within the ADK's LlmAgent and Runner.
MCP Tools are capabilities exposed by an MCP Server according to the protocol's schema. MCPToolset makes these look like ADK tools to an LlmAgent.
Langchain/CrewAI Tools are specific implementations within those libraries, often simple functions or classes, lacking the server/protocol structure of MCP. ADK offers wrappers (LangchainTool, CrewaiTool) for some interoperability.
Asynchronous nature: Both ADK and the MCP Python library are heavily based on the asyncio Python library. Tool implementations and server handlers should generally be async functions.

Stateful sessions (MCP): MCP establishes stateful, persistent connections between a client and server instance. This differs from typical stateless REST APIs.

Deployment: This statefulness can pose challenges for scaling and deployment, especially for remote servers handling many users. The original MCP design often assumed client and server were co-located. Managing these persistent connections requires careful infrastructure considerations (e.g., load balancing, session affinity).
ADK MCPToolset: Manages this connection lifecycle. The exit_stack pattern shown in the examples is crucial for ensuring the connection (and potentially the server process) is properly terminated when the ADK agent finishes.

OpenAPI Integration¶
python_only

Integrating REST APIs with OpenAPI¶
ADK simplifies interacting with external REST APIs by automatically generating callable tools directly from an OpenAPI Specification (v3.x). This eliminates the need to manually define individual function tools for each API endpoint.

Core Benefit

Use OpenAPIToolset to instantly create agent tools (RestApiTool) from your existing API documentation (OpenAPI spec), enabling agents to seamlessly call your web services.

Key Components¶
OpenAPIToolset: This is the primary class you'll use. You initialize it with your OpenAPI specification, and it handles the parsing and generation of tools.
RestApiTool: This class represents a single, callable API operation (like GET /pets/{petId} or POST /pets). OpenAPIToolset creates one RestApiTool instance for each operation defined in your spec.
How it Works¶
The process involves these main steps when you use OpenAPIToolset:

Initialization & Parsing:

You provide the OpenAPI specification to OpenAPIToolset either as a Python dictionary, a JSON string, or a YAML string.
The toolset internally parses the spec, resolving any internal references ($ref) to understand the complete API structure.
Operation Discovery:

It identifies all valid API operations (e.g., GET, POST, PUT, DELETE) defined within the paths object of your specification.
Tool Generation:

For each discovered operation, OpenAPIToolset automatically creates a corresponding RestApiTool instance.
Tool Name: Derived from the operationId in the spec (converted to snake_case, max 60 chars). If operationId is missing, a name is generated from the method and path.
Tool Description: Uses the summary or description from the operation for the LLM.
API Details: Stores the required HTTP method, path, server base URL, parameters (path, query, header, cookie), and request body schema internally.
RestApiTool Functionality: Each generated RestApiTool:

Schema Generation: Dynamically creates a FunctionDeclaration based on the operation's parameters and request body. This schema tells the LLM how to call the tool (what arguments are expected).
Execution: When called by the LLM, it constructs the correct HTTP request (URL, headers, query params, body) using the arguments provided by the LLM and the details from the OpenAPI spec. It handles authentication (if configured) and executes the API call using the requests library.
Response Handling: Returns the API response (typically JSON) back to the agent flow.
Authentication: You can configure global authentication (like API keys or OAuth - see Authentication for details) when initializing OpenAPIToolset. This authentication configuration is automatically applied to all generated RestApiTool instances.

Usage Workflow¶
Follow these steps to integrate an OpenAPI spec into your agent:

Obtain Spec: Get your OpenAPI specification document (e.g., load from a .json or .yaml file, fetch from a URL).
Instantiate Toolset: Create an OpenAPIToolset instance, passing the spec content and type (spec_str/spec_dict, spec_str_type). Provide authentication details (auth_scheme, auth_credential) if required by the API.


from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# Example with a JSON string
openapi_spec_json = '...' # Your OpenAPI JSON string
toolset = OpenAPIToolset(spec_str=openapi_spec_json, spec_str_type="json")

# Example with a dictionary
# openapi_spec_dict = {...} # Your OpenAPI spec as a dict
# toolset = OpenAPIToolset(spec_dict=openapi_spec_dict)
Add to Agent: Include the retrieved tools in your LlmAgent's tools list.


from google.adk.agents import LlmAgent

my_agent = LlmAgent(
    name="api_interacting_agent",
    model="gemini-2.0-flash", # Or your preferred model
    tools=[toolset], # Pass the toolset
    # ... other agent config ...
)
Instruct Agent: Update your agent's instructions to inform it about the new API capabilities and the names of the tools it can use (e.g., list_pets, create_pet). The tool descriptions generated from the spec will also help the LLM.

Run Agent: Execute your agent using the Runner. When the LLM determines it needs to call one of the APIs, it will generate a function call targeting the appropriate RestApiTool, which will then handle the HTTP request automatically.
Example¶
This example demonstrates generating tools from a simple Pet Store OpenAPI spec (using httpbin.org for mock responses) and interacting with them via an agent.

Code: Pet Store API
openapi_example.py

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import uuid # For unique session IDs
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- OpenAPI Tool Imports ---
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# --- Load Environment Variables (If ADK tools need them, e.g., API keys) ---
load_dotenv() # Create a .env file in the same directory if needed

# --- Constants ---
APP_NAME_OPENAPI = "openapi_petstore_app"
USER_ID_OPENAPI = "user_openapi_1"
SESSION_ID_OPENAPI = f"session_openapi_{uuid.uuid4()}" # Unique session ID
AGENT_NAME_OPENAPI = "petstore_manager_agent"
GEMINI_MODEL = "gemini-2.0-flash"

# --- Sample OpenAPI Specification (JSON String) ---
# A basic Pet Store API example using httpbin.org as a mock server
openapi_spec_string = """
{
  "openapi": "3.0.0",
  "info": {
    "title": "Simple Pet Store API (Mock)",
    "version": "1.0.1",
    "description": "An API to manage pets in a store, using httpbin for responses."
  },
  "servers": [
    {
      "url": "https://httpbin.org",
      "description": "Mock server (httpbin.org)"
    }
  ],
  "paths": {
    "/get": {
      "get": {
        "summary": "List all pets (Simulated)",
        "operationId": "listPets",
        "description": "Simulates returning a list of pets. Uses httpbin's /get endpoint which echoes query parameters.",
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "Maximum number of pets to return",
            "required": false,
            "schema": { "type": "integer", "format": "int32" }
          },
          {
             "name": "status",
             "in": "query",
             "description": "Filter pets by status",
             "required": false,
             "schema": { "type": "string", "enum": ["available", "pending", "sold"] }
          }
        ],
        "responses": {
          "200": {
            "description": "A list of pets (echoed query params).",
            "content": { "application/json": { "schema": { "type": "object" } } }
          }
        }
      }
    },
    "/post": {
      "post": {
        "summary": "Create a pet (Simulated)",
        "operationId": "createPet",
        "description": "Simulates adding a new pet. Uses httpbin's /post endpoint which echoes the request body.",
        "requestBody": {
          "description": "Pet object to add",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": ["name"],
                "properties": {
                  "name": {"type": "string", "description": "Name of the pet"},
                  "tag": {"type": "string", "description": "Optional tag for the pet"}
                }
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Pet created successfully (echoed request body).",
            "content": { "application/json": { "schema": { "type": "object" } } }
          }
        }
      }
    },
    "/get?petId={petId}": {
      "get": {
        "summary": "Info for a specific pet (Simulated)",
        "operationId": "showPetById",
        "description": "Simulates returning info for a pet ID. Uses httpbin's /get endpoint.",
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "description": "This is actually passed as a query param to httpbin /get",
            "required": true,
            "schema": { "type": "integer", "format": "int64" }
          }
        ],
        "responses": {
          "200": {
            "description": "Information about the pet (echoed query params)",
            "content": { "application/json": { "schema": { "type": "object" } } }
          },
          "404": { "description": "Pet not found (simulated)" }
        }
      }
    }
  }
}
"""

# --- Create OpenAPIToolset ---
petstore_toolset = OpenAPIToolset(
    spec_str=openapi_spec_string,
    spec_str_type='json',
    # No authentication needed for httpbin.org
)

# --- Agent Definition ---
root_agent = LlmAgent(
    name=AGENT_NAME_OPENAPI,
    model=GEMINI_MODEL,
    tools=[petstore_toolset], # Pass the list of RestApiTool objects
    instruction="""You are a Pet Store assistant managing pets via an API.
    Use the available tools to fulfill user requests.
    When creating a pet, confirm the details echoed back by the API.
    When listing pets, mention any filters used (like limit or status).
    When showing a pet by ID, state the ID you requested.
    """,
    description="Manages a Pet Store using tools generated from an OpenAPI spec."
)

# --- Session and Runner Setup ---
async def setup_session_and_runner():
    session_service_openapi = InMemorySessionService()
    runner_openapi = Runner(
        agent=root_agent,
        app_name=APP_NAME_OPENAPI,
        session_service=session_service_openapi,
    )
    await session_service_openapi.create_session(
        app_name=APP_NAME_OPENAPI,
        user_id=USER_ID_OPENAPI,
        session_id=SESSION_ID_OPENAPI,
    )
    return runner_openapi

# --- Agent Interaction Function ---
async def call_openapi_agent_async(query, runner_openapi):
    print("\n--- Running OpenAPI Pet Store Agent ---")
    print(f"Query: {query}")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not provide a final text response."
    try:
        async for event in runner_openapi.run_async(
            user_id=USER_ID_OPENAPI, session_id=SESSION_ID_OPENAPI, new_message=content
            ):
            # Optional: Detailed event logging for debugging
            # print(f"  DEBUG Event: Author={event.author}, Type={'Final' if event.is_final_response() else 'Intermediate'}, Content={str(event.content)[:100]}...")
            if event.get_function_calls():
                call = event.get_function_calls()[0]
                print(f"  Agent Action: Called function '{call.name}' with args {call.args}")
            elif event.get_function_responses():
                response = event.get_function_responses()[0]
                print(f"  Agent Action: Received response for '{response.name}'")
                # print(f"  Tool Response Snippet: {str(response.response)[:200]}...") # Uncomment for response details
            elif event.is_final_response() and event.content and event.content.parts:
                # Capture the last final text response
                final_response_text = event.content.parts[0].text.strip()

        print(f"Agent Final Response: {final_response_text}")

    except Exception as e:
        print(f"An error occurred during agent run: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for errors
    print("-" * 30)

# --- Run Examples ---
async def run_openapi_example():
    runner_openapi = await setup_session_and_runner()

    # Trigger listPets
    await call_openapi_agent_async("Show me the pets available.", runner_openapi)
    # Trigger createPet
    await call_openapi_agent_async("Please add a new dog named 'Dukey'.", runner_openapi)
    # Trigger showPetById
    await call_openapi_agent_async("Get info for pet with ID 123.", runner_openapi)

# --- Execute ---
if __name__ == "__main__":
    print("Executing OpenAPI example...")
    # Use asyncio.run() for top-level execution
    try:
        asyncio.run(run_openapi_example())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Info: Cannot run asyncio.run from a running event loop (e.g., Jupyter/Colab).")
            # If in Jupyter/Colab, you might need to run like this:
            # await run_openapi_example()
        else:
            raise e
    print("OpenAPI example finished.")

Authenticating with Tools¶
python_only

Core Concepts¶
Many tools need to access protected resources (like user data in Google Calendar, Salesforce records, etc.) and require authentication. ADK provides a system to handle various authentication methods securely.

The key components involved are:

AuthScheme: Defines how an API expects authentication credentials (e.g., as an API Key in a header, an OAuth 2.0 Bearer token). ADK supports the same types of authentication schemes as OpenAPI 3.0. To know more about what each type of credential is, refer to OpenAPI doc: Authentication. ADK uses specific classes like APIKey, HTTPBearer, OAuth2, OpenIdConnectWithConfig.
AuthCredential: Holds the initial information needed to start the authentication process (e.g., your application's OAuth Client ID/Secret, an API key value). It includes an auth_type (like API_KEY, OAUTH2, SERVICE_ACCOUNT) specifying the credential type.
The general flow involves providing these details when configuring a tool. ADK then attempts to automatically exchange the initial credential for a usable one (like an access token) before the tool makes an API call. For flows requiring user interaction (like OAuth consent), a specific interactive process involving the Agent Client application is triggered.

Supported Initial Credential Types¶
API_KEY: For simple key/value authentication. Usually requires no exchange.
HTTP: Can represent Basic Auth (not recommended/supported for exchange) or already obtained Bearer tokens. If it's a Bearer token, no exchange is needed.
OAUTH2: For standard OAuth 2.0 flows. Requires configuration (client ID, secret, scopes) and often triggers the interactive flow for user consent.
OPEN_ID_CONNECT: For authentication based on OpenID Connect. Similar to OAuth2, often requires configuration and user interaction.
SERVICE_ACCOUNT: For Google Cloud Service Account credentials (JSON key or Application Default Credentials). Typically exchanged for a Bearer token.
Configuring Authentication on Tools¶
You set up authentication when defining your tool:

RestApiTool / OpenAPIToolset: Pass auth_scheme and auth_credential during initialization

GoogleApiToolSet Tools: ADK has built-in 1st party tools like Google Calendar, BigQuery etc,. Use the toolset's specific method.

APIHubToolset / ApplicationIntegrationToolset: Pass auth_scheme and auth_credentialduring initialization, if the API managed in API Hub / provided by Application Integration requires authentication.

WARNING

Storing sensitive credentials like access tokens and especially refresh tokens directly in the session state might pose security risks depending on your session storage backend (SessionService) and overall application security posture.

InMemorySessionService: Suitable for testing and development, but data is lost when the process ends. Less risk as it's transient.
Database/Persistent Storage: Strongly consider encrypting the token data before storing it in the database using a robust encryption library (like cryptography) and managing encryption keys securely (e.g., using a key management service).
Secure Secret Stores: For production environments, storing sensitive credentials in a dedicated secret manager (like Google Cloud Secret Manager or HashiCorp Vault) is the most recommended approach. Your tool could potentially store only short-lived access tokens or secure references (not the refresh token itself) in the session state, fetching the necessary secrets from the secure store when needed.
Journey 1: Building Agentic Applications with Authenticated Tools¶
This section focuses on using pre-existing tools (like those from RestApiTool/ OpenAPIToolset, APIHubToolset, GoogleApiToolSet) that require authentication within your agentic application. Your main responsibility is configuring the tools and handling the client-side part of interactive authentication flows (if required by the tool).

1. Configuring Tools with Authentication¶
When adding an authenticated tool to your agent, you need to provide its required AuthScheme and your application's initial AuthCredential.

A. Using OpenAPI-based Toolsets (OpenAPIToolset, APIHubToolset, etc.)

Pass the scheme and credential during toolset initialization. The toolset applies them to all generated tools. Here are few ways to create tools with authentication in ADK.


API Key
OAuth2
Service Account
OpenID connect
Create a tool requiring an API Key.


from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.adk.tools.apihub_tool.apihub_toolset import APIHubToolset
auth_scheme, auth_credential = token_to_scheme_credential(
   "apikey", "query", "apikey", YOUR_API_KEY_STRING
)
sample_api_toolset = APIHubToolset(
   name="sample-api-requiring-api-key",
   description="A tool using an API protected by API Key",
   apihub_resource_name="...",
   auth_scheme=auth_scheme,
   auth_credential=auth_credential,
)

B. Using Google API Toolsets (e.g., calendar_tool_set)

These toolsets often have dedicated configuration methods.

Tip: For how to create a Google OAuth Client ID & Secret, see this guide: Get your Google API Client ID


# Example: Configuring Google Calendar Tools
from google.adk.tools.google_api_tool import calendar_tool_set

client_id = "YOUR_GOOGLE_OAUTH_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_GOOGLE_OAUTH_CLIENT_SECRET"

# Use the specific configure method for this toolset type
calendar_tool_set.configure_auth(
    client_id=oauth_client_id, client_secret=oauth_client_secret
)

# agent = LlmAgent(..., tools=calendar_tool_set.get_tool('calendar_tool_set'))
The sequence diagram of auth request flow (where tools are requesting auth credentials) looks like below:

Authentication

2. Handling the Interactive OAuth/OIDC Flow (Client-Side)¶
If a tool requires user login/consent (typically OAuth 2.0 or OIDC), the ADK framework pauses execution and signals your Agent Client application. There are two cases:

Agent Client application runs the agent directly (via runner.run_async) in the same process. e.g. UI backend, CLI app, or Spark job etc.
Agent Client application interacts with ADK's fastapi server via /run or /run_sse endpoint. While ADK's fastapi server could be setup on the same server or different server as Agent Client application
The second case is a special case of first case, because /run or /run_sse endpoint also invokes runner.run_async. The only differences are:

Whether to call a python function to run the agent (first case) or call a service endpoint to run the agent (second case).
Whether the result events are in-memory objects (first case) or serialized json string in http response (second case).
Below sections focus on the first case and you should be able to map it to the second case very straightforward. We will also describe some differences to handle for the second case if necessary.

Here's the step-by-step process for your client application:

Step 1: Run Agent & Detect Auth Request

Initiate the agent interaction using runner.run_async.
Iterate through the yielded events.
Look for a specific function call event whose function call has a special name: adk_request_credential. This event signals that user interaction is needed. You can use helper functions to identify this event and extract necessary information. (For the second case, the logic is similar. You deserialize the event from the http response).

# runner = Runner(...)
# session = await session_service.create_session(...)
# content = types.Content(...) # User's initial query

print("\nRunning agent...")
events_async = runner.run_async(
    session_id=session.id, user_id='user', new_message=content
)

auth_request_function_call_id, auth_config = None, None

async for event in events_async:
    # Use helper to check for the specific auth request event
    if (auth_request_function_call := get_auth_request_function_call(event)):
        print("--> Authentication required by agent.")
        # Store the ID needed to respond later
        if not (auth_request_function_call_id := auth_request_function_call.id):
            raise ValueError(f'Cannot get function call id from function call: {auth_request_function_call}')
        # Get the AuthConfig containing the auth_uri etc.
        auth_config = get_auth_config(auth_request_function_call)
        break # Stop processing events for now, need user interaction

if not auth_request_function_call_id:
    print("\nAuth not required or agent finished.")
    # return # Or handle final response if received
Helper functions helpers.py:


from google.adk.events import Event
from google.adk.auth import AuthConfig # Import necessary type
from google.genai import types

def get_auth_request_function_call(event: Event) -> types.FunctionCall:
    # Get the special auth request function call from the event
    if not event.content or event.content.parts:
        return
    for part in event.content.parts:
        if (
            part 
            and part.function_call 
            and part.function_call.name == 'adk_request_credential'
            and event.long_running_tool_ids 
            and part.function_call.id in event.long_running_tool_ids
        ):

            return part.function_call

def get_auth_config(auth_request_function_call: types.FunctionCall) -> AuthConfig:
    # Extracts the AuthConfig object from the arguments of the auth request function call
    if not auth_request_function_call.args or not (auth_config := auth_request_function_call.args.get('auth_config')):
        raise ValueError(f'Cannot get auth config from function call: {auth_request_function_call}')
    if not isinstance(auth_config, AuthConfig):
        raise ValueError(f'Cannot get auth config {auth_config} is not an instance of AuthConfig.')
    return auth_config
Step 2: Redirect User for Authorization

Get the authorization URL (auth_uri) from the auth_config extracted in the previous step.
Crucially, append your application's redirect_uri as a query parameter to this auth_uri. This redirect_uri must be pre-registered with your OAuth provider (e.g., Google Cloud Console, Okta admin panel).
Direct the user to this complete URL (e.g., open it in their browser).

# (Continuing after detecting auth needed)

if auth_request_function_call_id and auth_config:
    # Get the base authorization URL from the AuthConfig
    base_auth_uri = auth_config.exchanged_auth_credential.oauth2.auth_uri

    if base_auth_uri:
        redirect_uri = 'http://localhost:8000/callback' # MUST match your OAuth client app config
        # Append redirect_uri (use urlencode in production)
        auth_request_uri = base_auth_uri + f'&redirect_uri={redirect_uri}'
        # Now you need to redirect your end user to this auth_request_uri or ask them to open this auth_request_uri in their browser
        # This auth_request_uri should be served by the corresponding auth provider and the end user should login and authorize your applicaiton to access their data
        # And then the auth provider will redirect the end user to the redirect_uri you provided
        # Next step: Get this callback URL from the user (or your web server handler)
    else:
         print("ERROR: Auth URI not found in auth_config.")
         # Handle error
Step 3. Handle the Redirect Callback (Client):

Your application must have a mechanism (e.g., a web server route at the redirect_uri) to receive the user after they authorize the application with the provider.
The provider redirects the user to your redirect_uri and appends an authorization_code (and potentially state, scope) as query parameters to the URL.
Capture the full callback URL from this incoming request.
(This step happens outside the main agent execution loop, in your web server or equivalent callback handler.)
Step 4. Send Authentication Result Back to ADK (Client):

Once you have the full callback URL (containing the authorization code), retrieve the auth_request_function_call_id and the auth_config object saved in Client Step 1.
Set the captured callback URL into the exchanged_auth_credential.oauth2.auth_response_uri field. Also ensure exchanged_auth_credential.oauth2.redirect_uri contains the redirect URI you used.
Create a types.Content object containing a types.Part with a types.FunctionResponse.
Set name to "adk_request_credential". (Note: This is a special name for ADK to proceed with authentication. Do not use other names.)
Set id to the auth_request_function_call_id you saved.
Set response to the serialized (e.g., .model_dump()) updated AuthConfig object.
Call runner.run_async again for the same session, passing this FunctionResponse content as the new_message.

# (Continuing after user interaction)

    # Simulate getting the callback URL (e.g., from user paste or web handler)
    auth_response_uri = await get_user_input(
        f'Paste the full callback URL here:\n> '
    )
    auth_response_uri = auth_response_uri.strip() # Clean input

    if not auth_response_uri:
        print("Callback URL not provided. Aborting.")
        return

    # Update the received AuthConfig with the callback details
    auth_config.exchanged_auth_credential.oauth2.auth_response_uri = auth_response_uri
    # Also include the redirect_uri used, as the token exchange might need it
    auth_config.exchanged_auth_credential.oauth2.redirect_uri = redirect_uri

    # Construct the FunctionResponse Content object
    auth_content = types.Content(
        role='user', # Role can be 'user' when sending a FunctionResponse
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=auth_request_function_call_id,       # Link to the original request
                    name='adk_request_credential', # Special framework function name
                    response=auth_config.model_dump() # Send back the *updated* AuthConfig
                )
            )
        ],
    )

    # --- Resume Execution ---
    print("\nSubmitting authentication details back to the agent...")
    events_async_after_auth = runner.run_async(
        session_id=session.id,
        user_id='user',
        new_message=auth_content, # Send the FunctionResponse back
    )

    # --- Process Final Agent Output ---
    print("\n--- Agent Response after Authentication ---")
    async for event in events_async_after_auth:
        # Process events normally, expecting the tool call to succeed now
        print(event) # Print the full event for inspection
Step 5: ADK Handles Token Exchange & Tool Retry and gets Tool result

ADK receives the FunctionResponse for adk_request_credential.
It uses the information in the updated AuthConfig (including the callback URL containing the code) to perform the OAuth token exchange with the provider's token endpoint, obtaining the access token (and possibly refresh token).
ADK internally makes these tokens available by setting them in the session state).
ADK automatically retries the original tool call (the one that initially failed due to missing auth).
This time, the tool finds the valid tokens (via tool_context.get_auth_response()) and successfully executes the authenticated API call.
The agent receives the actual result from the tool and generates its final response to the user.
The sequence diagram of auth response flow (where Agent Client send back the auth response and ADK retries tool calling) looks like below:

Authentication

Journey 2: Building Custom Tools (FunctionTool) Requiring Authentication¶
This section focuses on implementing the authentication logic inside your custom Python function when creating a new ADK Tool. We will implement a FunctionTool as an example.

Prerequisites¶
Your function signature must include tool_context: ToolContext. ADK automatically injects this object, providing access to state and auth mechanisms.


from google.adk.tools import FunctionTool, ToolContext
from typing import Dict

def my_authenticated_tool_function(param1: str, ..., tool_context: ToolContext) -> dict:
    # ... your logic ...
    pass

my_tool = FunctionTool(func=my_authenticated_tool_function)
Authentication Logic within the Tool Function¶
Implement the following steps inside your function:

Step 1: Check for Cached & Valid Credentials:

Inside your tool function, first check if valid credentials (e.g., access/refresh tokens) are already stored from a previous run in this session. Credentials for the current sessions should be stored in tool_context.invocation_context.session.state (a dictionary of state) Check existence of existing credentials by checking tool_context.invocation_context.session.state.get(credential_name, None).


from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Inside your tool function
TOKEN_CACHE_KEY = "my_tool_tokens" # Choose a unique key
SCOPES = ["scope1", "scope2"] # Define required scopes

creds = None
cached_token_info = tool_context.state.get(TOKEN_CACHE_KEY)
if cached_token_info:
    try:
        creds = Credentials.from_authorized_user_info(cached_token_info, SCOPES)
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            tool_context.state[TOKEN_CACHE_KEY] = json.loads(creds.to_json()) # Update cache
        elif not creds.valid:
            creds = None # Invalid, needs re-auth
            tool_context.state[TOKEN_CACHE_KEY] = None
    except Exception as e:
        print(f"Error loading/refreshing cached creds: {e}")
        creds = None
        tool_context.state[TOKEN_CACHE_KEY] = None

if creds and creds.valid:
    # Skip to Step 5: Make Authenticated API Call
    pass
else:
    # Proceed to Step 2...
    pass
Step 2: Check for Auth Response from Client

If Step 1 didn't yield valid credentials, check if the client just completed the interactive flow by calling exchanged_credential = tool_context.get_auth_response().
This returns the updated exchanged_credential object sent back by the client (containing the callback URL in auth_response_uri).

# Use auth_scheme and auth_credential configured in the tool.
# exchanged_credential: AuthCredential | None

exchanged_credential = tool_context.get_auth_response(AuthConfig(
  auth_scheme=auth_scheme,
  raw_auth_credential=auth_credential,
))
# If exchanged_credential is not None, then there is already an exchanged credetial from the auth response. 
if exchanged_credential:
   # ADK exchanged the access token already for us
        access_token = exchanged_credential.oauth2.access_token
        refresh_token = exchanged_credential.oauth2.refresh_token
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri=auth_scheme.flows.authorizationCode.tokenUrl,
            client_id=auth_credential.oauth2.client_id,
            client_secret=auth_credential.oauth2.client_secret,
            scopes=list(auth_scheme.flows.authorizationCode.scopes.keys()),
        )
    # Cache the token in session state and call the API, skip to step 5
Step 3: Initiate Authentication Request

If no valid credentials (Step 1.) and no auth response (Step 2.) are found, the tool needs to start the OAuth flow. Define the AuthScheme and initial AuthCredential and call tool_context.request_credential(). Return a response indicating authorization is needed.


# Use auth_scheme and auth_credential configured in the tool.

  tool_context.request_credential(AuthConfig(
    auth_scheme=auth_scheme,
    raw_auth_credential=auth_credential,
  ))
  return {'pending': true, 'message': 'Awaiting user authentication.'}

# By setting request_credential, ADK detects a pending authentication event. It pauses execution and ask end user to login.
Step 4: Exchange Authorization Code for Tokens

ADK automatically generates oauth authorization URL and presents it to your Agent Client application. your Agent Client application should follow the same way described in Journey 1 to redirect the user to the authorization URL (with redirect_uri appended). Once a user completes the login flow following the authorization URL and ADK extracts the authentication callback url from Agent Client applications, automatically parses the auth code, and generates auth token. At the next Tool call, tool_context.get_auth_response in step 2 will contain a valid credential to use in subsequent API calls.

Step 5: Cache Obtained Credentials

After successfully obtaining the token from ADK (Step 2) or if the token is still valid (Step 1), immediately store the new Credentials object in tool_context.state (serialized, e.g., as JSON) using your cache key.


# Inside your tool function, after obtaining 'creds' (either refreshed or newly exchanged)
# Cache the new/refreshed tokens
tool_context.state[TOKEN_CACHE_KEY] = json.loads(creds.to_json())
print(f"DEBUG: Cached/updated tokens under key: {TOKEN_CACHE_KEY}")
# Proceed to Step 6 (Make API Call)
Step 6: Make Authenticated API Call

Once you have a valid Credentials object (creds from Step 1 or Step 4), use it to make the actual call to the protected API using the appropriate client library (e.g., googleapiclient, requests). Pass the credentials=creds argument.
Include error handling, especially for HttpError 401/403, which might mean the token expired or was revoked between calls. If you get such an error, consider clearing the cached token (tool_context.state.pop(...)) and potentially returning the auth_required status again to force re-authentication.

# Inside your tool function, using the valid 'creds' object
# Ensure creds is valid before proceeding
if not creds or not creds.valid:
   return {"status": "error", "error_message": "Cannot proceed without valid credentials."}

try:
   service = build("calendar", "v3", credentials=creds) # Example
   api_result = service.events().list(...).execute()
   # Proceed to Step 7
except Exception as e:
   # Handle API errors (e.g., check for 401/403, maybe clear cache and re-request auth)
   print(f"ERROR: API call failed: {e}")
   return {"status": "error", "error_message": f"API call failed: {e}"}
Step 7: Return Tool Result

After a successful API call, process the result into a dictionary format that is useful for the LLM.
Crucially, include a along with the data.

# Inside your tool function, after successful API call
    processed_result = [...] # Process api_result for the LLM
    return {"status": "success", "data": processed_result}
Full Code

Tools and Agent
Agent CLI
Helper
Spec
tools_and_agent.py

import os

from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents.llm_agent import LlmAgent

# --- Authentication Configuration ---
# This section configures how the agent will handle authentication using OpenID Connect (OIDC),
# often layered on top of OAuth 2.0.

# Define the Authentication Scheme using OpenID Connect.
# This object tells the ADK *how* to perform the OIDC/OAuth2 flow.
# It requires details specific to your Identity Provider (IDP), like Google OAuth, Okta, Auth0, etc.
# Note: Replace the example Okta URLs and credentials with your actual IDP details.
# All following fields are required, and available from your IDP.
auth_scheme = OpenIdConnectWithConfig(
    # The URL of the IDP's authorization endpoint where the user is redirected to log in.
    authorization_endpoint="https://your-endpoint.okta.com/oauth2/v1/authorize",
    # The URL of the IDP's token endpoint where the authorization code is exchanged for tokens.
    token_endpoint="https://your-token-endpoint.okta.com/oauth2/v1/token",
    # The scopes (permissions) your application requests from the IDP.
    # 'openid' is standard for OIDC. 'profile' and 'email' request user profile info.
    scopes=['openid', 'profile', "email"]
)

# Define the Authentication Credentials for your specific application.
# This object holds the client identifier and secret that your application uses
# to identify itself to the IDP during the OAuth2 flow.
# !! SECURITY WARNING: Avoid hardcoding secrets in production code. !!
# !! Use environment variables or a secret management system instead. !!
auth_credential = AuthCredential(
  auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
  oauth2=OAuth2Auth(
    client_id="CLIENT_ID",
    client_secret="CIENT_SECRET",
  )
)


# --- Toolset Configuration from OpenAPI Specification ---
# This section defines a sample set of tools the agent can use, configured with Authentication
# from steps above.
# This sample set of tools use endpoints protected by Okta and requires an OpenID Connect flow
# to acquire end user credentials.
with open(os.path.join(os.path.dirname(__file__), 'spec.yaml'), 'r') as f:
    spec_content = f.read()

userinfo_toolset = OpenAPIToolset(
   spec_str=spec_content,
   spec_str_type='yaml',
   # ** Crucially, associate the authentication scheme and credentials with these tools. **
   # This tells the ADK that the tools require the defined OIDC/OAuth2 flow.
   auth_scheme=auth_scheme,
   auth_credential=auth_credential,
)

# --- Agent Configuration ---
# Configure and create the main LLM Agent.
root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='enterprise_assistant',
    instruction='Help user integrate with multiple enterprise systems, including retrieving user information which may require authentication.',
    tools=userinfo_toolset.get_tools(),
)

# --- Ready for Use ---
# The `root_agent` is now configured with tools protected by OIDC/OAuth2 authentication.
# When the agent attempts to use one of these tools, the ADK framework will automatically
# trigger the authentication flow defined by `auth_scheme` and `auth_credential`
# if valid credentials are not already available in the session.
# The subsequent interaction flow would guide the user through the login process and handle
# token exchanging, and automatically attach the exchanged token to the endpoint defined in
# the tool.

Runtime¶
What is runtime?¶
The ADK Runtime is the underlying engine that powers your agent application during user interactions. It's the system that takes your defined agents, tools, and callbacks and orchestrates their execution in response to user input, managing the flow of information, state changes, and interactions with external services like LLMs or storage.

Think of the Runtime as the "engine" of your agentic application. You define the parts (agents, tools), and the Runtime handles how they connect and run together to fulfill a user's request.

Core Idea: The Event Loop¶
At its heart, the ADK Runtime operates on an Event Loop. This loop facilitates a back-and-forth communication between the Runner component and your defined "Execution Logic" (which includes your Agents, the LLM calls they make, Callbacks, and Tools).

intro_components.png

In simple terms:

The Runner receives a user query and asks the main Agent to start processing.
The Agent (and its associated logic) runs until it has something to report (like a response, a request to use a tool, or a state change) – it then yields or emits an Event.
The Runner receives this Event, processes any associated actions (like saving state changes via Services), and forwards the event onwards (e.g., to the user interface).
Only after the Runner has processed the event does the Agent's logic resume from where it paused, now potentially seeing the effects of the changes committed by the Runner.
This cycle repeats until the agent has no more events to yield for the current user query.
This event-driven loop is the fundamental pattern governing how ADK executes your agent code.

The Heartbeat: The Event Loop - Inner workings¶
The Event Loop is the core operational pattern defining the interaction between the Runner and your custom code (Agents, Tools, Callbacks, collectively referred to as "Execution Logic" or "Logic Components" in the design document). It establishes a clear division of responsibilities:

Note

The specific method names and parameter names may vary slightly by SDK language (e.g., agent_to_run.runAsync(...) in Java, agent_to_run.run_async(...) in Python). Refer to the language-specific API documentation for details.

Runner's Role (Orchestrator)¶
The Runner acts as the central coordinator for a single user invocation. Its responsibilities in the loop are:

Initiation: Receives the end user's query (new_message) and typically appends it to the session history via the SessionService.
Kick-off: Starts the event generation process by calling the main agent's execution method (e.g., agent_to_run.run_async(...)).
Receive & Process: Waits for the agent logic to yield or emit an Event. Upon receiving an event, the Runner promptly processes it. This involves:
Using configured Services (SessionService, ArtifactService, MemoryService) to commit changes indicated in event.actions (like state_delta, artifact_delta).
Performing other internal bookkeeping.
Yield Upstream: Forwards the processed event onwards (e.g., to the calling application or UI for rendering).
Iterate: Signals the agent logic that processing is complete for the yielded event, allowing it to resume and generate the next event.
Conceptual Runner Loop:


Python
Java

# Simplified view of Runner's main loop logic
def run(new_query, ...) -> Generator[Event]:
    # 1. Append new_query to session event history (via SessionService)
    session_service.append_event(session, Event(author='user', content=new_query))

    # 2. Kick off event loop by calling the agent
    agent_event_generator = agent_to_run.run_async(context)

    async for event in agent_event_generator:
        # 3. Process the generated event and commit changes
        session_service.append_event(session, event) # Commits state/artifact deltas etc.
        # memory_service.update_memory(...) # If applicable
        # artifact_service might have already been called via context during agent run

        # 4. Yield event for upstream processing (e.g., UI rendering)
        yield event
        # Runner implicitly signals agent generator can continue after yielding

Execution Logic's Role (Agent, Tool, Callback)¶
Your code within agents, tools, and callbacks is responsible for the actual computation and decision-making. Its interaction with the loop involves:

Execute: Runs its logic based on the current InvocationContext, including the session state as it was when execution resumed.
Yield: When the logic needs to communicate (send a message, call a tool, report a state change), it constructs an Event containing the relevant content and actions, and then yields this event back to the Runner.
Pause: Crucially, execution of the agent logic pauses immediately after the yield statement (or return in RxJava). It waits for the Runner to complete step 3 (processing and committing).
Resume: Only after the Runner has processed the yielded event does the agent logic resume execution from the statement immediately following the yield.
See Updated State: Upon resumption, the agent logic can now reliably access the session state (ctx.session.state) reflecting the changes that were committed by the Runner from the previously yielded event.
Conceptual Execution Logic:


Python
Java

# Simplified view of logic inside Agent.run_async, callbacks, or tools

# ... previous code runs based on current state ...

# 1. Determine a change or output is needed, construct the event
# Example: Updating state
update_data = {'field_1': 'value_2'}
event_with_state_change = Event(
    author=self.name,
    actions=EventActions(state_delta=update_data),
    content=types.Content(parts=[types.Part(text="State updated.")])
    # ... other event fields ...
)

# 2. Yield the event to the Runner for processing & commit
yield event_with_state_change
# <<<<<<<<<<<< EXECUTION PAUSES HERE >>>>>>>>>>>>

# <<<<<<<<<<<< RUNNER PROCESSES & COMMITS THE EVENT >>>>>>>>>>>>

# 3. Resume execution ONLY after Runner is done processing the above event.
# Now, the state committed by the Runner is reliably reflected.
# Subsequent code can safely assume the change from the yielded event happened.
val = ctx.session.state['field_1']
# here `val` is guaranteed to be "value_2" (assuming Runner committed successfully)
print(f"Resumed execution. Value of field_1 is now: {val}")

# ... subsequent code continues ...
# Maybe yield another event later...

This cooperative yield/pause/resume cycle between the Runner and your Execution Logic, mediated by Event objects, forms the core of the ADK Runtime.

Key components of the Runtime¶
Several components work together within the ADK Runtime to execute an agent invocation. Understanding their roles clarifies how the event loop functions:

Runner¶
Role: The main entry point and orchestrator for a single user query (run_async).
Function: Manages the overall Event Loop, receives events yielded by the Execution Logic, coordinates with Services to process and commit event actions (state/artifact changes), and forwards processed events upstream (e.g., to the UI). It essentially drives the conversation turn by turn based on yielded events. (Defined in google.adk.runners.runner).
Execution Logic Components¶
Role: The parts containing your custom code and the core agent capabilities.
Components:
Agent (BaseAgent, LlmAgent, etc.): Your primary logic units that process information and decide on actions. They implement the _run_async_impl method which yields events.
Tools (BaseTool, FunctionTool, AgentTool, etc.): External functions or capabilities used by agents (often LlmAgent) to interact with the outside world or perform specific tasks. They execute and return results, which are then wrapped in events.
Callbacks (Functions): User-defined functions attached to agents (e.g., before_agent_callback, after_model_callback) that hook into specific points in the execution flow, potentially modifying behavior or state, whose effects are captured in events.
Function: Perform the actual thinking, calculation, or external interaction. They communicate their results or needs by yielding Event objects and pausing until the Runner processes them.
Event¶
Role: The message passed back and forth between the Runner and the Execution Logic.
Function: Represents an atomic occurrence (user input, agent text, tool call/result, state change request, control signal). It carries both the content of the occurrence and the intended side effects (actions like state_delta).
Services¶
Role: Backend components responsible for managing persistent or shared resources. Used primarily by the Runner during event processing.
Components:
SessionService (BaseSessionService, InMemorySessionService, etc.): Manages Session objects, including saving/loading them, applying state_delta to the session state, and appending events to the event history.
ArtifactService (BaseArtifactService, InMemoryArtifactService, GcsArtifactService, etc.): Manages the storage and retrieval of binary artifact data. Although save_artifact is called via context during execution logic, the artifact_delta in the event confirms the action for the Runner/SessionService.
MemoryService (BaseMemoryService, etc.): (Optional) Manages long-term semantic memory across sessions for a user.
Function: Provide the persistence layer. The Runner interacts with them to ensure changes signaled by event.actions are reliably stored before the Execution Logic resumes.
Session¶
Role: A data container holding the state and history for one specific conversation between a user and the application.
Function: Stores the current state dictionary, the list of all past events (event history), and references to associated artifacts. It's the primary record of the interaction, managed by the SessionService.
Invocation¶
Role: A conceptual term representing everything that happens in response to a single user query, from the moment the Runner receives it until the agent logic finishes yielding events for that query.
Function: An invocation might involve multiple agent runs (if using agent transfer or AgentTool), multiple LLM calls, tool executions, and callback executions, all tied together by a single invocation_id within the InvocationContext.
These players interact continuously through the Event Loop to process a user's request.

How It Works: A Simplified Invocation¶
Let's trace a simplified flow for a typical user query that involves an LLM agent calling a tool:

intro_components.png

Step-by-Step Breakdown¶
User Input: The User sends a query (e.g., "What's the capital of France?").
Runner Starts: Runner.run_async begins. It interacts with the SessionService to load the relevant Session and adds the user query as the first Event to the session history. An InvocationContext (ctx) is prepared.
Agent Execution: The Runner calls agent.run_async(ctx) on the designated root agent (e.g., an LlmAgent).
LLM Call (Example): The Agent_Llm determines it needs information, perhaps by calling a tool. It prepares a request for the LLM. Let's assume the LLM decides to call MyTool.
Yield FunctionCall Event: The Agent_Llm receives the FunctionCall response from the LLM, wraps it in an Event(author='Agent_Llm', content=Content(parts=[Part(function_call=...)])), and yields or emits this event.
Agent Pauses: The Agent_Llm's execution pauses immediately after the yield.
Runner Processes: The Runner receives the FunctionCall event. It passes it to the SessionService to record it in the history. The Runner then yields the event upstream to the User (or application).
Agent Resumes: The Runner signals that the event is processed, and Agent_Llm resumes execution.
Tool Execution: The Agent_Llm's internal flow now proceeds to execute the requested MyTool. It calls tool.run_async(...).
Tool Returns Result: MyTool executes and returns its result (e.g., {'result': 'Paris'}).
Yield FunctionResponse Event: The agent (Agent_Llm) wraps the tool result into an Event containing a FunctionResponse part (e.g., Event(author='Agent_Llm', content=Content(role='user', parts=[Part(function_response=...)]))). This event might also contain actions if the tool modified state (state_delta) or saved artifacts (artifact_delta). The agent yields this event.
Agent Pauses: Agent_Llm pauses again.
Runner Processes: Runner receives the FunctionResponse event. It passes it to SessionService which applies any state_delta/artifact_delta and adds the event to history. Runner yields the event upstream.
Agent Resumes: Agent_Llm resumes, now knowing the tool result and any state changes are committed.
Final LLM Call (Example): Agent_Llm sends the tool result back to the LLM to generate a natural language response.
Yield Final Text Event: Agent_Llm receives the final text from the LLM, wraps it in an Event(author='Agent_Llm', content=Content(parts=[Part(text=...)])), and yields it.
Agent Pauses: Agent_Llm pauses.
Runner Processes: Runner receives the final text event, passes it to SessionService for history, and yields it upstream to the User. This is likely marked as the is_final_response().
Agent Resumes & Finishes: Agent_Llm resumes. Having completed its task for this invocation, its run_async generator finishes.
Runner Completes: The Runner sees the agent's generator is exhausted and finishes its loop for this invocation.
This yield/pause/process/resume cycle ensures that state changes are consistently applied and that the execution logic always operates on the most recently committed state after yielding an event.

Important Runtime Behaviors¶
Understanding a few key aspects of how the ADK Runtime handles state, streaming, and asynchronous operations is crucial for building predictable and efficient agents.

State Updates & Commitment Timing¶
The Rule: When your code (in an agent, tool, or callback) modifies the session state (e.g., context.state['my_key'] = 'new_value'), this change is initially recorded locally within the current InvocationContext. The change is only guaranteed to be persisted (saved by the SessionService) after the Event carrying the corresponding state_delta in its actions has been yield-ed by your code and subsequently processed by the Runner.

Implication: Code that runs after resuming from a yield can reliably assume that the state changes signaled in the yielded event have been committed.


Python
Java

# Inside agent logic (conceptual)

# 1. Modify state
ctx.session.state['status'] = 'processing'
event1 = Event(..., actions=EventActions(state_delta={'status': 'processing'}))

# 2. Yield event with the delta
yield event1
# --- PAUSE --- Runner processes event1, SessionService commits 'status' = 'processing' ---

# 3. Resume execution
# Now it's safe to rely on the committed state
current_status = ctx.session.state['status'] # Guaranteed to be 'processing'
print(f"Status after resuming: {current_status}")

"Dirty Reads" of Session State¶
Definition: While commitment happens after the yield, code running later within the same invocation, but before the state-changing event is actually yielded and processed, can often see the local, uncommitted changes. This is sometimes called a "dirty read".
Example:

Python
Java

# Code in before_agent_callback
callback_context.state['field_1'] = 'value_1'
# State is locally set to 'value_1', but not yet committed by Runner

# ... agent runs ...

# Code in a tool called later *within the same invocation*
# Readable (dirty read), but 'value_1' isn't guaranteed persistent yet.
val = tool_context.state['field_1'] # 'val' will likely be 'value_1' here
print(f"Dirty read value in tool: {val}")

# Assume the event carrying the state_delta={'field_1': 'value_1'}
# is yielded *after* this tool runs and is processed by the Runner.

Implications:
Benefit: Allows different parts of your logic within a single complex step (e.g., multiple callbacks or tool calls before the next LLM turn) to coordinate using state without waiting for a full yield/commit cycle.
Caveat: Relying heavily on dirty reads for critical logic can be risky. If the invocation fails before the event carrying the state_delta is yielded and processed by the Runner, the uncommitted state change will be lost. For critical state transitions, ensure they are associated with an event that gets successfully processed.
Streaming vs. Non-Streaming Output (partial=True)¶
This primarily relates to how responses from the LLM are handled, especially when using streaming generation APIs.

Streaming: The LLM generates its response token-by-token or in small chunks.
The framework (often within BaseLlmFlow) yields multiple Event objects for a single conceptual response. Most of these events will have partial=True.
The Runner, upon receiving an event with partial=True, typically forwards it immediately upstream (for UI display) but skips processing its actions (like state_delta).
Eventually, the framework yields a final event for that response, marked as non-partial (partial=False or implicitly via turn_complete=True).
The Runner fully processes only this final event, committing any associated state_delta or artifact_delta.
Non-Streaming: The LLM generates the entire response at once. The framework yields a single event marked as non-partial, which the Runner processes fully.
Why it Matters: Ensures that state changes are applied atomically and only once based on the complete response from the LLM, while still allowing the UI to display text progressively as it's generated.
Async is Primary (run_async)¶
Core Design: The ADK Runtime is fundamentally built on asynchronous libraries (like Python's asyncio and Java's RxJava) to handle concurrent operations (like waiting for LLM responses or tool executions) efficiently without blocking.
Main Entry Point: Runner.run_async is the primary method for executing agent invocations. All core runnable components (Agents, specific flows) use asynchronous methods internally.
Synchronous Convenience (run): A synchronous Runner.run method exists mainly for convenience (e.g., in simple scripts or testing environments). However, internally, Runner.run typically just calls Runner.run_async and manages the async event loop execution for you.
Developer Experience: We recommend designing your applications (e.g., web servers using ADK) to be asynchronous for best performance. In Python, this means using asyncio; in Java, leverage RxJava's reactive programming model.
Sync Callbacks/Tools: The ADK framework supports both asynchronous and synchronous functions for tools and callbacks.
Blocking I/O: For long-running synchronous I/O operations, the framework attempts to prevent stalls. Python ADK may use asyncio.to_thread, while Java ADK often relies on appropriate RxJava schedulers or wrappers for blocking calls.
CPU-Bound Work: Purely CPU-intensive synchronous tasks will still block their execution thread in both environments.
Understanding these behaviors helps you write more robust ADK applications and debug issues related to state consistency, streaming updates, and asynchronous execution.

State: The Session's Scratchpad¶
Within each Session (our conversation thread), the state attribute acts like the agent's dedicated scratchpad for that specific interaction. While session.events holds the full history, session.state is where the agent stores and updates dynamic details needed during the conversation.

What is session.state?¶
Conceptually, session.state is a collection (dictionary or Map) holding key-value pairs. It's designed for information the agent needs to recall or track to make the current conversation effective:

Personalize Interaction: Remember user preferences mentioned earlier (e.g., 'user_preference_theme': 'dark').
Track Task Progress: Keep tabs on steps in a multi-turn process (e.g., 'booking_step': 'confirm_payment').
Accumulate Information: Build lists or summaries (e.g., 'shopping_cart_items': ['book', 'pen']).
Make Informed Decisions: Store flags or values influencing the next response (e.g., 'user_is_authenticated': True).
Key Characteristics of State¶
Structure: Serializable Key-Value Pairs

Data is stored as key: value.
Keys: Always strings (str). Use clear names (e.g., 'departure_city', 'user:language_preference').
Values: Must be serializable. This means they can be easily saved and loaded by the SessionService. Stick to basic types in the specific languages (Python/ Java) like strings, numbers, booleans, and simple lists or dictionaries containing only these basic types. (See API documentation for precise details).
⚠️ Avoid Complex Objects: Do not store non-serializable objects (custom class instances, functions, connections, etc.) directly in the state. Store simple identifiers if needed, and retrieve the complex object elsewhere.
Mutability: It Changes

The contents of the state are expected to change as the conversation evolves.
Persistence: Depends on SessionService

Whether state survives application restarts depends on your chosen service:
InMemorySessionService: Not Persistent. State is lost on restart.
DatabaseSessionService / VertexAiSessionService: Persistent. State is saved reliably.
Note

The specific parameters or method names for the primitives may vary slightly by SDK language (e.g., session.state['current_intent'] = 'book_flight' in Python, session.state().put("current_intent", "book_flight) in Java). Refer to the language-specific API documentation for details.

Organizing State with Prefixes: Scope Matters¶
Prefixes on state keys define their scope and persistence behavior, especially with persistent services:

No Prefix (Session State):

Scope: Specific to the current session (id).
Persistence: Only persists if the SessionService is persistent (Database, VertexAI).
Use Cases: Tracking progress within the current task (e.g., 'current_booking_step'), temporary flags for this interaction (e.g., 'needs_clarification').
Example: session.state['current_intent'] = 'book_flight'
user: Prefix (User State):

Scope: Tied to the user_id, shared across all sessions for that user (within the same app_name).
Persistence: Persistent with Database or VertexAI. (Stored by InMemory but lost on restart).
Use Cases: User preferences (e.g., 'user:theme'), profile details (e.g., 'user:name').
Example: session.state['user:preferred_language'] = 'fr'
app: Prefix (App State):

Scope: Tied to the app_name, shared across all users and sessions for that application.
Persistence: Persistent with Database or VertexAI. (Stored by InMemory but lost on restart).
Use Cases: Global settings (e.g., 'app:api_endpoint'), shared templates.
Example: session.state['app:global_discount_code'] = 'SAVE10'
temp: Prefix (Temporary Session State):

Scope: Specific to the current session processing turn.
Persistence: Never Persistent. Guaranteed to be discarded, even with persistent services.
Use Cases: Intermediate results needed only immediately, data you explicitly don't want stored.
Example: session.state['temp:raw_api_response'] = {...}
How the Agent Sees It: Your agent code interacts with the combined state through the single session.state collection (dict/ Map). The SessionService handles fetching/merging state from the correct underlying storage based on prefixes.

How State is Updated: Recommended Methods¶
State should always be updated as part of adding an Event to the session history using session_service.append_event(). This ensures changes are tracked, persistence works correctly, and updates are thread-safe.

1. The Easy Way: output_key (for Agent Text Responses)

This is the simplest method for saving an agent's final text response directly into the state. When defining your LlmAgent, specify the output_key:


Python
Java

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai.types import Content, Part

# Define agent with output_key
greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash", # Use a valid model
    instruction="Generate a short, friendly greeting.",
    output_key="last_greeting" # Save response to state['last_greeting']
)

# --- Setup Runner and Session ---
app_name, user_id, session_id = "state_app", "user1", "session1"
session_service = InMemorySessionService()
runner = Runner(
    agent=greeting_agent,
    app_name=app_name,
    session_service=session_service
)
session = await session_service.create_session(app_name=app_name, 
                                    user_id=user_id, 
                                    session_id=session_id)
print(f"Initial state: {session.state}")

# --- Run the Agent ---
# Runner handles calling append_event, which uses the output_key
# to automatically create the state_delta.
user_message = Content(parts=[Part(text="Hello")])
for event in runner.run(user_id=user_id, 
                        session_id=session_id, 
                        new_message=user_message):
    if event.is_final_response():
      print(f"Agent responded.") # Response text is also in event.content

# --- Check Updated State ---
updated_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
print(f"State after agent run: {updated_session.state}")
# Expected output might include: {'last_greeting': 'Hello there! How can I help you today?'}

Behind the scenes, the Runner uses the output_key to create the necessary EventActions with a state_delta and calls append_event.

2. The Standard Way: EventActions.state_delta (for Complex Updates)

For more complex scenarios (updating multiple keys, non-string values, specific scopes like user: or app:, or updates not tied directly to the agent's final text), you manually construct the state_delta within EventActions.


Python
Java

from google.adk.sessions import InMemorySessionService, Session
from google.adk.events import Event, EventActions
from google.genai.types import Part, Content
import time

# --- Setup ---
session_service = InMemorySessionService()
app_name, user_id, session_id = "state_app_manual", "user2", "session2"
session = await session_service.create_session(
    app_name=app_name,
    user_id=user_id,
    session_id=session_id,
    state={"user:login_count": 0, "task_status": "idle"}
)
print(f"Initial state: {session.state}")

# --- Define State Changes ---
current_time = time.time()
state_changes = {
    "task_status": "active",              # Update session state
    "user:login_count": session.state.get("user:login_count", 0) + 1, # Update user state
    "user:last_login_ts": current_time,   # Add user state
    "temp:validation_needed": True        # Add temporary state (will be discarded)
}

# --- Create Event with Actions ---
actions_with_update = EventActions(state_delta=state_changes)
# This event might represent an internal system action, not just an agent response
system_event = Event(
    invocation_id="inv_login_update",
    author="system", # Or 'agent', 'tool' etc.
    actions=actions_with_update,
    timestamp=current_time
    # content might be None or represent the action taken
)

# --- Append the Event (This updates the state) ---
await session_service.append_event(session, system_event)
print("`append_event` called with explicit state delta.")

# --- Check Updated State ---
updated_session = await session_service.get_session(app_name=app_name,
                                            user_id=user_id, 
                                            session_id=session_id)
print(f"State after event: {updated_session.state}")
# Expected: {'user:login_count': 1, 'task_status': 'active', 'user:last_login_ts': <timestamp>}
# Note: 'temp:validation_needed' is NOT present.

3. Via CallbackContext or ToolContext (Recommended for Callbacks and Tools)

Modifying state within agent callbacks (e.g., on_before_agent_call, on_after_agent_call) or tool functions is best done using the state attribute of the CallbackContext or ToolContext provided to your function.

callback_context.state['my_key'] = my_value
tool_context.state['my_key'] = my_value
These context objects are specifically designed to manage state changes within their respective execution scopes. When you modify context.state, the ADK framework ensures that these changes are automatically captured and correctly routed into the EventActions.state_delta for the event being generated by the callback or tool. This delta is then processed by the SessionService when the event is appended, ensuring proper persistence and tracking.

This method abstracts away the manual creation of EventActions and state_delta for most common state update scenarios within callbacks and tools, making your code cleaner and less error-prone.

For more comprehensive details on context objects, refer to the Context documentation.


Python
Java

# In an agent callback or tool function
from google.adk.agents import CallbackContext # or ToolContext

def my_callback_or_tool_function(context: CallbackContext, # Or ToolContext
                                 # ... other parameters ...
                                ):
    # Update existing state
    count = context.state.get("user_action_count", 0)
    context.state["user_action_count"] = count + 1

    # Add new state
    context.state["temp:last_operation_status"] = "success"

    # State changes are automatically part of the event's state_delta
    # ... rest of callback/tool logic ...

What append_event Does:

Adds the Event to session.events.
Reads the state_delta from the event's actions.
Applies these changes to the state managed by the SessionService, correctly handling prefixes and persistence based on the service type.
Updates the session's last_update_time.
Ensures thread-safety for concurrent updates.
⚠️ A Warning About Direct State Modification¶
Avoid directly modifying the session.state collection (dictionary/Map) on a Session object that was obtained directly from the SessionService (e.g., via session_service.get_session() or session_service.create_session()) outside of the managed lifecycle of an agent invocation (i.e., not through a CallbackContext or ToolContext). For example, code like retrieved_session = await session_service.get_session(...); retrieved_session.state['key'] = value is problematic.

State modifications within callbacks or tools using CallbackContext.state or ToolContext.state are the correct way to ensure changes are tracked, as these context objects handle the necessary integration with the event system.

Why direct modification (outside of contexts) is strongly discouraged:

Bypasses Event History: The change isn't recorded as an Event, losing auditability.
Breaks Persistence: Changes made this way will likely NOT be saved by DatabaseSessionService or VertexAiSessionService. They rely on append_event to trigger saving.
Not Thread-Safe: Can lead to race conditions and lost updates.
Ignores Timestamps/Logic: Doesn't update last_update_time or trigger related event logic.
Recommendation: Stick to updating state via output_key, EventActions.state_delta (when manually creating events), or by modifying the state property of CallbackContext or ToolContext objects when within their respective scopes. These methods ensure reliable, trackable, and persistent state management. Use direct access to session.state (from a SessionService-retrieved session) only for reading state.

Best Practices for State Design Recap¶
Minimalism: Store only essential, dynamic data.
Serialization: Use basic, serializable types.
Descriptive Keys & Prefixes: Use clear names and appropriate prefixes (user:, app:, temp:, or none).
Shallow Structures: Avoid deep nesting where possible.
Standard Update Flow: Rely on append_event.

Types of Callbacks¶
The framework provides different types of callbacks that trigger at various stages of an agent's execution. Understanding when each callback fires and what context it receives is key to using them effectively.

Agent Lifecycle Callbacks¶
These callbacks are available on any agent that inherits from BaseAgent (including LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, etc).

Note

The specific method names or return types may vary slightly by SDK language (e.g., return None in Python, return Optional.empty() or Maybe.empty() in Java). Refer to the language-specific API documentation for details.

Before Agent Callback¶
When: Called immediately before the agent's _run_async_impl (or _run_live_impl) method is executed. It runs after the agent's InvocationContext is created but before its core logic begins.

Purpose: Ideal for setting up resources or state needed only for this specific agent's run, performing validation checks on the session state (callback_context.state) before execution starts, logging the entry point of the agent's activity, or potentially modifying the invocation context before the core logic uses it.

Code

Python
Java

# # --- Setup Instructions ---
# # 1. Install the ADK package:
# !pip install google-adk
# # Make sure to restart kernel if using colab/jupyter notebooks

# # 2. Set up your Gemini API Key:
# #    - Get a key from Google AI Studio: https://aistudio.google.com/app/apikey
# #    - Set it as an environment variable:
# import os
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE" # <--- REPLACE with your actual key
# # Or learn about other authentication methods (like Vertex AI):
# # https://google.github.io/adk-docs/agents/models/

# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import InMemoryRunner # Use InMemoryRunner
from google.genai import types # For types.Content
from typing import Optional

# Define the model - Use the specific model name requested
GEMINI_2_FLASH="gemini-2.0-flash"

# --- 1. Define the Callback Function ---
def check_if_agent_should_run(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Logs entry and checks 'skip_llm_agent' in session state.
    If True, returns Content to skip the agent's execution.
    If False or not present, returns None to allow execution.
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    current_state = callback_context.state.to_dict()

    print(f"\n[Callback] Entering agent: {agent_name} (Inv: {invocation_id})")
    print(f"[Callback] Current State: {current_state}")

    # Check the condition in session state dictionary
    if current_state.get("skip_llm_agent", False):
        print(f"[Callback] State condition 'skip_llm_agent=True' met: Skipping agent {agent_name}.")
        # Return Content to skip the agent's run
        return types.Content(
            parts=[types.Part(text=f"Agent {agent_name} skipped by before_agent_callback due to state.")],
            role="model" # Assign model role to the overriding response
        )
    else:
        print(f"[Callback] State condition not met: Proceeding with agent {agent_name}.")
        # Return None to allow the LlmAgent's normal execution
        return None

# --- 2. Setup Agent with Callback ---
llm_agent_with_before_cb = LlmAgent(
    name="MyControlledAgent",
    model=GEMINI_2_FLASH,
    instruction="You are a concise assistant.",
    description="An LLM agent demonstrating stateful before_agent_callback",
    before_agent_callback=check_if_agent_should_run # Assign the callback
)

# --- 3. Setup Runner and Sessions using InMemoryRunner ---
async def main():
    app_name = "before_agent_demo"
    user_id = "test_user"
    session_id_run = "session_will_run"
    session_id_skip = "session_will_skip"

    # Use InMemoryRunner - it includes InMemorySessionService
    runner = InMemoryRunner(agent=llm_agent_with_before_cb, app_name=app_name)
    # Get the bundled session service to create sessions
    session_service = runner.session_service

    # Create session 1: Agent will run (default empty state)
    session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id_run
        # No initial state means 'skip_llm_agent' will be False in the callback check
    )

    # Create session 2: Agent will be skipped (state has skip_llm_agent=True)
    session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id_skip,
        state={"skip_llm_agent": True} # Set the state flag here
    )

    # --- Scenario 1: Run where callback allows agent execution ---
    print("\n" + "="*20 + f" SCENARIO 1: Running Agent on Session '{session_id_run}' (Should Proceed) " + "="*20)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id_run,
        new_message=types.Content(role="user", parts=[types.Part(text="Hello, please respond.")])
    ):
        # Print final output (either from LLM or callback override)
        if event.is_final_response() and event.content:
            print(f"Final Output: [{event.author}] {event.content.parts[0].text.strip()}")
        elif event.is_error():
             print(f"Error Event: {event.error_details}")

    # --- Scenario 2: Run where callback intercepts and skips agent ---
    print("\n" + "="*20 + f" SCENARIO 2: Running Agent on Session '{session_id_skip}' (Should Skip) " + "="*20)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id_skip,
        new_message=types.Content(role="user", parts=[types.Part(text="This message won't reach the LLM.")])
    ):
         # Print final output (either from LLM or callback override)
         if event.is_final_response() and event.content:
            print(f"Final Output: [{event.author}] {event.content.parts[0].text.strip()}")
         elif event.is_error():
             print(f"Error Event: {event.error_details}")

# --- 4. Execute ---
# In a Python script:
# import asyncio
# if __name__ == "__main__":
#     # Make sure GOOGLE_API_KEY environment variable is set if not using Vertex AI auth
#     # Or ensure Application Default Credentials (ADC) are configured for Vertex AI
#     asyncio.run(main())

# In a Jupyter Notebook or similar environment:
await main()

Note on the before_agent_callback Example:

What it Shows: This example demonstrates the before_agent_callback. This callback runs right before the agent's main processing logic starts for a given request.
How it Works: The callback function (check_if_agent_should_run) looks at a flag (skip_llm_agent) in the session's state.
If the flag is True, the callback returns a types.Content object. This tells the ADK framework to skip the agent's main execution entirely and use the callback's returned content as the final response.
If the flag is False (or not set), the callback returns None or an empty object. This tells the ADK framework to proceed with the agent's normal execution (calling the LLM in this case).
Expected Outcome: You'll see two scenarios:
In the session with the skip_llm_agent: True state, the agent's LLM call is bypassed, and the output comes directly from the callback ("Agent... skipped...").
In the session without that state flag, the callback allows the agent to run, and you see the actual response from the LLM (e.g., "Hello!").
Understanding Callbacks: This highlights how before_ callbacks act as gatekeepers, allowing you to intercept execution before a major step and potentially prevent it based on checks (like state, input validation, permissions).
After Agent Callback¶
When: Called immediately after the agent's _run_async_impl (or _run_live_impl) method successfully completes. It does not run if the agent was skipped due to before_agent_callback returning content or if end_invocation was set during the agent's run.

Purpose: Useful for cleanup tasks, post-execution validation, logging the completion of an agent's activity, modifying final state, or augmenting/replacing the agent's final output.

Code

Python
Java

# # --- Setup Instructions ---
# # 1. Install the ADK package:
# !pip install google-adk
# # Make sure to restart kernel if using colab/jupyter notebooks

# # 2. Set up your Gemini API Key:
# #    - Get a key from Google AI Studio: https://aistudio.google.com/app/apikey
# #    - Set it as an environment variable:
# import os
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE" # <--- REPLACE with your actual key
# # Or learn about other authentication methods (like Vertex AI):
# # https://google.github.io/adk-docs/agents/models/


# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import InMemoryRunner # Use InMemoryRunner
from google.genai import types # For types.Content
from typing import Optional

# Define the model - Use the specific model name requested
GEMINI_2_FLASH="gemini-2.0-flash"

# --- 1. Define the Callback Function ---
def modify_output_after_agent(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Logs exit from an agent and checks 'add_concluding_note' in session state.
    If True, returns new Content to *replace* the agent's original output.
    If False or not present, returns None, allowing the agent's original output to be used.
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    current_state = callback_context.state.to_dict()

    print(f"\n[Callback] Exiting agent: {agent_name} (Inv: {invocation_id})")
    print(f"[Callback] Current State: {current_state}")

    # Example: Check state to decide whether to modify the final output
    if current_state.get("add_concluding_note", False):
        print(f"[Callback] State condition 'add_concluding_note=True' met: Replacing agent {agent_name}'s output.")
        # Return Content to *replace* the agent's own output
        return types.Content(
            parts=[types.Part(text=f"Concluding note added by after_agent_callback, replacing original output.")],
            role="model" # Assign model role to the overriding response
        )
    else:
        print(f"[Callback] State condition not met: Using agent {agent_name}'s original output.")
        # Return None - the agent's output produced just before this callback will be used.
        return None

# --- 2. Setup Agent with Callback ---
llm_agent_with_after_cb = LlmAgent(
    name="MySimpleAgentWithAfter",
    model=GEMINI_2_FLASH,
    instruction="You are a simple agent. Just say 'Processing complete!'",
    description="An LLM agent demonstrating after_agent_callback for output modification",
    after_agent_callback=modify_output_after_agent # Assign the callback here
)

# --- 3. Setup Runner and Sessions using InMemoryRunner ---
async def main():
    app_name = "after_agent_demo"
    user_id = "test_user_after"
    session_id_normal = "session_run_normally"
    session_id_modify = "session_modify_output"

    # Use InMemoryRunner - it includes InMemorySessionService
    runner = InMemoryRunner(agent=llm_agent_with_after_cb, app_name=app_name)
    # Get the bundled session service to create sessions
    session_service = runner.session_service

    # Create session 1: Agent output will be used as is (default empty state)
    session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id_normal
        # No initial state means 'add_concluding_note' will be False in the callback check
    )
    # print(f"Session '{session_id_normal}' created with default state.")

    # Create session 2: Agent output will be replaced by the callback
    session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id_modify,
        state={"add_concluding_note": True} # Set the state flag here
    )
    # print(f"Session '{session_id_modify}' created with state={{'add_concluding_note': True}}.")


    # --- Scenario 1: Run where callback allows agent's original output ---
    print("\n" + "="*20 + f" SCENARIO 1: Running Agent on Session '{session_id_normal}' (Should Use Original Output) " + "="*20)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id_normal,
        new_message=types.Content(role="user", parts=[types.Part(text="Process this please.")])
    ):
        # Print final output (either from LLM or callback override)
        if event.is_final_response() and event.content:
            print(f"Final Output: [{event.author}] {event.content.parts[0].text.strip()}")
        elif event.is_error():
             print(f"Error Event: {event.error_details}")

    # --- Scenario 2: Run where callback replaces the agent's output ---
    print("\n" + "="*20 + f" SCENARIO 2: Running Agent on Session '{session_id_modify}' (Should Replace Output) " + "="*20)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id_modify,
        new_message=types.Content(role="user", parts=[types.Part(text="Process this and add note.")])
    ):
         # Print final output (either from LLM or callback override)
         if event.is_final_response() and event.content:
            print(f"Final Output: [{event.author}] {event.content.parts[0].text.strip()}")
         elif event.is_error():
             print(f"Error Event: {event.error_details}")

# --- 4. Execute ---
# In a Python script:
# import asyncio
# if __name__ == "__main__":
#     # Make sure GOOGLE_API_KEY environment variable is set if not using Vertex AI auth
#     # Or ensure Application Default Credentials (ADC) are configured for Vertex AI
#     asyncio.run(main())

# In a Jupyter Notebook or similar environment:
await main()

Note on the after_agent_callback Example:

What it Shows: This example demonstrates the after_agent_callback. This callback runs right after the agent's main processing logic has finished and produced its result, but before that result is finalized and returned.
How it Works: The callback function (modify_output_after_agent) checks a flag (add_concluding_note) in the session's state.
If the flag is True, the callback returns a new types.Content object. This tells the ADK framework to replace the agent's original output with the content returned by the callback.
If the flag is False (or not set), the callback returns None or an empty object. This tells the ADK framework to use the original output generated by the agent.
Expected Outcome: You'll see two scenarios:
In the session without the add_concluding_note: True state, the callback allows the agent's original output ("Processing complete!") to be used.
In the session with that state flag, the callback intercepts the agent's original output and replaces it with its own message ("Concluding note added...").
Understanding Callbacks: This highlights how after_ callbacks allow post-processing or modification. You can inspect the result of a step (the agent's run) and decide whether to let it pass through, change it, or completely replace it based on your logic.
LLM Interaction Callbacks¶
These callbacks are specific to LlmAgent and provide hooks around the interaction with the Large Language Model.

Before Model Callback¶
When: Called just before the generate_content_async (or equivalent) request is sent to the LLM within an LlmAgent's flow.

Purpose: Allows inspection and modification of the request going to the LLM. Use cases include adding dynamic instructions, injecting few-shot examples based on state, modifying model config, implementing guardrails (like profanity filters), or implementing request-level caching.

Return Value Effect:
If the callback returns None (or a Maybe.empty() object in Java), the LLM continues its normal workflow. If the callback returns an LlmResponse object, then the call to the LLM is skipped. The returned LlmResponse is used directly as if it came from the model. This is powerful for implementing guardrails or caching.

Code

Python
Java

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.runners import Runner
from typing import Optional
from google.genai import types 
from google.adk.sessions import InMemorySessionService

GEMINI_2_FLASH="gemini-2.0-flash"

# --- Define the Callback Function ---
def simple_before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    # Inspect the last user message in the request contents
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    # --- Modification Example ---
    # Add a prefix to the system instruction
    original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
    prefix = "[Modified by Callback] "
    # Ensure system_instruction is Content and parts list exists
    if not isinstance(original_instruction, types.Content):
         # Handle case where it might be a string (though config expects Content)
         original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
    if not original_instruction.parts:
        original_instruction.parts.append(types.Part(text="")) # Add an empty part if none exist

    # Modify the text of the first part
    modified_text = prefix + (original_instruction.parts[0].text or "")
    original_instruction.parts[0].text = modified_text
    llm_request.config.system_instruction = original_instruction
    print(f"[Callback] Modified system instruction to: '{modified_text}'")

    # --- Skip Example ---
    # Check if the last user message contains "BLOCK"
    if "BLOCK" in last_user_message.upper():
        print("[Callback] 'BLOCK' keyword found. Skipping LLM call.")
        # Return an LlmResponse to skip the actual LLM call
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="LLM call was blocked by before_model_callback.")],
            )
        )
    else:
        print("[Callback] Proceeding with LLM call.")
        # Return None to allow the (modified) request to go to the LLM
        return None


# Create LlmAgent and Assign Callback
my_llm_agent = LlmAgent(
        name="ModelCallbackAgent",
        model=GEMINI_2_FLASH,
        instruction="You are a helpful assistant.", # Base instruction
        description="An LLM agent demonstrating before_model_callback",
        before_model_callback=simple_before_model_modifier # Assign the function here
)

APP_NAME = "guardrail_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=my_llm_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
  content = types.Content(role='user', parts=[types.Part(text=query)])
  events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

  for event in events:
      if event.is_final_response():
          final_response = event.content.parts[0].text
          print("Agent Response: ", final_response)

call_agent("callback example")

After Model Callback¶
When: Called just after a response (LlmResponse) is received from the LLM, before it's processed further by the invoking agent.

Purpose: Allows inspection or modification of the raw LLM response. Use cases include

logging model outputs,
reformatting responses,
censoring sensitive information generated by the model,
parsing structured data from the LLM response and storing it in callback_context.state
or handling specific error codes.
Code

Python
Java

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from typing import Optional
from google.genai import types 
from google.adk.sessions import InMemorySessionService
from google.adk.models import LlmResponse

GEMINI_2_FLASH="gemini-2.0-flash"

# --- Define the Callback Function ---
def simple_after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM response after it's received."""
    agent_name = callback_context.agent_name
    print(f"[Callback] After model call for agent: {agent_name}")

    # --- Inspection ---
    original_text = ""
    if llm_response.content and llm_response.content.parts:
        # Assuming simple text response for this example
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            print(f"[Callback] Inspected original response text: '{original_text[:100]}...'") # Log snippet
        elif llm_response.content.parts[0].function_call:
             print(f"[Callback] Inspected response: Contains function call '{llm_response.content.parts[0].function_call.name}'. No text modification.")
             return None # Don't modify tool calls in this example
        else:
             print("[Callback] Inspected response: No text content found.")
             return None
    elif llm_response.error_message:
        print(f"[Callback] Inspected response: Contains error '{llm_response.error_message}'. No modification.")
        return None
    else:
        print("[Callback] Inspected response: Empty LlmResponse.")
        return None # Nothing to modify

    # --- Modification Example ---
    # Replace "joke" with "funny story" (case-insensitive)
    search_term = "joke"
    replace_term = "funny story"
    if search_term in original_text.lower():
        print(f"[Callback] Found '{search_term}'. Modifying response.")
        modified_text = original_text.replace(search_term, replace_term)
        modified_text = modified_text.replace(search_term.capitalize(), replace_term.capitalize()) # Handle capitalization

        # Create a NEW LlmResponse with the modified content
        # Deep copy parts to avoid modifying original if other callbacks exist
        modified_parts = [copy.deepcopy(part) for part in llm_response.content.parts]
        modified_parts[0].text = modified_text # Update the text in the copied part

        new_response = LlmResponse(
             content=types.Content(role="model", parts=modified_parts),
             # Copy other relevant fields if necessary, e.g., grounding_metadata
             grounding_metadata=llm_response.grounding_metadata
             )
        print(f"[Callback] Returning modified response.")
        return new_response # Return the modified response
    else:
        print(f"[Callback] '{search_term}' not found. Passing original response through.")
        # Return None to use the original llm_response
        return None


# Create LlmAgent and Assign Callback
my_llm_agent = LlmAgent(
        name="AfterModelCallbackAgent",
        model=GEMINI_2_FLASH,
        instruction="You are a helpful assistant.",
        description="An LLM agent demonstrating after_model_callback",
        after_model_callback=simple_after_model_modifier # Assign the function here
)

APP_NAME = "guardrail_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=my_llm_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
  content = types.Content(role='user', parts=[types.Part(text=query)])
  events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

  for event in events:
      if event.is_final_response():
          final_response = event.content.parts[0].text
          print("Agent Response: ", final_response)

call_agent("callback example")

Tool Execution Callbacks¶
These callbacks are also specific to LlmAgent and trigger around the execution of tools (including FunctionTool, AgentTool, etc.) that the LLM might request.

Before Tool Callback¶
When: Called just before a specific tool's run_async method is invoked, after the LLM has generated a function call for it.

Purpose: Allows inspection and modification of tool arguments, performing authorization checks before execution, logging tool usage attempts, or implementing tool-level caching.

Return Value Effect:

If the callback returns None (or a Maybe.empty() object in Java), the tool's run_async method is executed with the (potentially modified) args.
If a dictionary (or Map in Java) is returned, the tool's run_async method is skipped. The returned dictionary is used directly as the result of the tool call. This is useful for caching or overriding tool behavior.
Code

Python
Java

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from typing import Optional
from google.genai import types 
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from typing import Dict, Any


GEMINI_2_FLASH="gemini-2.0-flash"

def get_capital_city(country: str) -> str:
    """Retrieves the capital city of a given country."""
    print(f"--- Tool 'get_capital_city' executing with country: {country} ---")
    country_capitals = {
        "united states": "Washington, D.C.",
        "canada": "Ottawa",
        "france": "Paris",
        "germany": "Berlin",
    }
    return country_capitals.get(country.lower(), f"Capital not found for {country}")

capital_tool = FunctionTool(func=get_capital_city)

def simple_before_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """Inspects/modifies tool args or skips the tool call."""
    agent_name = tool_context.agent_name
    tool_name = tool.name
    print(f"[Callback] Before tool call for tool '{tool_name}' in agent '{agent_name}'")
    print(f"[Callback] Original args: {args}")

    if tool_name == 'get_capital_city' and args.get('country', '').lower() == 'canada':
        print("[Callback] Detected 'Canada'. Modifying args to 'France'.")
        args['country'] = 'France'
        print(f"[Callback] Modified args: {args}")
        return None

    # If the tool is 'get_capital_city' and country is 'BLOCK'
    if tool_name == 'get_capital_city' and args.get('country', '').upper() == 'BLOCK':
        print("[Callback] Detected 'BLOCK'. Skipping tool execution.")
        return {"result": "Tool execution was blocked by before_tool_callback."}

    print("[Callback] Proceeding with original or previously modified args.")
    return None

my_llm_agent = LlmAgent(
        name="ToolCallbackAgent",
        model=GEMINI_2_FLASH,
        instruction="You are an agent that can find capital cities. Use the get_capital_city tool.",
        description="An LLM agent demonstrating before_tool_callback",
        tools=[capital_tool],
        before_tool_callback=simple_before_tool_modifier
)

APP_NAME = "guardrail_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=my_llm_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
  content = types.Content(role='user', parts=[types.Part(text=query)])
  events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

  for event in events:
      if event.is_final_response():
          final_response = event.content.parts[0].text
          print("Agent Response: ", final_response)

call_agent("callback example")

After Tool Callback¶
When: Called just after the tool's run_async method completes successfully.

Purpose: Allows inspection and modification of the tool's result before it's sent back to the LLM (potentially after summarization). Useful for logging tool results, post-processing or formatting results, or saving specific parts of the result to the session state.

Return Value Effect:

If the callback returns None (or a Maybe.empty() object in Java), the original tool_response is used.
If a new dictionary is returned, it replaces the original tool_response. This allows modifying or filtering the result seen by the LLM.
Code

Python
Java

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from typing import Optional
from google.genai import types 
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from typing import Dict, Any
from copy import deepcopy

GEMINI_2_FLASH="gemini-2.0-flash"

# --- Define a Simple Tool Function (Same as before) ---
def get_capital_city(country: str) -> str:
    """Retrieves the capital city of a given country."""
    print(f"--- Tool 'get_capital_city' executing with country: {country} ---")
    country_capitals = {
        "united states": "Washington, D.C.",
        "canada": "Ottawa",
        "france": "Paris",
        "germany": "Berlin",
    }
    return {"result": country_capitals.get(country.lower(), f"Capital not found for {country}")}

# --- Wrap the function into a Tool ---
capital_tool = FunctionTool(func=get_capital_city)

# --- Define the Callback Function ---
def simple_after_tool_modifier(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    """Inspects/modifies the tool result after execution."""
    agent_name = tool_context.agent_name
    tool_name = tool.name
    print(f"[Callback] After tool call for tool '{tool_name}' in agent '{agent_name}'")
    print(f"[Callback] Args used: {args}")
    print(f"[Callback] Original tool_response: {tool_response}")

    # Default structure for function tool results is {"result": <return_value>}
    original_result_value = tool_response.get("result", "")
    # original_result_value = tool_response

    # --- Modification Example ---
    # If the tool was 'get_capital_city' and result is 'Washington, D.C.'
    if tool_name == 'get_capital_city' and original_result_value == "Washington, D.C.":
        print("[Callback] Detected 'Washington, D.C.'. Modifying tool response.")

        # IMPORTANT: Create a new dictionary or modify a copy
        modified_response = deepcopy(tool_response)
        modified_response["result"] = f"{original_result_value} (Note: This is the capital of the USA)."
        modified_response["note_added_by_callback"] = True # Add extra info if needed

        print(f"[Callback] Modified tool_response: {modified_response}")
        return modified_response # Return the modified dictionary

    print("[Callback] Passing original tool response through.")
    # Return None to use the original tool_response
    return None


# Create LlmAgent and Assign Callback
my_llm_agent = LlmAgent(
        name="AfterToolCallbackAgent",
        model=GEMINI_2_FLASH,
        instruction="You are an agent that finds capital cities using the get_capital_city tool. Report the result clearly.",
        description="An LLM agent demonstrating after_tool_callback",
        tools=[capital_tool], # Add the tool
        after_tool_callback=simple_after_tool_modifier # Assign the callback
    )

APP_NAME = "guardrail_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

# Session and Runner
session_service = InMemorySessionService()
session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=my_llm_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
async def call_agent(query):
  content = types.Content(role='user', parts=[types.Part(text=query)])
  events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

  async for event in events:
      if event.is_final_response():
          final_response = event.content.parts[0].text
          print("Agent Response: ", final_response)

await call_agent("united states")

Design Patterns and Best Practices for Callbacks¶
Callbacks offer powerful hooks into the agent lifecycle. Here are common design patterns illustrating how to leverage them effectively in ADK, followed by best practices for implementation.

Design Patterns¶
These patterns demonstrate typical ways to enhance or control agent behavior using callbacks:

1. Guardrails & Policy Enforcement¶
Pattern: Intercept requests before they reach the LLM or tools to enforce rules.
How: Use before_model_callback to inspect the LlmRequest prompt or before_tool_callback to inspect tool arguments. If a policy violation is detected (e.g., forbidden topics, profanity), return a predefined response (LlmResponse or dict/ Map) to block the operation and optionally update context.state to log the violation.
Example: A before_model_callback checks llm_request.contents for sensitive keywords and returns a standard "Cannot process this request" LlmResponse if found, preventing the LLM call.
2. Dynamic State Management¶
Pattern: Read from and write to session state within callbacks to make agent behavior context-aware and pass data between steps.
How: Access callback_context.state or tool_context.state. Modifications (state['key'] = value) are automatically tracked in the subsequent Event.actions.state_delta for persistence by the SessionService.
Example: An after_tool_callback saves a transaction_id from the tool's result to tool_context.state['last_transaction_id']. A later before_agent_callback might read state['user_tier'] to customize the agent's greeting.
3. Logging and Monitoring¶
Pattern: Add detailed logging at specific lifecycle points for observability and debugging.
How: Implement callbacks (e.g., before_agent_callback, after_tool_callback, after_model_callback) to print or send structured logs containing information like agent name, tool name, invocation ID, and relevant data from the context or arguments.
Example: Log messages like INFO: [Invocation: e-123] Before Tool: search_api - Args: {'query': 'ADK'}.
4. Caching¶
Pattern: Avoid redundant LLM calls or tool executions by caching results.
How: In before_model_callback or before_tool_callback, generate a cache key based on the request/arguments. Check context.state (or an external cache) for this key. If found, return the cached LlmResponse or result directly, skipping the actual operation. If not found, allow the operation to proceed and use the corresponding after_ callback (after_model_callback, after_tool_callback) to store the new result in the cache using the key.
Example: before_tool_callback for get_stock_price(symbol) checks state[f"cache:stock:{symbol}"]. If present, returns the cached price; otherwise, allows the API call and after_tool_callback saves the result to the state key.
5. Request/Response Modification¶
Pattern: Alter data just before it's sent to the LLM/tool or just after it's received.
How:
before_model_callback: Modify llm_request (e.g., add system instructions based on state).
after_model_callback: Modify the returned LlmResponse (e.g., format text, filter content).
before_tool_callback: Modify the tool args dictionary (or Map in Java).
after_tool_callback: Modify the tool_response dictionary (or Map in Java).
Example: before_model_callback appends "User language preference: Spanish" to llm_request.config.system_instruction if context.state['lang'] == 'es'.
6. Conditional Skipping of Steps¶
Pattern: Prevent standard operations (agent run, LLM call, tool execution) based on certain conditions.
How: Return a value from a before_ callback (Content from before_agent_callback, LlmResponse from before_model_callback, dict from before_tool_callback). The framework interprets this returned value as the result for that step, skipping the normal execution.
Example: before_tool_callback checks tool_context.state['api_quota_exceeded']. If True, it returns {'error': 'API quota exceeded'}, preventing the actual tool function from running.
7. Tool-Specific Actions (Authentication & Summarization Control)¶
Pattern: Handle actions specific to the tool lifecycle, primarily authentication and controlling LLM summarization of tool results.
How: Use ToolContext within tool callbacks (before_tool_callback, after_tool_callback).
Authentication: Call tool_context.request_credential(auth_config) in before_tool_callback if credentials are required but not found (e.g., via tool_context.get_auth_response or state check). This initiates the auth flow.
Summarization: Set tool_context.actions.skip_summarization = True if the raw dictionary output of the tool should be passed back to the LLM or potentially displayed directly, bypassing the default LLM summarization step.
Example: A before_tool_callback for a secure API checks for an auth token in state; if missing, it calls request_credential. An after_tool_callback for a tool returning structured JSON might set skip_summarization = True.
8. Artifact Handling¶
Pattern: Save or load session-related files or large data blobs during the agent lifecycle.
How: Use callback_context.save_artifact / await tool_context.save_artifact to store data (e.g., generated reports, logs, intermediate data). Use load_artifact to retrieve previously stored artifacts. Changes are tracked via Event.actions.artifact_delta.
Example: An after_tool_callback for a "generate_report" tool saves the output file using await tool_context.save_artifact("report.pdf", report_part). A before_agent_callback might load a configuration artifact using callback_context.load_artifact("agent_config.json").
Best Practices for Callbacks¶
Keep Focused: Design each callback for a single, well-defined purpose (e.g., just logging, just validation). Avoid monolithic callbacks.
Mind Performance: Callbacks execute synchronously within the agent's processing loop. Avoid long-running or blocking operations (network calls, heavy computation). Offload if necessary, but be aware this adds complexity.
Handle Errors Gracefully: Use try...except/ catch blocks within your callback functions. Log errors appropriately and decide if the agent invocation should halt or attempt recovery. Don't let callback errors crash the entire process.
Manage State Carefully:
Be deliberate about reading from and writing to context.state. Changes are immediately visible within the current invocation and persisted at the end of the event processing.
Use specific state keys rather than modifying broad structures to avoid unintended side effects.
Consider using state prefixes (State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX) for clarity, especially with persistent SessionService implementations.
Consider Idempotency: If a callback performs actions with external side effects (e.g., incrementing an external counter), design it to be idempotent (safe to run multiple times with the same input) if possible, to handle potential retries in the framework or your application.
Test Thoroughly: Unit test your callback functions using mock context objects. Perform integration tests to ensure callbacks function correctly within the full agent flow.
Ensure Clarity: Use descriptive names for your callback functions. Add clear docstrings explaining their purpose, when they run, and any side effects (especially state modifications).
Use Correct Context Type: Always use the specific context type provided (CallbackContext for agent/model, ToolContext for tools) to ensure access to the appropriate methods and properties.
By applying these patterns and best practices, you can effectively use callbacks to create more robust, observable, and customized agent behaviors in ADK.

Events¶
Events are the fundamental units of information flow within the Agent Development Kit (ADK). They represent every significant occurrence during an agent's interaction lifecycle, from initial user input to the final response and all the steps in between. Understanding events is crucial because they are the primary way components communicate, state is managed, and control flow is directed.

What Events Are and Why They Matter¶
An Event in ADK is an immutable record representing a specific point in the agent's execution. It captures user messages, agent replies, requests to use tools (function calls), tool results, state changes, control signals, and errors.


Python
Java
Technically, it's an instance of the google.adk.events.Event class, which builds upon the basic LlmResponse structure by adding essential ADK-specific metadata and an actions payload.


# Conceptual Structure of an Event (Python)
# from google.adk.events import Event, EventActions
# from google.genai import types

# class Event(LlmResponse): # Simplified view
#     # --- LlmResponse fields ---
#     content: Optional[types.Content]
#     partial: Optional[bool]
#     # ... other response fields ...

#     # --- ADK specific additions ---
#     author: str          # 'user' or agent name
#     invocation_id: str   # ID for the whole interaction run
#     id: str              # Unique ID for this specific event
#     timestamp: float     # Creation time
#     actions: EventActions # Important for side-effects & control
#     branch: Optional[str] # Hierarchy path
#     # ...

Events are central to ADK's operation for several key reasons:

Communication: They serve as the standard message format between the user interface, the Runner, agents, the LLM, and tools. Everything flows as an Event.

Signaling State & Artifact Changes: Events carry instructions for state modifications and track artifact updates. The SessionService uses these signals to ensure persistence. In Python changes are signaled via event.actions.state_delta and event.actions.artifact_delta.

Control Flow: Specific fields like event.actions.transfer_to_agent or event.actions.escalate act as signals that direct the framework, determining which agent runs next or if a loop should terminate.

History & Observability: The sequence of events recorded in session.events provides a complete, chronological history of an interaction, invaluable for debugging, auditing, and understanding agent behavior step-by-step.

In essence, the entire process, from a user's query to the agent's final answer, is orchestrated through the generation, interpretation, and processing of Event objects.

Understanding and Using Events¶
As a developer, you'll primarily interact with the stream of events yielded by the Runner. Here's how to understand and extract information from them:

Note

The specific parameters or method names for the primitives may vary slightly by SDK language (e.g., event.content() in Python, event.content().get().parts() in Java). Refer to the language-specific API documentation for details.

Identifying Event Origin and Type¶
Quickly determine what an event represents by checking:

Who sent it? (event.author)
'user': Indicates input directly from the end-user.
'AgentName': Indicates output or action from a specific agent (e.g., 'WeatherAgent', 'SummarizerAgent').
What's the main payload? (event.content and event.content.parts)

Text: Indicates a conversational message. For Python, check if event.content.parts[0].text exists. For Java, check if event.content() is present, its parts() are present and not empty, and the first part's text() is present.
Tool Call Request: Check event.get_function_calls(). If not empty, the LLM is asking to execute one or more tools. Each item in the list has .name and .args.
Tool Result: Check event.get_function_responses(). If not empty, this event carries the result(s) from tool execution(s). Each item has .name and .response (the dictionary returned by the tool). Note: For history structuring, the role inside the content is often 'user', but the event author is typically the agent that requested the tool call.
Is it streaming output? (event.partial) Indicates whether this is an incomplete chunk of text from the LLM.

True: More text will follow.
False or None/Optional.empty(): This part of the content is complete (though the overall turn might not be finished if turn_complete is also false).

Python
Java

# Pseudocode: Basic event identification (Python)
# async for event in runner.run_async(...):
#     print(f"Event from: {event.author}")
#
#     if event.content and event.content.parts:
#         if event.get_function_calls():
#             print("  Type: Tool Call Request")
#         elif event.get_function_responses():
#             print("  Type: Tool Result")
#         elif event.content.parts[0].text:
#             if event.partial:
#                 print("  Type: Streaming Text Chunk")
#             else:
#                 print("  Type: Complete Text Message")
#         else:
#             print("  Type: Other Content (e.g., code result)")
#     elif event.actions and (event.actions.state_delta or event.actions.artifact_delta):
#         print("  Type: State/Artifact Update")
#     else:
#         print("  Type: Control Signal or Other")

Extracting Key Information¶
Once you know the event type, access the relevant data:

Text Content: Always check for the presence of content and parts before accessing text. In Python its text = event.content.parts[0].text.

Function Call Details:


Python
Java

calls = event.get_function_calls()
if calls:
    for call in calls:
        tool_name = call.name
        arguments = call.args # This is usually a dictionary
        print(f"  Tool: {tool_name}, Args: {arguments}")
        # Application might dispatch execution based on this

Function Response Details:


Python
Java

responses = event.get_function_responses()
if responses:
    for response in responses:
        tool_name = response.name
        result_dict = response.response # The dictionary returned by the tool
        print(f"  Tool Result: {tool_name} -> {result_dict}")

Identifiers:

event.id: Unique ID for this specific event instance.
event.invocation_id: ID for the entire user-request-to-final-response cycle this event belongs to. Useful for logging and tracing.
Detecting Actions and Side Effects¶
The event.actions object signals changes that occurred or should occur. Always check if event.actions and it's fields/ methods exists before accessing them.

State Changes: Gives you a collection of key-value pairs that were modified in the session state during the step that produced this event.


Python
Java
delta = event.actions.state_delta (a dictionary of {key: value} pairs).


if event.actions and event.actions.state_delta:
    print(f"  State changes: {event.actions.state_delta}")
    # Update local UI or application state if necessary

Artifact Saves: Gives you a collection indicating which artifacts were saved and their new version number (or relevant Part information).


Python
Java
artifact_changes = event.actions.artifact_delta (a dictionary of {filename: version}).


if event.actions and event.actions.artifact_delta:
    print(f"  Artifacts saved: {event.actions.artifact_delta}")
    # UI might refresh an artifact list

Control Flow Signals: Check boolean flags or string values:


Python
Java
event.actions.transfer_to_agent (string): Control should pass to the named agent.
event.actions.escalate (bool): A loop should terminate.
event.actions.skip_summarization (bool): A tool result should not be summarized by the LLM.

if event.actions:
    if event.actions.transfer_to_agent:
        print(f"  Signal: Transfer to {event.actions.transfer_to_agent}")
    if event.actions.escalate:
        print("  Signal: Escalate (terminate loop)")
    if event.actions.skip_summarization:
        print("  Signal: Skip summarization for tool result")

Determining if an Event is a "Final" Response¶
Use the built-in helper method event.is_final_response() to identify events suitable for display as the agent's complete output for a turn.

Purpose: Filters out intermediate steps (like tool calls, partial streaming text, internal state updates) from the final user-facing message(s).
When True?
The event contains a tool result (function_response) and skip_summarization is True.
The event contains a tool call (function_call) for a tool marked as is_long_running=True. In Java, check if the longRunningToolIds list is empty:
event.longRunningToolIds().isPresent() && !event.longRunningToolIds().get().isEmpty() is true.
OR, all of the following are met:
No function calls (get_function_calls() is empty).
No function responses (get_function_responses() is empty).
Not a partial stream chunk (partial is not True).
Doesn't end with a code execution result that might need further processing/display.
Usage: Filter the event stream in your application logic.


Python
Java

# Pseudocode: Handling final responses in application (Python)
# full_response_text = ""
# async for event in runner.run_async(...):
#     # Accumulate streaming text if needed...
#     if event.partial and event.content and event.content.parts and event.content.parts[0].text:
#         full_response_text += event.content.parts[0].text
#
#     # Check if it's a final, displayable event
#     if event.is_final_response():
#         print("\n--- Final Output Detected ---")
#         if event.content and event.content.parts and event.content.parts[0].text:
#              # If it's the final part of a stream, use accumulated text
#              final_text = full_response_text + (event.content.parts[0].text if not event.partial else "")
#              print(f"Display to user: {final_text.strip()}")
#              full_response_text = "" # Reset accumulator
#         elif event.actions and event.actions.skip_summarization and event.get_function_responses():
#              # Handle displaying the raw tool result if needed
#              response_data = event.get_function_responses()[0].response
#              print(f"Display raw tool result: {response_data}")
#         elif hasattr(event, 'long_running_tool_ids') and event.long_running_tool_ids:
#              print("Display message: Tool is running in background...")
#         else:
#              # Handle other types of final responses if applicable
#              print("Display: Final non-textual response or signal.")

By carefully examining these aspects of an event, you can build robust applications that react appropriately to the rich information flowing through the ADK system.

How Events Flow: Generation and Processing¶
Events are created at different points and processed systematically by the framework. Understanding this flow helps clarify how actions and history are managed.

Generation Sources:

User Input: The Runner typically wraps initial user messages or mid-conversation inputs into an Event with author='user'.
Agent Logic: Agents (BaseAgent, LlmAgent) explicitly yield Event(...) objects (setting author=self.name) to communicate responses or signal actions.
LLM Responses: The ADK model integration layer translates raw LLM output (text, function calls, errors) into Event objects, authored by the calling agent.
Tool Results: After a tool executes, the framework generates an Event containing the function_response. The author is typically the agent that requested the tool, while the role inside the content is set to 'user' for the LLM history.
Processing Flow:

Yield/Return: An event is generated and yielded (Python) or returned/emitted (Java) by its source.
Runner Receives: The main Runner executing the agent receives the event.
SessionService Processing: The Runner sends the event to the configured SessionService. This is a critical step:
Applies Deltas: The service merges event.actions.state_delta into session.state and updates internal records based on event.actions.artifact_delta. (Note: The actual artifact saving usually happened earlier when context.save_artifact was called).
Finalizes Metadata: Assigns a unique event.id if not present, may update event.timestamp.
Persists to History: Appends the processed event to the session.events list.
External Yield: The Runner yields (Python) or returns/emits (Java) the processed event outwards to the calling application (e.g., the code that invoked runner.run_async).
This flow ensures that state changes and history are consistently recorded alongside the communication content of each event.

Common Event Examples (Illustrative Patterns)¶
Here are concise examples of typical events you might see in the stream:

User Input:

{
  "author": "user",
  "invocation_id": "e-xyz...",
  "content": {"parts": [{"text": "Book a flight to London for next Tuesday"}]}
  // actions usually empty
}
Agent Final Text Response: (is_final_response() == True)

{
  "author": "TravelAgent",
  "invocation_id": "e-xyz...",
  "content": {"parts": [{"text": "Okay, I can help with that. Could you confirm the departure city?"}]},
  "partial": false,
  "turn_complete": true
  // actions might have state delta, etc.
}
Agent Streaming Text Response: (is_final_response() == False)

{
  "author": "SummaryAgent",
  "invocation_id": "e-abc...",
  "content": {"parts": [{"text": "The document discusses three main points:"}]},
  "partial": true,
  "turn_complete": false
}
// ... more partial=True events follow ...
Tool Call Request (by LLM): (is_final_response() == False)

{
  "author": "TravelAgent",
  "invocation_id": "e-xyz...",
  "content": {"parts": [{"function_call": {"name": "find_airports", "args": {"city": "London"}}}]}
  // actions usually empty
}
Tool Result Provided (to LLM): (is_final_response() depends on skip_summarization)

{
  "author": "TravelAgent", // Author is agent that requested the call
  "invocation_id": "e-xyz...",
  "content": {
    "role": "user", // Role for LLM history
    "parts": [{"function_response": {"name": "find_airports", "response": {"result": ["LHR", "LGW", "STN"]}}}]
  }
  // actions might have skip_summarization=True
}
State/Artifact Update Only: (is_final_response() == False)

{
  "author": "InternalUpdater",
  "invocation_id": "e-def...",
  "content": null,
  "actions": {
    "state_delta": {"user_status": "verified"},
    "artifact_delta": {"verification_doc.pdf": 2}
  }
}
Agent Transfer Signal: (is_final_response() == False)

{
  "author": "OrchestratorAgent",
  "invocation_id": "e-789...",
  "content": {"parts": [{"function_call": {"name": "transfer_to_agent", "args": {"agent_name": "BillingAgent"}}}]},
  "actions": {"transfer_to_agent": "BillingAgent"} // Added by framework
}
Loop Escalation Signal: (is_final_response() == False)

{
  "author": "CheckerAgent",
  "invocation_id": "e-loop...",
  "content": {"parts": [{"text": "Maximum retries reached."}]}, // Optional content
  "actions": {"escalate": true}
}
Additional Context and Event Details¶
Beyond the core concepts, here are a few specific details about context and events that are important for certain use cases:

ToolContext.function_call_id (Linking Tool Actions):

When an LLM requests a tool (FunctionCall), that request has an ID. The ToolContext provided to your tool function includes this function_call_id.
Importance: This ID is crucial for linking actions like authentication back to the specific tool request that initiated them, especially if multiple tools are called in one turn. The framework uses this ID internally.
How State/Artifact Changes are Recorded:

When you modify state or save an artifact using CallbackContext or ToolContext, these changes aren't immediately written to persistent storage.
Instead, they populate the state_delta and artifact_delta fields within the EventActions object.
This EventActions object is attached to the next event generated after the change (e.g., the agent's response or a tool result event).
The SessionService.append_event method reads these deltas from the incoming event and applies them to the session's persistent state and artifact records. This ensures changes are tied chronologically to the event stream.
State Scope Prefixes (app:, user:, temp:):

When managing state via context.state, you can optionally use prefixes:
app:my_setting: Suggests state relevant to the entire application (requires a persistent SessionService).
user:user_preference: Suggests state relevant to the specific user across sessions (requires a persistent SessionService).
temp:intermediate_result or no prefix: Typically session-specific or temporary state for the current invocation.
The underlying SessionService determines how these prefixes are handled for persistence.
Error Events:

An Event can represent an error. Check the event.error_code and event.error_message fields (inherited from LlmResponse).
Errors might originate from the LLM (e.g., safety filters, resource limits) or potentially be packaged by the framework if a tool fails critically. Check tool FunctionResponse content for typical tool-specific errors.

// Example Error Event (conceptual)
{
  "author": "LLMAgent",
  "invocation_id": "e-err...",
  "content": null,
  "error_code": "SAFETY_FILTER_TRIGGERED",
  "error_message": "Response blocked due to safety settings.",
  "actions": {}
}
These details provide a more complete picture for advanced use cases involving tool authentication, state persistence scope, and error handling within the event stream.

Best Practices for Working with Events¶
To use events effectively in your ADK applications:

Clear Authorship: When building custom agents, ensure correct attribution for agent actions in the history. The framework generally handles authorship correctly for LLM/tool events.


Python
Java
Use yield Event(author=self.name, ...) in BaseAgent subclasses.


Semantic Content & Actions: Use event.content for the core message/data (text, function call/response). Use event.actions specifically for signaling side effects (state/artifact deltas) or control flow (transfer, escalate, skip_summarization).

Idempotency Awareness: Understand that the SessionService is responsible for applying the state/artifact changes signaled in event.actions. While ADK services aim for consistency, consider potential downstream effects if your application logic re-processes events.
Use is_final_response(): Rely on this helper method in your application/UI layer to identify complete, user-facing text responses. Avoid manually replicating its logic.
Leverage History: The session's event list is your primary debugging tool. Examine the sequence of authors, content, and actions to trace execution and diagnose issues.
Use Metadata: Use invocation_id to correlate all events within a single user interaction. Use event.id to reference specific, unique occurrences.
Treating events as structured messages with clear purposes for their content and actions is key to building, debugging, and managing complex agent behaviors in ADK.

Context¶
What are Context¶
In the Agent Development Kit (ADK), "context" refers to the crucial bundle of information available to your agent and its tools during specific operations. Think of it as the necessary background knowledge and resources needed to handle a current task or conversation turn effectively.

Agents often need more than just the latest user message to perform well. Context is essential because it enables:

Maintaining State: Remembering details across multiple steps in a conversation (e.g., user preferences, previous calculations, items in a shopping cart). This is primarily managed through session state.
Passing Data: Sharing information discovered or generated in one step (like an LLM call or a tool execution) with subsequent steps. Session state is key here too.
Accessing Services: Interacting with framework capabilities like:
Artifact Storage: Saving or loading files or data blobs (like PDFs, images, configuration files) associated with the session.
Memory: Searching for relevant information from past interactions or external knowledge sources connected to the user.
Authentication: Requesting and retrieving credentials needed by tools to access external APIs securely.
Identity and Tracking: Knowing which agent is currently running (agent.name) and uniquely identifying the current request-response cycle (invocation_id) for logging and debugging.
Tool-Specific Actions: Enabling specialized operations within tools, such as requesting authentication or searching memory, which require access to the current interaction's details.
The central piece holding all this information together for a single, complete user-request-to-final-response cycle (an invocation) is the InvocationContext. However, you typically won't create or manage this object directly. The ADK framework creates it when an invocation starts (e.g., via runner.run_async) and passes the relevant contextual information implicitly to your agent code, callbacks, and tools.


Python
Java

# Conceptual Pseudocode: How the framework provides context (Internal Logic)

# runner = Runner(agent=my_root_agent, session_service=..., artifact_service=...)
# user_message = types.Content(...)
# session = session_service.get_session(...) # Or create new

# --- Inside runner.run_async(...) ---
# 1. Framework creates the main context for this specific run
# invocation_context = InvocationContext(
#     invocation_id="unique-id-for-this-run",
#     session=session,
#     user_content=user_message,
#     agent=my_root_agent, # The starting agent
#     session_service=session_service,
#     artifact_service=artifact_service,
#     memory_service=memory_service,
#     # ... other necessary fields ...
# )
#
# 2. Framework calls the agent's run method, passing the context implicitly
#    (The agent's method signature will receive it, e.g., runAsyncImpl(InvocationContext invocationContext))
# await my_root_agent.run_async(invocation_context)
#   --- End Internal Logic ---
#
# As a developer, you work with the context objects provided in method arguments.

The Different types of Context¶
While InvocationContext acts as the comprehensive internal container, ADK provides specialized context objects tailored to specific situations. This ensures you have the right tools and permissions for the task at hand without needing to handle the full complexity of the internal context everywhere. Here are the different "flavors" you'll encounter:

InvocationContext

Where Used: Received as the ctx argument directly within an agent's core implementation methods (_run_async_impl, _run_live_impl).
Purpose: Provides access to the entire state of the current invocation. This is the most comprehensive context object.
Key Contents: Direct access to session (including state and events), the current agent instance, invocation_id, initial user_content, references to configured services (artifact_service, memory_service, session_service), and fields related to live/streaming modes.
Use Case: Primarily used when the agent's core logic needs direct access to the overall session or services, though often state and artifact interactions are delegated to callbacks/tools which use their own contexts. Also used to control the invocation itself (e.g., setting ctx.end_invocation = True).

Python
Java

# Pseudocode: Agent implementation receiving InvocationContext
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

class MyAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Direct access example
        agent_name = ctx.agent.name
        session_id = ctx.session.id
        print(f"Agent {agent_name} running in session {session_id} for invocation {ctx.invocation_id}")
        # ... agent logic using ctx ...
        yield # ... event ...

ReadonlyContext

Where Used: Provided in scenarios where only read access to basic information is needed and mutation is disallowed (e.g., InstructionProvider functions). It's also the base class for other contexts.
Purpose: Offers a safe, read-only view of fundamental contextual details.
Key Contents: invocation_id, agent_name, and a read-only view of the current state.

Python
Java

# Pseudocode: Instruction provider receiving ReadonlyContext
from google.adk.agents import ReadonlyContext

def my_instruction_provider(context: ReadonlyContext) -> str:
    # Read-only access example
    user_tier = context.state().get("user_tier", "standard") # Can read state
    # context.state['new_key'] = 'value' # This would typically cause an error or be ineffective
    return f"Process the request for a {user_tier} user."

CallbackContext

Where Used: Passed as callback_context to agent lifecycle callbacks (before_agent_callback, after_agent_callback) and model interaction callbacks (before_model_callback, after_model_callback).
Purpose: Facilitates inspecting and modifying state, interacting with artifacts, and accessing invocation details specifically within callbacks.
Key Capabilities (Adds to ReadonlyContext):
Mutable state Property: Allows reading and writing to session state. Changes made here (callback_context.state['key'] = value) are tracked and associated with the event generated by the framework after the callback.
Artifact Methods: load_artifact(filename) and save_artifact(filename, part) methods for interacting with the configured artifact_service.
Direct user_content access.

Python
Java

# Pseudocode: Callback receiving CallbackContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types
from typing import Optional

def my_before_model_cb(callback_context: CallbackContext, request: LlmRequest) -> Optional[types.Content]:
    # Read/Write state example
    call_count = callback_context.state.get("model_calls", 0)
    callback_context.state["model_calls"] = call_count + 1 # Modify state

    # Optionally load an artifact
    # config_part = callback_context.load_artifact("model_config.json")
    print(f"Preparing model call #{call_count + 1} for invocation {callback_context.invocation_id}")
    return None # Allow model call to proceed

ToolContext

Where Used: Passed as tool_context to the functions backing FunctionTools and to tool execution callbacks (before_tool_callback, after_tool_callback).
Purpose: Provides everything CallbackContext does, plus specialized methods essential for tool execution, like handling authentication, searching memory, and listing artifacts.
Key Capabilities (Adds to CallbackContext):
Authentication Methods: request_credential(auth_config) to trigger an auth flow, and get_auth_response(auth_config) to retrieve credentials provided by the user/system.
Artifact Listing: list_artifacts() to discover available artifacts in the session.
Memory Search: search_memory(query) to query the configured memory_service.
function_call_id Property: Identifies the specific function call from the LLM that triggered this tool execution, crucial for linking authentication requests or responses back correctly.
actions Property: Direct access to the EventActions object for this step, allowing the tool to signal state changes, auth requests, etc.

Python
Java

# Pseudocode: Tool function receiving ToolContext
from google.adk.tools import ToolContext
from typing import Dict, Any

# Assume this function is wrapped by a FunctionTool
def search_external_api(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    api_key = tool_context.state.get("api_key")
    if not api_key:
        # Define required auth config
        # auth_config = AuthConfig(...)
        # tool_context.request_credential(auth_config) # Request credentials
        # Use the 'actions' property to signal the auth request has been made
        # tool_context.actions.requested_auth_configs[tool_context.function_call_id] = auth_config
        return {"status": "Auth Required"}

    # Use the API key...
    print(f"Tool executing for query '{query}' using API key. Invocation: {tool_context.invocation_id}")

    # Optionally search memory or list artifacts
    # relevant_docs = tool_context.search_memory(f"info related to {query}")
    # available_files = tool_context.list_artifacts()

    return {"result": f"Data for {query} fetched."}

Understanding these different context objects and when to use them is key to effectively managing state, accessing services, and controlling the flow of your ADK application. The next section will detail common tasks you can perform using these contexts.

Common Tasks Using Context¶
Now that you understand the different context objects, let's focus on how to use them for common tasks when building your agents and tools.

Accessing Information¶
You'll frequently need to read information stored within the context.

Reading Session State: Access data saved in previous steps or user/app-level settings. Use dictionary-like access on the state property.


Python
Java

# Pseudocode: In a Tool function
from google.adk.tools import ToolContext

def my_tool(tool_context: ToolContext, **kwargs):
    user_pref = tool_context.state.get("user_display_preference", "default_mode")
    api_endpoint = tool_context.state.get("app:api_endpoint") # Read app-level state

    if user_pref == "dark_mode":
        # ... apply dark mode logic ...
        pass
    print(f"Using API endpoint: {api_endpoint}")
    # ... rest of tool logic ...

# Pseudocode: In a Callback function
from google.adk.agents.callback_context import CallbackContext

def my_callback(callback_context: CallbackContext, **kwargs):
    last_tool_result = callback_context.state.get("temp:last_api_result") # Read temporary state
    if last_tool_result:
        print(f"Found temporary result from last tool: {last_tool_result}")
    # ... callback logic ...

Getting Current Identifiers: Useful for logging or custom logic based on the current operation.


Python
Java

# Pseudocode: In any context (ToolContext shown)
from google.adk.tools import ToolContext

def log_tool_usage(tool_context: ToolContext, **kwargs):
    agent_name = tool_context.agent_nameSystem.out.println("Found temporary result from last tool: " + lastToolResult);
    inv_id = tool_context.invocation_id
    func_call_id = getattr(tool_context, 'function_call_id', 'N/A') # Specific to ToolContext

    print(f"Log: Invocation={inv_id}, Agent={agent_name}, FunctionCallID={func_call_id} - Tool Executed.")

Accessing the Initial User Input: Refer back to the message that started the current invocation.


Python
Java

# Pseudocode: In a Callback
from google.adk.agents.callback_context import CallbackContext

def check_initial_intent(callback_context: CallbackContext, **kwargs):
    initial_text = "N/A"
    if callback_context.user_content and callback_context.user_content.parts:
        initial_text = callback_context.user_content.parts[0].text or "Non-text input"

    print(f"This invocation started with user input: '{initial_text}'")

# Pseudocode: In an Agent's _run_async_impl
# async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
#     if ctx.user_content and ctx.user_content.parts:
#         initial_text = ctx.user_content.parts[0].text
#         print(f"Agent logic remembering initial query: {initial_text}")
#     ...

Managing Session State¶
State is crucial for memory and data flow. When you modify state using CallbackContext or ToolContext, the changes are automatically tracked and persisted by the framework.

How it Works: Writing to callback_context.state['my_key'] = my_value or tool_context.state['my_key'] = my_value adds this change to the EventActions.state_delta associated with the current step's event. The SessionService then applies these deltas when persisting the event.
Passing Data Between Tools:


Python
Java

# Pseudocode: Tool 1 - Fetches user ID
from google.adk.tools import ToolContext
import uuid

def get_user_profile(tool_context: ToolContext) -> dict:
    user_id = str(uuid.uuid4()) # Simulate fetching ID
    # Save the ID to state for the next tool
    tool_context.state["temp:current_user_id"] = user_id
    return {"profile_status": "ID generated"}

# Pseudocode: Tool 2 - Uses user ID from state
def get_user_orders(tool_context: ToolContext) -> dict:
    user_id = tool_context.state.get("temp:current_user_id")
    if not user_id:
        return {"error": "User ID not found in state"}

    print(f"Fetching orders for user ID: {user_id}")
    # ... logic to fetch orders using user_id ...
    return {"orders": ["order123", "order456"]}

Updating User Preferences:


Python
Java

# Pseudocode: Tool or Callback identifies a preference
from google.adk.tools import ToolContext # Or CallbackContext

def set_user_preference(tool_context: ToolContext, preference: str, value: str) -> dict:
    # Use 'user:' prefix for user-level state (if using a persistent SessionService)
    state_key = f"user:{preference}"
    tool_context.state[state_key] = value
    print(f"Set user preference '{preference}' to '{value}'")
    return {"status": "Preference updated"}

State Prefixes: While basic state is session-specific, prefixes like app: and user: can be used with persistent SessionService implementations (like DatabaseSessionService or VertexAiSessionService) to indicate broader scope (app-wide or user-wide across sessions). temp: can denote data only relevant within the current invocation.

Working with Artifacts¶
Use artifacts to handle files or large data blobs associated with the session. Common use case: processing uploaded documents.

Document Summarizer Example Flow:

Ingest Reference (e.g., in a Setup Tool or Callback): Save the path or URI of the document, not the entire content, as an artifact.


Python
Java

# Pseudocode: In a callback or initial tool
from google.adk.agents import CallbackContext # Or ToolContext
from google.genai import types

def save_document_reference(context: CallbackContext, file_path: str) -> None:
    # Assume file_path is something like "gs://my-bucket/docs/report.pdf" or "/local/path/to/report.pdf"
    try:
        # Create a Part containing the path/URI text
        artifact_part = types.Part(text=file_path)
        version = context.save_artifact("document_to_summarize.txt", artifact_part)
        print(f"Saved document reference '{file_path}' as artifact version {version}")
        # Store the filename in state if needed by other tools
        context.state["temp:doc_artifact_name"] = "document_to_summarize.txt"
    except ValueError as e:
        print(f"Error saving artifact: {e}") # E.g., Artifact service not configured
    except Exception as e:
        print(f"Unexpected error saving artifact reference: {e}")

# Example usage:
# save_document_reference(callback_context, "gs://my-bucket/docs/report.pdf")

Summarizer Tool: Load the artifact to get the path/URI, read the actual document content using appropriate libraries, summarize, and return the result.


Python
Java

# Pseudocode: In the Summarizer tool function
from google.adk.tools import ToolContext
from google.genai import types
# Assume libraries like google.cloud.storage or built-in open are available
# Assume a 'summarize_text' function exists
# from my_summarizer_lib import summarize_text

def summarize_document_tool(tool_context: ToolContext) -> dict:
    artifact_name = tool_context.state.get("temp:doc_artifact_name")
    if not artifact_name:
        return {"error": "Document artifact name not found in state."}

    try:
        # 1. Load the artifact part containing the path/URI
        artifact_part = tool_context.load_artifact(artifact_name)
        if not artifact_part or not artifact_part.text:
            return {"error": f"Could not load artifact or artifact has no text path: {artifact_name}"}

        file_path = artifact_part.text
        print(f"Loaded document reference: {file_path}")

        # 2. Read the actual document content (outside ADK context)
        document_content = ""
        if file_path.startswith("gs://"):
            # Example: Use GCS client library to download/read
            # from google.cloud import storage
            # client = storage.Client()
            # blob = storage.Blob.from_string(file_path, client=client)
            # document_content = blob.download_as_text() # Or bytes depending on format
            pass # Replace with actual GCS reading logic
        elif file_path.startswith("/"):
             # Example: Use local file system
             with open(file_path, 'r', encoding='utf-8') as f:
                 document_content = f.read()
        else:
            return {"error": f"Unsupported file path scheme: {file_path}"}

        # 3. Summarize the content
        if not document_content:
             return {"error": "Failed to read document content."}

        # summary = summarize_text(document_content) # Call your summarization logic
        summary = f"Summary of content from {file_path}" # Placeholder

        return {"summary": summary}

    except ValueError as e:
         return {"error": f"Artifact service error: {e}"}
    except FileNotFoundError:
         return {"error": f"Local file not found: {file_path}"}
    # except Exception as e: # Catch specific exceptions for GCS etc.
    #      return {"error": f"Error reading document {file_path}: {e}"}

Listing Artifacts: Discover what files are available.


Python
Java

# Pseudocode: In a tool function
from google.adk.tools import ToolContext

def check_available_docs(tool_context: ToolContext) -> dict:
    try:
        artifact_keys = tool_context.list_artifacts()
        print(f"Available artifacts: {artifact_keys}")
        return {"available_docs": artifact_keys}
    except ValueError as e:
        return {"error": f"Artifact service error: {e}"}

Handling Tool Authentication¶
python_only

Securely manage API keys or other credentials needed by tools.


# Pseudocode: Tool requiring auth
from google.adk.tools import ToolContext
from google.adk.auth import AuthConfig # Assume appropriate AuthConfig is defined

# Define your required auth configuration (e.g., OAuth, API Key)
MY_API_AUTH_CONFIG = AuthConfig(...)
AUTH_STATE_KEY = "user:my_api_credential" # Key to store retrieved credential

def call_secure_api(tool_context: ToolContext, request_data: str) -> dict:
    # 1. Check if credential already exists in state
    credential = tool_context.state.get(AUTH_STATE_KEY)

    if not credential:
        # 2. If not, request it
        print("Credential not found, requesting...")
        try:
            tool_context.request_credential(MY_API_AUTH_CONFIG)
            # The framework handles yielding the event. The tool execution stops here for this turn.
            return {"status": "Authentication required. Please provide credentials."}
        except ValueError as e:
            return {"error": f"Auth error: {e}"} # e.g., function_call_id missing
        except Exception as e:
            return {"error": f"Failed to request credential: {e}"}

    # 3. If credential exists (might be from a previous turn after request)
    #    or if this is a subsequent call after auth flow completed externally
    try:
        # Optionally, re-validate/retrieve if needed, or use directly
        # This might retrieve the credential if the external flow just completed
        auth_credential_obj = tool_context.get_auth_response(MY_API_AUTH_CONFIG)
        api_key = auth_credential_obj.api_key # Or access_token, etc.

        # Store it back in state for future calls within the session
        tool_context.state[AUTH_STATE_KEY] = auth_credential_obj.model_dump() # Persist retrieved credential

        print(f"Using retrieved credential to call API with data: {request_data}")
        # ... Make the actual API call using api_key ...
        api_result = f"API result for {request_data}"

        return {"result": api_result}
    except Exception as e:
        # Handle errors retrieving/using the credential
        print(f"Error using credential: {e}")
        # Maybe clear the state key if credential is invalid?
        # tool_context.state[AUTH_STATE_KEY] = None
        return {"error": "Failed to use credential"}
Remember: request_credential pauses the tool and signals the need for authentication. The user/system provides credentials, and on a subsequent call, get_auth_response (or checking state again) allows the tool to proceed. The tool_context.function_call_id is used implicitly by the framework to link the request and response.
Leveraging Memory¶
python_only

Access relevant information from the past or external sources.


# Pseudocode: Tool using memory search
from google.adk.tools import ToolContext

def find_related_info(tool_context: ToolContext, topic: str) -> dict:
    try:
        search_results = tool_context.search_memory(f"Information about {topic}")
        if search_results.results:
            print(f"Found {len(search_results.results)} memory results for '{topic}'")
            # Process search_results.results (which are SearchMemoryResponseEntry)
            top_result_text = search_results.results[0].text
            return {"memory_snippet": top_result_text}
        else:
            return {"message": "No relevant memories found."}
    except ValueError as e:
        return {"error": f"Memory service error: {e}"} # e.g., Service not configured
    except Exception as e:
        return {"error": f"Unexpected error searching memory: {e}"}
Advanced: Direct InvocationContext Usage¶
python_only

While most interactions happen via CallbackContext or ToolContext, sometimes the agent's core logic (_run_async_impl/_run_live_impl) needs direct access.


# Pseudocode: Inside agent's _run_async_impl
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

class MyControllingAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Example: Check if a specific service is available
        if not ctx.memory_service:
            print("Memory service is not available for this invocation.")
            # Potentially change agent behavior

        # Example: Early termination based on some condition
        if ctx.session.state.get("critical_error_flag"):
            print("Critical error detected, ending invocation.")
            ctx.end_invocation = True # Signal framework to stop processing
            yield Event(author=self.name, invocation_id=ctx.invocation_id, content="Stopping due to critical error.")
            return # Stop this agent's execution

        # ... Normal agent processing ...
        yield # ... event ...
Setting ctx.end_invocation = True is a way to gracefully stop the entire request-response cycle from within the agent or its callbacks/tools (via their respective context objects which also have access to modify the underlying InvocationContext's flag).

Key Takeaways & Best Practices¶
Use the Right Context: Always use the most specific context object provided (ToolContext in tools/tool-callbacks, CallbackContext in agent/model-callbacks, ReadonlyContext where applicable). Use the full InvocationContext (ctx) directly in _run_async_impl / _run_live_impl only when necessary.
State for Data Flow: context.state is the primary way to share data, remember preferences, and manage conversational memory within an invocation. Use prefixes (app:, user:, temp:) thoughtfully when using persistent storage.
Artifacts for Files: Use context.save_artifact and context.load_artifact for managing file references (like paths or URIs) or larger data blobs. Store references, load content on demand.
Tracked Changes: Modifications to state or artifacts made via context methods are automatically linked to the current step's EventActions and handled by the SessionService.
Start Simple: Focus on state and basic artifact usage first. Explore authentication, memory, and advanced InvocationContext fields (like those for live streaming) as your needs become more complex.
By understanding and effectively using these context objects, you can build more sophisticated, stateful, and capable agents with ADK.

Model Context Protocol (MCP)¶
What is Model Context Protocol (MCP)?¶
The Model Context Protocol (MCP) is an open standard designed to standardize how Large Language Models (LLMs) like Gemini and Claude communicate with external applications, data sources, and tools. Think of it as a universal connection mechanism that simplifies how LLMs obtain context, execute actions, and interact with various systems.

How does MCP work?¶
MCP follows a client-server architecture, defining how data (resources), interactive templates (prompts), and actionable functions (tools) are exposed by an MCP server and consumed by an MCP client (which could be an LLM host application or an AI agent).

MCP Tools in ADK¶
ADK helps you both use and consume MCP tools in your agents, whether you're trying to build a tool to call an MCP service, or exposing an MCP server for other developers or agents to interact with your tools.

Refer to the MCP Tools documentation for code samples and design patterns that help you use ADK together with MCP servers, including:

Using Existing MCP Servers within ADK: An ADK agent can act as an MCP client and use tools provided by external MCP servers.
Exposing ADK Tools via an MCP Server: How to build an MCP server that wraps ADK tools, making them accessible to any MCP client.
MCP Toolbox for Databases¶
MCP Toolbox for Databases is an open source MCP server that helps you build Gen AI tools so that your agents can access data in your database. Google’s Agent Development Kit (ADK) has built in support for The MCP Toolbox for Databases.

Refer to the MCP Toolbox for Databases documentation on how you can use ADK together with the MCP Toolbox for Databases. For getting started with the MCP Toolbox for Databases, a blog post Tutorial : MCP Toolbox for Databases - Exposing Big Query Datasets and Codelab MCP Toolbox for Databases:Making BigQuery datasets available to MCP clients are also available.

GenAI Toolbox

ADK Agent and FastMCP server¶
FastMCP handles all the complex MCP protocol details and server management, so you can focus on building great tools. It's designed to be high-level and Pythonic; in most cases, decorating a function is all you need.

Refer to the MCP Tools documentation documentation on how you can use ADK together with the FastMCP server running on Cloud Run.