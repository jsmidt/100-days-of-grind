import os
from typing import TypedDict, List, Dict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import json

# ----------------------------------------------------------------------------------------
# Agent Definitions
# ----------------------------------------------------------------------------------------

class AgentState(TypedDict):
    """Represents the state of our agent."""
    messages: List[BaseMessage]
    steps: int

# --- LLM Configuration ---
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") # Set API key

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7) # Choose your Gemini model

# --- Tools (Example - you'll need real implementations) ---
def web_search(query: str) -> str:
    """Searches the web for relevant information."""
    # Replace with your web search implementation (e.g., using SerpAPI, Google Search API, etc.)
    return f"Web search results for: {query} (This is a placeholder)"

def execute_code(code: str) -> str:
    """Executes Python code and returns the output."""
    # Replace with a safe code execution environment (e.g., a Docker container, a sandboxed environment)
    # **WARNING: Executing arbitrary code is extremely dangerous.  Implement robust security measures.**
    try:
        # You'll need a secure way to execute code.  This is just a placeholder!
        # exec(code)  # DANGEROUS - Don't do this directly!
        return f"Code executed successfully (placeholder). Result: Code ran but output cannot be displayed"
    except Exception as e:
        return f"Error executing code: {e}"

web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Useful for when you need to search the web to answer questions.",
)

code_execution_tool = Tool(
    name="execute_code",
    func=execute_code,
    description="Useful for executing Python code.  Use with extreme caution.",
)

tools = [web_search_tool, code_execution_tool]

# --- Agent Logic (Using create_react_agent) ---
def create_agent(name: str, instructions: str, tools: list[Tool]) -> AgentExecutor:
    prompt = PromptTemplate.from_template(
        f"""You are {name}.  Your goal is to accomplish tasks as instructed.

        Instructions:
        {instructions}

        You have access to the following tools:
        {{tools}}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{{tool_names}}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {{input}}
        Thought:{{agent_scratchpad}}
        """
    )

    agent_executor = create_react_agent(llm, tools, prompt)
    return agent_executor

def agent_node(agent: AgentExecutor, state: AgentState):
    """Stateful agent node that returns an action based on state."""
    # Extract the *last* HumanMessage's content from the message history
    # The ReAct agent expects a single string as input
    relevant_messages = [message for message in state["messages"] if isinstance(message, HumanMessage)]
    if not relevant_messages:
        return {"action": "final_answer", "value": "No human message found."}  # Handle the case where there's no initial input

    agent_input = relevant_messages[-1].content

    # Prepare the input dictionary for the ReAct agent, including intermediate_steps
    react_input = {"input": agent_input,
                   "intermediate_steps": []}

    result = agent.invoke(react_input)


    # Handle the case where the agent returns an AgentAction
    if isinstance(result, dict) and "intermediate_steps" in result:
      action = result["intermediate_steps"][0][0].tool
      action_input = result["intermediate_steps"][0][0].tool_input
      return {"action": action, "value": action_input}

    # Handle the case where the agent returns the final answer directly
    else:
      final_answer = result
      print(f"Final answer: {final_answer}")
      return {"action": "final_answer", "value": final_answer}


# ----------------------------------------------------------------------------------------
# Agent Instances
# ----------------------------------------------------------------------------------------

supervisor_instructions = """Your job is to oversee the other agents and decide what to do next.
You either respond with the agent to route to for the next step, or final_answer if the goal is complete.
"""

web_search_instructions = """You are a web search expert.  Use the web_search tool to find information relevant to the user's request."""

coding_instructions = """You are a coding expert.  Use the execute_code tool to write and execute Python code to solve problems."""

# Create agents using create_react_agent
supervisor_agent = create_agent("Supervisor", supervisor_instructions, tools)  # Supervisor needs tools for routing
web_search_agent = create_agent("Web Search Agent", web_search_instructions, [web_search_tool])
coding_agent = create_agent("Coding Agent", coding_instructions, [code_execution_tool])

# ----------------------------------------------------------------------------------------
# LangGraph Definition
# ----------------------------------------------------------------------------------------

def route(state):
    """Routes based on the agent's action"""
    agent_output = state['action']
    print (f"Agent's action: {agent_output}")

    if agent_output["action"] == "web_search":
        return "web_search"
    elif agent_output["action"] == "execute_code":
        return "coding"
    elif agent_output["action"] == "final_answer":
        return "end"
    else:
        # Default to supervisor if action is not recognized
        return "supervisor"

# Define the LangGraph graph
graph = StateGraph(AgentState)

graph.add_node("supervisor", agent_node(supervisor_agent))
graph.add_node("web_search", agent_node(web_search_agent))
graph.add_node("coding", agent_node(coding_agent))

# Add direct edges between the nodes
graph.add_edge("supervisor", "web_search")
graph.add_edge("supervisor", "coding")
graph.add_edge("web_search", "supervisor")
graph.add_edge("coding", "supervisor")

graph.set_entry_point("supervisor")

# The conditional edges are added directly to the nodes
graph.add_conditional_edges(
    "supervisor",  # Starting node
    route,
    {
        "web_search": "web_search",
        "coding": "coding",
        "end": END,
        "supervisor": "supervisor"
    }
)
graph.add_conditional_edges(
    "web_search",  # Starting node
    route,
    {
        "web_search": "web_search",
        "coding": "coding",
        "end": END,
        "supervisor": "supervisor"
    }
)
graph.add_conditional_edges(
    "coding",  # Starting node
    route,
    {
        "web_search": "web_search",
        "coding": "coding",
        "end": END,
        "supervisor": "supervisor"
    }
)

# Compile the graph
app = graph.compile()

# ----------------------------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------------------------

# Initialize the state
initial_state = {
    "messages": [
        HumanMessage(content="What is the capital of France?")
    ],
    "steps": 0
}

# Run the graph
results = app.invoke(initial_state)

# Print the final result
print("Final Result:")
print(results)