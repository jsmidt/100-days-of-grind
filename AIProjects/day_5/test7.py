#############################
# langgraph_app.py
#############################

from langgraph import Node
from langgraph.prebuilt import create_react_agent
from langgraph.llms.google.generationai import ChatGoogleGenerativeAI
from typing import Any, Dict

# ----------------------------------------------------------------------------
# Set up the Gemini llm
# ----------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-preview-02-05",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # add any other parameters you'd like...
)

# ----------------------------------------------------------------------------
# Create each agent
# ----------------------------------------------------------------------------

# The Supervisor agent (e.g., a coordinator that decides where to route the request)
supervisor_agent = create_react_agent(
    llm=llm,
    # You can provide a prompt prefix or special instructions as needed
    prompt_prefix="You are the supervisor agent. Decide if we should do web search, code generation, or handle it directly.\n\n",
)

# A Web-Search agent (assuming you have the search tools set up)
web_search_agent = create_react_agent(
    llm=llm,
    # Possibly configured with some web search tool or plugin
    prompt_prefix="You are the web search agent. You have access to a search tool.\n\n",
    # tools=[my_search_tool]  # if you have a tool set up
)

# A Coding agent
coding_agent = create_react_agent(
    llm=llm,
    # Possibly configured with some code execution tool
    prompt_prefix="You are the coding agent. Generate or fix code.\n\n",
    # tools=[my_code_tool]  # if you have a tool set up
)

# ----------------------------------------------------------------------------
# Helper condition functions
# ----------------------------------------------------------------------------

def wants_web_search(inputs: Dict[str, Any], **kwargs) -> bool:
    """Check if user wants to do a web search."""
    user_text = inputs.get("text", "").lower()
    return "search" in user_text or "find" in user_text

def wants_coding(inputs: Dict[str, Any], **kwargs) -> bool:
    """Check if user is asking for code or programming help."""
    user_text = inputs.get("text", "").lower()
    return "code" in user_text or "program" in user_text

def wants_supervisor_back(inputs: Dict[str, Any], **kwargs) -> bool:
    """Check if we want to bounce back to the supervisor (e.g. done searching or coding)."""
    user_text = inputs.get("text", "").lower()
    # Arbitrarily, if the user says 'done' or 'back', we return to the supervisor
    return "done" in user_text or "back" in user_text

# ----------------------------------------------------------------------------
# Create the graph nodes
# ----------------------------------------------------------------------------
supervisor_node = Node(
    name="SupervisorNode",
    agent=supervisor_agent,
    prompt_template=lambda inputs: inputs["text"],  # Simple pass-through for demonstration
)

web_search_node = Node(
    name="WebSearchNode",
    agent=web_search_agent,
    prompt_template=lambda inputs: inputs["text"],
)

coding_node = Node(
    name="CodingNode",
    agent=coding_agent,
    prompt_template=lambda inputs: inputs["text"],
)

# ----------------------------------------------------------------------------
# Set up the conditional edges
# ----------------------------------------------------------------------------
# From the supervisor node, route to web search or coding
supervisor_node.add_edge(
    condition=wants_web_search,
    target=web_search_node
)
supervisor_node.add_edge(
    condition=wants_coding,
    target=coding_node
)
# (If none of those conditions match, we could simply stay in the supervisor node itself,
# or you could add a fallback that just uses the supervisor agent to handle the request.)

# Web search node can route back to the supervisor
web_search_node.add_edge(
    condition=wants_supervisor_back,
    target=supervisor_node
)

# Coding node can route back to the supervisor
coding_node.add_edge(
    condition=wants_supervisor_back,
    target=supervisor_node
)

# ----------------------------------------------------------------------------
# Compile the LangGraph app
# ----------------------------------------------------------------------------
app = LangGraphApp(
    nodes=[
        supervisor_node,
        web_search_node,
        coding_node,
    ],
    start_node=supervisor_node  # We'll start at the supervisor
)

# If you're running this as a script, you might have a simple CLI like:
if __name__ == "__main__":
    # Quick example of how you could run a loop
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting LangGraph app.")
            break
        response = app.run({"text": user_input})
        print(f"App: {response}")