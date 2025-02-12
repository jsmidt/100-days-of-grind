from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults

import json

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            print ('*** Tool name:', tool_call["name"])
            print ('*** Tool args:', tool_call["args"])
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


search = DuckDuckGoSearchResults()
tools = [search]


llm = ChatOllama(
    model="llama3.3",
    temperature=0,
    # other params...
)
llm = llm.bind_tools(tools)

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def chatbot(state: State):
    print ('##########\nThis is what is passed to the LLM Start:\n')
    print (state)
    for m in state["messages"]:
        #try:
        #    print ("*** TOOL CALL ***:", m.tool_calls)
        #except:
        #    pass
        m.pretty_print()
    print ('\nThat was is passed to the LLM Stop.\n##########\n')
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

def stream_graph_updates(user_input: str):
    sys_message = SystemMessage('Always responds like a pirate.')
    for event in graph.stream({"messages": [sys_message, HumanMessage(user_input)]}):
        print ('$$$$$$$$$$\nThis is what was returned from the LLM Start:\n')
        print (event)
        for value in event.values():
            
            for m in value["messages"]:
                #try:
                #    print ("*** TOOL CALL ***:", m.tool_calls)
                #except:
                #    pass
                print (m.pretty_print())
            print ('This is what was returned from the LLM Stop.\n$$$$$$$$$$\n')


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break