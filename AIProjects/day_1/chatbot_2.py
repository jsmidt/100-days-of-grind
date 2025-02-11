from typing import Annotated

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition



class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


search = DuckDuckGoSearchResults()
tools = [search]
llm = ChatOllama(
    model="llama3.3",
    temperature=0,
    # other params...
)
llm = llm.bind_tools(tools)


def chatbot(state: State):
    #for m in state["messages"]:
    #    m.pretty_print()
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    sys_message = SystemMessage('Always responds like a pirate.')
    for event in graph.stream({"messages": [sys_message, HumanMessage(user_input)]}):
        #for e in event:
        #    print (e)
        for value in event.values():
            #print ('-----\n')
            #print (value)
            #print ('\n-----')
            #print("Assistant:", value["messages"][-1].content)
            value["messages"][-1].pretty_print()


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

