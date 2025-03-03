from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool



llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


###
#
# Two
#
###

class State2(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot2(state: State2):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder2 = StateGraph(State2)

graph_builder2.add_node("chatbot2", chatbot2)
graph_builder2.set_entry_point("chatbot2")
graph_builder2.set_finish_point("chatbot2")
graph2 = graph_builder2.compile()

@tool
def get_weather(location: str) -> str:
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊

    
    prompt = "Tell me that it is sunny in San diego"
    my_prompt = {"messages": [{"role": "user", "content": prompt}]}

    #print (my_prompt)
    
    msg = []
    for event_type, event in graph2.stream(my_prompt,stream_mode=["updates"]):
        #print ('event:', event)
        for k, v in event.items():
            for msgs in v:
                msg.append(msgs.content)
                print (' - ', msgs.content)
            #    for msg in msgs:
            #        print (msg.content)
    
    return msg[-1]

#model = model.bind_tools(tools)

###
#
# One
#
###

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool, get_weather]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

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

prompt = "Get the weather in San Diego"
my_prompt = {"messages": [{"role": "user", "content": prompt}]}

for event_type, event in graph.stream(my_prompt,stream_mode=["updates"]):
    for k, v in event.items():
        for msgs in v.values():
            for msg in msgs:
                print ('Yo:', msg.content)