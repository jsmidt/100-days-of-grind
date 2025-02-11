from typing import Annotated

from langchain_ollama import ChatOllama
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    # other params...
)


def chatbot(state: State):
    #for m in state["messages"]:
    #    m.pretty_print()
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
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

