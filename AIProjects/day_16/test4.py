from typing import Annotated
import asyncio
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk, ToolMessage, SystemMessage

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


from gradio import ChatMessage





class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

search_tool = TavilySearchResults(max_results=2)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


tools = [search_tool, human_assistance]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Note to chatbot
graph_builder.add_node("chatbot", chatbot)

# Note to tools
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot",tools_condition,)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# Define graph and a thread config
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "2"}}

# Update the system prompt
system_prompt ="""
Always talk like a pirate
"""
graph.update_state(config, {"messages": [SystemMessage(system_prompt)]})



async def interact_with_stream(prompt, history):
    messages = []
    yield messages
    messages.append('')
    partial_message = ''
    my_prompt = None

    #snapshot = graph.get_state(config)

    #print (graph.get_state(config))
    state = next(iter(graph.get_state_history(config)))
    for task in state.tasks:
        if task.interrupts:
            for interrupt in task.interrupts:
                my_prompt = Command(resume={"data": prompt})
    if my_prompt is None:
        my_prompt = {"messages": [{"role": "user", "content": prompt}]}

    async for event_type, event in graph.astream(my_prompt,config,stream_mode=["updates", "messages"]):
        
        # If the messages are streamiing:
        if event_type == 'messages':
            for chunk in event:
                if isinstance(chunk, AIMessageChunk):
                    partial_message += chunk.content
                    messages[-1] = partial_message
                    yield messages
        
        # If the messages are not streaming, like from tools
        else:
            for event_key, event_values in event.items():
                if event_key == '__interrupt__':
                    messages.append(event_values[0].value["query"])
                    yield messages
                else:
                    for msgs in event_values.values():
                        for msg in msgs:

                            # If a tool call, name it
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    messages.append(ChatMessage(role="assistant", content=f'Calling tool {tool_call.get('name')}. This may take a minute...\n'))
                                    yield messages
                                    messages.append('')
                                    partial_message = ''

                            elif isinstance(msg,ToolMessage):
                                messages.append(ChatMessage(role="assistant", content=msg.content, metadata={"title": f"🛠️ Output from tool {msg.name}."}))
                                yield messages
                                messages.append('')
                                partial_message = ''
            


demo = gr.ChatInterface(
    interact_with_stream,
    chatbot= gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
    )
)

'''
gui=True
if __name__ == '__main__':
    if gui:
        demo.launch()
    else:
        asyncio.run(main())
'''




demo.launch()


