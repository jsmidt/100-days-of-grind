from typing import Annotated
import asyncio
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langchain_core.messages import AIMessageChunk

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

async def interact_with_stream(prompt, history):
    messages = []
    yield messages
    #async for event in graph.astream({"messages": [{"role": "user", "content": "Tell me about Albert Einstien."}]},stream_mode=["updates", "messages"]):
    messages.append('')
    partial_message = ''
    async for event_type, event in graph.astream({"messages": [{"role": "user", "content": prompt}]},stream_mode=["updates", "messages"]):
        if event_type == 'messages':
            for chunk in event:
                if isinstance(chunk, AIMessageChunk):
                    #messages.append(chunk.content)
                    partial_message += chunk.content
                    print 
                    messages[-1] = partial_message
                    yield messages
            
'''
if __name__ == '__main__':
    asyncio.run(main())
'''

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

demo.launch()
