from typing import Literal
from typing_extensions import TypedDict


from langchain import hub
from typing import Annotated

from gradio import ChatMessage
import gradio as gr
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import MessagesState, END
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-preview-02-05",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

search_tool = DuckDuckGoSearchResults()

repl = PythonREPL()


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]


graph = create_react_agent(llm, tools=tools)


#print(sys_prompt)
lg_messages = []#SystemMessage(sys_prompt)]

async def interact_with_langchain_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    lg_messages.append(HumanMessage(prompt))

    async for event in graph.astream({"messages": lg_messages}):
        for e in event.values():
            msg = e["messages"][-1]

            lg_messages.append(msg)
            if len(msg.content) < 2:
                continue
            elif isinstance(msg, ToolMessage):
                messages.append(ChatMessage(role="assistant", content=msg.content, metadata={"title": f"🛠️  From tool {msg.name}:"}))
            else:
                messages.append(ChatMessage(role="assistant", content=msg.content))
            yield messages




with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
     # Function to clear chat history
    def clear_chat():
        return []

    with gr.Row():
        input_box = gr.Textbox(lines=1, label="Chat Message",submit_btn=True, scale=10)
        clear_btn = gr.Button("Clear Chat")  # Button to reset history

    # Clear chat when clicking "Clear Chat"
    clear_btn.click(clear_chat, [], chatbot)
    input_box.submit(interact_with_langchain_agent, [input_box, chatbot], [chatbot])

demo.launch()
