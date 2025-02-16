# First we initialize the model we want to use.
import gradio as gr


from typing import Literal

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


st.set_page_config(layout="wide")           

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-preview-02-05",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# For this tutorial we will use custom tool that returns pre-defined values for weather in two cities (NYC & SF)



@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]


# Define the graph


graph = create_react_agent(llm, tools=tools)


message = st.chat_input("Type message here", on_submit=submit_message, key="message")
#inputs = {"messages": [("user", "what is the weather in sf")]}
for event in graph.stream(inputs, stream_mode="values"):
    print ('\n-----\n')
    message = event["messages"][-1]
    st.write(message.content)
    message.pretty_print()


'''
def chat_with_agent(history):
    """
    Takes chat history as input, sends user message to the agent, and streams responses.
    """
    # Extract latest user message
    user_message = history[-1][0] if history else "Hello!"

    # Prepare input for LangGraph agent
    inputs = {"messages": [("user", user_message)]}

    # Stream responses from the agent
    bot_response = ""
    for event in graph.stream(inputs, stream_mode="values"):
        message = event["messages"][-1]  # Get latest message
        bot_response = message.content  # Extract the text

        yield [(user_message, bot_response)]  # Stream updates to UI
        #yield [message.content]

# Create Gradio Chatbot Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 LangGraph Chatbot")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message:")
    clear = gr.Button("Clear Chat")

    def user(user_message, history):
        return history + [(user_message, None)]  # Append user message

    msg.submit(user, [msg, chatbot], chatbot, queue=False).then(
        chat_with_agent, chatbot, chatbot
    )

    clear.click(lambda: [], None, chatbot, queue=False)

# Launch the chatbot app
demo.launch()


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]


# Define the graph

graph = create_react_agent(llm, tools=tools)

system_message = "You are a helpful assistant who acts like a pirate."

def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_message))

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    if message is not None:
        history_langchain_format.append(HumanMessage(content=message))
        partial_message = ""
        #for response in llm.stream(history_langchain_format):
        #    partial_message += response.content
        #    yield partial_message

        partial_message = ""
        for event in graph.stream(history_langchain_format, stream_mode="values"):  # Stream LangGraph response
            last_message = event["messages"][-1]  # Get the latest response
            partial_message = last_message.content  # Extract text
            yield partial_message  # Send updates to Gradio UI


demo_interface = gr.ChatInterface(

    stream_response,
    textbox=gr.Textbox(placeholder="Send to the LLM...",
                       container=False,
                       autoscroll=True,
                       scale=7),
)

demo_interface.launch(share=True, debug=True)
'''