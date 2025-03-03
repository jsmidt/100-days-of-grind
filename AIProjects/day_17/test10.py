from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.gemini import Gemini
import gradio as gr
from gradio import ChatMessage

llm = Gemini(model="models/gemini-2.0-flash")
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
agent = ReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

messages = []
async def predict(prompt, history):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages    
    messages.append(ChatMessage(role="assistant",content=''))
    full_response = ""
    response = agent.stream_chat(message=prompt)
    for token in response.response_gen:
        full_response += token
        messages[-1] = ChatMessage(role="assistant",content=full_response)
        yield messages

demo = gr.ChatInterface(
    predict,
    type="messages"
)

demo.launch()