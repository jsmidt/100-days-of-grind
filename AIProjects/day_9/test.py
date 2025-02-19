from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

llm = Gemini(model="models/gemini-2.0-flash")
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)