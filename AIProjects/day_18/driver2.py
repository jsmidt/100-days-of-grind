
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import AgentStream


from dotenv import load_dotenv
load_dotenv()

llm = Gemini(
    model="models/gemini-1.5-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

#response_gen = agent.stream_chat("What is 20+2*4? Calculate step by step")
#response_gen.print_response_stream()

async def main():
    response = agent.chat("What's the weather like in San Francisco?")

    print (response)
    
    #async for event in handler.stream_events():
    #    if isinstance(event, AgentStream):
    #        print(event.delta, end="", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
