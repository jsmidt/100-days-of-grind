
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.gemini import Gemini
import asyncio

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


llm = Gemini(
    model="models/gemini-2.0-flash",
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

class GreetingWorkflow(Workflow):
    @step
    def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        #print ('Hello 1')
        return StopEvent(result="Hello, world!")


greeting_workflow = GreetingWorkflow()

async def greeting(topic: str) -> str:
    """Tool for saying a greeting"""
    #print (topic)
    #hello_workflow = HelloWorkflow()
    #hand = await hello_workflow.run(user_msg=topic)
    #print ('result:', result)
    #for event in hand.stream_events():
    #    print ('event',event)
    #result = await greeting_workflow.run(user_msg=topic)
    #print ('topic:', topic)
    #print ('topic:', result)
    return "Yes"

greeting_tool = FunctionTool.from_defaults(fn=greeting)


agent = ReActAgent.from_tools([multiply_tool, add_tool, greeting_tool], llm=llm, verbose=True)

'''
response_gen = agent.stream_chat("What is 20+2*4? Calculate step by step")
response_gen.print_response_stream()

response_gen = agent.stream_chat("Prompt the greeting tool with Hola, and return the greeting")
response_gen.print_response_stream()
'''


async def main():
    agent = ReActAgent.from_tools([multiply_tool, add_tool, greeting_tool], llm=llm, verbose=True)
    response_gen = await agent.astream_chat("Prompt the greeting tool with Hola, and return the greeting")
    async for event in response_gen.async_response_gen():
        print(event)

if __name__ == "__main__":
    asyncio.run(main())