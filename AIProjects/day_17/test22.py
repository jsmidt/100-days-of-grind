
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.gemini import Gemini
import asyncio

llm = Gemini(
    model="models/gemini-1.5-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

async def greeting(topic: str) -> str:
    """Tool for saying a greeting"""
    return "Yes"

greeting_tool = FunctionTool.from_defaults(async_fn=greeting)

async def main():
    agent = ReActAgent.from_tools([greeting_tool], llm=llm, verbose=True)
    try:
        response_gen = await agent.astream_chat("Prompt the greeting tool with Hola, and return the greeting")
        async for event in response_gen.async_response_gen():
            print(event)
    finally:
        # Explicitly close the Gemini client
        #llm.close()
        exit()
        #pass

if __name__ == "__main__":
    #asyncio.run(main())
    #exit()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
