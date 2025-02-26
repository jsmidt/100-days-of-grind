from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.gemini import Gemini
import asyncio



# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = FunctionAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant.",
    tools=[
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),
    ],
    llm = Gemini(model="models/gemini-2.0-flash"),
)



retriever_agent = FunctionAgent(
    name="retriever",
    description="Manages data retrieval",
    system_prompt="You are a retrieval assistant.",
    llm=Gemini(model="models/gemini-2.0-flash"),
)

# Create and run the workflow
workflow = AgentWorkflow(
    agents=[calculator_agent, retriever_agent], root_agent="calculator"
)



#  Or stream the events
async def main():

    # Run the system
    #response = await workflow.run(user_msg="Can you add 5 and 3?")
    handler = workflow.run(user_msg="Can you add 5 and 3?")
    async for event in handler.stream_events():
        if hasattr(event, "delta"):
            print(event.delta, end="", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
