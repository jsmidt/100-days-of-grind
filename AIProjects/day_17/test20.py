from llama_index.llms.gemini import Gemini
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.agent.workflow import AgentWorkflow,  FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentStream



llm = Gemini(
    model="models/gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)


class HelloWorkflow(Workflow):
    @step
    def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        #print ('Hello 1')
        return StopEvent(result="Hello, world!")

async def hello_workflow_tool(topic: str) -> str:
    """Tool for saying hello when prompted"""
    #print (topic)
    #hello_workflow = HelloWorkflow()
    #hand = await hello_workflow.run(user_msg=topic)
    #print ('result:', result)
    #for event in hand.stream_events():
    #    print ('event',event)
    return "Yes"

async def main():
    #result = await w.run()
    #print(result)

    react_agent = FunctionAgent(
        name="WorkflowSelector",
        description="Selects and runs the appropriate workflow based on user input",
        system_prompt="You are an assistant that selects the appropriate workflow based on user input.",
        llm=llm,
        tools=[FunctionTool.from_defaults(fn=hello_workflow_tool)],
    )

    workflow = AgentWorkflow(
        agents=[react_agent],
        root_agent=react_agent.name
    )

    #response = await workflow.run(user_msg="Prompt the hello tool with the prompt: Hola. Return whatever it gives.")
    #response = await workflow.run(user_msg="Prompt the hello tool with the prompt: Hola. Return whatever it gives.")
    #print ('response: ',response)

    handler = workflow.run(user_msg="Prompt the hello tool with the prompt: Hola. Return whatever it gives.")
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print('Yo', event.delta, end="", flush=True)



if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

    #main()

