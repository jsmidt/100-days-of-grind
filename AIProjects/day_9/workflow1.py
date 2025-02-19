from llama_index.llms.gemini import Gemini
from tavily import AsyncTavilyClient
from llama_index.core.agent.workflow import AgentWorkflow
import asyncio
from rich.markdown import Markdown
from rich.console import Console
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.core.workflow import (
    Context,
    InputRequiredEvent,
    HumanResponseEvent,
)

llm = Gemini(
    model="models/gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)


async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient()
    return str(await client.search(query))


workflow = AgentWorkflow.from_tools_or_functions(
    [search_web],
    llm=llm,
    system_prompt="You are a helpful assistant that can search the web for information.",
)


async def main():

    async def dangerous_task(ctx: Context) -> str:
        """A dangerous task that requires human confirmation."""
        ctx.write_event_to_stream(
            InputRequiredEvent(
                prefix="Are you sure you want to proceed?",
                user_name="Logan",
            )
        )

        response = await ctx.wait_for_event(
            HumanResponseEvent, requirements={"user_name": "Logan"}
        )
        if response.response == "yes":
            return "Dangerous task completed successfully."
        else:
            return "Dangerous task aborted."


    workflow = AgentWorkflow.from_tools_or_functions(
        [dangerous_task],
        llm=llm,
        system_prompt="You are a helpful assistant that can perform dangerous tasks.",
    )


    handler = workflow.run(user_msg="I want to proceed with the dangerous task.")

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            response = input(event.prefix).strip().lower()
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    user_name=event.user_name,
                )
            )

    response = await handler
    print(str(response))

    '''
    async def set_name(ctx: Context, name: str) -> str:
        state = await ctx.get("state")
        state["name"] = name
        await ctx.set("state", state)
        return f"Name set to {name}"


    workflow = AgentWorkflow.from_tools_or_functions(
        [set_name],
        llm=llm,
        system_prompt="You are a helpful assistant that can set a name.",
        initial_state={"name": "unset"},
    )

    ctx = Context(workflow)

    response = await workflow.run(user_msg="My name is Logan", ctx=ctx)
    print(str(response))

    state = await ctx.get("state")
    print(state["name"])
    
    handler = workflow.run(user_msg="What is the weather in Saskatoon?")

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            #print(event.delta, end="", flush=True)
            #print ('event:',event)
            print(event.delta, end="", flush=True)

            #print(event.response)  # the current full response
            #print(event.raw)  # the raw llm api response
            #print(event.current_agent_name)  # the current agent name
        # elif isinstance(event, AgentInput):
        #    print(event.input)  # the current input messages
        #    print(event.current_agent_name)  # the current agent name
        # elif isinstance(event, AgentOutput):
        #    print(event.response)  # the current full response
        #    print(event.tool_calls)  # the selected tool calls, if any
        #    print(event.raw)  # the raw llm api response
        elif isinstance(event, ToolCallResult):
             print('* ToolCallResult:', event.tool_name)  # the tool name
        #    print(event.tool_kwargs)  # the tool kwargs
             print('* ToolCalloutput:', event.tool_output)  # the tool output
        elif isinstance(event, ToolCall):
             print('* Calling tool name:', event.tool_name)  # the tool name
             print('* Calling tool as:', event.tool_kwargs)  # the tool kwargs
        '''

    '''
    ctx = Context(workflow)
    #response = await workflow.run(user_msg="What is the weather in San Francisco? Please be detailed and return with markdown")

    response = await workflow.run(
        user_msg="My name is Logan, nice to meet you!", ctx=ctx
    )
    console = Console()
    console.print(Markdown(response.response.content))

    ctx_dict = ctx.to_dict(serializer=JsonSerializer())

    print (ctx_dict)
    '''




if __name__ == "__main__":

    asyncio.run(main())

