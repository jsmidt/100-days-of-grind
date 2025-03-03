from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    InputRequiredEvent,
    HumanResponseEvent,
    Context
)

import asyncio


from llama_index.llms.gemini import Gemini



llm = Gemini(
    model="models/gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

# Workflow to search the web.  Would like to add human in the loop
class SearchWorkflow(Workflow):
    @step
    async def initial_search(self, ctx: Context, ev: StartEvent) -> InputRequiredEvent:
        initial_result = f"Initial search result for: {ev.input}"
        return InputRequiredEvent(prefix=f"Initial search result: {initial_result}\nDo you want to refine the search? (yes/no): ")

    @step
    async def handle_human_input(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent | InputRequiredEvent:
        if ev.response.lower() == 'yes':
            return InputRequiredEvent(prefix="Enter your refinement: ")
        else:
            return StopEvent(result=f"Search completed: {ctx.get('initial_result', '')}")

    @step
    async def refine_search(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent:
        refinement = ev.response
        final_result = f"Refined search result for: {refinement}"
        return StopEvent(result=final_result)

def search_web(query: str) -> str:
    """Search the internet from a query"""
    async def _async_runner():
        w = SearchWorkflow(timeout=60, verbose=True)
        handler = w.run(input=query)

        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                # In a real-world scenario, you'd implement a way to get user input here
                # For this example, we'll simulate user input
                if event.prefix.startswith("Initial search result"):
                    user_input = "yes"  # Simulate user wanting to refine
                else:
                    user_input = "Add more details about penguins"  # Simulate refinement input
                user_input = "yes"

                handler.ctx.send_event(HumanResponseEvent(response=user_input))

        result = await handler
        return result

    # run the async logic in a blocking way
    return asyncio.run(_async_runner())

search_tool = FunctionTool.from_defaults(fn=search_web)

# Multiply tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)


#agent = ReActAgent.from_tools([multiply_tool, add_tool, search_tool], llm=llm, verbose=True)


class PausableReActAgent(ReActAgent):
    def init(self, args, **kwargs):
        super().init(args, **kwargs)
        self.paused_state = None

    async def achat(self, message: str, **kwargs):
        if self.paused_state:
            # Resume from paused state
            task = self.paused_state
            self.paused_state = None
        else:
            # Start a new chat
            task = self.create_task(message)

        async for event in self.astep_stream(task):
            if isinstance(event, InputRequiredEvent):
                # Pause the agent and return the event
                self.paused_state = task
                return event
            # Handle other events as needed

        # If we get here, the task is complete
        return self.finalize_response(task)

    def resume_with_input(self, user_input: str):
        if not self.paused_state:
            raise ValueError("Agent is not paused")

agent = PausableReActAgent.from_tools([search_tool], llm=llm, verbose=True)

async def chat_with_agent(message):
    response = await agent.achat(message)
    if isinstance(response, InputRequiredEvent):
        # Return to the GUI for user input
        return response.prefix
    else:
        # Return the final response
        return str(response)

async def submit_user_input(user_input):
    response = await agent.resume_with_input(user_input)
    if isinstance(response, InputRequiredEvent):
        # Still need more input
        return response.prefix
    else:
        # Task is complete
        return str(response)


#response = agent.chat("What is 20+(2*4)? Calculate step by step ")
#print (response)

#response = agent.chat("Search the web using your tool for penguins.")
#print (response)



from llama_index.packs.gradio_agent_chat import GradioAgentChatPack
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini

llm = Gemini(model="models/gemini-2.0-flash")

def search_web(query: str) -> str:
    """Search the internet from a query"""
    return f"Here are the search results for: {query}"

#search_tool = FunctionTool.from_defaults(fn=search_web)

#agent = ReActAgent.from_tools([search_tool], llm=llm, verbose=True)

gradio_pack = GradioAgentChatPack(
    agent=agent,
    env_name="My Chatbot",
    description="This is a chatbot that can search the web.",
)
gradio_pack.run()

