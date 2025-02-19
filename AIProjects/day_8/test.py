from llama_index.llms.gemini import Gemini
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows


llm = Gemini(
    model="models/gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

#resp = llm.complete("Write a poem about a magic backpack")
#print(resp)

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)


if __name__ == "__main__":
    import asyncio

    draw_all_possible_flows(MyWorkflow, filename="basic_workflow.html")
    asyncio.run(main())
