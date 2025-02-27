import json
import asyncio

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)

from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent


from pydantic import BaseModel, Field

from llama_index.llms.gemini import Gemini

from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

from rich.markdown import Markdown
from rich.console import Console


from dotenv import load_dotenv

load_dotenv();


class Joke(BaseModel):
    """Data model for joke and notes why it's funny."""
    joke: str
    notes: str


class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):
    llm = Gemini(model="models/gemini-2.0-flash",)

    notes = {}

    @step
    async def do_research(self, ctx: Context, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        await ctx.set('notes',[ev.topic])

        prompt = f"Write your best joke about {topic}.  Then take some notes why the joke is funny."
        sllm = self.llm.as_structured_llm(output_cls=Joke)
        response = await sllm.acomplete(prompt)
        data = json.loads(response.text)
        self.notes['generate_joke'] = data['notes']

        # Return joke event and write to stream
        joke_event = JokeEvent(joke=data['joke'])
        ctx.write_event_to_stream(joke_event)
        return joke_event

    @step
    async def write_report(self, ctx: Context, ev: JokeEvent) -> StopEvent:
        joke = ev.joke
        notes = ''

        #for key, value in self.notes.items():
        #    print (f"Agent {key} noted:\n\n{value}")
        #    notes.join(f"Agent {key} noted:\n\n{value}")

        notes = "".join(f"Agent {key} noted:\n\n{value}" for key, value in self.notes.items())

        #print ('Notes: ', notes)
        
        prompt = f"Give a thorough analysis and critique of the following joke: {joke}. Also consider using these notes to aid in your analysis: {notes}. Write a detailed report in Markdown."
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))




async def main():
    console = Console()
    w = JokeFlow( verbose=False)
    handler = w.run(topic="pirates")

    async for ev in handler.stream_events():
        if isinstance(ev, JokeEvent):
            print(ev.joke)

    final_result = await handler
    console.print( Markdown(str(final_result)))
    #draw_all_possible_flows(JokeFlow, filename="streaming_workflow.html")


if __name__ == "__main__":
    asyncio.run(main())