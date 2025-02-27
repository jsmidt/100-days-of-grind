import json
import asyncio
import sys

from gradio import ChatMessage
import gradio as gr
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)


from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent

from pydantic import BaseModel

from llama_index.llms.gemini import Gemini


from rich.markdown import Markdown
from rich.console import Console


from dotenv import load_dotenv
load_dotenv()


class Joke(BaseModel):
    """Data model for joke and notes why it's funny."""
    joke: str
    notes: str

class ResearchEvent(Event):
    joke: str

class BetterPromptEvent(Event):
    better_prompt: str

class ResearchFlow(Workflow):
    llm = Gemini(model="models/gemini-2.0-flash")

    notes = {}

    @step
    async def step1(self, ctx: Context, ev: StartEvent) -> InputRequiredEvent:
        topic = ev.topic
        await ctx.set('topic',topic)
        prompt = (f"You been asked to write a joke on {topic}.  Before you do, you will do well "
                  f"to have more information. Ask the user three questions that if answered would "
                  f"help you write a better joke.")
        response = await self.llm.acomplete(prompt)
        print (str(response))
        await ctx.set('questions',str(response))
        return InputRequiredEvent(prefix=str(response))

    @step
    async def better_prompt(self, ctx: Context, ev: HumanResponseEvent) -> BetterPromptEvent:
        answers= ev.response
        questions = await ctx.get('questions')
        topic = await ctx.get('topic')

        print ('topic: ', topic)
        print ('questions: ', questions)
        print ('response: ', answers)
        answers = str(answers)
        prompt = (f"You are an expert prompt writing assistant. A user wants to write a joke "
                  f"about {topic}. To give more information, the user was as these questions: "
                  f"{questions}.  The user provided these answers: {str(answers)}. Use this "
                  f"information to write a much better prompt for generating a joke.")
        response = await self.llm.acomplete(prompt)
        return BetterPromptEvent(better_prompt=str(response))


    @step
    async def do_research(self, ctx: Context, ev: BetterPromptEvent) -> ResearchEvent:
        
        prompt = ev.better_prompt + '\n\nAlso, take some notes on why this joke is funny.'
        print (prompt)
        sllm = self.llm.as_structured_llm(output_cls=Joke)
        response = await sllm.acomplete(prompt)
        data = json.loads(response.text)
        self.notes['generate_joke'] = data['notes']

        # Return joke event and write to stream
        research_event = ResearchEvent(joke=data['joke'])
        ctx.write_event_to_stream(research_event)
        return research_event

    @step
    async def write_report(self, ctx: Context, ev: ResearchEvent) -> StopEvent:
        joke = ev.joke
        notes = "".join(f"Agent {key} noted:\n\n{value}" for key, value in self.notes.items())
        
        prompt = (f"Give a thorough analysis and critique of the following joke: {joke}. "
                  f"Also consider using these notes to aid in your analysis: {notes}. "
                  f"Write a detailed report in Markdown.")
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))
    

console = Console()
w = ResearchFlow(timeout=600, verbose=True)

async def main():
    handler = w.run(topic="pirates")

    async for ev in handler.stream_events():
        if isinstance(ev, ResearchEvent):
            print(ev.joke)
        elif isinstance(ev, InputRequiredEvent):
            # here, we can handle human input however you want
            # this means using input(), websockets, accessing async state, etc.
            # here, we just use input()
            response = input(ev.prefix)
            handler.ctx.send_event(HumanResponseEvent(response=response))

    final_result = await handler
    console.print( Markdown(str(final_result)))




async def interact_with_langchain_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages

    handler = w.run(topic=prompt)
    async for ev in handler.stream_events():
        if isinstance(ev, ResearchEvent):
            messages.append(ChatMessage(role="assistant", content=ev.joke,
                                metadata={"title": "🛠️ Used tool Joke"}))
            yield messages
        elif isinstance(ev, InputRequiredEvent):
            # here, we can handle human input however you want
            # this means using input(), websockets, accessing async state, etc.
            # here, we just use input()
            response = input(ev.prefix)
            messages.append(ChatMessage(role="user", content=str(response)))
            yield messages
            handler.ctx.send_event(HumanResponseEvent(response=response))



    final_result = await handler
    messages.append(ChatMessage(role="assistant", content=str(final_result)))
    yield messages
    

with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    input_box = gr.Textbox(lines=1, label="Chat Message")
    input_box.submit(interact_with_langchain_agent, [input_box, chatbot], [chatbot])


if __name__ == "__main__":
    if sys.argv[-1] == 'gui':
        demo.launch()
    else:
        asyncio.run(main())
        