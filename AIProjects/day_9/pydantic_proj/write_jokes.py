from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
import asyncio


# Output for joke agent
class JokeOutput(BaseModel):
    joke_setup: str = Field(description='The joke setup') 
    joke_punchline: str = Field(description='The joke punchline') 

# Output from better prompt agent
class BetterPrompt(BaseModel):
    better_joke_prompt: str = Field(description='A better prompt for generating a high quality joke') 


class State(BaseModel):
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


prompt_writer_agent = Agent(
    'google-gla:gemini-2.0-flash-001',
    result_type=BetterPrompt,
    system_prompt='You are an agent reciving a prompt from a user to generate a joke. You are not to generate teh joke, but to improve the prompt so that the joke making agent can write a hilarous joke.',
)

joke_writer_agent = Agent(
    'google-gla:gemini-2.0-flash-001',
    result_type=JokeOutput,
    system_prompt='You are a world class joke writer. Write a hilarous joke based on the prompt.',
)



@dataclass
class WritePrompt(BaseNode[State]):

    async def run(self, ctx: GraphRunContext[State]) -> WriteJoke:

        prompt = 'Write a joke about turtles'
        result = await prompt_writer_agent .run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        print ('Here1')
        ctx.state.write_agent_messages += result.all_messages()
        return WriteJoke(result.data)



@dataclass
class WriteJoke(BaseNode[State, None, BetterPrompt]):
    bprompt: BetterPrompt

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> End[JokeOutput]:
        print ('Here2')
        print (self.bprompt)
        result = await joke_writer_agent.run(self.bprompt.better_joke_prompt)
        #print (result)

        return End(result.data)


async def main():
    state = State()
    feedback_graph = Graph(nodes=(WritePrompt, WriteJoke))
    feedback_graph.mermaid_image(start_node=WritePrompt,image_type='png')
    feedback_graph.mermaid_save('mygraph.png')
    joke, _ = await feedback_graph.run(WritePrompt(), state=state)
    print ()
    print(f' * Joke setup: {joke.joke_setup}')
    print(f' * Joke punchline: {joke.joke_punchline}')
    """
    Email(
        subject='Welcome to our tech blog!',
        body='Hello John, Welcome to our tech blog! ...',
    )
    """




'''
async def main():
    result = await prompt_writer_agent.run('Write a joke about zebras')
    for m in result.all_messages():
        for p in m.parts:
            print ()
            print (p)
    print (result.data.better_joke_prompt)
'''
asyncio.run(main())