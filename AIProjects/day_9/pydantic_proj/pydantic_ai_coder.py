from pydantic_ai import Agent
from pydantic import BaseModel, Field
from graph_example import DivisibleBy5, fives_graph


class JokeOutput(BaseModel):
    joke_setup: str = Field(description='The joke setup') 
    joke_punchline: str = Field(description='The joke punchline') 


pydantic_ai_coder = Agent(
    'google-gla:gemini-2.0-flash-001',
    system_prompt=('You are an expert joke writer. You write jokes based on the prompt give. '  
                   'These jokes always have a funny setup and punchline.'
                   ),
    result_type=JokeOutput,
    retries=2
)

result = pydantic_ai_coder.run_sync('Tell me a joke about zebras?')


print(result.data)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""

'''
pydantic_ai_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)
'''
