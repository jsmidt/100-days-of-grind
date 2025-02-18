# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel

from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(model_id="gemini/gemini-2.0-flash") # Could use 'gpt-4o'
'''
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)
'''

agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
