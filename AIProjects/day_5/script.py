import argparse
import os
import threading

from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import (
    CodeAgent,
    HfApiModel,
    LiteLLMModel,
    ToolCallingAgent,
    VisitWebpageTool,
    DuckDuckGoSearchTool,
    GradioUI,
)
import os


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "matplotlib",
    "seaborn",
    "IPython",
]


text_limit = 100000

os.environ["GEMINI_API_KEY"]  = os.environ["GOOGLE_API_KEY"] 
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
model = LiteLLMModel(
    #"gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-lite-preview-02-05",
    custom_role_conversions=custom_role_conversions,
    max_completion_tokens=8192,
)

#model = HfApiModel()
print (model)

text_webbrowser_agent = ToolCallingAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    max_steps=20,
    verbosity_level=2,
    planning_interval=4,
    name="search_agent",
    description="""A team member that will search the internet to answer your question.
Ask him for all your questions that require browsing the web.
Provide him as much context as possible, in particular if you need to search on a specific timeframe!
And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
""",
    provide_run_summary=True,
)
text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

manager_agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    max_steps=25,
    verbosity_level=2,
    additional_authorized_imports=AUTHORIZED_IMPORTS,
    planning_interval=4,
    managed_agents=[],
)

GradioUI(manager_agent).launch()
