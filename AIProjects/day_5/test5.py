from smolagents import CodeAgent, GradioUI, HfApiModel, LiteLLMModel

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
]

import os
os.environ["GEMINI_API_KEY"]  = os.environ["GOOGLE_API_KEY"] 

print ("Hello!")

# Let's setup the instrumentation first

# Then we run the agentic part!
#model = HfApiModel()
#model = LiteLLMModel(model_id="gemini/gemini-2.0-flash")

agent = CodeAgent(tools=[], model=HfApiModel(), max_steps=20, verbosity_level=2,additional_authorized_imports=AUTHORIZED_IMPORTS)

GradioUI(agent).launch()
