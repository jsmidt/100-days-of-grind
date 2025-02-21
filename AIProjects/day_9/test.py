from smolagents import CodeAgent, GradioUI, HfApiModel, LiteLLMModel

AUTHORIZED_IMPORTS = [
    "numpy",
    "matplotlib",
    "seaborn",
    "PIL",
    "io"
]

agent = CodeAgent(tools=[], model=HfApiModel(), max_steps=20, verbosity_level=2,additional_authorized_imports=AUTHORIZED_IMPORTS)

GradioUI(agent).launch()
