from smolagents import CodeAgent, GradioUI, HfApiModel, LiteLLMModel

AUTHORIZED_IMPORTS = [
    "numpy",
    "matplotlib",
    "seaborn",
]

agent = CodeAgent(tools=[], model=HfApiModel(), max_steps=20, verbosity_level=2,additional_authorized_imports=AUTHORIZED_IMPORTS)

GradioUI(agent).launch()

    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
