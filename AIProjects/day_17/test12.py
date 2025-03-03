from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.gemini import Gemini
import gradio as gr
from gradio import ChatMessage
from llama_index.core.base.llms.types import ChatResponse
#from llama_index.core.agent.types import ActionReasoningStep, ObservationReasoningStep
from typing import List, Sequence
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.base.memory.types import BaseMemory
from llama_index.core.agent.types import AgentOutput
from llama_index.core.agent.types import Context

class CustomReActAgent(ReActAgent):
    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        output = await super().take_step(ctx, llm_input, tools, memory)
        reasoning_steps = await ctx.get(self.reasoning_key, default=[])
        output.reasoning_steps = reasoning_steps
        return output

llm = Gemini(model="models/gemini-2.0-flash")

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b
multiply_tool = FunctionTool.from_defaults(fn=multiply)
agent = CustomReActAgent.from_tools([multiply_tool], llm=llm, verbose=True)

messages = []
async def predict(prompt, history):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    messages.append(ChatMessage(role="assistant", content=''))
    full_response = ""
    response = agent.stream_chat(message=prompt)

    async for step_output in response.async_response_gen():
        if isinstance(step_output, AgentOutput):
            for step in step_output.reasoning_steps:
                if isinstance(step, ActionReasoningStep):
                    step_content = f"Thought: {step.thought}\nAction: {step.action}\nAction Input: {step.action_input}"
                elif isinstance(step, ObservationReasoningStep):
                    step_content = f"Observation: {step.observation}"
                else:
                    step_content = step.get_content()
                messages.append(ChatMessage(role="assistant", content=step_content))
                yield messages
        elif isinstance(step_output, str):
            full_response += step_output
            messages[-1] = ChatMessage(role="assistant", content=full_response)
            yield messages

demo = gr.ChatInterface(
    predict,
    type="messages"
)

demo.launch()


