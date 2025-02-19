from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.workflow import AgentWorkflow
from gradio import ChatMessage
import gradio as gr
from llama_index.core.workflow import Context

from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
from llama_index.utils.workflow import draw_all_possible_flows

llm = Gemini(model="models/gemini-2.0-flash")

supervisor_agent = ReActAgent(
    name="SupervisorAgent",
    description="Supervisor agent to answer questions.",
    system_prompt=(
        "You are the SupervisorAgent tasked with answering questions. You always respond as a pirate"
    ),
    llm=llm,
    tools=[],
    can_handoff_to=[],
    verbose=True,
)

agent_workflow = AgentWorkflow(agents=[supervisor_agent])
ctx = Context(agent_workflow)


async def interact_with_llamaindex_agent(prompt, messages):
    # Need to add user prompt to gradio chat history
    messages.append(ChatMessage(role="user", content=prompt))

    # Return prompt message to update gradio App
    yield messages

    # Initialize assistant messages
    messages.append(ChatMessage(role="assistant", content=""))
    # async with agent.run_stream(prompt, message_history=msgs) as result:
    handler = agent_workflow.run(user_msg=prompt, ctx=ctx)
    async for event in handler.stream_events():
        # Stream text
        if isinstance(event, AgentStream):
            messages[-1].content += event.delta
            print(event.delta, end="", flush=True)
            yield messages
            # print(event.response)  # the current full response
            # print(event.raw)  # the raw llm api response
            # print(event.current_agent_name)  # the current agent name
        elif isinstance(event, AgentInput):
            print("\n * AgentInput input: ", event.input)  # the current input messages
            print(
                "\n * AgentInput input: ", event.current_agent_name
            )  # the current agent name
        elif isinstance(event, AgentOutput):
            print(
                "\n * AgentOutput response: ", event.response
            )  # the current full response
            print(
                "\n * AgentOutput tool_calls:", event.tool_calls
            )  # the selected tool calls, if any
            print("\n * AgentOutput raw:", event.raw)  # the raw llm api response
        elif isinstance(event, ToolCallResult):
            print("\n * ToolCallResult tool_name: ", event.tool_name)  # the tool name
            print(
                "\n * ToolCallResult tool_kwargs: ", event.tool_kwargs
            )  # the tool kwargs
            print(
                "\n * ToolCallResult tool_output: ", event.tool_output
            )  # the tool output
        elif isinstance(event, ToolCall):
            print("\n * ToolCall tool_name: ", event.tool_name)  # the tool name
            print("\n * ToolCall tool_kwargs: ", event.tool_kwargs)  # the tool kwargs


with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LlamaIndex 🦙 and see its thoughts 💭 and tools 🛠")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    input_box = gr.Textbox(lines=1, label="Chat Message")
    input_box.submit(interact_with_llamaindex_agent, [input_box, chatbot], [chatbot])

draw_all_possible_flows(agent_workflow, filename="multi_step_workflow.html")
demo.launch()
