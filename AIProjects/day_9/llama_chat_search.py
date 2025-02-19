from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.workflow import AgentWorkflow
from tavily import AsyncTavilyClient
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
verbose = False

##############################
#
# Supervisor Agent
#
##############################

supervisor_agent = ReActAgent(
    name="SupervisorAgent",
    description="Supervisor agent to oversee answering questions by orchestrating other agents.",
    system_prompt=(
        "You are the SupervisorAgent tasked with answering questions . You always respond as a pirate."
        "When in doubt how to answer a question, use the ResearchAgent."
    ),
    llm=llm,
    tools=[],
    can_handoff_to=["ResearchAgent"],
)



##############################
#
# Research Agent
#
##############################

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient()
    return str(await client.search(query))


research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful agent for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic. "
        "Once information is obtained, you should hand off control back to the SupervisorAgent."
    ),
    llm=llm,
    tools=[search_web],
    can_handoff_to=["SupervisorAgent"],
)



##############################
#
#  Agent WorkflowResearch Agent
#
##############################

agent_workflow = AgentWorkflow(agents=[supervisor_agent, research_agent], root_agent=supervisor_agent.name)
ctx = Context(agent_workflow)


async def interact_with_llamaindex_agent(prompt, messages):
    # Need to add user prompt to gradio chat history
    messages.append(ChatMessage(role="user", content=prompt))

    # Return prompt message to update gradio App
    yield messages

    # Initialize assistant messages
    # async with agent.run_stream(prompt, message_history=msgs) as result:
    handler = agent_workflow.run(user_msg=prompt, ctx=ctx)
    current_agent = None
    current_tool_calls = ""
    not_streaming = True
    async for event in handler.stream_events():

        # Print what agent we are on
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            # Only create a new chat message if we aren't in one
            if not_streaming:
                messages.append(ChatMessage(role="assistant", content=""))
                not_streaming = False
            current_agent = event.current_agent_name
            partial_message = ''
            partial_message += f"\n{'='*50}"
            partial_message += f"🤖 Agent: {current_agent}"
            partial_message += f"{'='*50}\n\n"
            messages[-1].content += partial_message
            yield messages
        
        # Stream the Agent's tokens if it's in streaming moce
        elif isinstance(event, AgentStream):
            # Only create a new chat message if we aren't in one
            if not_streaming:
                messages.append(ChatMessage(role="assistant", content=""))
                not_streaming = False
            messages[-1].content += event.delta
            yield messages

        # Speical output if the Agent is thking to use a tool or handoff to other agent
        elif isinstance(event, AgentOutput):
            if event.tool_calls:
                not_streaming = True
                messages.append(ChatMessage(role="assistant", content="", metadata={"title": '🧠 Thinking'}))
                partial_message = ''
                partial_message += f"📖 Planning to use tools: {[call.tool_name for call in event.tool_calls]}\n"
                messages[-1].content += partial_message
                yield messages   

        # If a new tool is called, let user know in a special box
        elif isinstance(event, ToolCall):
            not_streaming = True
            messages.append(ChatMessage(role="assistant", content="", metadata={"title": f'New action: {event.tool_name}'}))
            partial_message = ''
            partial_message += f"🔨 Calling Tool: {event.tool_name}\n"
            partial_message += f"  With arguments: {event.tool_kwargs}\n"
            messages[-1].content += partial_message
            yield messages

        # Print the results of the tool call in a special tool box
        elif isinstance(event, ToolCallResult):
            #not_streaming = True
            #messages.append(ChatMessage(role="assistant", content="", metadata={"title": f'New action: {event.tool_name}'}))
            partial_message = ''
            partial_message += f"🔧 Tool Result ({event.tool_name}):\n"
            partial_message += f"  Arguments: {event.tool_kwargs}\n"
            partial_message += f"  Output: {event.tool_output}\n"
            messages[-1].content += partial_message
            yield messages


        # Print to
        if verbose:
            if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
            ):
                current_agent = event.current_agent_name
                print(f"\n{'='*50}")
                print(f"🤖 Agent: {current_agent}")
                print(f"{'='*50}\n")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print("📤 Output:", event.response.content)
                if event.tool_calls:
                    print(
                        "🛠️  Planning to use tools:",
                        [call.tool_name for call in event.tool_calls],
                    )
            elif isinstance(event, ToolCallResult):
                print(f"🔧 Tool Result ({event.tool_name}):")
                print(f"  Arguments: {event.tool_kwargs}")
                print(f"  Output: {event.tool_output}")
            elif isinstance(event, ToolCall):
                print(f"🔨 Calling Tool: {event.tool_name}")
                print(f"  With arguments: {event.tool_kwargs}")




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
