from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
from tavily import AsyncTavilyClient
from gradio import ChatMessage
import gradio as gr
from llama_index.core.workflow import Context
from llama_index.core.tools import BaseTool, FunctionTool

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


sys_prompt = 'You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.\n\n## Tools\n\nYou have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.\nThis may require breaking the task into subtasks and using different tools to complete each subtask.\n\nYou have access to the following tools:\n{tool_desc}\n\n\n## Output Format\n\nPlease answer in the same language as the question and use the following format:\n\n```\nThought: The current language of the user is: (user\'s language). I need to use a tool to help me answer the question.\nAction: tool name (one of {tool_names}) if using a tool.\nAction Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})\n```\n\nPlease ALWAYS start with a Thought.\n\nNEVER surround your response with markdown code markers. You may use code markers within your response if you need to.\n\nPlease use a valid JSON format for the Action Input. Do NOT do this {{\'input\': \'hello world\', \'num_beams\': 5}}.\n\nIf this format is used, the tool will respond in the following format:\n\n```\nObservation: tool response\n```\n\nYou should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:\n\n```\nThought: I can answer without using any more tools. I\'ll use the user\'s language to answer\nAnswer: [your answer here (In the same language as the user\'s question)]\n```\n\n```\nThought: I cannot answer the question with the provided tools.\nAnswer: [your answer here (In the same language as the user\'s question)]\n```\n\n## Current Conversation\n\nBelow is the current conversation consisting of interleaving human and assistant messages.\n' 


##############################
#
# Research Agent
#
##############################

code_spec = CodeInterpreterToolSpec()


async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient()
    return str(await client.search(query))

search_tool = FunctionTool.from_defaults(fn=search_web)

supervisor_agent = ReActAgent(
    name="SupervisorAgent",
    description="Useful agent for executing python code to answer questions.",
    system_prompt=sys_prompt,
    llm=llm,
    tools=code_spec.to_tool_list() + [search_tool],
    #can_handoff_to=["ResearchAgent"],
)



##############################
#
#  Agent WorkflowResearch Agent
#
##############################

agent_workflow = AgentWorkflow(agents=[supervisor_agent], root_agent=supervisor_agent.name)

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
                
        # Print the results of the tool call in a special tool box
        elif isinstance(event, ToolCallResult):
            not_streaming = True
            messages.append(ChatMessage(role="assistant", content="", metadata={"title": f'New action: {event.tool_name}'}))
            partial_message = ''
            partial_message += f"🔧 Tool Result ({event.tool_name}):\n"
            partial_message += f"  Arguments: {event.tool_kwargs}\n"
            partial_message += f"  Output: {event.tool_output}\n"
            messages[-1].content += partial_message
            yield messages
        # If a new tool is called, let user know in a special box

        # Print to terminal if verbose. 
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
