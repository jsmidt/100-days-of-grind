from pydantic_ai import Agent, RunContext
from typing import Annotated
from dataclasses import dataclass
import asyncio
from gradio import ChatMessage
import gradio as gr
from pydantic_ai.messages import ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL


"""
sys_prompt = ("You are an expert assistant who can solve any task executing python code blobs"
" using the `python_code_execution` tool or researching a question by searhing the web with the"
" `search_web` tool. You will be given a task to solve as best you can. To do so, you have been"
" To solve the task, you must plan forward to proceed in a series of steps, in a cycle of"
" 'Thought:', 'Research:', Code:', and 'Observation:' sequences."
""
" At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use."
" Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence."
" During each intermediate step, you can use 'print()' to save whatever important information you will then need."
" These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step."
" In the end you have to return a final answer using the `final_answer` tool."
"  Here are a couple examples using notional tools:"
" ----"
"  Task: 'What is the result of the following operation: 5 + 3 + 1294.678?'"
""
"  Thought: I will use python code to compute the result of the operation and then return the final answer using the `python_code_execution` tool"
"  Code:"
"  ```python"
"  result = 5 + 3 + 1294.678"
"  final_answer(result)"
"  ```<end_code>"
" ----"
"  Task: 'How fast does it take a cheetah to cross the Golden Gate Bridge?'"
"  Thought: I will use the `search_web` to reasrch teh speed at which a cheetah can run and the length or the Golden Gate Bridge."
"  I will then use the `python_code_execution` to do the math to find the right answer"
"  Code:"
"  ```python"
"  length_of_gg_bridge = 2737  # Golden Gate Bridge length in meters taken from `search_web` research"
"  speed_of_cheetah = 30       # Average cheetah speed in m/s (can reach ~30 m/s) taken from `search_web` research"
"  time_seconds = length_of_gg_bridge / speed_of_cheetah"
"  final_answer(time_seconds)"
"  ```<end_code>"
""
" Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.")
"""

sys_prompt = (
    "You are an expert assistant who can solve any task either executing python code blobs"
    " using the `python_code_execution` tool or researching a question by searhing the web with the"
    " `search_web` tool. You will be given a task to solve as best you can. To do so, you have been"
    " To solve the task, you must plan forward to proceed in a series of steps, in a cycle of"
    " 'Thought:', 'Research:', Code:', and 'Observation:' sequences."
    " At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use."
    " You then will need to call the tools according to your plan."
    " Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000."
)


print(sys_prompt)


chat_agent = Agent("google-gla:gemini-2.0-flash", system_prompt=sys_prompt)


@dataclass
class SearchResult:
    title: str
    description: str
    url: str


@chat_agent.tool
async def search_web(ctx: RunContext, web_query: str) -> list[SearchResult]:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """

    search = DuckDuckGoSearchResults(output_format="list", max_results=5)
    web_results = search.invoke(web_query)

    results = []
    for item in web_results[:5]:
        title = item.get("title", "")
        description = item.get("snippet", "")
        url = item.get("link", "")
        results.append(SearchResult(title=title, description=description, url=url))

    return results


repl = PythonREPL()


@chat_agent.tool
def python_code_execution(
    ctx: RunContext,
    code: str,
):
    """Tool to perform tasks with python.

    Args:
        ctx: The context.
        code: The python code to execute to perform tasks or answer questions

    Returns:
        str: The search results as a formatted string.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n```python\n{code}\n```\nWith Stdout: {result}"
    )
    return result_str


msgs = []


async def interact_with_pydantic_agent(prompt, messages):
    # Need to add user prompt to gradio chat history
    messages.append(ChatMessage(role="user", content=prompt))

    # Return prompt message to update gradio App
    yield messages

    # print (msgs)
    result = await chat_agent.run(prompt, message_history=msgs)

    # For each new message, including tool messages, we need to send to gradio and add to message history
    for msg in result.new_messages():
        # We loop over messages finding which goes to assistant message what which is a tool message
        for part in msg.parts:
            # print (part)
            # Assistant messages
            if isinstance(part, TextPart):
                messages.append(ChatMessage(role="assistant", content=part.content))
            # Tool Calls
            if isinstance(part, ToolCallPart):
                if part.tool_name == "search_web":
                    tool_call = f'🛠️ From tool {part.tool_name} searching "{part.args["web_query"]}":'
                else:
                    tool_call = f"🛠️ From tool {part.tool_name}:"

            # Tool Messages
            elif isinstance(part, ToolReturnPart):
                # Web search tools have special formating to take into account
                if part.tool_name == "search_web":
                    tool_msg = ""
                    for m in part.content:
                        tool_msg += f"* Title: {m.title}\n  Description: {m.description}\n  url: {m.url}\n"
                else:
                    tool_msg = part.content

                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=tool_msg,
                        metadata={"title": tool_call},
                    )
                )

            yield messages

        msgs.append(msg)


with gr.Blocks() as demo:
    gr.Markdown("# Chat with a Pydantic Agent 🦜⛓️ and see its thoughts 💭")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    input_box = gr.Textbox(lines=1, label="Chat Message")
    input_box.submit(interact_with_pydantic_agent, [input_box, chatbot], [chatbot])

demo.launch()
