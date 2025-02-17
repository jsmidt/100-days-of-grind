from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import asyncio
from gradio import ChatMessage
import gradio as gr
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolReturnPart,
    ToolCallPart
)
from langchain_community.tools import DuckDuckGoSearchResults


chat_agent = Agent('google-gla:gemini-1.5-flash',system_prompt='Be a helpful assistant who responds like a pirate. Always use the `search_web` tool when you are uncertain about something.')


@dataclass
class SearchResult:
    title: str
    description: str
    url: str

@chat_agent.tool
async def search_web(
    ctx: RunContext,
    web_query: str
) -> list[SearchResult]:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """

    search = DuckDuckGoSearchResults(output_format="list",max_results=5)
    web_results = search.invoke(web_query)

    results = []
    for item in web_results[:5]:
        title = item.get('title', '')
        description = item.get('snippet', '')
        url = item.get('link', '')
        results.append(SearchResult(title=title, description=description, url=url))

    return results


msgs = []

async def interact_with_pydantic_agent(prompt, messages):

    # Need to add user prompt to gradio chat history
    messages.append(ChatMessage(role="user", content=prompt))

    # Return prompt message to update gradio App
    yield messages

    #print (msgs)
    result = await chat_agent.run(prompt, message_history=msgs)

    # For each new message, including tool messages, we need to send to gradio and add to message history
    for msg in result.new_messages():
        # We loop over messages finding which goes to assistant message what which is a tool message
        for part in msg.parts:
            # Assistant messages
            if isinstance(part, TextPart):
                messages.append(ChatMessage(role="assistant", content=part.content)) 
            # Tool Calls
            if isinstance(part, ToolCallPart):
                tool_call = f'🛠️ From tool {part.tool_name} searching "{part.args["web_query"]}":'
            
            # Tool Messages
            elif isinstance(part, ToolReturnPart):
                # Web search tools have special formating to take into account
                if part.tool_name == 'search_web':
                    tool_msg = ''
                    for m in part.content:
                        tool_msg += f"* Title: {m.title}\n  Description: {m.description}\n  url: {m.url}\n"
                    
                messages.append(ChatMessage(role="assistant", content=tool_msg, metadata={"title": tool_call}))

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
