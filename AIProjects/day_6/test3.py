from pydantic_ai import Agent
import asyncio
from gradio import ChatMessage
import gradio as gr
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

agent = Agent('google-gla:gemini-1.5-flash',system_prompt='Be a helpful assistant who responds like a pirate.')
msgs = []

async def interact_with_pydantic_agent(prompt, messages):

    # Need to add user prompt to gradio chat history
    messages.append(ChatMessage(role="user", content=prompt))

    # Return prompt message to update gradio App
    yield messages

    # Initialize assistant messages
    messages.append(ChatMessage(role="assistant", content=""))    
    async with agent.run_stream(prompt, message_history=msgs) as result:
        # Stream text
        async for partial_message in result.stream_text(delta=True):
            # Keep adding each delta of the message for streaming
            messages[-1].content += partial_message
            # Return assistant message to update gradio app
            yield messages

        # Update new messages from user
        for m in result.new_messages():
            msgs.append(m)
    
        # Update new messages from model response
        msgs.append(ModelResponse(parts=[TextPart(content=messages[-1].content)]))

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
