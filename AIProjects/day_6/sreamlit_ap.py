from dotenv import load_dotenv
from httpx import AsyncClient
from datetime import datetime
import streamlit as st
import asyncio
import json
import os

from pydantic_ai.messages import ModelResponse, UserPromptPart, TextPart

from web_search2 import web_search_agent, Deps

load_dotenv()
model = 'google-gla:gemini-2.0-flash'

async def prompt_ai(messages):
    async with AsyncClient() as client:
        #brave_api_key = os.getenv('BRAVE_API_KEY', None)
        deps = Deps(client=client)

        async with web_search_agent.run_stream(
            messages[-1].content, deps=deps, message_history=messages[:-1]
        ) as result:
            async for message in result.stream_text(delta=True):  
                yield message  

async def main():
    st.title("Pydantic AI Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message.role
        if role in ["user", "model-text-response"]:
            with st.chat_message("human" if role == "user" else "ai"):
                st.markdown(message.content)        

    # React to user input
    if prompt := st.chat_input("What would you like research today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(UserPromptPart(content=prompt))

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in prompt_ai(st.session_state.messages):
                response_content += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)
        
        st.session_state.messages.append( ModelResponse(parts=[TextPart(content=response_content)]))


if __name__ == "__main__":
    asyncio.run(main())