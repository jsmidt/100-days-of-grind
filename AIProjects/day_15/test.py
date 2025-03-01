import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Streaming Bot", page_icon="🤖")

st.title("Streaming bot")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

user_query = st.chat_input("Your messaage", key="user_input")


# Get response
def get_response(query, chat_history):
    msgs = [
        SystemMessage("You are a helpful assistant that always responds like a pirate")
    ] + chat_history
    return llm.astream(msgs)


# Conversation History:
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)


# User Input
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(
            get_response(user_query, st.session_state.chat_history)
        )

    st.session_state.chat_history.append(AIMessage(ai_response))

    get_response(user_query, st.session_state.chat_history)
