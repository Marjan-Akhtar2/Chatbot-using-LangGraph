import streamlit as st
from Langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# ************************ UTILITY FUNCTIONS *********************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    st.session_state['message_history'] = []

def add_thread(t_id):
    if t_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(t_id)

def load_conv(t_id):
    config = {'configurable': {'thread_id': t_id}}
    state = chatbot.get_state(config=config)
    return state.values.get('messages', [])

# ************************ SESSION SETUP **************************************

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# *********************** SIDEBAR UI *****************************************
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

# Sidebar buttons list
for t_id in st.session_state['chat_threads']:
    btn_label = f"Chat {t_id[:8]}..."
    if st.sidebar.button(btn_label, key=str(t_id)):
        st.session_state['thread_id'] = t_id
        messages = load_conv(t_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            content = msg.content
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get("text", "")
            temp_messages.append({'role': role, 'content': content})

        st.session_state['message_history'] = temp_messages
        st.rerun()

# ************************ MAIN UI ********************************************

# Display history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here...')

if user_input:
    # 1. Thread ko conversation history list mein tabhi add karein jab message send ho
    add_thread(st.session_state['thread_id'])

    # 2. Add User Input to State & UI
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    current_config = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # 3. Stream Assistant Response
    with st.chat_message('assistant'):
        def generate_response():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=current_config,
                stream_mode='messages'
            ):
                content = message_chunk.content
                if isinstance(content, list) and len(content) > 0:
                    yield content[0].get("text", "")
                elif isinstance(content, str):
                    yield content

        ai_message = st.write_stream(generate_response())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
