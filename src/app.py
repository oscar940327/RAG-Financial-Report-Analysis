import hmac
import streamlit as st
from query import ask_question

REPORTS_MAP = {
    "Tesla Q4 2025": "Tesla_collection",
    "Palantir Q4 2025": "Palantir_collection"
}

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    pw = st.text_input("Password", type="password")
    if pw and hmac.compare_digest(pw, st.secrets["APP_PASSWORD"]):
        st.session_state.authenticated = True
        st.rerun()

    st.stop()

check_password()

st.title("RAG Financial Report Analysis")

if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_chat_history():
    st.session_state.messages = []

st.sidebar.title("Setting")
selected_report_name = st.sidebar.selectbox(
    "Select financial report...", 
    options=list(REPORTS_MAP.keys()),
    on_change=clear_chat_history
    )
target_collection = REPORTS_MAP[selected_report_name]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("What question do you have about the financial statements?")

if user_question:
    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.write(user_question)

    ai_answer = ask_question(user_question, target_collection, st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        st.write(ai_answer)

    st.session_state.messages.append({"role": "assistant", "content": ai_answer})