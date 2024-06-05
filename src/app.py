import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage

def get_response(user_input):
    return "idk"

st.set_page_config(page_title="AskMe")
st.title("Ask me!")

if "chat_hist" not in st.session_state:

    st.session_state.chat_hist=[
   AIMessage(content="Hello I am a bot. How can i help you?")
]
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter url here: ")

user_query= st.chat_input("Type your message here: ")

if user_query!="" and user_query is not None:
    response = get_response(user_query)
    st.session_state.chat_hist.append(HumanMessage(content=user_query))
    st.session_state.chat_hist.append(AIMessage(content=response))



for message in st.session_state.chat_hist:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)