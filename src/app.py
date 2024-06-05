import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
# from platform import python_version
# print(python_version())
def get_response(user_input):
    return "idk"


def get_vector_store(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents
st.set_page_config(page_title="AskMe")
st.title("Ask me!")

if "chat_hist" not in st.session_state:

    st.session_state.chat_hist=[
   AIMessage(content="Hello I am a bot. How can i help you?")
]

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter url here: ")
if(website_url==""):
    st.info("Please enter a url!")
else:
    docs = get_vector_store(website_url)
    with st.sidebar:
        st.write(docs)
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