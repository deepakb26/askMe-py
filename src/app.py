import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv
import os

# load_dotenv("./.env")
# print(load_dotenv("./.env"))
# from platform import python_version
# print(python_version())

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
# print(GOOGLE_API_KEY)
st.set_page_config(page_title="AskMe")
st.title("Ask me!")
#creates a knowledgebase from url into the vector store model
def get_vector_store(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    txt_split = RecursiveCharacterTextSplitter()
    doc_chunks = txt_split.split_documents(document)
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vector_store = Chroma.from_documents(doc_chunks,hf)

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GOOGLE_API_KEY)
    retriever = vector_store.as_retriever()
        
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),  # passes in the context of the chat if it exists into the query 
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)  #finds relevant docs 
    
    return retriever_chain

def get_conv_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GOOGLE_API_KEY)      
    prompt = ChatPromptTemplate.from_messages([
    ("user", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),   #appends history if it exist
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)  #creates a chain of docs and context that have been created in the conv
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)  #combines docs and retriever chain to give o/p

#maintains chat history
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conv_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    # return response
    return response['answer']


# SIDEBAR details
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter url here: ")

#check for url    
if(website_url==""):
    st.info("Please enter a url!")
else:
    if "chat_history" not in st.session_state:

        st.session_state.chat_history=[
    AIMessage(content="Hello I am a bot. How can i help you?")
    ]
    if "vectore_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store(website_url)  #make vector store of website persistent, minimises llm utilisation


    retriever_chain = get_context_retriever_chain(st.session_state.vector_store )

    #user input
    user_query= st.chat_input("Type your message here: ")
    if user_query!="" and user_query is not None:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


#message
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)