_input):
#     return "idk"


# #creates a knowledgebase from url into the vector store model
# def get_vector_store(url):
#     loader = WebBaseLoader(url)
#     document = loader.load()
#     txt_split = RecursiveCharacterTextSplitter()
#     doc_chunks = txt_split.split_documents(document)
#     model_name = "BAAI/bge-small-en"
#     model_kwargs = {"device": "cpu"}
#     encode_kwargs = {"normalize_embeddings": True}
#     hf = HuggingFaceBgeEmbeddings(
#         model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
#     )
#     vector_store = Chroma.from_documents(doc_chunks,hf)

#     return vector_store
# st.set_page_config(page_title="AskMe")
# st.title("Ask me!")

# def get_context_retriever_chain(vector_store):
#     llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=GOOGLE_API_KEY)
#     retriever = vector_store.as_retriever()
        
#     prompt = ChatPromptTemplate.from_messages([
#       MessagesPlaceholder(variable_name="chat_history"),  # passes in the context of the chat if it exists into the query 
#       ("user", "{input}"),
#       ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
#     ])
    
#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
#     return retriever_chain
# #maintains chat history
# if "chat_hist" not in st.session_state:

#     st.session_state.chat_hist=[
#    AIMessage(content="Hello I am a bot. How can i help you?")
# ]


# # SIDEBAR details
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Enter url here: ")

# #check for url    
# if(website_url==""):
#     st.info("Please enter a url!")
# else:
#     vector_store = get_vector_store(website_url)
    
#     user_query= st.chat_input("Type your message here: ")
#     retriever_chain = get_context_retriever_chain(vector_store)
#     if user_query!="" and user_query is not None:
#         response = get_response(user_query)
#         st.session_state.chat_hist.append(HumanMessage(content=user_query))
#         st.session_state.chat_hist.append(AIMessage(content=response))
#         retrieved_docs =  retriever_chain.invoke({
#             "chat_history":st.session_state.chat_hist,
#             "input":user_query
#         })
#         st.write(retrieved_docs)


#     for message in st.session_state.chat_hist:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.write(message.content)