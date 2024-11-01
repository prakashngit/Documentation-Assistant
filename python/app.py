from dotenv import load_dotenv
load_dotenv()

from retriever import ReadTheDocsRetriever
import streamlit as st
from streamlit import chat_input, chat_message
import os

st.header("Gramine Documentation Chatbot")

st.session_state.script_dir = os.path.dirname(os.path.abspath(__file__))
st.session_state.retriever = ReadTheDocsRetriever(collection_name="gramine_docs", chroma_persist_directory=os.path.join(st.session_state.script_dir, "./chroma_db"))

if "user_prompt_history" not in st.session_state:
    st.session_state.user_prompt_history = []

if "response_history" not in st.session_state:
    st.session_state.response_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.text_input("Prompt", placeholder="Ask anything about Gramine SGX")

if prompt:
    with st.spinner("Generating response..."):
        answer, context = st.session_state.retriever.chat(prompt, return_context=True, chat_history=st.session_state.chat_history)
        source_documents =  set([doc.metadata["source"] for doc in context])
        formatted_response = f"Answer: {answer}\n\nSource Documents: {source_documents}"
        
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.response_history.append(formatted_response)
        st.session_state.chat_history.append(("human", prompt))
        st.session_state.chat_history.append(("assistant", formatted_response))    
        
if st.session_state.chat_history:   
    for response, query in zip(st.session_state.response_history, st.session_state.user_prompt_history): 
        chat_message("user").write(query)
        chat_message("assistant").write(response)