from dotenv import load_dotenv
load_dotenv()

from retriever import ReadTheDocsRetriever
import streamlit as st
from streamlit import chat_input, chat_message
import os

# Configure Streamlit page settings
st.set_page_config(
    page_title="Gramine Documentation Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #ff3333;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    h1, h2, h3 {
        color: #262730;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .user-info {
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://www.gravatar.com/avatar/", width=130)  # Replace with actual image path
    st.title("User Profile")
    
    # Add user info with custom styling
    with st.container():
        if st.experimental_user.email:
            st.markdown(f"""
                <div class='user-info'>
                    <h3 style='margin:0'>👤 {st.experimental_user.email}</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='user-info'>
                    <h3 style='margin:0'>👤 Guest</h3>
                </div>
            """, unsafe_allow_html=True)
    
    # Add additional profile information
    st.markdown("---")
    st.markdown("""
    ### About
    Welcome to the Gramine Documentation Chatbot! 
    This bot helps you find answers about Gramine SGX.
    
    ### Contact
    For support: support@example.com
    """)
    
    # Styled logout button
    if st.button("Logout", key="logout_button"):
        # Add logout logic here if needed
        pass

# Main app content with custom styling
st.title("Gramine Documentation Chatbot")
st.markdown("<p style='font-size: 1.2em; color: #666;'>Ask me anything about Gramine SGX!</p>", unsafe_allow_html=True)

st.session_state.script_dir = os.path.dirname(os.path.abspath(__file__))
st.session_state.retriever = ReadTheDocsRetriever(collection_name="gramine_docs", chroma_persist_directory=os.path.join(st.session_state.script_dir, "./chroma_db"))

if "user_prompt_history" not in st.session_state:
    st.session_state.user_prompt_history = []

if "response_history" not in st.session_state:
    st.session_state.response_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Styled input box
prompt = st.text_input("", placeholder="Ask anything about Gramine SGX", key="prompt_input")

if prompt:
    with st.spinner("Generating response..."):
        answer, context = st.session_state.retriever.chat(prompt, return_context=True, chat_history=st.session_state.chat_history)
        source_documents = set([doc.metadata["source"] for doc in context])
        formatted_response = f"Answer: {answer}\n\nSource Documents: {source_documents}"
        
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.response_history.append(formatted_response)
        st.session_state.chat_history.append(("human", prompt))
        st.session_state.chat_history.append(("assistant", formatted_response))    

# Display chat history with custom styling
if st.session_state.chat_history:   
    for response, query in zip(st.session_state.response_history, st.session_state.user_prompt_history): 
        with chat_message("user"):
            st.markdown(f"<p style='color: #262730;'>{query}</p>", unsafe_allow_html=True)
        with chat_message("assistant"):
            st.markdown(f"<p style='color: #262730;'>{response}</p>", unsafe_allow_html=True)