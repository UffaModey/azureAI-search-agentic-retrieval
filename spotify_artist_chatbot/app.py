import streamlit as st

from azure.identity import DefaultAzureCredential
from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

from azure_demo.agentic_retrieval import (
    search_endpoint,
    knowledge_base_name,
    knowledge_source_name,
    ask_knowledge_base,
)


# Initialize clients once per session
@st.cache_resource
def get_agent_client():
    credential = DefaultAzureCredential()
    client = KnowledgeBaseRetrievalClient(
        endpoint=search_endpoint,
        knowledge_base_name=knowledge_base_name,
        credential=credential,
    )
    return client


agent_client = get_agent_client()

st.set_page_config(page_title="Spotify Artists Chat", page_icon="🎵")
st.title("🎵 Spotify Artists Chatbot")

st.markdown(
    "Ask questions about **Spotify artists**: popularity, followers, genres, recommendations, and more from 71K+ artists dataset."
)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Input box at bottom
if user_input := st.chat_input("Ask about artists, genres, popularity, followers..."):
    # show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, new_history = ask_knowledge_base(
                agent_client=agent_client,
                knowledge_source_name=knowledge_source_name,
                user_question=user_input,
                chat_history=st.session_state.chat_history,
            )
            st.markdown(answer)

    # update history
    st.session_state.chat_history = new_history

# Sidebar with examples
with st.sidebar:
    st.header("💡 Try these:")
    example_queries = [
        "Top 5 most popular hip-hop artists?",
        "Electronic artists with over 1M followers",
        "Artists similar to The Beach Boys",
        "Rock artists with highest popularity",
    ]
    for query in example_queries:
        if st.button(query, key=query):
            st.session_state.chat_history = []
            st.rerun()
