import streamlit as st

from azure.identity import DefaultAzureCredential
from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

from azure_demo.agentic_retrieval import (  # rename if your file is named differently
    search_endpoint,
    knowledge_base_name,
    knowledge_source_name,
    ask_knowledge_base,
)


# --- Initialize clients once per session ---
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

st.set_page_config(page_title="Earth at Night Chat", page_icon="🌍")
st.title("Earth at Night – Agentic Retrieval Chat")

st.markdown(
    "Ask questions about the **Earth** at night using your Azure AI Search knowledge base."
)

# --- Session state for chat history ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Input box at bottom
if user_input := st.chat_input("Ask a question about Earth at night..."):
    # show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # call backend
    answer, new_history = ask_knowledge_base(
        agent_client=agent_client,
        knowledge_source_name=knowledge_source_name,
        user_question=user_input,
        chat_history=st.session_state.chat_history,
    )

    # show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    # update history
    st.session_state.chat_history = new_history
