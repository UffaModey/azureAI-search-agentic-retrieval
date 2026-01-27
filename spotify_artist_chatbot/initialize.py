"""
Initialization script for Spotify Artists Knowledge Base.
Run this once to set up all required Azure resources before starting the Streamlit app.

Usage:
    python3 initialize.py
"""

from azure_demo.agentic_retrieval import (
    create_search_index,
    upload_documents,
    create_knowledge_source,
    create_knowledge_base,
)


def initialize_resources():
    """Initialize all Azure Search resources for the Spotify Artists chatbot."""
    print("🚀 Starting initialization of Spotify Artists Knowledge Base...\n")

    try:
        print("1️⃣  Creating search index...")
        create_search_index()
        print("✅ Search index created successfully.\n")

        print("2️⃣  Uploading documents...")
        upload_documents()
        print("✅ Documents uploaded successfully.\n")

        print("3️⃣  Creating knowledge source...")
        create_knowledge_source()
        print("✅ Knowledge source created successfully.\n")

        print("4️⃣  Creating knowledge base...")
        create_knowledge_base()
        print("✅ Knowledge base created successfully.\n")

        print("=" * 80)
        print("✨ Initialization complete! You can now run the Streamlit app.")
        print("   Command: python3 -m streamlit run app.py")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        print("\nIf you get an error about 'Existing field(s) cannot be deleted',")
        print("the resources may already exist. Try running the app anyway.")
        raise


if __name__ == "__main__":
    initialize_resources()
