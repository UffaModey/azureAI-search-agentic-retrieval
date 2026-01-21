from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SearchIndexKnowledgeSource,
    SearchIndexKnowledgeSourceParameters,
    SearchIndexFieldReference,
    KnowledgeBase,
    KnowledgeBaseAzureOpenAIModel,
    KnowledgeSourceReference,
    KnowledgeRetrievalOutputMode,
    KnowledgeRetrievalLowReasoningEffort,
)
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchIndexingBufferedSender
from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    SearchIndexKnowledgeSourceParams,
)
import json
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()

# Define variables
search_endpoint = os.getenv('SEARCH_ENDPOINT')
aoai_endpoint = os.getenv('AOAI_ENDPOINT')
aoai_embedding_model = "text-embedding-3-large"
aoai_embedding_deployment = "text-embedding-3-large"
aoai_gpt_model = "gpt-5-mini"
aoai_gpt_deployment = "gpt-5-mini"
index_name = "spotify-artists"
knowledge_source_name = "spotify-knowledge-source"
knowledge_base_name = "spotify-knowledge-base"
search_api_version = "2025-11-01-preview"

# System instructions (defined once)
SYSTEM_INSTRUCTIONS = """
A Q&A agent that can answer questions about Spotify artists including their followers, popularity, genres, and main genre.
If you don't have the answer, respond with "I don't know".
"""

# Initialize credentials and clients once
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://search.azure.com/.default"
)
azure_openai_token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)
index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)


def create_search_index():
    """Create the search index with vector and semantic search capabilities."""
    index = SearchIndex(
        name=index_name,
        fields=[
            SearchField(
                name="id",
                type="Edm.String",
                key=True,
                filterable=True,
                sortable=True,
                facetable=True,
            ),
            SearchField(
                name="name",
                type="Edm.String",
                filterable=True,
                sortable=True,
                facetable=True,
            ),
            SearchField(
                name="followers",
                type="Edm.Int64",
                filterable=True,
                sortable=True,
                facetable=True,
            ),
            SearchField(
                name="popularity",
                type="Edm.Int32",
                filterable=True,
                sortable=True,
                facetable=True,
            ),
            SearchField(
                name="genres",
                type="Collection(Edm.String)",
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="main_genre",
                type="Edm.String",
                filterable=True,
                sortable=True,
                facetable=True,
            ),
            SearchField(
                name="artist_text",
                type="Edm.String",
                filterable=False,
                sortable=False,
                facetable=False,
            ),
            SearchField(
                name="artist_embedding_text_3_large",
                type="Collection(Edm.Single)",
                stored=False,
                vector_search_dimensions=3072,
                vector_search_profile_name="hnsw_text_3_large",
            ),
        ],
        vector_search=VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="hnsw_text_3_large",
                    algorithm_configuration_name="alg",
                    vectorizer_name="azure_openai_text_3_large",
                )
            ],
            algorithms=[HnswAlgorithmConfiguration(name="alg")],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="azure_openai_text_3_large",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=aoai_endpoint,
                        deployment_name=aoai_embedding_deployment,
                        model_name=aoai_embedding_model,
                    ),
                )
            ],
        ),
        semantic_search=SemanticSearch(
            default_configuration_name="semantic_config",
            configurations=[
                SemanticConfiguration(
                    name="semantic_config",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[
                            SemanticField(field_name="artist_text"),
                            SemanticField(field_name="name"),
                            SemanticField(field_name="genres"),
                        ]
                    ),
                )
            ],
        ),
    )

    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' created or updated successfully.")


def upload_documents():
    """Load and upload documents to the search index."""
    with open("artists.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []
    for row in raw_data[1:]:  # skip header row
        doc = {
            "id": row["id"],
            "name": row["name"],
            "followers": int(row["followers"]) if row["followers"].isdigit() else 0,
            "popularity": int(row["popularity"]) if row["popularity"].isdigit() else 0,
            "main_genre": row["main_genre"],
            "artist_text": f"{row['name']}: {row['main_genre']} artist with {row['followers']} followers and popularity {row['popularity']}. Genres: {row['genres']}",
        }

        # Parse genres string to list
        genres_str = row["genres"].strip("[]").replace("'", "").replace('"', "")
        doc["genres"] = [g.strip() for g in genres_str.split(",")] if genres_str else []
        documents.append(doc)

    with SearchIndexingBufferedSender(
        endpoint=search_endpoint, index_name=index_name, credential=credential
    ) as client:
        client.upload_documents(documents=documents)

    print(
        f"Documents uploaded to index '{index_name}' successfully. Processed {len(documents)} artists."
    )


def create_knowledge_source():
    """Create a knowledge source from the search index."""
    ks = SearchIndexKnowledgeSource(
        name=knowledge_source_name,
        description="Knowledge source for Spotify artists data",
        search_index_parameters=SearchIndexKnowledgeSourceParameters(
            search_index_name=index_name,
            source_data_fields=[
                SearchIndexFieldReference(name="id"),
                SearchIndexFieldReference(name="name"),
            ],
        ),
    )

    index_client.create_or_update_knowledge_source(knowledge_source=ks)
    print(
        f"Knowledge source '{knowledge_source_name}' created or updated successfully."
    )


def create_knowledge_base():
    """Create a knowledge base with the knowledge source."""
    aoai_params = AzureOpenAIVectorizerParameters(
        resource_url=aoai_endpoint,
        deployment_name=aoai_gpt_deployment,
        model_name=aoai_gpt_model,
    )

    knowledge_base = KnowledgeBase(
        name=knowledge_base_name,
        models=[KnowledgeBaseAzureOpenAIModel(azure_open_ai_parameters=aoai_params)],
        knowledge_sources=[KnowledgeSourceReference(name=knowledge_source_name)],
        output_mode=KnowledgeRetrievalOutputMode.ANSWER_SYNTHESIS,
        answer_instructions="Provide a concise and informative answer about Spotify artists based on the retrieved data.",
    )

    index_client.create_or_update_knowledge_base(knowledge_base)
    print(f"Knowledge base '{knowledge_base_name}' created or updated successfully.")


def ask_knowledge_base(
    agent_client, knowledge_source_name, user_question, chat_history
):
    """
    Reusable retrieval helper for querying the knowledge base.

    Args:
        agent_client: KnowledgeBaseRetrievalClient already initialized
        knowledge_source_name: name of the search index knowledge source
        user_question: latest user message (string)
        chat_history: list of {"role": "...", "content": "..."} messages

    Returns:
        tuple: (answer_text, new_messages_list)
    """
    # Rebuild messages from chat_history + new user question
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_question})

    req = KnowledgeBaseRetrievalRequest(
        messages=[
            KnowledgeBaseMessage(
                role=m["role"],
                content=[KnowledgeBaseMessageTextContent(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ],
        knowledge_source_params=[
            SearchIndexKnowledgeSourceParams(
                knowledge_source_name=knowledge_source_name,
                include_references=True,
                include_reference_source_data=True,
                always_query_source=True,
            )
        ],
        include_activity=True,
        retrieval_reasoning_effort=KnowledgeRetrievalLowReasoningEffort,
    )

    result = agent_client.retrieve(retrieval_request=req)

    # Extract response text
    response_parts = []
    for resp in result.response:
        for content in resp.content:
            response_parts.append(content.text)
    answer = "\n\n".join(response_parts) if response_parts else "No response found."

    # Extend chat history
    new_history = chat_history + [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": answer},
    ]

    return answer, new_history


def cleanup_resources():
    """Delete knowledge base, knowledge source, and index."""
    index_client.delete_knowledge_base(knowledge_base_name)
    print(f"Knowledge base '{knowledge_base_name}' deleted successfully.")

    index_client.delete_knowledge_source(knowledge_source=knowledge_source_name)
    print(f"Knowledge source '{knowledge_source_name}' deleted successfully.")

    index_client.delete_index(index_name)
    print(f"Index '{index_name}' deleted successfully.")


if __name__ == "__main__":
    # Create resources
    create_search_index()
    upload_documents()
    create_knowledge_source()
    create_knowledge_base()

    # Initialize agent client
    agent_client = KnowledgeBaseRetrievalClient(
        endpoint=search_endpoint,
        knowledge_base_name=knowledge_base_name,
        credential=credential,
    )

    # First query
    query_1 = """
    Who are the most popular electronic artists? 
    Which artists have the most followers in hip-hop?
    """

    answer_1, chat_history, result_1 = ask_knowledge_base(
        agent_client=agent_client,
        knowledge_source_name=knowledge_source_name,
        user_question=query_1,
        chat_history=[],
    )

    print("=" * 80)
    print("QUERY 1 RESPONSE:")
    print(answer_1)
    print("\nACTIVITY:")
    if result_1.activity:
        print(json.dumps([a.as_dict() for a in result_1.activity], indent=2))
    else:
        print("No activity found")
    print("\nREFERENCES:")
    if result_1.references:
        print(json.dumps([r.as_dict() for r in result_1.references], indent=2))
    else:
        print("No references found")

    # Second query (continuing conversation)
    query_2 = "Find artists similar to The Beach Boys."

    answer_2, chat_history, result_2 = ask_knowledge_base(
        agent_client=agent_client,
        knowledge_source_name=knowledge_source_name,
        user_question=query_2,
        chat_history=chat_history,
    )

    print("\n" + "=" * 80)
    print("QUERY 2 RESPONSE:")
    print(answer_2)
    print("\nACTIVITY:")
    if result_2.activity:
        print(json.dumps([a.as_dict() for a in result_2.activity], indent=2))
    else:
        print("No activity found")
    print("\nREFERENCES:")
    if result_2.references:
        print(json.dumps([r.as_dict() for r in result_2.references], indent=2))
    else:
        print("No references found")

    # Cleanup
    cleanup_resources()
