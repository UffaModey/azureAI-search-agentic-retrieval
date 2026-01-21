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

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://search.azure.com/.default"
)

# Create an index
azure_openai_token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

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

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
index_client.create_or_update_index(index)
print(f"Index '{index_name}' created or updated successfully.")

# Upload documents - Load local artists.json
with open("artists.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Clean data: skip header row, parse genres, create searchable text
documents = []
for row in raw_data[1:]:  # skip first row {"id": "id", ...}
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
    if genres_str:
        doc["genres"] = [g.strip() for g in genres_str.split(",")]
    else:
        doc["genres"] = []
    documents.append(doc)

with SearchIndexingBufferedSender(
    endpoint=search_endpoint, index_name=index_name, credential=credential
) as client:
    client.upload_documents(documents=documents)

print(
    f"Documents uploaded to index '{index_name}' successfully. Processed {len(documents)} artists."
)

# Create a knowledge source
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

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
index_client.create_or_update_knowledge_source(knowledge_source=ks)
print(f"Knowledge source '{knowledge_source_name}' created or updated successfully.")

# Create a knowledge base
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

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
index_client.create_or_update_knowledge_base(knowledge_base)
print(f"Knowledge base '{knowledge_base_name}' created or updated successfully.")


# --- Reuseable retrieval helper for frontend ---
def ask_knowledge_base(
    agent_client, knowledge_source_name, user_question, chat_history
):
    """
    agent_client: KnowledgeBaseRetrievalClient already initialized
    knowledge_source_name: name of the search index knowledge source
    user_question: latest user message (string)
    chat_history: list of {"role": "...", "content": "..."} messages
    returns: (answer_text, new_messages_list)
    """
    # system instructions
    instructions = """
    A Q&A agent that can answer questions about Spotify artists including their followers, popularity, genres, and main genre.
    If you don't have the answer, respond with "I don't know".
    """

    # rebuild messages from chat_history + new user question
    messages = [{"role": "system", "content": instructions}]
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

    response_parts = []
    for resp in result.response:
        for content in resp.content:
            response_parts.append(content.text)
    answer = "\n\n".join(response_parts) if response_parts else "No response found."

    # extend chat history for the caller
    new_history = chat_history + [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": answer},
    ]
    return answer, new_history


# Set up messages
instructions = """
A Q&A agent that can answer questions about Spotify artists.
If you don't have the answer, respond with "I don't know".
"""

messages = [{"role": "system", "content": instructions}]

# Run agentic retrieval
agent_client = KnowledgeBaseRetrievalClient(
    endpoint=search_endpoint,
    knowledge_base_name=knowledge_base_name,
    credential=credential,
)
query_1 = """
Who are the most popular electronic artists? 
Which artists have the most followers in hip-hop?
"""

messages.append({"role": "user", "content": query_1})

req = KnowledgeBaseRetrievalRequest(
    messages=[
        KnowledgeBaseMessage(
            role=m["role"], content=[KnowledgeBaseMessageTextContent(text=m["content"])]
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
print(f"Retrieved content from '{knowledge_base_name}' successfully.")

# Display the response, activity, and references
response_contents = []
activity_contents = []
references_contents = []

response_parts = []
for resp in result.response:
    for content in resp.content:
        response_parts.append(content.text)
response_content = (
    "\n\n".join(response_parts) if response_parts else "No response found on 'result'"
)

response_contents.append(response_content)

# Print the three string values
print("response_content:\n", response_content, "\n")

messages.append({"role": "assistant", "content": response_content})

if result.activity:
    activity_content = json.dumps([a.as_dict() for a in result.activity], indent=2)
else:
    activity_content = "No activity found on 'result'"

activity_contents.append(activity_content)
print("activity_content:\n", activity_content, "\n")

if result.references:
    references_content = json.dumps([r.as_dict() for r in result.references], indent=2)
else:
    references_content = "No references found on 'result'"

references_contents.append(references_content)
print("references_content:\n", references_content)

# Continue the conversation
query_2 = "Find artists similar to The Beach Boys."
messages.append({"role": "user", "content": query_2})

req = KnowledgeBaseRetrievalRequest(
    messages=[
        KnowledgeBaseMessage(
            role=m["role"], content=[KnowledgeBaseMessageTextContent(text=m["content"])]
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
print(f"Retrieved content from '{knowledge_base_name}' successfully.")

# Display the new retrieval response, activity, and references
response_parts = []
for resp in result.response:
    for content in resp.content:
        response_parts.append(content.text)
response_content = (
    "\n\n".join(response_parts) if response_parts else "No response found on 'result'"
)

response_contents.append(response_content)

# Print the three string values
print("response_content:\n", response_content, "\n")

if result.activity:
    activity_content = json.dumps([a.as_dict() for a in result.activity], indent=2)
else:
    activity_content = "No activity found on 'result'"

activity_contents.append(activity_content)
print("activity_content:\n", activity_content, "\n")

if result.references:
    references_content = json.dumps([r.as_dict() for r in result.references], indent=2)
else:
    references_content = "No references found on 'result'"

references_contents.append(references_content)
print("references_content:\n", references_content)

if __name__ == "__main__":
    # Clean up resources
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    index_client.delete_knowledge_base(knowledge_base_name)
    print(f"Knowledge base '{knowledge_base_name}' deleted successfully.")

    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    index_client.delete_knowledge_source(knowledge_source=knowledge_source_name)
    print(f"Knowledge source '{knowledge_source_name}' deleted successfully.")

    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    index_client.delete_index(index_name)
    print(f"Index '{index_name}' deleted successfully.")
