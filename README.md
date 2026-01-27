# 🎵 Spotify Artists Chatbot

An AI-powered chatbot that answers questions about Spotify artists using Azure Search and Azure OpenAI. Ask about artist popularity, followers, genres, recommendations, and more from a dataset of 71K+ artists.

## What It Does

This chatbot uses agentic retrieval with Azure Search's knowledge base to provide intelligent answers about Spotify artists. It leverages:
- **Azure Search**: Vector and semantic search over artist data
- **Azure OpenAI**: LLM for answer generation and embeddings
- **Streamlit**: User-friendly web interface

## Prerequisites

Before you begin, make sure you have:
- Python 3.10+
- An Azure account with:
  - Azure Search resource
  - Azure OpenAI resource (with `text-embedding-3-large` and `gpt-4-mini` deployments)
- A `.env` file with the following variables:
  ```
  SEARCH_ENDPOINT=https://<your-search>.search.windows.net
  AOAI_ENDPOINT=https://<your-openai>.openai.azure.com/
  OPENAI_API_KEY=<your-openai-api-key>
  LANGSMITH_API_KEY=<your-langsmith-api-key>
  ```

## Setup Instructions

### 1. Clone/Navigate to Project
```bash
cd spotify_artist_chatbot
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Azure Resources (Run Once)
```bash
python3 initialize.py
```

This script will:
- Create the Azure Search index
- Upload artist documents
- Create a knowledge source
- Set up the knowledge base

**Note**: If you get an error about existing fields, the resources may already be set up. You can proceed to the next step.

### 5. Run the Streamlit App
```bash
python3 -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8505`

## Usage

1. Type your question in the chat input box at the bottom
2. Ask questions like:
   - "Top 5 most popular hip-hop artists?"
   - "Electronic artists with over 1M followers"
   - "Artists similar to The Beach Boys"
   - "Rock artists with highest popularity"

3. The chatbot will retrieve relevant artist information and provide an answer with references

## Project Structure

```
spotify_artist_chatbot/
├── app.py                      # Streamlit frontend
├── agentic_retrieval_n.py      # Core retrieval logic and functions
├── initialize.py               # One-time setup script
├── artists.json                # Artist data
├── artists.csv                 # Artist data (CSV format)
├── data/                       # Data processing utilities
│   ├── artists.csv
│   └── convert_csv_to_json.py
└── __pycache__/
```

## Troubleshooting

**Error: "API deployment does not exist"**
- Check your Azure OpenAI deployments match the configuration in `agentic_retrieval_n.py`

**Error: "Existing field(s) cannot be deleted"**
- Your Azure Search index already exists. You can either:
  - Delete it manually in Azure portal, then run `initialize.py` again
  - Or just proceed to running the app

**"ModuleNotFoundError: No module named..."**
- Make sure you've installed all requirements: `pip install -r requirements.txt`
- Verify your virtual environment is activated

## Running on Streamlit Cloud

To deploy this on Streamlit Cloud:
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your environment variables in the Streamlit Cloud secrets manager

## License

See LICENSE file for details.
