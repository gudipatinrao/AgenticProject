# NewsGenie Demo 06

## Overview

This demo implements NewsGenie, a Streamlit-based unified assistant for:
- real-time news retrieval using NewsAPI
- external web search using SerpAPI
- general question answering using Azure OpenAI via LangGraph workflow

The demo uses a LangGraph workflow to classify requests, fetch news or search results, and generate a final answer.

## Files

- `Demo_06_NewsGenie_Unified_News_and_QA_with_LangGraph.py`: main demo application
- `.env`: environment configuration file for API keys and endpoints

## Required Python Packages

Install the following packages before running the demo:

```bash
pip install streamlit requests python-dotenv openai langgraph
```

The workspace also includes these package versions in nearby demos:
- `streamlit==1.30.0`
- `requests` (latest compatible)
- `python-dotenv==1.0.1`
- `openai==1.64.0`
- `langgraph==0.3.0`

## Environment Variables

The demo uses the following `.env` values:

- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version (default: `2025-04-01-preview`)
- `NEWS_API_KEY`: NewsAPI key for fetching news articles
- `NEWSAPI_ENDPOINT`: Optional custom NewsAPI endpoint (default: `https://newsapi.org/v2/everything`)
- `SERPAPI_API_KEY`: SerpAPI key for external web search
- `SERPAPI_ENDPOINT`: SerpAPI endpoint URL (default: `https://serpapi.com/search`)

## How It Works

1. **Request classification**
   - The user query is classified as either a news request or a general question.
2. **News retrieval**
   - If classified as news, the demo fetches articles from NewsAPI using the `everything` endpoint.
   - News results are included in the final response.
3. **Web search**
   - A separate SerpAPI web search is used to gather external search snippets.
4. **Answer generation**
   - Azure OpenAI is used to generate the final NewsGenie response, combining news and search context.

## LangGraph Workflow Visualization

The Demo 06 workflow is built with LangGraph and follows these main stages:

```mermaid
flowchart TD
    U[User request]
    C[Classify request]
    N[Fetch news (NewsAPI)]
    B[Build search query]
    W[Web search (SerpAPI)]
    A[Generate answer (Azure OpenAI)]
    O[Format output]

    U --> C
    C --> N
    C --> B
    B --> W
    N --> A
    W --> A
    A --> O
```

This means the system can fetch news and prepare search context in parallel, then combine both sources into a single answer.

## Running the Demo

From the demo folder, run:

```bash
streamlit run Demo_06_NewsGenie_Unified_News_and_QA_with_LangGraph.py
```

## Debugging

The app includes debug print statements for:
- request classification
- news fetch trigger and NewsAPI request details
- SerpAPI web search trigger and request details

These prints appear in the terminal where Streamlit is running.

## Notes

- `NewsAPI` is used for live news article retrieval.
- `SerpAPI` is used only for web search results, not news feed retrieval.
- If the NewsAPI returns zero articles, check the query, category, API key, and endpoint in `.env`.
- The app uses `openai.AzureOpenAI` with the `gpt-5.4-nano` model.

## Example `.env`

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-azure-endpoint.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-04-01-preview

NEWS_API_KEY=your_newsapi_key
NEWSAPI_ENDPOINT=https://newsapi.org/v2/everything

SERPAPI_API_KEY=your_serpapi_key
SERPAPI_ENDPOINT=https://serpapi.com/search
```
