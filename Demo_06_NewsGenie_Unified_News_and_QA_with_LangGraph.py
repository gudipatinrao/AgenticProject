import json
import os
import requests
import streamlit as st
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import openai

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
SERPAPI_ENDPOINT = os.getenv("SERPAPI_ENDPOINT", "https://serpapi.com/search")
NEWSAPI_ENDPOINT = os.getenv("NEWSAPI_ENDPOINT", "https://newsapi.org/v2/everything")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

client: Optional[Any] = None
if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
    client = openai.AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def llm_chat(messages: List[Dict[str, str]], max_completion_tokens: int = 300, temperature: float = 0.7) -> str:
    """Call the configured OpenAI client and return the response text."""
    if client is None:
        raise RuntimeError(
            "OpenAI client is not configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env."
        )
    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    return response.choices[0].message.content.strip()


class NewsGenieState(TypedDict):
    query: str
    news_category: str
    request_type: str
    search_query: str
    news_results: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    answer: str
    output: str
    error: str


def classify_request(state: NewsGenieState) -> Dict[str, Any]:
    """Decide whether the user query is a news request or a general question."""
    prompt = (
        "You are a smart assistant that classifies whether a user request is asking for the latest news "
        "or for a general question/answer. Reply with exactly one word: 'news' or 'general'. "
        "If the user asks for headlines, current events, or a news category, choose 'news'. "
        "Otherwise choose 'general'.\n\n"
        f"User request: {state['query']}"
    )
    try:
        classification = llm_chat(
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_completion_tokens=20,
            temperature=0.0,
        ).lower()
        if "news" in classification:
            print(f"[DEBUG] Request classified as: NEWS")
            return {"request_type": "news"}
    except Exception:
        # Fallback to keyword-based detection when LLM classification fails.
        fallback_terms = ["news", "headlines", "latest", "update", "what's happening", "current events"]
        if any(term in state["query"].lower() for term in fallback_terms):
            print(f"[DEBUG] Request classified as: NEWS (keyword fallback)")
            return {"request_type": "news"}
    print(f"[DEBUG] Request classified as: GENERAL")
    return {"request_type": "general"}



def fetch_news(state: NewsGenieState) -> Dict[str, Any]:
    """Retrieve top headlines or topic-specific news from NewsAPI."""
    if state["request_type"] != "news":
        print(f"[DEBUG] fetch_news() skipped - request_type is '{state['request_type']}', not 'news'")
        return {
            "news_results": [],
        }

    print(f"[DEBUG] fetch_news() triggered - fetching from NewsAPI")
    if not NEWS_API_KEY:
        print(f"[DEBUG] NewsAPI key not configured")
        return {
            "news_results": [],
            "error": "News API key not configured. Set NEWS_API_KEY in .env to enable news retrieval.",
        }

    endpoint = NEWSAPI_ENDPOINT
    params = {
        "apiKey": NEWS_API_KEY,
        "pageSize": 5,
        "language": "en",
    }
   
    if state["query"]:
        params["q"] = state["query"]

    print(f"[DEBUG] NewsAPI request - Query: {state['query']}, Category: {state['news_category']}")
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        news_results = [
            {
                "title": article.get("title", ""),
                "source": article.get("source", {}).get("name", ""),
                "url": article.get("url", ""),
                "description": article.get("description", ""),
            }
            for article in articles
        ]
        print(f"[DEBUG] NewsAPI returned {len(news_results)} articles")
        return {"news_results": news_results}
    except Exception as exc:
        print(f"[DEBUG] NewsAPI request failed: {str(exc)}")
        return {
            "news_results": [],
            "error": f"News fetch failed: {str(exc)}",
        }


def build_search_query(state: NewsGenieState) -> Dict[str, Any]:
    """Turn a user request into a search query for web research."""
    prompt = (
        "Rewrite the user's question into a concise web search query. "
        "Return only the best search phrase without any additional text.\n\n"
        f"User request: {state['query']}"
    )
    try:
        result = llm_chat(
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_completion_tokens=50,
            temperature=0.2,
        )
        return {"search_query": result}
    except Exception:
        return {"search_query": state["query"]}


def web_search(state: NewsGenieState) -> Dict[str, Any]:
    """Fetch external search results using a web search API."""
    query = state["search_query"] or state["query"]
    if not query:
        print(f"[DEBUG] web_search() skipped - no query provided")
        return {"search_results": []}

    print(f"[DEBUG] web_search() triggered - fetching from SerpAPI")
    if not SERPAPI_API_KEY or not SERPAPI_ENDPOINT:
        print(f"[DEBUG] SerpAPI key or endpoint not configured")
        return {
            "search_results": [],
            "error": "Web search is unavailable. Set SERPAPI_API_KEY in .env.",
        }

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 5,
        "hl": "en",
    }

    print(f"[DEBUG] SerpAPI request - Query: {query}")
    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        web_results = []
        for item in data.get("organic_results", []):
            web_results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        print(f"[DEBUG] SerpAPI returned {len(web_results)} results")
        return {"search_results": web_results}
    except Exception as exc:
        print(f"[DEBUG] SerpAPI request failed: {str(exc)}")
        return {
            "search_results": [],
            "error": f"Web search failed: {str(exc)}",
        }


def generate_answer(state: NewsGenieState) -> Dict[str, Any]:
    """Create a response that combines news and search information when available."""
    system_prompt = (
        "You are NewsGenie, an assistant that provides up-to-date news summaries and reliable answers. "
        "Use any available news or web search results to support the response. "
        "If the user asked for news, prioritize headlines and a short curated summary. "
        "If the user asked a general question, answer directly and cite the best external sources when possible."
    )

    news_context = ""
    if state["news_results"]:
        news_context = "\n\nLatest news articles:\n"
        for article in state["news_results"]:
            news_context += f"- {article['title']} ({article['source']}): {article['description']}\n  {article['url']}\n"

    search_context = ""
    if state["search_results"]:
        search_context = "\n\nWeb search results:\n"
        for result in state["search_results"]:
            search_context += f"- {result['title']}: {result['snippet']}\n  {result['url']}\n"

    if state["request_type"] == "news" and not state["news_results"]:
        news_context += "\nNo recent news articles were available for this request."

    user_prompt = (
        f"User request: {state['query']}\n"
        f"News category: {state['news_category']}\n"
        f"Request type: {state['request_type']}\n"
        f"{news_context}\n{search_context}\n"
        "Provide a short, clear, and factual response. "
        "If you cannot fetch live articles or external sources, explain that gracefully and still answer the query as best as possible."
    )

    try:
        answer_text = llm_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=400,
            temperature=0.5,
        )
    except Exception as exc:
        answer_text = (
            "NewsGenie could not generate a live answer using the language model. "
            f"Error: {str(exc)}"
        )

    return {"answer": answer_text}


def format_output(state: NewsGenieState) -> Dict[str, Any]:
    """Create the final assistant output based on fetched results and answer text."""
    output_lines = [
        f"Request type: {state['request_type'].capitalize()}",
        "",
        state["answer"],
    ]

    if state["news_results"]:
        output_lines.extend(["", "Top curated headlines:"])
        for item in state["news_results"]:
            output_lines.append(f"- {item['title']} ({item['source']})")

    if state.get("error"):
        output_lines.extend(["", "Fallback notes:", state["error"]])

    return {"output": "\n".join(output_lines)}


def build_workflow() -> Any:
    """Construct the NewsGenie LangGraph workflow."""
    workflow = StateGraph(NewsGenieState)
    workflow.add_node("classify_request", classify_request)
    workflow.add_node("fetch_news", fetch_news)
    workflow.add_node("build_search_query", build_search_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("format_output", format_output)

    workflow.add_edge(START, "classify_request")
    workflow.add_edge("classify_request", "fetch_news")
    workflow.add_edge("classify_request", "build_search_query")
    workflow.add_edge("build_search_query", "web_search")
    workflow.add_edge("fetch_news", "generate_answer")
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", "format_output")
    workflow.add_edge("format_output", END)

    return workflow.compile()


def run_streamlit_app() -> None:
    """Build the Streamlit interface and manage session state."""
    st.set_page_config(page_title="NewsGenie", layout="wide")
    st.title("NewsGenie: Unified News + QA Assistant")
    st.write(
        "NewsGenie helps you get fast answers and live news updates in one place. "
        "Choose a category, enter your query, and let the system decide whether to fetch news or provide a general answer."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    category = st.selectbox(
        "Select a news category:",
        ["", "business", "entertainment", "general", "health", "science", "sports", "technology"],
        index=0,
    )
    query = st.text_input("Ask NewsGenie anything:", value="")
    max_articles = st.slider("Max news headlines", min_value=1, max_value=10, value=5)

    if st.button("Submit Request"):
        if not query:
            st.warning("Enter a query or news topic before submitting.")
        else:
            workflow = build_workflow()
            state = workflow.invoke(
                {
                    "query": query,
                    "news_category": category,
                    "search_query": "",
                    "news_results": [],
                    "search_results": [],
                    "answer": "",
                    "output": "",
                    "error": "",
                }
            )
            result = state["output"]
            error_message = state.get("error", "")

            st.session_state.history.append(
                {
                    "query": query,
                    "request_type": state["request_type"],
                    "category": category,
                    "result": result,
                    "error": error_message,
                }
            )

            st.subheader("NewsGenie Output")
            st.text_area("Result", value=result, height=320)

            if state["news_results"]:
                with st.expander("Curated headlines"):
                    for article in state["news_results"][:max_articles]:
                        st.markdown(
                            f"- **{article['title']}** ({article['source']})  \n"
                            f"  {article['description']}  \n"
                            f"  [Read more]({article['url']})"
                        )

            if error_message:
                st.error(error_message)

    if st.session_state.history:
        with st.expander("Request history"):
            for idx, entry in enumerate(reversed(st.session_state.history), start=1):
                st.markdown(
                    f"**{idx}. [{entry['request_type'].upper()}] {entry['query']}" 
                    f"(Category: {entry['category'] or 'none'})**"
                )
                st.write(entry["result"])
                if entry["error"]:
                    st.write(f"⚠️ {entry['error']}")
                st.markdown("---")

    st.sidebar.header("Setup & Notes")
    st.sidebar.write("Make sure your .env file contains the required keys:")
    st.sidebar.write(
        "- AZURE_OPENAI_API_KEY\n- AZURE_OPENAI_ENDPOINT\n- NEWS_API_KEY\n- SERPAPI_API_KEY"
    )
    st.sidebar.write(
        "If the web search or news API is unavailable, NewsGenie will still attempt to answer using the language model and fallback logic."
    )


if __name__ == "__main__":
    run_streamlit_app()
