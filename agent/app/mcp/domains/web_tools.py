# agent/app/mcp/domains/web_tools.py
import logging
import os
import tiktoken
from tavily import TavilyClient
from app.core.settings import settings
from app.services.query_transformer import transform_user_query
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import asyncio

logger = logging.getLogger("McpRuntimeGateway.WebTools")
base_llm_kwargs = {
    "api_key": settings.openai.api_key.get_secret_value() if settings.openai.api_key else None,
    "base_url": settings.openai.base_url,
    "streaming": True
}

# TIER 1: Fast & Cheap (Router)
llm_tier1_fast = ChatOpenAI(model=settings.openai.tier1_fast_model, **base_llm_kwargs)

# Initialize Tavily Client utilizing centralized validated configurations
tavily_key = settings.tavily.api_key.get_secret_value() if settings.tavily.api_key else None
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

compression_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a factual summarizer. Extract ONLY relevant facts, technical specifications, or direct answers from the web snippet that match the user query. Remove marketing fluff, navigation text, and boilerplate. Be extremely concise. Keep it under 150 words."),
    ("human", "User Query: {query}\nWeb Snippet: {snippet}")
])
compressor_chain = compression_prompt | llm_tier1_fast | StrOutputParser()

async def search_web_logic(query: str, history_summary: str = "") -> str:
    """
    Executes a real-time AI search query and compiles high-density context snippets.
    Integrates dynamic token calculation to strictly enforce payload boundaries, 
    preventing Context Window Overflow across downstream LLM inferences.
    """
    if not tavily_client:
        logger.error("❌ [TAVILY CONFIGURATION ERROR] Missing TAVILY_API_KEY inside environment scopes.")
        return "Search Component Error: Tavily API Key is not configured."

    try:
        logger.info(f"==> [Tavily Search] Dispatched optimized query request: '{query}'")
        clean_query = await transform_user_query(query, history_summary)
        # Execute search directly optimized for LLM agents
        response = tavily_client.search(
            query=clean_query, 
            search_depth="advanced",
            max_results=5
        )
        
        results = response.get("results", [])
        if not results:
            logger.warning(f"==> [Tavily Search] Query returned empty results for: '{query}'")
            return f"No verified web context found for query: '{query}'."

        # ─── TOKEN BUDGET ENFORCEMENT ───
        # Allocate a strict maximum token budget exclusively for search results (e.g., 2000 tokens).
        # This guarantees safety for the remaining context window (Chat History, System Prompt).
        MAX_TOOL_TOKEN_BUDGET = settings.tavily.max_token_budget  # Configurable via environment variables for flexible tuning

        try:
            encoding = tiktoken.encoding_for_model(settings.openai.tier1_fast_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        compiled_context_chunks = []
        current_token_count = 0

        tasks = []
        for result in results:
            raw_snippet = result.get("content", "")
            if len(raw_snippet) > 100: # Chỉ nén các snippet đủ dài
                tasks.append(compressor_chain.ainvoke({"query": clean_query, "snippet": raw_snippet}))
        compressed_snippets = await asyncio.gather(*tasks)

        for idx, result in enumerate(results):
            title = result.get("title", "No Title")
            link = result.get("url", "No Source URL")
            fact_body = compressed_snippets[idx] if idx < len(compressed_snippets) else result.get("content", "")

            formatted_chunk = f"[{idx+1}] Source: {title}\n   Link: {link}\n   Fact: {fact_body}\n\n"
            chunk_tokens = len(encoding.encode(formatted_chunk))

            if current_token_count + chunk_tokens > MAX_TOOL_TOKEN_BUDGET:
                remaining_budget = MAX_TOOL_TOKEN_BUDGET - current_token_count
                if remaining_budget > 20: 
                    truncated_text = encoding.decode(encoding.encode(formatted_chunk)[:remaining_budget])
                    compiled_context_chunks.append(f"{truncated_text}... [BUDGET LIMIT]")
                break
            
            compiled_context_chunks.append(formatted_chunk)
            current_token_count += chunk_tokens

        logger.info(f"==> [Tavily Search] Compiled payload utilizing {current_token_count}/{MAX_TOOL_TOKEN_BUDGET} allocated tokens.")
        return "".join(compiled_context_chunks)

    except Exception as search_err:
        logger.error(f"❌ [TOOL COMPONENT FAILURE] Tavily API operation failed -> Trace: {str(search_err)}")
        return f"External Intelligence System Error: Unable to complete search task due to: {str(search_err)}"