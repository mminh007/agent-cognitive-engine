# app/services/query_transformer.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.settings import settings
from app.core.logger import setup_app_logger

logger = setup_app_logger("QueryTransformer")

transform_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an information retrieval optimizer. Convert the user's latest message and short context into a single, clean, highly effective search query for vector search or search engines. Output ONLY the query, no explanations."),
    ("human", "Context: {context}\nLatest Message: {latest_message}")
])

base_llm_kwargs = {
    "api_key": settings.openai.api_key.get_secret_value() if settings.openai.api_key else None,
    "base_url": settings.openai.base_url,
    "streaming": True
}

# TIER 1: Fast & Cheap (Router)
llm_tier1_fast = ChatOpenAI(model=settings.openai.tier1_fast_model, **base_llm_kwargs)

async def transform_user_query(latest_message: str, history_summary: str = "") -> str:
    """Transform the user's latest message into an optimized search query."""
    chain = transform_prompt | llm_tier1_fast
    response = await chain.ainvoke({"context": history_summary, "latest_message": latest_message})
    optimized_query = str(response.content).strip().strip('"')
    logger.info(f"==> [Query Transformation] '{latest_message[:30]}...' -> '{optimized_query}'")
    return optimized_query