#app/services/memory_worker.py
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.vector_db import vector_db_service
from app.core.settings import settings
from app.core.logger import setup_app_logger

logger = setup_app_logger("AsyncMemoryWorker")

class FactItem(BaseModel):
    category: Literal["user_preference", "verified_knowledge", "system_constraint"] = Field(
        description="Classify the fact archetype profile context."
    )
    text: str = Field(description="The isolated factual insight extracted cleanly out of the transcript.")
    semantic_anchors: List[str] = Field(
        default_factory=list, 
        description="3 predictable query sentences a human user might provide to look for this exact fact again."
    )
    importance: float = Field(default=0.5)

class LongTermFacts(BaseModel):
    facts: List[FactItem] = Field(default_factory=list)

# 🚀 FIX: Configured to pull API parameters dynamically from centralized settings definitions
memory_llm = ChatOpenAI(
    model=settings.openai.tier1_fast_model, 
    temperature=0,
    api_key=settings.openai.api_key.get_secret_value() if settings.openai.api_key else None,
    base_url=settings.openai.base_url
)
structured_memory_llm = memory_llm.with_structured_output(LongTermFacts)

async def extract_and_save_facts(user_id: str, target_collection_name: str, chat_history_messages: list):
    """Asynchronous worker analyzing raw chat dialogue turns to condense non-transient contextual indicators."""
    conversation_text = ""
    valid_messages = [msg for msg in chat_history_messages if msg.type in ["human", "ai"]]

    for msg in valid_messages[-4:]: 
        role = "User" if msg.__class__.__name__ == "HumanMessage" else "AI"
        conversation_text += f"{role}: {msg.content}\n"

    if not conversation_text.strip():
        return
    
    prompt_instruction = (
        "You are an elite enterprise long-term memory extraction agent.\n"
        "Filter out greeting noise, trivial conversation turns, or failed code snippets.\n"
        "Isolate and structure permanent core insights. Always fulfill the semantic_anchors array parameter."
    )

    try:
        logger.info(f"==> [Memory Extraction Engine] Processing trace tokens target box: [{target_collection_name.upper()}]")
        
        extracted_data: LongTermFacts = await structured_memory_llm.ainvoke([
            SystemMessage(content=prompt_instruction),
            HumanMessage(content=conversation_text)
        ])
        
        for item in extracted_data.facts:
            # Anchor text synthesis allowing wide search convergence vectors
            combined_document_text = f"[{item.category.upper()}] Fact: {item.text}\nSearch Triggers: {', '.join(item.semantic_anchors)}"
            
            # 🚀 Route fact upsert task to its corresponding domain-isolated database collection
            await vector_db_service.add_fact(
                user_id=user_id,
                fact_text=combined_document_text,
                category=item.category,
                semantic_anchors=item.semantic_anchors,
                collection_name=target_collection_name,
                score_importance=item.importance
            )
            logger.info(f"==> [VectorDB Upsert] Successfully index classified fact into slot. Target Box: {target_collection_name}\n")
                
    except Exception as extract_err:
        logger.error(f"❌ [CRITICAL WORKER FAILURE] Data synthesis sequence aborted for User: {user_id} | Trace: {str(extract_err)}\n")

        