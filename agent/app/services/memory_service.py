# app/service/memory_service.py
import time
from app.interfaces import MemoryStore
from app.core.logger import setup_app_logger

logger = setup_app_logger("MemoryService")

class MemoryService:
    """
    Service layer handling business logic related to Long-term Memory.
    """
    def __init__(self, memory_store: MemoryStore):
        # Dependency Injection of the storage interface
        self.memory_store = memory_store

    async def add_fact(
        self, 
        user_id: str, 
        fact_text: str, 
        category: str, 
        semantic_anchors: list, 
        collection_name: str, 
        score_importance: float = 0.5
    ):
        """
        Receives facts from the worker, pre-processes metadata, and saves them to the Store.
        """
        if not fact_text or str(fact_text).strip() == "":
            return

        # Generate a unique ID for the fact 
        fact_id = f"fact_{int(time.time() * 1000)}"
        
        # Flatten the anchors array into a string to support the Sparse Lexical BM25 parser later 
        anchors_str = " ".join(semantic_anchors)

        # Package all metadata following the legacy logic
        current_time = int(time.time())
        metadata = {
            "user_id": user_id,
            "category": category,
            "semantic_anchors": anchors_str,
            "timestamp": current_time,
            "last_accessed": current_time,
            "score_importance": score_importance
        }

        try:
            # Delegate storage to the MemoryStore interface (vector/LangGraph processing is handled downstream)
            await self.memory_store.add(
                collection=collection_name,
                doc_id=fact_id,
                text=fact_text,
                metadata=metadata
            )
            
            logger.info(f"==> [MemoryService] Successfully processed and stored fact. Target Box: {collection_name} | User: {user_id}\n")
            
        except Exception as e:
            logger.error(f"❌ [MemoryService Error] Failed to route fact data for User: {user_id} | Trace: {str(e)}\n")