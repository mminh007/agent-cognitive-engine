# app/services/memory_worker.py
from app.core.logger import setup_app_logger
from app.services.fact_extractor import FactExtractor
from app.services.memory_service import MemoryService

logger = setup_app_logger("AsyncMemoryWorker")

class MemoryWorker:
    """
    """
    def __init__(self, memory_service: MemoryService, extractor: FactExtractor):
        self.memory_service = memory_service
        self.extractor = extractor

    async def process_extraction_task(
            self,
            user_id: str, 
            target_collection_name: str, 
            chat_history_messages: list
    ):
        """
        Orchestration worker: Analyzes conversation via Extractor and routes storage via MemoryService.
        """
        try:
            logger.info(f"==> [Memory Extraction Engine] Initiating extraction process. Target Box: [{target_collection_name.upper()}]\n")
            
            # 1. Delegate the analysis task to the Extractor
            extracted_data = await self.extractor.extract(chat_history_messages)
            
            if not extracted_data or not extracted_data.facts:
                logger.info("==> [Memory Extraction Engine] No permanent core insights detected in recent payload. Skipping upsert.\n")
                return

            # 2. Route the data to MemoryService for storage
            for item in extracted_data.facts:
                # Anchor text synthesis allowing wide search convergence vectors
                combined_document_text = f"[{item.category.upper()}] Fact: {item.text}\nSearch Triggers: {', '.join(item.semantic_anchors)}"
                
                # Delegate storage structure logic to MemoryService instead of calling the DB directly
                await self.memory_service.add_fact(
                    user_id=user_id,
                    fact_text=combined_document_text,
                    category=item.category,
                    semantic_anchors=item.semantic_anchors,
                    collection_name=target_collection_name,
                    score_importance=item.importance
                )
                
                logger.info(f"==> [Memory Worker] Successfully routed classified fact to DB Service. Target Box: {target_collection_name}\n")
                    
        except Exception as extract_err:
            logger.error(f"❌ [CRITICAL WORKER FAILURE] Data synthesis sequence aborted for User: {user_id} | Trace: {str(extract_err)}\n")