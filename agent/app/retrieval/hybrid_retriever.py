from typing import List, Dict
from app.interfaces import Retriever
from app.interfaces import VectorStore
from app.interfaces import EmbeddingProvider
from app.core.logger import setup_app_logger

logger = setup_app_logger("HybridRetriever")

class HybridRetriever(Retriever):
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedding_provider: EmbeddingProvider
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def retrieve(self, user_id: str, query: str, collection: str, k: int = 3) -> str:
        """
        Redefine retrieve_context:
        Orchestrates Hybrid Search (Dense + Sparse) natively via Qdrant.
        """
        try:
            filters = {"user_id": user_id}
            
            # Embed the query to dense vector
            query_vector = await self.embedding_provider.embed(query)
            
            # Delegate entirely to Qdrant Native Hybrid Search with Prefetch & RRF
            hybrid_results = await self.vector_store.hybrid_search(
                collection_name=collection,
                query_text=query,
                dense_embedding=query_vector,
                top_k=k,
                filters=filters
            )
            
            final_top_k = []
            if hybrid_results and hybrid_results.get("ids") and hybrid_results["ids"][0]:
                for i in range(len(hybrid_results["ids"][0])):
                    final_top_k.append({
                        "id": hybrid_results["ids"][0][i],
                        "text": hybrid_results["documents"][0][i],
                        "metadata": hybrid_results["metadatas"][0][i]
                    })

            # ─── FORMAT OUTPUT RESULTS ───
            if final_top_k:
                facts = [doc["text"] for doc in final_top_k]
                context = "\n- ".join(facts)
                
                # Keep the log structure separated by a blank line as requested
                logger.info(f"==> [Hybrid Search] Successfully aggregated {len(final_top_k)} contexts from [{collection.upper()}] via Qdrant Native RRF.\n")
                
                return f"Known context extracted out of dedicated partition [{collection.upper()}]:\n- {context}"
                
        except Exception as e:
            logger.error(f"❌ [Hybrid Retriever Error] Failed pipeline processing on box {collection}: {str(e)}\n")
        
        return ""