#app/retrieval/hybird_retriever.py
from typing import List, Dict
from app.interfaces import Retriever
from app.interfaces import VectorStore
from app.interfaces import EmbeddingProvider
from .bm25_search import BM25Retriever
from .rrf_search import RRF
from app.core.logger import setup_app_logger

logger = setup_app_logger("HybridRetriever")

class HybridRetriever(Retriever):
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedding_provider: EmbeddingProvider, 
        bm25_retriever: BM25Retriever
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_retriever = bm25_retriever

    async def retrieve(self, user_id: str, query: str, collection: str, k: int = 3) -> str:
        """
        Redefine retrieve_context:
        Orchestrates Hybrid Search (Dense + BM25) and blends them using RRF.
        """
        try:
            # ─── STEP 1: PREPARE CORPUS DATA FOR BM25 ───
            filters = {"user_id": user_id}
            
            # Fetch all user documents within the collection (Requires a get method in VectorStore)
            all_docs = await self.vector_store.get(collection_name=collection, filters=filters)
            
            if not all_docs or not all_docs.get("documents"):
                return ""

            corpus_docs = []
            for i in range(len(all_docs["ids"])):
                corpus_docs.append({
                    "id": all_docs["ids"][i],
                    "text": all_docs["documents"][i],
                    "metadata": all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
                })

            # ─── STEP 2: STREAM LAYER 1 - DENSE SEMANTIC VECTORS ───
            query_vector = await self.embedding_provider.embed(query)
            
            chroma_raw = await self.vector_store.search(
                collection_name=collection,
                embedding=query_vector,
                top_k=min(k * 2, len(corpus_docs)),
                filters=filters
            )
            
            dense_results = []
            if chroma_raw and chroma_raw.get("ids") and chroma_raw["ids"][0]:
                for i in range(len(chroma_raw["ids"][0])):
                    dense_results.append({
                        "id": chroma_raw["ids"][0][i],
                        "text": chroma_raw["documents"][0][i],
                        "metadata": chroma_raw["metadatas"][0][i]
                    })

            # ─── STEP 3: STREAM LAYER 2 - SPARSE LEXICAL BM25 ───
            lexical_results = self.bm25_retriever.retrieve(
                query=query, 
                corpus_docs=corpus_docs, 
                limit=k
            )

            # ─── STEP 4: HYBRID SYNTHESIS (Using your RFF class) ───
            fused_candidates = RRF._reciprocal_rank_fusion(dense_results, lexical_results, k=60)
            final_top_k = fused_candidates[:k]

            # ─── STEP 5: FORMAT OUTPUT RESULTS ───
            if final_top_k:
                facts = [doc["text"] for doc in final_top_k]
                context = "\n- ".join(facts)
                
                # Keep the log structure separated by a blank line as requested
                logger.info(f"==> [Hybrid Search] Successfully aggregated {len(final_top_k)} contexts from [{collection.upper()}] via RRF Mixer.\n")
                
                return f"Known context extracted out of dedicated partition [{collection.upper()}]:\n- {context}"
                
        except Exception as e:
            logger.error(f"❌ [Hybrid Retriever Error] Failed pipeline processing on box {collection}: {str(e)}\n")
        
        return ""