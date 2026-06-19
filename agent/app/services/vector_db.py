# app/services/vector_db.py
from typing import Optional
import time
import chromadb
import os
import asyncio
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.core.settings import settings
from langchain_openai import OpenAIEmbeddings
from app.core.logger import setup_app_logger

logger = setup_app_logger("VectorDBService")

class VectorDBService:
    """
    Service layer managing domain-isolated vector storage partitions inside ChromaDB.
    Supports Dense Semantic Retrieval (via OpenAI) and Sparse Lexical Retrieval (via BM25)
    fused seamlessly through Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        """
        Initializes OpenAI Embeddings and persistent ChromaDB storage backends 
        utilizing validated central system configuration parameters.
        """
        self.embeddings = OpenAIEmbeddings(
            model=settings.chroma.embedding_model,
            openai_api_key=settings.openai.api_key.get_secret_value() if settings.openai.api_key else None,
            base_url=settings.openai.base_url
        )

        chroma_host = os.getenv("CHROMA_SERVER_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_SERVER_PORT", 8000))
        
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            tenant="default_tenant",
            database="default_database"   
        )

    def _get_isolated_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Validates or instantiates individual collection partitions safely at runtime.

        Args:
            collection_name (str): The destination collection segment key inside ChromaDB.

        Returns:
            chromadb.Collection: The fetched or newly declared partition metadata instance.
        """
        return self.chroma_client.get_or_create_collection(name=collection_name)

    async def add_fact(self, user_id: str, fact_text: str, category: str, semantic_anchors: list, collection_name: str, score_importance: float = 0.5):
        """
        Vectorizes a processed text insight and stores it into the targeted domain partition.

        Args:
            user_id (str): Multi-tenant identification partition boundary token.
            fact_text (str): Raw descriptive knowledge string or fact to ingest.
            collection_name (str): Destination partition domain ('general_memory', etc.).
            score_importance (float, optional): Node persistence coefficient rank. Defaults to 0.5.
        """
        if not fact_text or str(fact_text).strip() == "":
            return
        
        vector = await self.embeddings.embed_query(fact_text)
        fact_id = f"fact_{int(time.time() * 1000)}"
        
        # Flatten anchors into a highly indexable string format for the Lexical BM25 parser
        anchors_str = " ".join(semantic_anchors)

        target_box = await asyncio.to_thread(self._get_isolated_collection(collection_name))
        await asyncio.to_thread(
            target_box.add,
            ids=[fact_id],
            embeddings=[vector],
            documents=[fact_text],
            metadatas=[{
                "user_id": user_id,
                "category": category,
                "semantic_anchors": anchors_str,
                "timestamp": int(time.time()),
                "last_accessed": int(time.time()),
                "score_importance": score_importance
            }]
        )

        logger.info(f"==> [VectorDB] Saved fact inside partition [{collection_name.upper()}] for user {user_id}")

    def _reciprocal_rank_fusion(self, dense_results: List[Dict], lexical_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Executes Reciprocal Rank Fusion (RRF) to score and re-rank documents 
        based on their ordinal positional indices retrieved from multi-modal pipelines.

        Args:
            dense_results (List[Dict]): Candidates extracted via dense semantic vector lookup.
            lexical_results (List[Dict]): Candidates extracted via keyword-matching BM25 calculations.
            k (int, optional): RRF constant parameter scale mitigating rank noise anomalies. Defaults to 60.

        Returns:
            List[Dict]: Consolidated corpus elements sorted descending by aggregated RRF scores.
        """
        rrf_scores = {}
        doc_mapping = {}

        # Process position ranks originating from Dense Semantic Retrieval
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            doc_mapping[doc_id] = doc
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        # Process position ranks originating from Lexical Keyword Search (BM25)
        for rank, doc in enumerate(lexical_results):
            doc_id = doc["id"]
            doc_mapping[doc_id] = doc
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        # Re-rank elements by descending score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_docs:
            hydrated_doc = doc_mapping[doc_id].copy()
            hydrated_doc["rrf_score"] = score
            final_results.append(hydrated_doc)
            
        return final_results

    async def check_semantic_cache(self, query: str, threshold: float = 0.12) -> Optional[str]:
        """
        Evaluates if a similar query exists within the cache.
        ChromaDB utilizes L2 distance by default; a smaller distance implies higher similarity.
        """
        try:
            cache_box = await asyncio.to_thread(self._get_isolated_collection, "semantic_cache")
            query_vector = await self.embeddings.aembed_query(query)
            
            results = cache_box.query(
                query_embeddings=[query_vector],
                n_results=1
            )
            
            if results and results["distances"] and results["distances"][0]:
                distance = results["distances"][0][0]
                if distance <= threshold:
                    logger.info(f"==> [Semantic Cache HIT] Distance: {distance}\n")
                    return results["documents"][0][0]
        except Exception as e:
            logger.error(f"Cache check error: {e}\n")
        return None

    async def set_semantic_cache(self, query: str, response: str):
        """Persists the prompt-response pair into the cache for future request bypasses."""
        try:
            cache_box = await asyncio.to_thread(self._get_isolated_collection, "semantic_cache")
            vector = await self.embeddings.aembed_query(query)
            cache_id = f"cache_{int(time.time() * 1000)}"
            
            await asyncio.to_thread(
                cache_box.add,
                ids=[cache_id],
                embeddings=[vector],
                documents=[response],
                metadatas=[{"query": query, "timestamp": int(time.time())}]
            ) 
            logger.info("==> [Semantic Cache SET] Successfully cached response.\n")
        except Exception as e:
            logger.error(f"Cache set error: {e}\n")

    async def retrieve_context(self, user_id: str, query: str, collection_name: str, limit: int = 3) -> str:
        """
        Queries the partitioned vector database utilizing synchronous Hybrid Search mechanisms 
        (Dense Embeddings + BM25 Okapi Lexical Analysis) blended through Reciprocal Rank Fusion.

        Args:
            user_id (str): Multi-tenant identification boundary query parameter constraint.
            query (str): Active runtime chat input string generated by the user interface.
            collection_name (str): Isolated target storage space to query ('general_memory', etc.).
            limit (int, optional): Maximum high-density context chunks to assemble. Defaults to 3.

        Returns:
            str: Compiled, markdown-formatted systemic background reference block injected into the LLM prompt layer.
        """
        try:
            target_box = await asyncio.to_thread(self._get_isolated_collection, collection_name)
            
            # Extract full contextual data elements scoped under active multi-tenant constraints for BM25 feeding
            all_docs = await asyncio.to_thread(target_box.get, where={"user_id": user_id})
            if not all_docs or not all_docs["documents"]:
                return ""

            # Standardize structural properties into clean map arrays
            corpus_docs = []
            for i in range(len(all_docs["ids"])):
                corpus_docs.append({
                    "id": all_docs["ids"][i],
                    "text": all_docs["documents"][i],
                    "metadata": all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                })

            # ─── STREAM LAYER 1: DENSE SEMANTIC VECTORS ───
            query_vector = await self.embeddings.aembed_query(query)
           
           # 🚀 Thread wrapper for DB QUERY
            chroma_raw = await asyncio.to_thread(
                target_box.query,
                query_embeddings=[query_vector],
                n_results=min(limit * 2, len(corpus_docs)),
                where={"user_id": user_id}
            )
            
            dense_results = []
            if chroma_raw and chroma_raw["ids"][0]:
                for i in range(len(chroma_raw["ids"][0])):
                    dense_results.append({
                        "id": chroma_raw["ids"][0][i],
                        "text": chroma_raw["documents"][0][i],
                        "metadata": chroma_raw["metadatas"][0][i]
                    })

            # ─── STREAM LAYER 2: SPARSE LEXICAL BM25OKAPI ───
            # Tokenize queries and documents via raw white-space and lowercase parsing bounds
            tokenized_query = query.lower().split(" ")
            tokenized_corpus = [doc["text"].lower().split(" ") for doc in corpus_docs]
            
            bm25_engine = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25_engine.get_scores(tokenized_query)
            
            lexical_results = []
            for idx, score in enumerate(bm25_scores):
                if score > 0:  # Retain exclusively contextual items showing genuine term overlap scores
                    doc_copy = corpus_docs[idx].copy()
                    doc_copy["bm25_score"] = score
                    lexical_results.append(doc_copy)
            
            # Sort and isolate candidate sequences showing highest keyword correspondence density scales
            lexical_results = sorted(lexical_results, key=lambda x: x["bm25_score"], reverse=True)[:limit * 2]

            # ─── HYBRID SYNTHESIS COMPILATION (RRF PIPELINE IMPLEMENTATION) ───
            fused_candidates = self._reciprocal_rank_fusion(dense_results, lexical_results, k=60)
            final_top_k = fused_candidates[:limit]

            if final_top_k:
                facts = [doc["text"] for doc in final_top_k]
                context = "\n- ".join(facts)
                logger.info(f"==> [Hybrid Search] Successfully aggregated {len(final_top_k)} contexts from [{collection_name.upper()}] via RRF Mixer.")
                return f"Known context extracted out of dedicated partition [{collection_name.upper()}]:\n- {context}"
                
        except Exception as e:
            logger.error(f"❌ [VectorDB Hybrid Search Error] Failed pipeline processing on box {collection_name}: {str(e)}")
        
        return ""

vector_db_service = VectorDBService()