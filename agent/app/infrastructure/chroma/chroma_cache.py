# infrastructure/chorma/chroma_cache.py
import time
from app.interfaces import SemanticCache
from app.interfaces import VectorStore
from app.interfaces import EmbeddingProvider
from app.core.logger import setup_app_logger
from app.core.metrics import SEMANTIC_CACHE_LOOKUPS

logger = setup_app_logger("ChromaSemanticCache")

class ChromaSemanticCache(SemanticCache):
    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider, threshold: float = 0.12):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.threshold = threshold
        self.cache_collection = "semantic_cache"

    async def get(self, query: str) -> str | None:
        try:
            query_vector = await self.embedding_provider.embed(query)
            
            results = await self.vector_store.search(
                collection_name=self.cache_collection,
                embedding=query_vector,
                top_k=1
            )
            
            if results and results.get("distances") and results["distances"][0]:
                distance = results["distances"][0][0]
                logger.info(f"==> [Semantic Cache HIT] Distance: {distance}\n")
                SEMANTIC_CACHE_LOOKUPS.labels(status="hit").inc()
                return results["documents"][0][0]
            
            SEMANTIC_CACHE_LOOKUPS.labels(status="miss").inc()
            
        except Exception as e:
            logger.error(f"Cache check error: {e}\n")
            SEMANTIC_CACHE_LOOKUPS.labels(status="error").inc()
            
        return None

    async def set(self, query: str, response: str):
        try:
            vector = await self.embedding_provider.embed(query)
            cache_id = f"cache_{int(time.time() * 1000)}"
            
            await self.vector_store.add(
                collection_name=self.cache_collection,
                ids=[cache_id],
                embeddings=[vector],
                documents=[response],
                metadatas=[{"query": query, "timestamp": int(time.time())}]
            )
            
            logger.info("==> [Semantic Cache SET] Successfully cached response.\n")
        except Exception as e:
            logger.error(f"Cache set error: {e}\n")