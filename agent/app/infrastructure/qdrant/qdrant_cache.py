import hashlib
import asyncio
from app.interfaces.ISemantic_cache import SemanticCache
from app.interfaces.IVector_store import VectorStore
from app.interfaces.IEmbedding_provider import EmbeddingProvider
from app.core.logger import setup_app_logger
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = setup_app_logger("QdrantSemanticCache")


class QdrantSemanticCache(SemanticCache):
    """
    Semantic Cache backed by Qdrant with two production-grade features:

    1. Lightweight Hybrid Reranker:
       Flow: embed → Qdrant ANN top-5 (with score_threshold pre-filter)
             → rerank(semantic_score × alpha + lexical_score × (1-alpha))
             → pick best reranked candidate → HIT

       The ANN graph can be noisy and near-tie intents (same question, different phrasing)
       can cause the true best match to land at rank 2-3. Fetching top-5 + reranking
       guards against both issues without adding any heavyweight ML dependency.

    2. Cache Versioning by LLM model name:
       Every cached entry stores a `model_version` field (e.g., "gemini-2.0-flash").
       The `get()` filter includes this version, so when you upgrade the LLM:
         - Old cache entries for v1 are silently bypassed (version mismatch → MISS).
         - The new LLM generates a fresh answer → stored under v2.
         - Old data is NOT deleted; it remains for potential rollback analysis.
       Zero downtime, zero manual cache invalidation, zero stale responses.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        model_version: str = "default",
        threshold: float = 0.92,
        top_k: int = 5,
        rerank_alpha: float = 0.8,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.model_version = model_version
        self.threshold = threshold
        self.top_k = top_k
        # alpha weights: semantic vs lexical in the rerank formula
        # Higher alpha → trust ANN scores more; lower alpha → trust keyword overlap more
        self.rerank_alpha = rerank_alpha
        self.collection = "semantic_cache"

    # ─────────────────────────────────────────────────────────────────────────────
    # PRIVATE: Lightweight Hybrid Reranker
    # ─────────────────────────────────────────────────────────────────────────────

    def _lexical_score(self, query: str, cached_query: str) -> float:
        """
        Compute Jaccard token overlap between the incoming query and a cached query.

        Complexity: O(|q| + |c|) — effectively free at cache-lookup scale.
        Rationale: ANN gives us semantic similarity; Jaccard gives us surface-form overlap.
        Combining both reduces false positives where two sentences are semantically close
        but lexically very different (e.g., "How does Redis work?" vs. "Explain in-memory DB").
        """
        q_tokens = set(query.lower().split())
        c_tokens = set(cached_query.lower().split())
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)

    def _rerank(self, query: str, candidates: list) -> list[tuple[float, object]]:
        """
        Hybrid reranker: combines Qdrant cosine score and Jaccard lexical overlap.

        Final score = alpha × semantic_score + (1 - alpha) × lexical_score

        Both signals are already in [0, 1], so the combined score is also in [0, 1].
        Candidates that passed Qdrant's score_threshold are guaranteed semantic_score ≥ threshold.
        Reranking only changes their ORDER — the threshold guarantee is preserved.

        Returns list of (final_score, candidate) sorted descending.
        """
        scored = []
        for candidate in candidates:
            semantic = candidate.score
            lexical = self._lexical_score(query, candidate.payload.get("query", ""))
            final = self.rerank_alpha * semantic + (1 - self.rerank_alpha) * lexical
            scored.append((final, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # ─────────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────────────────────

    async def get(self, query: str) -> str | None:
        """
        Lookup pipeline:
          embed(query)
          → Qdrant ANN top-{top_k} filtered by (score_threshold AND model_version)
          → lightweight rerank (semantic + lexical)
          → return best candidate's cached response, or None on MISS
        """
        embedding = await self.embedding_provider.embed(query)

        # ─── VERSIONING FILTER: Only consider entries from the current LLM version ───
        # When the LLM upgrades (model_version changes), this filter causes all old
        # cache entries to be invisibly bypassed → MISS → fresh LLM call → stored as new version.
        version_filter = Filter(
            must=[FieldCondition(key="model_version", match=MatchValue(value=self.model_version))]
        )

        # ─── STEP 1: Qdrant ANN search with DB-level pre-filtering ───
        # score_threshold is enforced at the Qdrant engine level (Rust), not in Python.
        # This means bad candidates never cross the network boundary.
        candidates = await asyncio.to_thread(
            self.vector_store.client.search,
            collection_name=self.collection,
            query_vector=("text-dense", embedding),
            limit=self.top_k,
            with_payload=True,
            query_filter=version_filter,
            score_threshold=self.threshold,
        )

        if not candidates:
            logger.info(
                f"==> [SemanticCache] MISS | threshold={self.threshold} | "
                f"version={self.model_version} | top_k={self.top_k}"
            )
            return None

        # ─── STEP 2: Lightweight hybrid rerank ───
        # At this point, all candidates already have score ≥ threshold (guaranteed by Qdrant).
        # Reranking only reorders them; it does NOT invalidate the threshold guarantee.
        reranked = self._rerank(query, candidates)
        best_score, best = reranked[0]

        logger.info(
            f"==> [SemanticCache] HIT | "
            f"semantic={best.score:.4f} | reranked={best_score:.4f} | "
            f"version={self.model_version} | evaluated={len(candidates)}/{self.top_k}"
        )
        return best.payload.get("document") or None

    async def set(self, query: str, response: str) -> None:
        """
        Store a (query, response) pair in the cache.

        The doc_id is hashed from (model_version + query) so that the same question
        answered by different LLM versions can coexist in the collection without collision.
        """
        embedding = await self.embedding_provider.embed(query)

        # Include model_version in the hash → different LLM versions get different doc_ids.
        # This means upgrading the LLM doesn't overwrite old cached answers;
        # it creates a new entry under a new ID, preserving history.
        doc_id = hashlib.sha256(
            f"{self.model_version}:{query}".encode("utf-8")
        ).hexdigest()

        await self.vector_store.add(
            collection_name=self.collection,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[response],
            metadatas=[{
                "query": query,
                "model_version": self.model_version,
            }],
        )
