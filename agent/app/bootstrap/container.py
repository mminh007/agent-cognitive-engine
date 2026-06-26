# app/bootstrap/container.py
from app.infrastructure.embedding_provider import EmbeddingProvider
from app.infrastructure.qdrant import QdrantMemoryStore, QdrantSemanticCache, QdrantVectorStore
from app.services import MemoryService, MemoryWorker
from app.services import FactExtractor
from app.retrieval import HybridRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core.settings import settings
from app.core.logger import setup_app_logger
import redis.asyncio as redis_async

logger = setup_app_logger("Container")


def _resolve_default_provider() -> str:
    """
    Determine the system-level default provider based on which API key
    is configured in .env. Priority: openai -> claude.
    Note: Claude has no native embedding API, so OpenAI is used for embeddings.
    """
    if settings.openai.api_key:
        return "openai"
    return "claude"


def _resolve_cache_model_version(provider: str) -> str:
    """
    Derive a stable, human-readable version tag from the active LLM model name.
    This tag is embedded into every cached entry so that upgrading the LLM model
    automatically bypasses all stale cache entries — zero manual invalidation.
    """
    if provider == "claude":
        return settings.claude.tier2_balanced_model
    return settings.openai.tier2_balanced_model


def _resolve_default_embedding(provider: str):
    """
    Create the embedding model instance.
    Since only OpenAI embeddings are supported in the current stack, 
    this always returns OpenAIEmbeddings regardless of the provider.
    Embedding MUST remain fixed per deployment to keep vector spaces compatible.
    """
    api_key = settings.openai.api_key.get_secret_value() if settings.openai.api_key else None
    logger.info(f"==> [Container] Using OpenAI embedding model")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        base_url=settings.openai.base_url
    )


def _resolve_default_memory_llm(provider: str):
    """
    Create the memory LLM instance (used by FactExtractor) based on
    the system default provider. This is a background job, not per-user.
    """
    if provider == "claude":
        api_key = settings.claude.api_key.get_secret_value() if settings.claude.api_key else None
        model = settings.claude.tier1_fast_model
        logger.info(f"==> [Container] Using Claude memory LLM: {model}")
        return ChatAnthropic(
            model=model, temperature=0,
            api_key=api_key
        )
    else:
        api_key = settings.openai.api_key.get_secret_value() if settings.openai.api_key else None
        model = settings.openai.tier1_fast_model
        logger.info(f"==> [Container] Using OpenAI memory LLM: {model}")
        return ChatOpenAI(
            model=model, temperature=0,
            api_key=api_key, base_url=settings.openai.base_url
        )


class Container:

    def __init__(self):
        self.memory_service = None
        self.memory_worker = None
        self.embedding_provider = None
        self.vector_store = None
        self.memory_store = None
        self.extractor = None
        self.semantic_cache = None
        self.hybrid_search = None
        self.redis_client = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.redis_client = redis_async.from_url(settings.redis.url)

        # Dynamically resolve embedding + memory LLM from .env default
        default_provider = _resolve_default_provider()
        logger.info(f"==> [Container] Resolved system default provider: {default_provider}")

        embedding_model = _resolve_default_embedding(default_provider)
        memory_llm = _resolve_default_memory_llm(default_provider)

        self.embedding_provider = EmbeddingProvider(embedding_model)

        self.vector_store = QdrantVectorStore()

        self.memory_store = QdrantMemoryStore(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )

        # ─── Derive LLM model version for semantic cache versioning ───
        # Changing the LLM in .env automatically invalidates stale cache entries.
        cache_model_version = _resolve_cache_model_version(default_provider)
        logger.info(f"==> [Container] Semantic cache version anchor: '{cache_model_version}'")

        self.semantic_cache = QdrantSemanticCache(
            self.vector_store,
            self.embedding_provider,
            model_version=cache_model_version
        )

        self.memory_service = MemoryService(self.memory_store)

        self.extractor = FactExtractor(memory_llm)

        self.memory_worker = MemoryWorker(
            memory_service=self.memory_service,
            extractor=self.extractor
        )

        self.hybrid_search = HybridRetriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store
        )

    async def shutdown(self):
        if self.redis_client:
            await self.redis_client.aclose()

container = Container()