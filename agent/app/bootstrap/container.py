# app/bootstrap/container.py
from app.infrastructure.embedding_provider import EmbeddingProvider
from app.infrastructure.chroma import ChromaMemoryStore, ChromaSemanticCache, ChromaVectorStore
from app.services import MemoryService, MemoryWorker
from app.services import FactExtractor
from app.retrieval import HybridRetriever, BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from app.core.settings import settings
from app.core.logger import setup_app_logger
import redis.asyncio as redis_async

logger = setup_app_logger("Container")


def _resolve_default_provider() -> str:
    """
    Determine the system-level default provider based on which API key
    is configured in .env. Priority: gemini → llama → deepseek → openai.
    Claude has no embedding API, so it cannot be the system default for infrastructure.
    """
    if settings.gemini.api_key:
        return "gemini"
    elif settings.llama.api_key:
        return "llama"
    elif settings.deepseek.api_key:
        return "deepseek"
    return "openai"


def _resolve_default_embedding(provider: str):
    """
    Create the embedding model instance based on the system default provider.
    Embedding MUST remain fixed per deployment to keep vector spaces compatible.
    """
    if provider == "gemini":
        api_key = settings.gemini.api_key.get_secret_value() if settings.gemini.api_key else None
        logger.info(f"==> [Container] Using Gemini embedding model: {settings.gemini.embedding_model}")
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini.embedding_model,
            google_api_key=api_key
        )
    else:
        # Fallback to OpenAI embedding for openai, claude, llama, deepseek
        api_key = settings.openai.api_key.get_secret_value() if settings.openai.api_key else None
        logger.info(f"==> [Container] Using OpenAI embedding model: {settings.chroma.embedding_model}")
        return OpenAIEmbeddings(
            model=settings.chroma.embedding_model,
            openai_api_key=api_key,
            base_url=settings.openai.base_url
        )


def _resolve_default_memory_llm(provider: str):
    """
    Create the memory LLM instance (used by FactExtractor) based on
    the system default provider. This is a background job, not per-user.
    """
    if provider == "gemini":
        api_key = settings.gemini.api_key.get_secret_value() if settings.gemini.api_key else None
        base_url = settings.gemini.base_url
        model = settings.gemini.tier1_fast_model
        logger.info(f"==> [Container] Using Gemini memory LLM: {model}")
        if base_url and "openai" in base_url:
            return ChatOpenAI(
                model=model, temperature=0,
                api_key=api_key, base_url=base_url
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model, temperature=0,
                google_api_key=api_key
            )
    elif provider == "llama":
        api_key = settings.llama.api_key.get_secret_value() if settings.llama.api_key else None
        model = settings.llama.tier1_fast_model
        logger.info(f"==> [Container] Using Llama memory LLM: {model}")
        return ChatOpenAI(
            model=model, temperature=0,
            api_key=api_key, base_url=settings.llama.base_url
        )
    elif provider == "deepseek":
        api_key = settings.deepseek.api_key.get_secret_value() if settings.deepseek.api_key else None
        model = settings.deepseek.tier1_fast_model
        logger.info(f"==> [Container] Using DeepSeek memory LLM: {model}")
        return ChatOpenAI(
            model=model, temperature=0,
            api_key=api_key, base_url=settings.deepseek.base_url
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

        self.vector_store = ChromaVectorStore()

        self.memory_store = ChromaMemoryStore(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )

        self.semantic_cache = ChromaSemanticCache(self.vector_store, self.embedding_provider)

        self.memory_service = MemoryService(self.memory_store)

        self.extractor = FactExtractor(memory_llm)

        self.memory_worker = MemoryWorker(
            memory_service=self.memory_service,
            extractor=self.extractor
        )

        self.hybrid_search = HybridRetriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            bm25_retriever=BM25Retriever()
        )

    async def shutdown(self):
        if self.redis_client:
            await self.redis_client.aclose()

container = Container()