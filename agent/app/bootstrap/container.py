# app/bootstrap/container.py
from app.infrastructure.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.chroma import ChromaMemoryStore, ChromaSemanticCache, ChromaVectorStore
from app.services import MemoryService, MemoryWorker
from app.services import FactExtractor
from app.retrieval import HybridRetriever, BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from app.core.settings import settings

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

    async def initialize(self):

        embedding_model = OpenAIEmbeddings(
            model=settings.chroma.embedding_model,
            openai_api_key=settings.openai.api_key.get_secret_value() if settings.openai.api_key else None,
            base_url=settings.openai.base_url
        )

        memory_llm = ChatOpenAI(
            model=settings.openai.tier1_fast_model, 
            temperature=0,
            api_key=settings.openai.api_key.get_secret_value() if settings.openai.api_key else None,
            base_url=settings.openai.base_url
        )

        self.embedding_provider = OpenAIEmbeddingProvider(embedding_model)

        self.vector_store = ChromaVectorStore()

        self.memory_store = ChromaMemoryStore(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )

        self.semantic_cache = ChromaSemanticCache(self.vector_store, self.embedding_provider)

        self.memory_service = MemoryService(self.memory_store)

        self.memory_worker = MemoryWorker(
            memory_service=self.memory_service,
            extractor= self.extractor
        )

        self.extractor = FactExtractor(memory_llm)

        self.hybrid_search = HybridRetriever(
        embedding_provider=self.embedding_provider,
        vector_store=self.vector_store,
        bm25_retriever= BM25Retriever()
    )

    async def shutdown(self):
        pass

container = Container()