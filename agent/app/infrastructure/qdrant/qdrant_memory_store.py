from app.interfaces.IMemory_store import MemoryStore
from app.interfaces.IVector_store import VectorStore
from app.interfaces.IEmbedding_provider import EmbeddingProvider

class QdrantMemoryStore(MemoryStore):
    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def add(self, collection: str, doc_id: str, text: str, metadata: dict):
        embedding = await self.embedding_provider.embed(text)
        await self.vector_store.add(
            collection_name=collection,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    async def get(self, collection: str, filters: dict):
        return await self.vector_store.get(collection_name=collection, filters=filters)

    async def delete(self, collection: str, doc_id: str):
        await self.vector_store.delete(collection_name=collection, ids=[doc_id])
