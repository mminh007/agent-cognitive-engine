# infrastructure/chorma/chroma_store.py
import asyncio
import chromadb
import os
from app.interfaces import VectorStore
from app.core.logger import setup_app_logger
from app.core.settings import settings

logger = setup_app_logger("ChromaVectorStore")


class ChromaVectorStore(VectorStore):
    def __init__(self):

        self.client = chromadb.HttpClient(
            host=settings.chroma.server_host,
            port=settings.chroma.server_port,
            tenant="default_tenant",
            database="default_database"
        )

    def _get_collection(self, collection_name: str):
        return self.client.get_or_create_collection(name=collection_name)

    async def add(self, collection_name: str, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        target_box = await asyncio.to_thread(self._get_collection, collection_name)
        
        await asyncio.to_thread(
            target_box.add,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"==> [VectorStore] Inserted {len(ids)} vectors into {collection_name}.\n")

    async def search(self, collection_name: str, embedding: list[float], top_k: int, filters: dict | None = None) -> dict:
        target_box = await asyncio.to_thread(self._get_collection, collection_name)
        
        results = await asyncio.to_thread(
            target_box.query,
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters
        )
        
        return results

    async def delete(self, collection_name: str, ids: list[str]):
        target_box = await asyncio.to_thread(self._get_collection, collection_name)
        await asyncio.to_thread(target_box.delete, ids=ids)
        logger.info(f"==> [VectorStore] Deleted {len(ids)} nodes from {collection_name}.\n")

    
    async def get(self, collection_name: str, filters: dict | None = None) -> dict:
        target_box = await asyncio.to_thread(self._get_collection, collection_name)
        
        results = await asyncio.to_thread(
            target_box.get,
            where=filters
        )
        logger.info(f"==> [VectorStore] Get {len(results)} nodes from {collection_name}.\n")

        return results