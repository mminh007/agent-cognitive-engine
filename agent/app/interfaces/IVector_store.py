# interfaces/vector_store.py
from abc import ABC, abstractmethod

class VectorStore(ABC):

    @abstractmethod
    async def add(
        self,
        collection_name: str,     
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict]
    ):
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,   
        embedding: list[float],
        top_k: int,
        filters: dict | None = None
    ):
        pass

    @abstractmethod
    async def delete(
        self,
        collection_name: str,       
        ids: list[str]
    ):
        pass

    @abstractmethod
    async def get(
        self,
        collection_name: str,
        filters: dict | None = None 
    ):
        pass