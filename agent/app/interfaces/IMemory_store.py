# interfaces/memory_store.py
from abc import ABC, abstractmethod

class MemoryStore(ABC):

    @abstractmethod
    async def add(
        self,
        collection: str,
        doc_id: str,
        text: str,
        metadata: dict
    ):
        pass

    @abstractmethod
    async def get(
        self,
        collection: str,
        filters: dict
    ):
        pass

    @abstractmethod
    async def delete(
        self,
        collection: str,
        doc_id: str
    ):
        pass