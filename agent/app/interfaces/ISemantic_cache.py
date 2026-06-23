# interfaces/semantic_cache.py
from abc import ABC, abstractmethod

class SemanticCache(ABC):
    
    @abstractmethod
    async def get(self, query: str) -> str | None:
        """
        """
        pass
    
    @abstractmethod
    async def set(self, query: str, response: str) -> None:
        """
        """
        pass