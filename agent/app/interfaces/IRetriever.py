# interfaces/retriever.py
from abc import ABC, abstractmethod


class Retriever(ABC):

    @abstractmethod
    async def retrieve(
        self,
        user_id: str,
        query: str,
        collection: str,
        k: int
    ):
        pass