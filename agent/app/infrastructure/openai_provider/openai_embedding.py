from app.interfaces import EmbeddingProvider
from langchain_openai import OpenAIEmbeddings
from app.core.settings import settings

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embedding_model: OpenAIEmbeddings):
        # Initialize OpenAI Embeddings using central configuration settings
        self.embeddings = embedding_model

    async def embed(self, text: str) -> list[float]:
        # Return a pure list of floats
        return await self.embeddings.aembed_query(text)