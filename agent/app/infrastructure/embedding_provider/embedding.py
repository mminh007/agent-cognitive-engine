from app.interfaces import EmbeddingProvider
from langchain_core.embeddings import Embeddings

class EmbeddingProvider(EmbeddingProvider):
    def __init__(self, embedding_model: Embeddings):
        # Initialize Embeddings using central configuration settings
        # Supports OpenAIEmbeddings, GoogleGenerativeAIEmbeddings, etc.
        self.embeddings = embedding_model

    async def embed(self, text: str) -> list[float]:
        # Return a pure list of floats
        return await self.embeddings.aembed_query(text)