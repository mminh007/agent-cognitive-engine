import json
from typing import Any, Optional
from app.interfaces import MemoryStore
from app.interfaces import VectorStore
from app.interfaces import EmbeddingProvider
from app.core.logger import setup_app_logger

logger = setup_app_logger("ChromaMemoryStore")

class ChromaMemoryStore(MemoryStore):
    """
    Implementation of the MemoryStore interface using ChromaDB as the underlying vector database.
    This class handles document storage, retrieval, and vector embeddings generation.
    """

    def __init__(self, vector_store: VectorStore, embedding_provider: EmbeddingProvider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def add(self, collection: str, doc_id: str, text: str, metadata: dict) -> None:
        """
        Embeds the input text and stores the document along with its vector and metadata into ChromaDB.

        Args:
            collection (str): The name of the collection to store the item.
            doc_id (str): The unique identifier for the document.
            text (str): The text content to be embedded and stored.
            metadata (dict): Additional metadata associated with the document.
        """
        try:
            # Generate vector embedding for the input text
            vector = await self.embedding_provider.embed(text)
            
            # Avoid mutating the original metadata dictionary
            final_metadata = metadata.copy() if metadata else {}
            
            # Save to the underlying vector store
            await self.vector_store.add(
                collection_name=collection,
                ids=[doc_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[final_metadata]
            )
            logger.info(f"==> [MemoryStore] Added item successfully. Collection: {collection} | ID: {doc_id}")
        except Exception as e:
            logger.error(f"==> [MemoryStore] Failed to add item to Collection: {collection} | ID: {doc_id}. Error: {str(e)}")
            raise e

    async def get(self, collection: str, filters: dict) -> Optional[dict]:
        """
        Retrieves items from the collection based on the provided metadata filters.

        Args:
            collection (str): The name of the collection to query.
            filters (dict): A dictionary representing ChromaDB metadata filter conditions.

        Returns:
            Optional[dict]: The raw results dictionary from the vector store if hits are found, otherwise None.
        """
        try:
            raw_results = await self.vector_store.get(collection_name=collection, filters=filters)
            
            # Verify if ChromaDB returned any valid results
            if raw_results and raw_results.get("ids") and len(raw_results["ids"]) > 0:
                logger.info(f"==> [MemoryStore] Get Item HIT. Collection: {collection} | Filters: {filters}")
                return raw_results
                
            logger.info(f"==> [MemoryStore] Get Item MISS. Collection: {collection} | Filters: {filters}")
            return None
        except Exception as e:
            logger.error(f"==> [MemoryStore] Error during get from Collection: {collection}. Error: {str(e)}")
            return None

    async def delete(self, collection: str, doc_id: str) -> None:
        """
        Deletes a specific document from the collection by its document ID.

        Args:
            collection (str): The name of the collection.
            doc_id (str): The unique identifier of the document to be deleted.
        """
        try:
            await self.vector_store.delete(collection_name=collection, ids=[doc_id])
            logger.info(f"==> [MemoryStore] Deleted item successfully. Collection: {collection} | ID: {doc_id}")
        except Exception as e:
            logger.error(f"==> [MemoryStore] Failed to delete item from Collection: {collection} | ID: {doc_id}. Error: {str(e)}")
            raise e

    # --- Optional Extension: Semantic Search ---
    async def search(self, collection: str, query: str, limit: int = 10, filters: Optional[dict] = None) -> list[dict]:
        """
        Performs a vector-based semantic search using a natural language query string.

        Args:
            collection (str): The name of the collection to search within.
            query (str): The natural language search query.
            limit (int): The maximum number of results to return.
            filters (Optional[dict]): Optional metadata filters to narrow down the search.

        Returns:
            list[dict]: A list of matching items/documents found.
        """
        try:
            query_vector = await self.embedding_provider.embed(query)
            raw_results = await self.vector_store.search(
                collection_name=collection,
                embedding=query_vector,
                top_k=limit,
                filters=filters
            )
            return raw_results
        except Exception as e:
            logger.error(f"==> [MemoryStore] Search failed in Collection: {collection}. Error: {str(e)}")
            return []