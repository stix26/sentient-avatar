from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Memory:
    """Lightweight memory wrapper around a vector store service for testing."""

    def __init__(self, vector_store_service, collection_name: str = "memories") -> None:
        self.vector_store_service = vector_store_service
        self.collection_name = collection_name

    async def store_memory(self, content: str) -> Dict[str, Any]:
        return await self.vector_store_service.upsert_points(content)

    async def search_memories(self, query: str) -> List[Dict[str, Any]]:
        result = await self.vector_store_service.search_points(query)
        points = result.get("points", []) if isinstance(result, dict) else result
        return points

    async def get_memory_context(self, query: str) -> str:
        results = await self.search_memories(query)
        texts = [p.get("payload", {}).get("text", "") for p in results]
        return "\n".join(texts)
