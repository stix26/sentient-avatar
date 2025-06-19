import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .base import BaseService

logger = logging.getLogger(__name__)


class VectorStoreService(BaseService):
    """Service connection for Vector Store (Qdrant)"""

    def __init__(
        self, base_url: str, collection_name: str = "sentient_avatar", timeout: int = 30
    ):
        super().__init__(base_url, timeout)
        self.collection_name = collection_name

    async def health_check(self) -> bool:
        """Check if Vector Store service is healthy"""
        try:
            response = await self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Vector Store health check failed: {str(e)}")
            return False

    async def initialize(self) -> None:
        """Initialize Vector Store service connection"""
        if not await self.health_check():
            raise RuntimeError("Vector Store service is not healthy")

        # Create collection if it doesn't exist
        try:
            await self._request("GET", f"/collections/{self.collection_name}")
        except Exception:
            await self.create_collection()

    async def cleanup(self) -> None:
        """Cleanup Vector Store service resources"""
        pass

    async def create_collection(
        self, vector_size: int = 1536, distance: str = "Cosine"
    ) -> Dict[str, Any]:
        """
        Create a new collection

        Args:
            vector_size: Size of vectors to store
            distance: Distance metric ('Cosine', 'Euclid', 'Dot')

        Returns:
            Dict containing collection info
        """
        payload = {
            "name": self.collection_name,
            "vectors": {"size": vector_size, "distance": distance},
        }

        try:
            response = await self._request("PUT", "/collections", json=payload)
            return response
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    async def upsert_points(
        self, points: List[Dict[str, Any]], batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Upsert points into collection

        Args:
            points: List of points with vectors and payloads
            batch_size: Number of points to process in each batch

        Returns:
            Dict containing operation status
        """
        # Process in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            payload = {"points": batch}

            try:
                await self._request(
                    "PUT", f"/collections/{self.collection_name}/points", json=payload
                )
            except Exception as e:
                logger.error(
                    f"Error upserting points batch {i//batch_size + 1}: {str(e)}"
                )
                raise

        return {"status": "success", "points_processed": len(points)}

    async def search_points(
        self,
        vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar points

        Args:
            vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter: Optional filter conditions

        Returns:
            Dict containing search results
        """
        payload = {"vector": vector, "limit": limit}

        if score_threshold:
            payload["score_threshold"] = score_threshold
        if filter:
            payload["filter"] = filter

        try:
            response = await self._request(
                "POST",
                f"/collections/{self.collection_name}/points/search",
                json=payload,
            )
            return response
        except Exception as e:
            logger.error(f"Error searching points: {str(e)}")
            raise

    async def delete_points(
        self, points_selector: Union[List[str], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Delete points from collection

        Args:
            points_selector: List of point IDs or filter conditions

        Returns:
            Dict containing operation status
        """
        if isinstance(points_selector, list):
            payload = {"points": points_selector}
        else:
            payload = {"filter": points_selector}

        try:
            response = await self._request(
                "POST",
                f"/collections/{self.collection_name}/points/delete",
                json=payload,
            )
            return response
        except Exception as e:
            logger.error(f"Error deleting points: {str(e)}")
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information

        Returns:
            Dict containing collection info and statistics
        """
        try:
            response = await self._request(
                "GET", f"/collections/{self.collection_name}"
            )
            return response
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
