from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)

class Memory:
    """Handles vector store operations for long-term memory"""
    
    def __init__(
        self,
        vector_store_service,
        collection_name: str = "conversation_memory",
        vector_size: int = 1536
    ):
        self.vector_store = vector_store_service
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._initialized = False
    
    async def initialize(self):
        """Initialize memory system"""
        if not self._initialized:
            try:
                # Check if collection exists
                collection_info = await self.vector_store.get_collection_info(self.collection_name)
                
                if not collection_info:
                    # Create collection if it doesn't exist
                    await self.vector_store.create_collection(
                        self.collection_name,
                        vector_size=self.vector_size
                    )
                
                self._initialized = True
                logger.info(f"Memory system initialized with collection: {self.collection_name}")
                
            except Exception as e:
                logger.error(f"Error initializing memory system: {e}")
                raise
    
    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None
    ) -> str:
        """
        Store a memory entry
        
        Args:
            content: Memory content
            metadata: Optional metadata
            vector: Optional vector embedding
            
        Returns:
            ID of stored memory
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Prepare point data
            point = {
                "id": str(datetime.utcnow().timestamp()),
                "vector": vector or [0.0] * self.vector_size,  # Placeholder vector
                "payload": {
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": metadata or {}
                }
            }
            
            # Store point
            await self.vector_store.upsert_points(
                self.collection_name,
                [point]
            )
            
            return point["id"]
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def search_memories(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search memories
        
        Args:
            query: Search query
            query_vector: Optional query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching memories
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Search points
            results = await self.vector_store.search_points(
                self.collection_name,
                query_vector or [0.0] * self.vector_size,  # Placeholder vector
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            memories = []
            for result in results:
                memory = {
                    "id": result["id"],
                    "content": result["payload"]["content"],
                    "timestamp": result["payload"]["timestamp"],
                    "metadata": result["payload"]["metadata"],
                    "score": result["score"]
                }
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
    
    async def delete_memory(self, memory_id: str):
        """
        Delete a memory entry
        
        Args:
            memory_id: ID of memory to delete
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Delete point
            await self.vector_store.delete_points(
                self.collection_name,
                [memory_id]
            )
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise
    
    async def get_memory_context(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> str:
        """
        Get memory context for a query
        
        Args:
            query: Query to get context for
            limit: Maximum number of memories to include
            score_threshold: Minimum similarity score
            
        Returns:
            Formatted context string
        """
        try:
            # Search relevant memories
            memories = await self.search_memories(
                query,
                limit=limit,
                score_threshold=score_threshold
            )
            
            if not memories:
                return ""
            
            # Format context
            context_parts = []
            for memory in memories:
                context_parts.append(
                    f"[{memory['timestamp']}] {memory['content']}"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup memory system"""
        try:
            if self._initialized:
                # TODO: Implement cleanup logic
                self._initialized = False
                
        except Exception as e:
            logger.error(f"Error cleaning up memory system: {e}")
            raise 