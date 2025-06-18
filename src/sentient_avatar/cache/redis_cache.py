import json
import pickle
from typing import Any, Optional, Union, Dict, List
import aioredis
from datetime import timedelta
import logging
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based caching system with support for different serialization methods."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis cache connection.

        Args:
            redis_url: Redis connection URL
        """
        self.redis = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        self.binary_redis = aioredis.from_url(
            redis_url, encoding=None
        )  # For binary data

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else default
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return default

    async def get_binary(self, key: str, default: Any = None) -> Any:
        """Get binary value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached binary value or default
        """
        try:
            value = await self.binary_redis.get(key)
            return pickle.loads(value) if value else default
        except Exception as e:
            logger.error(f"Error getting binary cache key {key}: {e}")
            return default

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds

        Returns:
            True if successful
        """
        try:
            await self.redis.set(key, json.dumps(value), ex=expire)
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def set_binary(
        self, key: str, value: Any, expire: Optional[int] = None
    ) -> bool:
        """Set binary value in cache.

        Args:
            key: Cache key
            value: Binary value to cache
            expire: Expiration time in seconds

        Returns:
            True if successful
        """
        try:
            await self.binary_redis.set(key, pickle.dumps(value), ex=expire)
            return True
        except Exception as e:
            logger.error(f"Error setting binary cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache.

        Args:
            key: Cache key
            amount: Amount to increment

        Returns:
            New value or None if error
        """
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return None

    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement counter in cache.

        Args:
            key: Cache key
            amount: Amount to decrement

        Returns:
            New value or None if error
        """
        try:
            return await self.redis.decrby(key, amount)
        except Exception as e:
            logger.error(f"Error decrementing cache key {key}: {e}")
            return None

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        try:
            values = await self.redis.mget(keys)
            return {k: json.loads(v) if v else None for k, v in zip(keys, values)}
        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {e}")
            return {}

    async def set_many(
        self, mapping: Dict[str, Any], expire: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache.

        Args:
            mapping: Dictionary of key-value pairs
            expire: Expiration time in seconds

        Returns:
            True if successful
        """
        try:
            pipeline = self.redis.pipeline()
            for key, value in mapping.items():
                pipeline.set(key, json.dumps(value), ex=expire)
            await pipeline.execute()
            return True
        except Exception as e:
            logger.error(f"Error setting multiple cache keys: {e}")
            return False

    async def delete_many(self, keys: List[str]) -> bool:
        """Delete multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            True if successful
        """
        try:
            await self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Error deleting multiple cache keys: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all values from cache.

        Returns:
            True if successful
        """
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def cached(self, expire: Optional[int] = None, key_prefix: str = ""):
        """Decorator for caching function results.

        Args:
            expire: Expiration time in seconds
            key_prefix: Prefix for cache keys

        Returns:
            Decorated function
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

                # Try to get from cache
                cached_value = await self.get(key)
                if cached_value is not None:
                    return cached_value

                # Call function and cache result
                result = await func(*args, **kwargs)
                await self.set(key, result, expire)
                return result

            return wrapper

        return decorator

    def cached_binary(self, expire: Optional[int] = None, key_prefix: str = ""):
        """Decorator for caching binary function results.

        Args:
            expire: Expiration time in seconds
            key_prefix: Prefix for cache keys

        Returns:
            Decorated function
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

                # Try to get from cache
                cached_value = await self.get_binary(key)
                if cached_value is not None:
                    return cached_value

                # Call function and cache result
                result = await func(*args, **kwargs)
                await self.set_binary(key, result, expire)
                return result

            return wrapper

        return decorator
